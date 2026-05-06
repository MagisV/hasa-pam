#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Dates
using Printf
using Random
using CairoMakie
using JLD2
using JSON3
using TranscranialFUS

function parse_cli(args)
    opts = Dict{String, String}(
        "source-model" => "squiggle",
        "sources-mm" => "30:0",
        "anchors-mm" => "45:0",
        "frequency-mhz" => "0.4",
        "fundamental-mhz" => "0.5",
        "amplitude-pa" => "1.0",
        "source-amplitudes-pa" => "",
        "source-frequencies-mhz" => "",
        "phases-deg" => "",
        "n-bubbles" => "10",
        "num-cycles" => "4",
        "harmonics" => "2,3,4",
        "harmonic-amplitudes" => "1.0,0.6,0.3",
        "cavitation-model" => "harmonic-cos",
        "gate-us" => "50",
        "taper-ratio" => "0.25",
        "axial-mm" => "80",
        "transverse-mm" => "102.4",
        "dx-mm" => "0.2",
        "dz-mm" => "0.2",
        "receiver-aperture-mm" => "full",
        "t-max-us" => "500",
        "dt-ns" => "20",
        "zero-pad-factor" => "4",
        "peak-suppression-radius-mm" => "8.0",
        "success-tolerance-mm" => "1.5",
        "aberrator" => "none",
        "ct-path" => DEFAULT_CT_PATH,
        "slice-index" => "250",
        "skull-transducer-distance-mm" => "30",
        "bottom-margin-mm" => "10",
        "hu-bone-thr" => "200",
        "lens-depth-mm" => "12",
        "lens-lateral-mm" => "0",
        "lens-axial-radius-mm" => "3",
        "lens-lateral-radius-mm" => "12",
        "aberrator-c" => "1700",
        "aberrator-rho" => "1150",
        "use-gpu" => "false",
        "recon-bandwidth-khz" => "500",
        "recon-step-um" => "50",
        "recon-mode" => "auto",
        "recon-window-us" => "20",
        "recon-hop-us" => "10",
        "recon-window-taper" => "hann",
        "recon-min-window-energy-ratio" => "0.001",
        "recon-progress" => "false",
        "phase-mode" => "geometric",
        "phase-jitter-rad" => "0.2",
        "random-seed" => "42",
        "source-phase-mode" => "random_phase_per_window",
        "n-realizations" => "1",
        "frequency-jitter-percent" => "1",
        "transducer-mm" => "-30:0",
        "delays-us" => "0",
        "vascular-length-mm" => "12",
        "vascular-squiggle-amplitude-mm" => "1.5",
        "vascular-squiggle-wavelength-mm" => "8",
        "vascular-squiggle-slope" => "0.0",
        "vascular-source-spacing-mm" => "0.8",
        "vascular-position-jitter-mm" => "0.15",
        "vascular-min-separation-mm" => "0.3",
        "vascular-max-sources-per-anchor" => "0",
        "vascular-radius-mm" => "1.0",
        "analysis-mode" => "auto",
        "detection-threshold-ratio" => "0.2",
        "boundary-threshold-ratios" => "0.6,0.65,0.7",
        "peak-method" => "argmax",
        "clean-loop-gain" => "0.1",
        "clean-max-iter" => "500",
        "clean-threshold-ratio" => "0.01",
        "from-run-dir" => "",
    )

    provided_keys = Set{String}()
    for arg in args
        startswith(arg, "--") || error("Unsupported argument format: $arg")
        parts = split(arg[3:end], "="; limit=2)
        length(parts) == 2 || error("Arguments must use --name=value, got: $arg")
        push!(provided_keys, parts[1])
        opts[parts[1]] = parts[2]
    end
    apply_model_defaults!(opts, provided_keys)
    return opts, provided_keys
end

slug_value(x; digits::Int=1) = replace(string(round(Float64(x); digits=digits)), "-" => "m", "." => "p")
parse_bool(s::AbstractString) = lowercase(strip(s)) in ("1", "true", "yes", "on")

function parse_float_list(spec::AbstractString)
    isempty(strip(spec)) && return Float64[]
    return [parse(Float64, strip(item)) for item in split(spec, ",") if !isempty(strip(item))]
end

function parse_int_list(spec::AbstractString)
    isempty(strip(spec)) && return Int[]
    return [parse(Int, strip(item)) for item in split(spec, ",") if !isempty(strip(item))]
end

function parse_threshold_ratios(spec::AbstractString)
    ratios = parse_float_list(spec)
    isempty(ratios) && error("At least one threshold ratio is required.")
    all(r -> r > 0, ratios) || error("Threshold ratios must be positive.")
    return sort(unique(ratios))
end

function parse_source_model(s::AbstractString)
    value = Symbol(lowercase(strip(s)))
    value in (:point, :squiggle) || error("--source-model must be point or squiggle, got: $s")
    return value
end

function apply_model_defaults!(opts, provided_keys::Set{String})
    source_model = parse_source_model(opts["source-model"])
    if source_model == :point
        !("source-phase-mode" in provided_keys) && (opts["source-phase-mode"] = "coherent")
        !("recon-bandwidth-khz" in provided_keys) && (opts["recon-bandwidth-khz"] = "0")
        !("receiver-aperture-mm" in provided_keys) && (opts["receiver-aperture-mm"] = "50")
        !("transverse-mm" in provided_keys) && (opts["transverse-mm"] = "60")
        !("dt-ns" in provided_keys) && (opts["dt-ns"] = "40")
        !("t-max-us" in provided_keys) && (opts["t-max-us"] = "60")
        !("axial-mm" in provided_keys) && (opts["axial-mm"] = "60")
        !("phase-mode" in provided_keys) && (opts["phase-mode"] = "coherent")
    end
    return opts
end

function parse_aberrator(s::AbstractString)
    value = Symbol(lowercase(strip(s)))
    value in (:none, :lens, :skull) || error("Unknown aberrator: $s")
    return value
end

function parse_cavitation_model(s::AbstractString)
    value = Symbol(replace(lowercase(strip(s)), "-" => "_"))
    value in (:harmonic_cos, :gaussian_pulse) ||
        error("--cavitation-model must be harmonic-cos or gaussian-pulse, got: $s")
    return value
end

function parse_source_phase_mode(s::AbstractString)
    value = Symbol(replace(lowercase(strip(s)), "-" => "_"))
    value in (:coherent, :random_static_phase, :random_phase_per_window, :random_phase_per_realization) ||
        error("--source-phase-mode must be coherent, random_static_phase, random_phase_per_window, or random_phase_per_realization, got: $s")
    return value
end

parse_source_variability(opts) = SourceVariabilityConfig(
    frequency_jitter_fraction=parse(Float64, opts["frequency-jitter-percent"]) / 100.0,
)

function source_variability_from_summary(summary)
    if isnothing(summary) || !hasproperty(summary, :source_variability)
        return SourceVariabilityConfig()
    end
    sv = summary.source_variability
    if hasproperty(sv, :frequency_jitter_percent)
        return SourceVariabilityConfig(frequency_jitter_fraction=Float64(sv.frequency_jitter_percent) / 100.0)
    end
    return SourceVariabilityConfig()
end

function parse_analysis_mode(s::AbstractString, source_model::Symbol)
    value = Symbol(lowercase(strip(s)))
    value == :auto && return source_model == :squiggle ? :detection : :localization
    value in (:localization, :detection) || error("--analysis-mode must be auto, localization, or detection, got: $s")
    return value
end

resolve_reconstruction_mode(s::AbstractString, source_model::Symbol) =
    TranscranialFUS.pam_reconstruction_mode(s, source_model)

function parse_window_taper(s::AbstractString)
    value = Symbol(replace(lowercase(strip(s)), "-" => "_"))
    value in (:hann, :none, :rect, :rectangular, :tukey) ||
        error("--recon-window-taper must be hann, none, rectangular, or tukey, got: $s")
    return value
end

function make_window_config(opts, reconstruction_mode::Symbol)
    return PAMWindowConfig(
        enabled=reconstruction_mode == :windowed,
        window_duration=parse(Float64, opts["recon-window-us"]) * 1e-6,
        hop=parse(Float64, opts["recon-hop-us"]) * 1e-6,
        taper=parse_window_taper(opts["recon-window-taper"]),
        min_energy_ratio=parse(Float64, opts["recon-min-window-energy-ratio"]),
        accumulation=:intensity,
    )
end

function parse_receiver_aperture_mm(s::AbstractString)
    value = lowercase(strip(s))
    value in ("none", "full", "all") && return nothing
    return parse(Float64, value) * 1e-3
end

function parse_transducer_mm(s::AbstractString)
    parts = split(strip(s), ":"; limit=2)
    length(parts) == 2 || error("--transducer-mm must be depth_mm:lateral_mm, got: $s")
    return parse(Float64, strip(parts[1])) * 1e-3, parse(Float64, strip(parts[2])) * 1e-3
end

point_pairs_m(points) = [[point[1], point[2]] for point in points]
centerlines_m(centerlines) = [point_pairs_m(line) for line in centerlines]

function expand_source_values(values::Vector{Float64}, n::Int, default::Float64)
    isempty(values) && return fill(default, n)
    length(values) == 1 && return fill(values[1], n)
    length(values) == n && return values
    error("Per-source parameter list must have length 1 or match the number of sources ($n).")
end

function parse_coordinate_pairs_mm(spec::AbstractString, option_name::AbstractString)
    coord_tokens = [strip(token) for token in split(spec, ",") if !isempty(strip(token))]
    1 <= length(coord_tokens) <= 20 || error("Provide between 1 and 20 coordinates via --$option_name=depth:lateral,...")
    pairs = Tuple{Float64, Float64}[]
    for token in coord_tokens
        parts = split(token, ":"; limit=2)
        length(parts) == 2 || error("Each coordinate must be depth_mm:lateral_mm, got: $token")
        push!(pairs, (parse(Float64, strip(parts[1])) * 1e-3, parse(Float64, strip(parts[2])) * 1e-3))
    end
    return pairs
end

function parse_point_sources(opts)
    coordinates = parse_coordinate_pairs_mm(opts["sources-mm"], "sources-mm")
    n_sources = length(coordinates)
    frequencies_mhz = expand_source_values(
        parse_float_list(opts["source-frequencies-mhz"]),
        n_sources,
        parse(Float64, opts["frequency-mhz"]),
    )
    amplitudes_pa = expand_source_values(
        parse_float_list(opts["source-amplitudes-pa"]),
        n_sources,
        parse(Float64, opts["amplitude-pa"]),
    )
    phases_deg = expand_source_values(parse_float_list(opts["phases-deg"]), n_sources, 0.0)
    delays_us = expand_source_values(parse_float_list(opts["delays-us"]), n_sources, 0.0)
    num_cycles = parse(Int, opts["num-cycles"])

    phase_mode = lowercase(strip(opts["phase-mode"]))
    phase_mode in ("coherent", "random", "jittered") ||
        error("Point --phase-mode must be coherent, random, or jittered, got: $phase_mode")
    rng = Random.MersenneTwister(parse(Int, opts["random-seed"]))
    if phase_mode == "random"
        phases_deg = rand(rng, n_sources) .* 360.0
    elseif phase_mode == "jittered"
        phases_deg = phases_deg .+ randn(rng, n_sources) .* (parse(Float64, opts["phase-jitter-rad"]) * 180 / pi)
    end

    sources = EmissionSource2D[]
    for (idx, (depth_m, lateral_m)) in pairs(coordinates)
        push!(sources, PointSource2D(
            depth=depth_m,
            lateral=lateral_m,
            frequency=frequencies_mhz[idx] * 1e6,
            amplitude=amplitudes_pa[idx],
            phase=phases_deg[idx] * pi / 180,
            delay=delays_us[idx] * 1e-6,
            num_cycles=num_cycles,
        ))
    end
    return sources, Dict{String, Any}(
        "source_model" => "point",
        "coordinates_m" => [collect(coord) for coord in coordinates],
        "n_coordinate_sources" => n_sources,
        "n_emission_sources" => length(sources),
        "physical_source_count" => length(sources),
        "emission_event_count" => length(sources),
        "activity_model" => Dict("activity_mode" => "point_tone_burst"),
        "phase_mode" => phase_mode,
        "frequencies_hz" => frequencies_mhz .* 1e6,
        "amplitudes_pa" => amplitudes_pa,
        "phases_rad" => phases_deg .* pi ./ 180,
        "delays_s" => delays_us .* 1e-6,
        "num_cycles" => num_cycles,
        "random_seed" => parse(Int, opts["random-seed"]),
    )
end

function parse_squiggle_sources(opts, cfg::PAMConfig)
    anchors = parse_coordinate_pairs_mm(opts["anchors-mm"], "anchors-mm")
    f0 = parse(Float64, opts["fundamental-mhz"]) * 1e6
    harmonics = parse_int_list(opts["harmonics"])
    isempty(harmonics) && error("--harmonics must be a non-empty integer list.")
    harmonic_amplitudes = parse_float_list(opts["harmonic-amplitudes"])
    length(harmonic_amplitudes) == length(harmonics) ||
        error("--harmonic-amplitudes must have the same length as --harmonics ($(length(harmonics))).")

    n_anchors = length(anchors)
    n_bubbles_per = expand_source_values(parse_float_list(opts["n-bubbles"]), n_anchors, 10.0)
    delays_us = expand_source_values(parse_float_list(opts["delays-us"]), n_anchors, 0.0)
    tx_depth, tx_lateral = parse_transducer_mm(opts["transducer-mm"])
    max_sources_raw = parse(Int, opts["vascular-max-sources-per-anchor"])
    max_sources = max_sources_raw <= 0 ? nothing : max_sources_raw
    phase_mode = Symbol(replace(lowercase(strip(opts["phase-mode"])), "-" => "_"))

    sources = EmissionSource2D[]
    all_centerlines_m = Any[]
    anchors_meta = Dict{String, Any}[]
    rng = Random.MersenneTwister(parse(Int, opts["random-seed"]))
    for (idx, anchor) in pairs(anchors)
        anchor_sources, anchor_meta = make_squiggle_bubble_sources(
            [anchor];
            root_length=parse(Float64, opts["vascular-length-mm"]) * 1e-3,
            squiggle_amplitude=parse(Float64, opts["vascular-squiggle-amplitude-mm"]) * 1e-3,
            squiggle_wavelength=parse(Float64, opts["vascular-squiggle-wavelength-mm"]) * 1e-3,
            squiggle_slope=parse(Float64, opts["vascular-squiggle-slope"]),
            source_spacing=parse(Float64, opts["vascular-source-spacing-mm"]) * 1e-3,
            position_jitter=parse(Float64, opts["vascular-position-jitter-mm"]) * 1e-3,
            min_separation=parse(Float64, opts["vascular-min-separation-mm"]) * 1e-3,
            max_sources_per_anchor=max_sources,
            depth_bounds=(0.0, Inf),
            lateral_bounds=(-cfg.transverse_dim / 2, cfg.transverse_dim / 2),
            fundamental=f0,
            amplitude=parse(Float64, opts["amplitude-pa"]),
            n_bubbles=n_bubbles_per[idx],
            harmonics=harmonics,
            harmonic_amplitudes=harmonic_amplitudes,
            cavitation_model=parse_cavitation_model(opts["cavitation-model"]),
            gate_duration=parse(Float64, opts["gate-us"]) * 1e-6,
            taper_ratio=parse(Float64, opts["taper-ratio"]),
            delay=delays_us[idx] * 1e-6,
            phase_mode=phase_mode,
            phase_jitter=parse(Float64, opts["phase-jitter-rad"]),
            transducer_depth=tx_depth,
            transducer_lateral=tx_lateral,
            c0=cfg.c0,
            rng=rng,
        )
        append!(sources, anchor_sources)
        anchor_centerlines = centerlines_m(anchor_meta[:centerlines])
        append!(all_centerlines_m, anchor_centerlines)
        push!(anchors_meta, Dict(
            "anchor_m" => collect(anchor),
            "source_count" => length(anchor_sources),
            "centerlines_m" => anchor_centerlines,
        ))
    end

    return sources, Dict{String, Any}(
        "source_model" => "squiggle",
        "anchor_clusters_m" => [collect(anchor) for anchor in anchors],
        "n_anchor_clusters" => length(anchors),
        "n_emission_sources" => length(sources),
        "physical_source_count" => length(sources),
        "emission_event_count" => length(sources),
        "activity_model" => Dict("activity_mode" => "random_phase_per_window"),
        "phase_mode" => String(phase_mode),
        "fundamental_hz" => f0,
        "harmonics" => harmonics,
        "harmonic_amplitudes" => harmonic_amplitudes,
        "cavitation_model" => opts["cavitation-model"],
        "gate_duration_s" => parse(Float64, opts["gate-us"]) * 1e-6,
        "transducer_m" => [tx_depth, tx_lateral],
        "phase_jitter_rad" => parse(Float64, opts["phase-jitter-rad"]),
        "random_seed" => parse(Int, opts["random-seed"]),
        "n_bubbles_per_cluster" => n_bubbles_per,
        "delays_s" => delays_us .* 1e-6,
        "squiggle" => Dict(
            "length_m" => parse(Float64, opts["vascular-length-mm"]) * 1e-3,
            "squiggle_amplitude_m" => parse(Float64, opts["vascular-squiggle-amplitude-mm"]) * 1e-3,
            "squiggle_wavelength_m" => parse(Float64, opts["vascular-squiggle-wavelength-mm"]) * 1e-3,
            "squiggle_slope" => parse(Float64, opts["vascular-squiggle-slope"]),
            "source_spacing_m" => parse(Float64, opts["vascular-source-spacing-mm"]) * 1e-3,
            "position_jitter_m" => parse(Float64, opts["vascular-position-jitter-mm"]) * 1e-3,
            "min_separation_m" => parse(Float64, opts["vascular-min-separation-mm"]) * 1e-3,
            "max_sources_per_anchor" => max_sources_raw,
            "truth_radius_m" => parse(Float64, opts["vascular-radius-mm"]) * 1e-3,
            "centerlines_m" => all_centerlines_m,
            "anchors" => anchors_meta,
        ),
    )
end

function parse_sources(opts, cfg::PAMConfig)
    source_model = parse_source_model(opts["source-model"])
    return source_model == :point ? parse_point_sources(opts) : parse_squiggle_sources(opts, cfg)
end

function default_output_dir(opts, sources, cfg, emission_meta)
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    source_model = lowercase(String(emission_meta["source_model"]))
    parts = String[
        timestamp,
        "run_pam",
        lowercase(opts["aberrator"]),
        source_model,
        "$(length(sources))src",
        "ax$(slug_value(cfg.axial_dim * 1e3; digits=0))mm",
        "lat$(slug_value(cfg.transverse_dim * 1e3; digits=0))mm",
    ]
    if source_model == "squiggle"
        insert!(parts, 5, "$(emission_meta["n_anchor_clusters"])anchors")
        push!(parts, "f$(slug_value(parse(Float64, opts["fundamental-mhz"]); digits=2))mhz")
        push!(parts, "h$(replace(opts["harmonics"], "," => ""))")
        push!(parts, replace(lowercase(opts["source-phase-mode"]), "_" => ""))
    else
        push!(parts, "f$(slug_value(parse(Float64, opts["frequency-mhz"]); digits=2))mhz")
    end
    if lowercase(opts["aberrator"]) == "skull"
        insert!(parts, length(parts), "slice" * opts["slice-index"])
        insert!(parts, length(parts), "st$(slug_value(parse(Float64, opts["skull-transducer-distance-mm"]); digits=1))mm")
    end
    return joinpath(pwd(), "outputs", join(parts, "_"))
end

function default_reconstruction_output_dir(source_dir::AbstractString)
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    source_name = basename(normpath(source_dir))
    return joinpath(pwd(), "outputs", "$(timestamp)_reconstruct_$(source_name)")
end

function reject_cached_simulation_options!(provided_keys::Set{String}, blocked_keys)
    illegal = sort(collect(intersect(provided_keys, Set(blocked_keys))))
    isempty(illegal) && return nothing
    formatted = join(["--$key" for key in illegal], ", ")
    error("--from-run-dir reuses the previous RF simulation, medium, sources, and grid. Remove simulation-specific option(s): $formatted")
end

function default_simulation_info(cfg::PAMConfig)
    return Dict{Symbol, Any}(
        :receiver_row => receiver_row(cfg),
        :receiver_cols => receiver_col_range(cfg),
        :source_indices => Tuple{Int, Int}[],
    )
end

function default_recon_frequencies(sources)
    freqs = Float64[]
    for src in sources
        append!(freqs, emission_frequencies(src))
    end
    return sort(unique(freqs))
end

function json3_to_any(x)
    if x isa JSON3.Object
        return Dict{String, Any}(String(k) => json3_to_any(v) for (k, v) in pairs(x))
    elseif x isa JSON3.Array
        return Any[json3_to_any(v) for v in x]
    else
        return x
    end
end

function source_model_from_meta(meta, sources)
    if haskey(meta, "source_model")
        model = Symbol(String(meta["source_model"]))
        model == :vascular && return :squiggle
        return model
    end
    if haskey(meta, "cluster_model")
        old = Symbol(String(meta["cluster_model"]))
        return old == :vascular ? :squiggle : old
    end
    return any(src -> src isa Union{BubbleCluster2D, GaussianPulseCluster2D}, sources) ? :squiggle : :point
end

function centerlines_from_emission_meta(meta)
    key = haskey(meta, "squiggle") ? "squiggle" : (haskey(meta, "vascular") ? "vascular" : "")
    isempty(key) && return nothing
    block = meta[key]
    haskey(block, "centerlines_m") || return nothing
    centerlines = Vector{Tuple{Float64, Float64}}[]
    for raw_line in block["centerlines_m"]
        line = Tuple{Float64, Float64}[]
        for point in raw_line
            push!(line, (Float64(point[1]), Float64(point[2])))
        end
        length(line) >= 2 && push!(centerlines, line)
    end
    return isempty(centerlines) ? nothing : centerlines
end

function detection_truth_mask_from_meta(meta, kgrid, cfg, radius::Real)
    centerlines = centerlines_from_emission_meta(meta)
    isnothing(centerlines) && return nothing
    return pam_centerline_truth_mask(centerlines, kgrid, cfg; radius=radius)
end

function map_db(map::AbstractMatrix{<:Real}, ref::Real)
    safe_ref = max(Float64(ref), eps(Float64))
    return 10 .* log10.(max.(Float64.(map), eps(Float64)) ./ safe_ref)
end

function source_pairs_mm(sources)
    return [(src.depth * 1e3, src.lateral * 1e3) for src in sources]
end

function scatter_sources!(ax, sources; color=:red, marker=nothing, markersize=nothing, strokewidth=2)
    truth = source_pairs_mm(sources)
    marker = isnothing(marker) ? (length(sources) > 20 ? :circle : :x) : marker
    markersize = isnothing(markersize) ? (length(sources) > 20 ? 4 : 14) : markersize
    scatter!(
        ax,
        last.(truth),
        first.(truth);
        color=color,
        marker=marker,
        markersize=markersize,
        strokewidth=strokewidth,
    )
end

function save_overview(path, c, rf, pam_geo, pam_hasa, kgrid, cfg, sources, stats_geo, stats_hasa)
    depth_mm = depth_coordinates(kgrid, cfg) .* 1e3
    lateral_mm = kgrid.y_vec .* 1e3
    time_us = collect(0:(size(rf, 2) - 1)) .* cfg.dt .* 1e6
    map_ref = max(maximum(Float64.(pam_geo)), maximum(Float64.(pam_hasa)), eps(Float64))

    fig = Figure(size=(1500, 1000))
    ax_medium = Axis(fig[1, 1]; title="Simulation Medium", xlabel="Lateral position [mm]", ylabel="Depth below receiver [mm]", aspect=DataAspect())
    hm_medium = heatmap!(ax_medium, lateral_mm, depth_mm, Float64.(c)'; colormap=:thermal)
    hlines!(ax_medium, [0.0]; color=:white, linestyle=:dash)
    scatter_sources!(ax_medium, sources)
    Colorbar(fig[1, 2], hm_medium; label="Sound speed [m/s]")

    ax_rf = Axis(fig[1, 3]; title="Recorded RF Data", xlabel="Time [us]", ylabel="Lateral position [mm]")
    rf_ref = max(maximum(abs.(rf)), eps(Float64))
    hm_rf = heatmap!(ax_rf, time_us, lateral_mm, Float64.(rf ./ rf_ref)'; colormap=:balance, colorrange=(-1, 1))
    Colorbar(fig[1, 4], hm_rf; label="Norm. pressure")

    ax_geo = Axis(fig[2, 1]; title="Geometric ASA PAM", xlabel="Lateral position [mm]", ylabel="Depth below receiver [mm]", aspect=DataAspect())
    hm_geo = heatmap!(ax_geo, lateral_mm, depth_mm, map_db(pam_geo, map_ref)'; colormap=:viridis, colorrange=(-30, 0))
    scatter_sources!(ax_geo, sources)
    Colorbar(fig[2, 2], hm_geo; label="dB")

    ax_hasa = Axis(fig[2, 3]; title="Corrected HASA PAM", xlabel="Lateral position [mm]", ylabel="Depth below receiver [mm]", aspect=DataAspect())
    hm_hasa = heatmap!(ax_hasa, lateral_mm, depth_mm, map_db(pam_hasa, map_ref)'; colormap=:viridis, colorrange=(-30, 0))
    scatter_sources!(ax_hasa, sources)
    Colorbar(fig[2, 4], hm_hasa; label="dB")

    metrics_text = join([
        "Geometric: $(summary_line(stats_geo))",
        "Corrected: $(summary_line(stats_hasa))",
    ], "\n")
    Label(fig[3, 1:4], metrics_text; tellwidth=false, halign=:left)
    save(path, fig)
end

function summary_line(stats)
    if haskey(stats, :mean_radial_error_mm)
        return "mean err=$(round(stats[:mean_radial_error_mm]; digits=2)) mm, success=$(get(stats, :num_success, 0))/$(get(stats, :num_truth_sources, 0))"
    elseif haskey(stats, :f1)
        return "F1=$(round(stats[:f1]; digits=3)), precision=$(round(stats[:precision]; digits=3)), recall=$(round(stats[:recall]; digits=3))"
    end
    return string(stats)
end

string_key_dict(d::AbstractDict) = Dict(String(k) => v for (k, v) in d)

function threshold_detection_stats(intensity, kgrid, cfg, sources; threshold_ratios, truth_radius, truth_mask, frequencies)
    return [
        merge(
            Dict(:threshold_ratio => ratio),
            analyse_pam_detection_2d(
                intensity,
                kgrid,
                cfg,
                sources;
                truth_radius=truth_radius,
                threshold_ratio=ratio,
                truth_mask=truth_mask,
                frequencies=frequencies,
            ),
        )
        for ratio in threshold_ratios
    ]
end

function lines_centerlines!(ax, centerlines; color=(:black, 0.45), linewidth=2)
    isnothing(centerlines) && return nothing
    for line in centerlines
        length(line) >= 2 || continue
        lines!(ax, [point[2] * 1e3 for point in line], [point[1] * 1e3 for point in line]; color=color, linewidth=linewidth)
    end
    return nothing
end

function add_threshold_panel!(
    fig,
    row,
    title,
    intensity,
    kgrid,
    cfg,
    sources;
    threshold_ratios,
    colors,
    global_ref,
    truth_mask,
    truth_centerlines,
)
    depth_mm = depth_coordinates(kgrid, cfg) .* 1e3
    lateral_mm = kgrid.y_vec .* 1e3
    ax = Axis(fig[row, 1]; title=title, xlabel="Lateral [mm]", ylabel="Depth [mm]", aspect=DataAspect())
    hm = heatmap!(ax, lateral_mm, depth_mm, Float64.(intensity ./ global_ref)'; colormap=:viridis, colorrange=(0, 1))
    if !isnothing(truth_mask) && any(truth_mask) && any(.!truth_mask)
        contour!(ax, lateral_mm, depth_mm, Float64.(truth_mask)'; levels=[0.5], color=(:white, 0.85), linewidth=2.3, linestyle=:dash)
    end
    lines_centerlines!(ax, truth_centerlines; color=(:white, 0.7), linewidth=1.3)
    local_ref = max(maximum(Float64.(intensity)), eps(Float64))
    for (idx, ratio) in pairs(threshold_ratios)
        contour!(ax, lateral_mm, depth_mm, Float64.(intensity .>= ratio * local_ref)'; levels=[0.5], color=colors[idx], linewidth=2)
    end
    scatter_sources!(ax, sources; color=(:white, 0.55), marker=:circle, markersize=2.5, strokewidth=0)
    return hm
end

function add_threshold_table!(fig, row, col, title, stats)
    lines = ["thr    F1    Prec  Recall  Jacc"]
    for entry in stats
        push!(lines, @sprintf("%.2f  %.3f  %.3f  %.3f  %.3f",
            Float64(entry[:threshold_ratio]),
            Float64(entry[:f1]),
            Float64(entry[:precision]),
            Float64(entry[:recall]),
            Float64(entry[:jaccard]),
        ))
    end
    Label(fig[row, col], title * "\n" * join(lines, "\n"); font="DejaVu Sans Mono", tellwidth=false, halign=:left)
end

function save_threshold_boundary_detection(path, pam_geo, pam_hasa, kgrid, cfg, sources; threshold_ratios, truth_radius, truth_mask, truth_centerlines, frequencies)
    global_ref = max(maximum(Float64.(pam_geo)), maximum(Float64.(pam_hasa)), eps(Float64))
    colors = [:red, :orange, :cyan, :magenta, :lime]
    while length(colors) < length(threshold_ratios)
        append!(colors, colors)
    end
    geo_stats = threshold_detection_stats(pam_geo, kgrid, cfg, sources; threshold_ratios=threshold_ratios, truth_radius=truth_radius, truth_mask=truth_mask, frequencies=frequencies)
    hasa_stats = threshold_detection_stats(pam_hasa, kgrid, cfg, sources; threshold_ratios=threshold_ratios, truth_radius=truth_radius, truth_mask=truth_mask, frequencies=frequencies)

    fig = Figure(size=(1000, 1300))
    hm = add_threshold_panel!(
        fig,
        1,
        "Uncorrected activity regions",
        pam_geo,
        kgrid,
        cfg,
        sources;
        threshold_ratios=threshold_ratios,
        colors=colors,
        global_ref=global_ref,
        truth_mask=truth_mask,
        truth_centerlines=truth_centerlines,
    )
    add_threshold_panel!(
        fig,
        2,
        "Corrected activity regions",
        pam_hasa,
        kgrid,
        cfg,
        sources;
        threshold_ratios=threshold_ratios,
        colors=colors,
        global_ref=global_ref,
        truth_mask=truth_mask,
        truth_centerlines=truth_centerlines,
    )
    Colorbar(fig[1:2, 2], hm; label="Norm. PAM intensity")
    legend_elements = [LineElement(color=colors[i], linewidth=3) for i in eachindex(threshold_ratios)]
    legend_labels = ["thr=$(round(r; digits=2))" for r in threshold_ratios]
    Legend(fig[3, 1], legend_elements, legend_labels; orientation=:horizontal, tellheight=true, framevisible=false)
    add_threshold_table!(fig, 4, 1, "Uncorrected quantitative region metrics", geo_stats)
    add_threshold_table!(fig, 5, 1, "Corrected quantitative region metrics", hasa_stats)
    save(path, fig)
    return Dict(
        "threshold_ratios" => threshold_ratios,
        "geometric" => [string_key_dict(stats) for stats in geo_stats],
        "hasa" => [string_key_dict(stats) for stats in hasa_stats],
    )
end

function compact_window_info(info)
    haskey(info, :used_window_count) || return nothing
    range_pairs(ranges) = [[first(range), last(range)] for range in ranges]
    return Dict(
        "total_window_count" => info[:total_window_count],
        "used_window_count" => info[:used_window_count],
        "skipped_window_count" => info[:skipped_window_count],
        "window_samples" => info[:window_samples],
        "hop_samples" => info[:hop_samples],
        "effective_window_duration_s" => info[:effective_window_duration_s],
        "effective_hop_s" => info[:effective_hop_s],
        "energy_threshold" => info[:energy_threshold],
        "used_window_ranges" => range_pairs(info[:used_window_ranges]),
        "skipped_window_ranges" => range_pairs(info[:skipped_window_ranges]),
        "accumulation" => String(info[:accumulation]),
    )
end

function source_summary(src::PointSource2D)
    return Dict(
        "kind" => "point",
        "depth_m" => src.depth,
        "lateral_m" => src.lateral,
        "frequency_hz" => src.frequency,
        "amplitude_pa" => src.amplitude,
        "phase_rad" => src.phase,
        "delay_s" => src.delay,
        "num_cycles" => src.num_cycles,
    )
end

function source_summary(src::Union{BubbleCluster2D, GaussianPulseCluster2D})
    return Dict(
        "kind" => String(cavitation_model(src)),
        "depth_m" => src.depth,
        "lateral_m" => src.lateral,
        "fundamental_hz" => src.fundamental,
        "amplitude_pa" => src.amplitude,
        "n_bubbles" => src.n_bubbles,
        "harmonics" => src.harmonics,
        "harmonic_amplitudes" => src.harmonic_amplitudes,
        "harmonic_phases_rad" => src.harmonic_phases,
        "cavitation_model" => String(cavitation_model(src)),
        "gate_duration_s" => src.gate_duration,
        "delay_s" => src.delay,
    )
end

opts, provided_keys = parse_cli(ARGS)
source_model = parse_source_model(opts["source-model"])
from_run_dir = strip(opts["from-run-dir"])
peak_method = Symbol(lowercase(strip(opts["peak-method"])))
peak_method in (:argmax, :clean) || error("--peak-method must be argmax or clean, got: $(opts["peak-method"])")
detection_truth_radius_m = parse(Float64, opts["vascular-radius-mm"]) * 1e-3
detection_threshold_ratio = parse(Float64, opts["detection-threshold-ratio"])
boundary_threshold_ratios = parse_threshold_ratios(opts["boundary-threshold-ratios"])

if isempty(from_run_dir)
    cfg_base = PAMConfig(
        dx=parse(Float64, opts["dx-mm"]) * 1e-3,
        dz=parse(Float64, opts["dz-mm"]) * 1e-3,
        axial_dim=parse(Float64, opts["axial-mm"]) * 1e-3,
        transverse_dim=parse(Float64, opts["transverse-mm"]) * 1e-3,
        receiver_aperture=parse_receiver_aperture_mm(opts["receiver-aperture-mm"]),
        t_max=parse(Float64, opts["t-max-us"]) * 1e-6,
        dt=parse(Float64, opts["dt-ns"]) * 1e-9,
        zero_pad_factor=parse(Int, opts["zero-pad-factor"]),
        peak_suppression_radius=parse(Float64, opts["peak-suppression-radius-mm"]) * 1e-3,
        success_tolerance=parse(Float64, opts["success-tolerance-mm"]) * 1e-3,
    )

    sources, emission_meta = parse_sources(opts, cfg_base)
    source_model = source_model_from_meta(emission_meta, sources)

    aberrator = parse_aberrator(opts["aberrator"])
    bottom_margin_m = parse(Float64, opts["bottom-margin-mm"]) * 1e-3
    cfg = fit_pam_config(
        cfg_base,
        sources;
        min_bottom_margin=bottom_margin_m,
        reference_depth=aberrator == :skull ? parse(Float64, opts["skull-transducer-distance-mm"]) * 1e-3 : nothing,
    )

    out_dir = if haskey(opts, "out-dir") && !isempty(strip(opts["out-dir"]))
        opts["out-dir"]
    else
        default_output_dir(opts, sources, cfg, emission_meta)
    end
    mkpath(out_dir)

    c, rho, medium_info = make_pam_medium(
        cfg;
        aberrator=aberrator,
        lens_center_depth=parse(Float64, opts["lens-depth-mm"]) * 1e-3,
        lens_center_lateral=parse(Float64, opts["lens-lateral-mm"]) * 1e-3,
        lens_axial_radius=parse(Float64, opts["lens-axial-radius-mm"]) * 1e-3,
        lens_lateral_radius=parse(Float64, opts["lens-lateral-radius-mm"]) * 1e-3,
        c_aberrator=parse(Float64, opts["aberrator-c"]),
        rho_aberrator=parse(Float64, opts["aberrator-rho"]),
        ct_path=opts["ct-path"],
        slice_index=parse(Int, opts["slice-index"]),
        skull_to_transducer=parse(Float64, opts["skull-transducer-distance-mm"]) * 1e-3,
        hu_bone_thr=parse(Int, opts["hu-bone-thr"]),
    )

    recon_frequencies = if haskey(opts, "recon-frequencies-mhz") && !isempty(strip(opts["recon-frequencies-mhz"]))
        parse_float_list(opts["recon-frequencies-mhz"]) .* 1e6
    else
        default_recon_frequencies(sources)
    end
    reconstruction_mode = resolve_reconstruction_mode(opts["recon-mode"], source_model)
    recon_bandwidth_hz = parse(Float64, opts["recon-bandwidth-khz"]) * 1e3
    window_config = make_window_config(opts, reconstruction_mode)
    analysis_mode = parse_analysis_mode(opts["analysis-mode"], source_model)
    truth_centerlines = centerlines_from_emission_meta(emission_meta)
    detection_truth_mask = detection_truth_mask_from_meta(emission_meta, pam_grid(cfg), cfg, detection_truth_radius_m)

    source_phase_mode = parse_source_phase_mode(opts["source-phase-mode"])
    n_realizations = parse(Int, opts["n-realizations"])
    rng_sim = Random.MersenneTwister(parse(Int, opts["random-seed"]) + 1)
    source_variability = parse_source_variability(opts)
    if source_model == :squiggle
        emission_meta["activity_model"] = Dict(
            "activity_mode" => String(source_phase_mode),
            "frequency_jitter_percent" => source_variability.frequency_jitter_fraction * 100.0,
        )
    end

    results = run_pam_case(
        c,
        rho,
        sources,
        cfg;
        frequencies=recon_frequencies,
        bandwidth=recon_bandwidth_hz,
        use_gpu=parse_bool(opts["use-gpu"]),
        reconstruction_axial_step=parse(Float64, opts["recon-step-um"]) * 1e-6,
        analysis_mode=analysis_mode,
        peak_method=peak_method,
        clean_loop_gain=parse(Float64, opts["clean-loop-gain"]),
        clean_max_iter=parse(Int, opts["clean-max-iter"]),
        clean_threshold_ratio=parse(Float64, opts["clean-threshold-ratio"]),
        detection_truth_radius=detection_truth_radius_m,
        detection_threshold_ratio=detection_threshold_ratio,
        detection_truth_mask=detection_truth_mask,
        reconstruction_mode=reconstruction_mode,
        window_config=window_config,
        source_phase_mode=source_phase_mode,
        n_realizations=n_realizations,
        rng=rng_sim,
        source_variability=source_variability,
        show_progress=parse_bool(opts["recon-progress"]),
    )
    reconstruction_source = Dict("mode" => "simulation")
else
    reject_cached_simulation_options!(
        provided_keys,
        (
            "source-model", "sources-mm", "anchors-mm", "frequency-mhz", "fundamental-mhz",
            "amplitude-pa", "source-amplitudes-pa", "source-frequencies-mhz", "phases-deg",
            "n-bubbles", "num-cycles", "harmonics", "harmonic-amplitudes", "cavitation-model",
            "gate-us", "taper-ratio", "axial-mm", "transverse-mm", "dx-mm", "dz-mm",
            "receiver-aperture-mm", "t-max-us", "dt-ns", "zero-pad-factor",
            "peak-suppression-radius-mm", "success-tolerance-mm", "aberrator", "ct-path",
            "slice-index", "skull-transducer-distance-mm", "bottom-margin-mm", "hu-bone-thr",
            "lens-depth-mm", "lens-lateral-mm", "lens-axial-radius-mm", "lens-lateral-radius-mm",
            "aberrator-c", "aberrator-rho", "phase-mode", "phase-jitter-rad", "random-seed",
            "transducer-mm", "delays-us", "vascular-length-mm", "vascular-squiggle-amplitude-mm",
            "vascular-squiggle-wavelength-mm", "vascular-squiggle-slope",
            "vascular-source-spacing-mm", "vascular-position-jitter-mm",
            "vascular-min-separation-mm", "vascular-max-sources-per-anchor",
            "source-phase-mode", "n-realizations", "frequency-jitter-percent",
        ),
    )
    cached_path = joinpath(from_run_dir, "result.jld2")
    isfile(cached_path) || error("--from-run-dir must contain result.jld2, missing: $cached_path")
    cached = load(cached_path)
    c = cached["c"]
    rho = haskey(cached, "rho") ? cached["rho"] : fill(Float32(cached["cfg"].rho0), size(c))
    cfg = cached["cfg"]
    sources = cached["clusters"]
    cached_results = cached["results"]
    rf = cached_results[:rf]
    medium_info = haskey(cached, "medium_info") ? cached["medium_info"] : Dict{Symbol, Any}(:aberrator => :cached)
    bottom_margin_m = nothing
    cached_summary_path = joinpath(from_run_dir, "summary.json")
    cached_summary = isfile(cached_summary_path) ? JSON3.read(read(cached_summary_path, String)) : nothing
    source_variability = source_variability_from_summary(cached_summary)
    emission_meta = if !isnothing(cached_summary) && hasproperty(cached_summary, :emission_meta)
        Dict{String, Any}(json3_to_any(cached_summary.emission_meta))
    else
        Dict{String, Any}(
            "source_model" => source_model_from_meta(Dict{String, Any}(), sources) |> String,
            "n_emission_sources" => length(sources),
        )
    end
    emission_meta["from_run_dir"] = abspath(from_run_dir)
    source_model = source_model_from_meta(emission_meta, sources)

    out_dir = if haskey(opts, "out-dir") && !isempty(strip(opts["out-dir"]))
        opts["out-dir"]
    else
        default_reconstruction_output_dir(from_run_dir)
    end
    mkpath(out_dir)

    recon_frequencies = if haskey(opts, "recon-frequencies-mhz") && !isempty(strip(opts["recon-frequencies-mhz"]))
        parse_float_list(opts["recon-frequencies-mhz"]) .* 1e6
    else
        default_recon_frequencies(sources)
    end
    reconstruction_mode = resolve_reconstruction_mode(opts["recon-mode"], source_model)
    recon_bandwidth_hz = parse(Float64, opts["recon-bandwidth-khz"]) * 1e3
    window_config = make_window_config(opts, reconstruction_mode)
    analysis_mode = parse_analysis_mode(opts["analysis-mode"], source_model)
    simulation_info = haskey(cached_results, :simulation) ? cached_results[:simulation] : default_simulation_info(cfg)
    truth_centerlines = centerlines_from_emission_meta(emission_meta)
    detection_truth_mask = detection_truth_mask_from_meta(emission_meta, pam_grid(cfg), cfg, detection_truth_radius_m)
    results = reconstruct_pam_case(
        rf,
        c,
        sources,
        cfg;
        simulation_info=simulation_info,
        frequencies=recon_frequencies,
        bandwidth=recon_bandwidth_hz,
        use_gpu=parse_bool(opts["use-gpu"]),
        reconstruction_axial_step=parse(Float64, opts["recon-step-um"]) * 1e-6,
        analysis_mode=analysis_mode,
        peak_method=peak_method,
        clean_loop_gain=parse(Float64, opts["clean-loop-gain"]),
        clean_max_iter=parse(Int, opts["clean-max-iter"]),
        clean_threshold_ratio=parse(Float64, opts["clean-threshold-ratio"]),
        detection_truth_radius=detection_truth_radius_m,
        detection_threshold_ratio=detection_threshold_ratio,
        detection_truth_mask=detection_truth_mask,
        reconstruction_mode=reconstruction_mode,
        window_config=window_config,
        show_progress=parse_bool(opts["recon-progress"]),
    )
    reconstruction_source = Dict(
        "mode" => "cached_rf",
        "from_run_dir" => abspath(from_run_dir),
        "from_result_jld2" => abspath(cached_path),
    )
end

medium_summary = Dict{String, Any}()
for (key, value) in medium_info
    key == :mask && continue
    medium_summary[String(key)] = value
end

save_overview(
    joinpath(out_dir, "overview.png"),
    c, results[:rf], results[:pam_geo], results[:pam_hasa],
    results[:kgrid], cfg, sources, results[:stats_geo], results[:stats_hasa],
)

activity_boundary_path = joinpath(out_dir, "activity_boundaries.png")
activity_boundary_metrics = save_threshold_boundary_detection(
    activity_boundary_path,
    results[:pam_geo],
    results[:pam_hasa],
    results[:kgrid],
    cfg,
    sources;
    threshold_ratios=boundary_threshold_ratios,
    truth_radius=detection_truth_radius_m,
    truth_mask=detection_truth_mask,
    truth_centerlines=truth_centerlines,
    frequencies=recon_frequencies,
)

summary = Dict(
    "out_dir" => out_dir,
    "reconstruction_source" => reconstruction_source,
    "activity_boundary_figure" => activity_boundary_path,
    "activity_boundary_metrics" => activity_boundary_metrics,
    "sources" => [source_summary(src) for src in sources],
    "clusters" => [source_summary(src) for src in sources],
    "emission_meta" => emission_meta,
    "config" => Dict(
        "dx" => cfg.dx,
        "dz" => cfg.dz,
        "axial_dim" => cfg.axial_dim,
        "transverse_dim" => cfg.transverse_dim,
        "receiver_aperture" => cfg.receiver_aperture,
        "t_max" => cfg.t_max,
        "dt" => cfg.dt,
        "c0" => cfg.c0,
        "rho0" => cfg.rho0,
        "zero_pad_factor" => cfg.zero_pad_factor,
        "peak_suppression_radius" => cfg.peak_suppression_radius,
        "success_tolerance" => cfg.success_tolerance,
        "bottom_margin" => bottom_margin_m,
    ),
    "medium" => medium_summary,
    "reconstruction_frequencies_hz" => recon_frequencies,
    "reconstruction_bandwidth_hz" => recon_bandwidth_hz,
    "reconstruction_mode" => String(results[:reconstruction_mode]),
    "reconstruction_progress" => parse_bool(opts["recon-progress"]),
    "source_phase_mode" => String(get(results, :source_phase_mode, :coherent)),
    "n_realizations" => Int(get(results, :n_realizations, 1)),
    "source_variability" => Dict(
        "frequency_jitter_percent" => source_variability.frequency_jitter_fraction * 100.0,
    ),
    "window_config" => string_key_dict(results[:window_config]),
    "window_info" => Dict(
        "geometric" => compact_window_info(results[:geo_info]),
        "hasa" => compact_window_info(results[:hasa_info]),
    ),
    "reconstruction_axial_step_m" => results[:geo_info][:axial_step],
    "reference_sound_speed_m_per_s" => results[:geo_info][:reference_sound_speed],
    "activity_model" => get(emission_meta, "activity_model", Dict("activity_mode" => "static")),
    "physical_source_count" => get(emission_meta, "physical_source_count", length(sources)),
    "emission_event_count" => get(emission_meta, "emission_event_count", length(sources)),
    "analysis_mode" => String(analysis_mode),
    "detection_truth_radius_m" => detection_truth_radius_m,
    "detection_truth_mode" => isnothing(detection_truth_mask) ? "source_disks" : "centerline_tube",
    "detection_threshold_ratio" => detection_threshold_ratio,
    "boundary_threshold_ratios" => boundary_threshold_ratios,
    "peak_method" => String(peak_method),
    "clean_loop_gain" => parse(Float64, opts["clean-loop-gain"]),
    "clean_max_iter" => parse(Int, opts["clean-max-iter"]),
    "clean_threshold_ratio" => parse(Float64, opts["clean-threshold-ratio"]),
    "simulation" => Dict(
        "receiver_row" => results[:simulation][:receiver_row],
        "receiver_cols" => [first(results[:simulation][:receiver_cols]), last(results[:simulation][:receiver_cols])],
        "source_indices" => [[row, col] for (row, col) in get(results[:simulation], :source_indices, Tuple{Int, Int}[])],
    ),
    "geometric" => results[:stats_geo],
    "hasa" => results[:stats_hasa],
)

open(joinpath(out_dir, "summary.json"), "w") do io
    JSON3.pretty(io, summary)
end

clusters = sources
@save joinpath(out_dir, "result.jld2") c rho cfg clusters results medium_info

println("Saved PAM outputs to $out_dir")
