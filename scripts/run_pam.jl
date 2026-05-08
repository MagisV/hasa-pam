#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Dates
using Printf
using Random
using Statistics
using CairoMakie
using JLD2
using JSON3
using TranscranialFUS

function parse_cli(args)
    opts = Dict{String, String}(
        "dimension" => "2",
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
        "dy-mm" => "",
        "transverse-y-mm" => "",
        "transverse-z-mm" => "",
        "axial-gain-power" => "1.5",
        "receiver-aperture-mm" => "full",
        "receiver-aperture-y-mm" => "",
        "receiver-aperture-z-mm" => "",
        "t-max-us" => "500",
        "dt-ns" => "20",
        "zero-pad-factor" => "4",
        "peak-suppression-radius-mm" => "8.0",
        "success-tolerance-mm" => "1.5",
        "aberrator" => "none",
        "sim-mode" => "auto",
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
        "benchmark" => "false",
        "window-batch" => "1",
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
        "vascular-squiggle-amplitude-x-mm" => "1.0",
        "vascular-squiggle-wavelength-mm" => "8",
        "vascular-squiggle-slope" => "0.0",
        "squiggle-phase-x-deg" => "90",
        "vascular-source-spacing-mm" => "0.5",
        "vascular-position-jitter-mm" => "0.05",
        "vascular-min-separation-mm" => "0.25",
        "vascular-max-sources-per-anchor" => "0",
        "vascular-radius-mm" => "1.0",
        "analysis-mode" => "auto",
        "detection-threshold-ratio" => "0.2",
        "boundary-threshold-ratios" => "0.5,0.55,0.6,0.65,0.7,0.75",
        "auto-threshold-search" => "true",
        "auto-threshold-min" => "0.10",
        "auto-threshold-max" => "0.95",
        "auto-threshold-step" => "0.01",
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

function parse_dimension(s::AbstractString)
    value = strip(s)
    value in ("2", "2d", "2D") && return 2
    value in ("3", "3d", "3D") && return 3
    error("--dimension must be 2 or 3, got: $s")
end

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

function parse_threshold_search_ratios(opts)
    min_ratio = parse(Float64, opts["auto-threshold-min"])
    max_ratio = parse(Float64, opts["auto-threshold-max"])
    step = parse(Float64, opts["auto-threshold-step"])
    min_ratio > 0 || error("--auto-threshold-min must be positive.")
    max_ratio >= min_ratio || error("--auto-threshold-max must be >= --auto-threshold-min.")
    step > 0 || error("--auto-threshold-step must be positive.")
    n = floor(Int, (max_ratio - min_ratio) / step + 1e-9)
    ratios = [round(min_ratio + i * step; digits=6) for i in 0:n]
    if isempty(ratios) || ratios[end] < max_ratio - 1e-9
        push!(ratios, round(max_ratio; digits=6))
    end
    return sort(unique(ratios))
end

function parse_source_model(s::AbstractString)
    value = Symbol(lowercase(strip(s)))
    value in (:point, :squiggle) || error("--source-model must be point or squiggle, got: $s")
    return value
end

function apply_model_defaults!(opts, provided_keys::Set{String})
    dimension = parse_dimension(opts["dimension"])
    if dimension == 3
        !("source-model" in provided_keys) && (opts["source-model"] = "point")
        !("sources-mm" in provided_keys) && (opts["sources-mm"] = "30:0:0")
        !("anchors-mm" in provided_keys) && (opts["anchors-mm"] = "45:0:0")
        !("vascular-squiggle-amplitude-x-mm" in provided_keys) && (opts["vascular-squiggle-amplitude-x-mm"] = "1.0")
        !("squiggle-phase-x-deg" in provided_keys) && (opts["squiggle-phase-x-deg"] = "90")
        !("frequency-mhz" in provided_keys) && (opts["frequency-mhz"] = "0.5")
        !("recon-bandwidth-khz" in provided_keys) && (opts["recon-bandwidth-khz"] = "0")
        !("receiver-aperture-mm" in provided_keys) && (opts["receiver-aperture-mm"] = "full")
        !("dx-mm" in provided_keys) && (opts["dx-mm"] = "0.2")
        !("dy-mm" in provided_keys) && (opts["dy-mm"] = "0.5")
        !("dz-mm" in provided_keys) && (opts["dz-mm"] = "0.5")
        !("axial-mm" in provided_keys) && (opts["axial-mm"] = "60")
        !("transverse-mm" in provided_keys) && (opts["transverse-mm"] = "32")
        !("dt-ns" in provided_keys) && (opts["dt-ns"] = "80")
        !("t-max-us" in provided_keys) && (opts["t-max-us"] = "60")
        !("zero-pad-factor" in provided_keys) && (opts["zero-pad-factor"] = "4")
        !("num-cycles" in provided_keys) && (opts["num-cycles"] = "5")
        !("phase-mode" in provided_keys) && (opts["phase-mode"] = "coherent")
        !("recon-step-um" in provided_keys) && (opts["recon-step-um"] = "50")
        !("use-gpu" in provided_keys) && (opts["use-gpu"] = "true")
    end
    source_model = parse_source_model(opts["source-model"])
    if dimension == 3 && source_model == :squiggle
        !("vascular-source-spacing-mm" in provided_keys) && (opts["vascular-source-spacing-mm"] = "0.5")
        !("vascular-min-separation-mm" in provided_keys) && (opts["vascular-min-separation-mm"] = "0.25")
        !("recon-window-us" in provided_keys) && (opts["recon-window-us"] = "40")
        !("recon-hop-us" in provided_keys) && (opts["recon-hop-us"] = "20")
        !("boundary-threshold-ratios" in provided_keys) && (opts["boundary-threshold-ratios"] = "0.5,0.55,0.6,0.65,0.7,0.75")
    end
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
    value in (:none, :water, :lens, :skull) || error("Unknown aberrator: $s")
    return value
end

function parse_sim_mode(s::AbstractString)
    value = Symbol(lowercase(strip(s)))
    value in (:auto, :analytic, :kwave) || error("Unknown --sim-mode: $s (must be auto, analytic, or kwave)")
    value == :auto && return :kwave
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

function parse_coordinate_triples_mm(spec::AbstractString, option_name::AbstractString)
    coord_tokens = [strip(token) for token in split(spec, ",") if !isempty(strip(token))]
    1 <= length(coord_tokens) <= 20 || error("Provide between 1 and 20 coordinates via --$option_name=depth:y:z,...")
    triples = NTuple{3, Float64}[]
    for token in coord_tokens
        parts = split(token, ":"; limit=3)
        length(parts) == 3 || error("3D coordinates must be depth:y:z in mm, got: $token")
        push!(triples, (
            parse(Float64, strip(parts[1])) * 1e-3,
            parse(Float64, strip(parts[2])) * 1e-3,
            parse(Float64, strip(parts[3])) * 1e-3,
        ))
    end
    return triples
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

function parse_point_sources_3d(opts)
    coordinates = parse_coordinate_triples_mm(opts["sources-mm"], "sources-mm")
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

    sources = EmissionSource3D[]
    for (idx, (depth_m, lateral_y_m, lateral_z_m)) in pairs(coordinates)
        push!(sources, PointSource3D(
            depth=depth_m,
            lateral_y=lateral_y_m,
            lateral_z=lateral_z_m,
            frequency=frequencies_mhz[idx] * 1e6,
            amplitude=amplitudes_pa[idx],
            phase=phases_deg[idx] * pi / 180,
            delay=delays_us[idx] * 1e-6,
            num_cycles=num_cycles,
        ))
    end
    return sources, Dict{String, Any}(
        "source_model" => "point3d",
        "coordinates_m" => [collect(coord) for coord in coordinates],
        "n_coordinate_sources" => n_sources,
        "n_emission_sources" => length(sources),
        "physical_source_count" => length(sources),
        "emission_event_count" => length(sources),
        "activity_model" => Dict("activity_mode" => "point_tone_burst_3d"),
        "phase_mode" => phase_mode,
        "frequencies_hz" => frequencies_mhz .* 1e6,
        "amplitudes_pa" => amplitudes_pa,
        "phases_rad" => phases_deg .* pi ./ 180,
        "delays_s" => delays_us .* 1e-6,
        "num_cycles" => num_cycles,
        "random_seed" => parse(Int, opts["random-seed"]),
    )
end

centerlines_m_3d(centerlines) = [[[p[1], p[2], p[3]] for p in line] for line in centerlines]

function parse_squiggle_sources_3d(opts, cfg::PAMConfig3D)
    anchors = parse_coordinate_triples_mm(opts["anchors-mm"], "anchors-mm")
    f0 = parse(Float64, opts["fundamental-mhz"]) * 1e6
    harmonics = parse_int_list(opts["harmonics"])
    isempty(harmonics) && error("--harmonics must be a non-empty integer list.")
    harmonic_amplitudes = parse_float_list(opts["harmonic-amplitudes"])
    length(harmonic_amplitudes) == length(harmonics) ||
        error("--harmonic-amplitudes must have the same length as --harmonics ($(length(harmonics))).")

    n_anchors = length(anchors)
    n_bubbles_per = expand_source_values(parse_float_list(opts["n-bubbles"]), n_anchors, 10.0)
    delays_us = expand_source_values(parse_float_list(opts["delays-us"]), n_anchors, 0.0)
    max_sources_raw = parse(Int, opts["vascular-max-sources-per-anchor"])
    max_sources = max_sources_raw <= 0 ? nothing : max_sources_raw
    phase_mode = Symbol(replace(lowercase(strip(opts["phase-mode"])), "-" => "_"))

    sources = EmissionSource3D[]
    all_centerlines_m = Any[]
    anchors_meta = Dict{String, Any}[]
    rng = Random.MersenneTwister(parse(Int, opts["random-seed"]))
    half_y = cfg.transverse_dim_y / 2
    half_z = cfg.transverse_dim_z / 2

    for (idx, anchor) in pairs(anchors)
        anchor_sources, anchor_meta = make_squiggle_bubble_sources_3d(
            [anchor];
            root_length = parse(Float64, opts["vascular-length-mm"]) * 1e-3,
            squiggle_amplitude_y = parse(Float64, opts["vascular-squiggle-amplitude-mm"]) * 1e-3,
            squiggle_amplitude_x = parse(Float64, opts["vascular-squiggle-amplitude-x-mm"]) * 1e-3,
            squiggle_wavelength = parse(Float64, opts["vascular-squiggle-wavelength-mm"]) * 1e-3,
            squiggle_phase_x = parse(Float64, opts["squiggle-phase-x-deg"]) * pi / 180,
            squiggle_slope_x = parse(Float64, opts["vascular-squiggle-slope"]),
            squiggle_slope_y = 0.0,
            source_spacing = parse(Float64, opts["vascular-source-spacing-mm"]) * 1e-3,
            position_jitter = parse(Float64, opts["vascular-position-jitter-mm"]) * 1e-3,
            min_separation = parse(Float64, opts["vascular-min-separation-mm"]) * 1e-3,
            max_sources_per_anchor = max_sources,
            depth_bounds = (0.0, Inf),
            lateral_y_bounds = (-half_y, half_y),
            lateral_z_bounds = (-half_z, half_z),
            fundamental = f0,
            amplitude = parse(Float64, opts["amplitude-pa"]),
            n_bubbles = n_bubbles_per[idx],
            harmonics = harmonics,
            harmonic_amplitudes = harmonic_amplitudes,
            gate_duration = parse(Float64, opts["gate-us"]) * 1e-6,
            taper_ratio = parse(Float64, opts["taper-ratio"]),
            delay = delays_us[idx] * 1e-6,
            phase_mode = phase_mode,
            phase_jitter = parse(Float64, opts["phase-jitter-rad"]),
            transducer_depth = -parse(Float64, opts["skull-transducer-distance-mm"]) * 1e-3,
            transducer_y = 0.0,
            transducer_z = 0.0,
            c0 = cfg.c0,
            rng = rng,
        )
        append!(sources, anchor_sources)
        anchor_cls = centerlines_m_3d(anchor_meta[:centerlines])
        append!(all_centerlines_m, anchor_cls)
        push!(anchors_meta, Dict(
            "anchor_m" => collect(anchor),
            "source_count" => length(anchor_sources),
            "centerlines_m" => anchor_cls,
        ))
    end

    return sources, Dict{String, Any}(
        "source_model" => "squiggle3d",
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
        "gate_duration_s" => parse(Float64, opts["gate-us"]) * 1e-6,
        "phase_jitter_rad" => parse(Float64, opts["phase-jitter-rad"]),
        "random_seed" => parse(Int, opts["random-seed"]),
        "n_bubbles_per_cluster" => n_bubbles_per,
        "delays_s" => delays_us .* 1e-6,
        "squiggle" => Dict(
            "length_m" => parse(Float64, opts["vascular-length-mm"]) * 1e-3,
            "squiggle_amplitude_y_m" => parse(Float64, opts["vascular-squiggle-amplitude-mm"]) * 1e-3,
            "squiggle_amplitude_x_m" => parse(Float64, opts["vascular-squiggle-amplitude-x-mm"]) * 1e-3,
            "squiggle_wavelength_m" => parse(Float64, opts["vascular-squiggle-wavelength-mm"]) * 1e-3,
            "squiggle_phase_x_deg" => parse(Float64, opts["squiggle-phase-x-deg"]),
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
    lateral_slug = if cfg isa PAMConfig3D
        "laty$(slug_value(cfg.transverse_dim_y * 1e3; digits=0))mm_latz$(slug_value(cfg.transverse_dim_z * 1e3; digits=0))mm"
    else
        "lat$(slug_value(cfg.transverse_dim * 1e3; digits=0))mm"
    end
    parts = String[
        timestamp,
        "run_pam",
        cfg isa PAMConfig3D ? "3d" : "2d",
        lowercase(opts["aberrator"]),
        source_model,
        "$(length(sources))src",
        "ax$(slug_value(cfg.axial_dim * 1e3; digits=0))mm",
        lateral_slug,
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

function default_simulation_info(cfg::PAMConfig3D)
    return Dict{Symbol, Any}(
        :receiver_row => receiver_row(cfg),
        :receiver_cols_y => receiver_col_range_y(cfg),
        :receiver_cols_z => receiver_col_range_z(cfg),
        :source_indices => NTuple{3, Int}[],
    )
end

function default_recon_frequencies(sources)
    freqs = Float64[]
    for src in sources
        append!(freqs, emission_frequencies(src))
    end
    return sort(unique(freqs))
end

function analytic_rf_for_point_sources_3d(cfg::PAMConfig3D, sources::AbstractVector{<:EmissionSource3D})
    grid = pam_grid_3d(cfg)
    ny, nz, nt = pam_Ny(cfg), pam_Nz(cfg), pam_Nt(cfg)
    rf = zeros(Float32, ny, nz, nt)
    for src in sources
        duration = src.num_cycles / src.frequency
        for iy in 1:ny, iz in 1:nz
            dy_src = grid.y[iy] - src.lateral_y
            dz_src = grid.z[iz] - src.lateral_z
            r = sqrt(src.depth^2 + dy_src^2 + dz_src^2)
            arrival = src.delay + r / cfg.c0
            for it in 1:nt
                te = (it - 1) * cfg.dt - arrival
                if 0 <= te <= duration
                    rf[iy, iz, it] += Float32(src.amplitude * sin(2pi * src.frequency * te + src.phase))
                end
            end
        end
    end
    return rf, grid, Dict{Symbol, Any}(
        :receiver_row => receiver_row(cfg),
        :receiver_cols_y => receiver_col_range_y(cfg),
        :receiver_cols_z => receiver_col_range_z(cfg),
        :source_indices => [source_grid_index_3d(src, cfg) for src in sources],
    )
end

function run_pam_case_3d(
    c::AbstractArray{<:Real, 3},
    rho::AbstractArray{<:Real, 3},
    sources::AbstractVector{<:EmissionSource3D},
    cfg::PAMConfig3D;
    frequencies::Union{Nothing, AbstractVector{<:Real}}=nothing,
    bandwidth::Real=0.0,
    use_gpu::Bool=false,
    reconstruction_axial_step::Union{Nothing, Real}=nothing,
    reconstruction_mode::Symbol=:full,
    window_config::PAMWindowConfig=PAMWindowConfig(),
    show_progress::Bool=false,
    benchmark::Bool=false,
    window_batch::Int=1,
    sim_mode::Symbol=:analytic,
    source_phase_mode::Symbol=:coherent,
    rng::Random.AbstractRNG=Random.default_rng(),
    source_variability::SourceVariabilityConfig=SourceVariabilityConfig(),
)
    use_gpu || error("3D PAM reconstruction currently requires --use-gpu=true.")
    recon_freqs = isnothing(frequencies) ? default_recon_frequencies(sources) : Float64.(frequencies)
    phase_mode = TranscranialFUS._normalize_source_phase_mode(source_phase_mode)
    recon_mode = phase_mode == :random_phase_per_window ?
        :windowed :
        TranscranialFUS._normalize_reconstruction_mode(reconstruction_mode)
    effective_window_config = PAMWindowConfig(;
        enabled=recon_mode == :windowed,
        window_duration=window_config.window_duration,
        hop=window_config.hop,
        taper=window_config.taper,
        min_energy_ratio=window_config.min_energy_ratio,
        accumulation=window_config.accumulation,
    )
    sim_sources = sources
    n_frames = 1
    if phase_mode == :random_static_phase
        sim_sources = TranscranialFUS._resample_source_phases_3d(sources, rng)
    elseif phase_mode == :random_phase_per_window
        sim_sources, n_frames = TranscranialFUS._expand_sources_per_window(
            sources,
            effective_window_config.window_duration,
            effective_window_config.hop,
            cfg.t_max,
            rng;
            variability=source_variability,
        )
    elseif phase_mode == :random_phase_per_realization
        error("3D PAM does not implement --source-phase-mode=random_phase_per_realization yet.")
    end
    rf, grid, sim_info = if sim_mode == :kwave
        simulate_point_sources_3d(c, rho, sim_sources, cfg; use_gpu=use_gpu)
    else
        analytic_rf_for_point_sources_3d(cfg, sim_sources)
    end
    recon_kwargs = (
        frequencies=recon_freqs,
        bandwidth=bandwidth,
        reference_sound_speed=TranscranialFUS._pam_reference_sound_speed(c, cfg, sources),
        axial_step=reconstruction_axial_step,
        use_gpu=use_gpu,
        show_progress=show_progress,
        benchmark=benchmark,
        window_batch=window_batch,
    )
    pam_geo, _, geo_info = if recon_mode == :windowed
        reconstruct_pam_windowed_3d(
            rf,
            c,
            cfg;
            recon_kwargs...,
            corrected=false,
            window_config=effective_window_config,
        )
    else
        reconstruct_pam_3d(rf, c, cfg; recon_kwargs..., corrected=false)
    end
    pam_hasa, _, hasa_info = if recon_mode == :windowed
        reconstruct_pam_windowed_3d(
            rf,
            c,
            cfg;
            recon_kwargs...,
            corrected=true,
            window_config=effective_window_config,
        )
    else
        reconstruct_pam_3d(rf, c, cfg; recon_kwargs..., corrected=true)
    end

    return Dict{Symbol, Any}(
        :rf => Float64.(rf),
        :kgrid => grid,
        :simulation => sim_info,
        :pam_geo => pam_geo,
        :pam_hasa => pam_hasa,
        :geo_info => geo_info,
        :hasa_info => hasa_info,
        :stats_geo => any(s -> s isa BubbleCluster3D, sources) ? Dict{Symbol,Any}() : analyse_pam_3d(pam_geo, grid, cfg, sources),
        :stats_hasa => any(s -> s isa BubbleCluster3D, sources) ? Dict{Symbol,Any}() : analyse_pam_3d(pam_hasa, grid, cfg, sources),
        :reconstruction_frequencies => recon_freqs,
        :analysis_mode => any(s -> s isa BubbleCluster3D, sources) ? :detection : :localization,
        :analysis_source_count => length(sources),
        :emission_event_count => length(sim_sources),
        :reconstruction_mode => recon_mode,
        :source_phase_mode => phase_mode,
        :n_frames => n_frames,
        :window_config => TranscranialFUS._window_config_info(effective_window_config),
        :use_gpu => use_gpu,
        :show_progress => show_progress,
    )
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

function map_norm(map::AbstractMatrix{<:Real}, ref::Real)
    safe_ref = max(Float64(ref), eps(Float64))
    return Float64.(map) ./ safe_ref
end

function source_pairs_mm(sources)
    return [(src.depth * 1e3, src.lateral * 1e3) for src in sources]
end

function source_triples_mm(sources::AbstractVector{<:EmissionSource3D})
    return [(src.depth * 1e3, src.lateral_y * 1e3, src.lateral_z * 1e3) for src in sources]
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
    overlay_skull_2d!(ax_geo, c, lateral_mm, depth_mm)
    scatter_sources!(ax_geo, sources)
    Colorbar(fig[2, 2], hm_geo; label="dB")

    ax_hasa = Axis(fig[2, 3]; title="Corrected HASA PAM", xlabel="Lateral position [mm]", ylabel="Depth below receiver [mm]", aspect=DataAspect())
    hm_hasa = heatmap!(ax_hasa, lateral_mm, depth_mm, map_db(pam_hasa, map_ref)'; colormap=:viridis, colorrange=(-30, 0))
    overlay_skull_2d!(ax_hasa, c, lateral_mm, depth_mm)
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

function source_detection_stats_3d(pred, grid, cfg::PAMConfig3D, sources; radius::Real)
    radius_m = Float64(radius)
    radius_m >= 0 || error("source detection radius must be non-negative.")
    isempty(sources) && return Dict{Symbol, Any}(
        :source_recall => 0.0,
        :detected_source_count => 0,
        :num_truth_sources => 0,
        :mean_detected_source_distance_mm => nothing,
        :max_detected_source_distance_mm => nothing,
    )
    x = collect(grid.x)
    y = collect(grid.y)
    z = collect(grid.z)
    r0 = receiver_row(cfg)
    row_r = ceil(Int, radius_m / cfg.dx)
    col_r_y = ceil(Int, radius_m / cfg.dy)
    col_r_z = ceil(Int, radius_m / cfg.dz)
    radius2 = radius_m^2

    detected = 0
    distances_mm = Float64[]
    for src in sources
        src_x = x[r0] + src.depth
        row0 = r0 + round(Int, src.depth / cfg.dx)
        col0_y = argmin(abs.(y .- src.lateral_y))
        col0_z = argmin(abs.(z .- src.lateral_z))
        best_d2 = Inf
        for row in max(r0 + 1, row0 - row_r):min(size(pred, 1), row0 + row_r)
            dx2 = (x[row] - src_x)^2
            for iy in max(1, col0_y - col_r_y):min(size(pred, 2), col0_y + col_r_y)
                dy2 = (y[iy] - src.lateral_y)^2
                for iz in max(1, col0_z - col_r_z):min(size(pred, 3), col0_z + col_r_z)
                    pred[row, iy, iz] || continue
                    d2 = dx2 + dy2 + (z[iz] - src.lateral_z)^2
                    if d2 <= radius2 && d2 < best_d2
                        best_d2 = d2
                    end
                end
            end
        end
        if isfinite(best_d2)
            detected += 1
            push!(distances_mm, sqrt(best_d2) * 1e3)
        end
    end

    return Dict{Symbol, Any}(
        :source_recall => detected / length(sources),
        :detected_source_count => detected,
        :num_truth_sources => length(sources),
        :mean_detected_source_distance_mm => isempty(distances_mm) ? nothing : mean(distances_mm),
        :max_detected_source_distance_mm => isempty(distances_mm) ? nothing : maximum(distances_mm),
    )
end

function threshold_detection_stats_3d(intensity, grid, cfg, sources; threshold_ratios, truth_radius, truth_mask)
    truth = isnothing(truth_mask) ? pam_truth_mask_3d(sources, grid, cfg; radius=truth_radius) : truth_mask
    local_ref = max(maximum(Float64.(intensity)), eps(Float64))
    return [
        begin
            pred = intensity .>= ratio * local_ref
            tp = count(pred .& truth)
            fp = count(pred .& .!truth)
            fn = count(.!pred .& truth)
            precision = tp + fp == 0 ? 0.0 : tp / (tp + fp)
            recall = tp + fn == 0 ? 0.0 : tp / (tp + fn)
            f1 = precision + recall == 0 ? 0.0 : 2 * precision * recall / (precision + recall)
            source_stats = source_detection_stats_3d(pred, grid, cfg, sources; radius=truth_radius)
            source_recall = Float64(source_stats[:source_recall])
            source_f1 = precision + source_recall == 0 ? 0.0 : 2 * precision * source_recall / (precision + source_recall)
            merge(Dict(
                :threshold_ratio => ratio,
                :f1 => source_f1,
                :source_f1 => source_f1,
                :voxel_f1 => f1,
                :precision => precision,
                :recall => source_recall,
                :voxel_recall => recall,
                :true_positive_voxels => tp,
                :false_positive_voxels => fp,
                :false_negative_voxels => fn,
                :predicted_voxels => count(pred),
                :truth_voxels => count(truth),
            ), source_stats)
        end
        for ratio in threshold_ratios
    ]
end

function best_threshold_entry_3d(stats)
    isempty(stats) && error("No 3D threshold stats available.")
    best = first(stats)
    for entry in stats[2:end]
        score_metric = haskey(entry, :source_f1) ? :source_f1 : :f1
        best_metric = haskey(best, :source_f1) ? :source_f1 : :f1
        score = (Float64(entry[score_metric]), Float64(entry[:precision]), Float64(entry[:threshold_ratio]))
        best_score = (Float64(best[best_metric]), Float64(best[:precision]), Float64(best[:threshold_ratio]))
        if score > best_score
            best = entry
        end
    end
    return best
end

_metric_value(entry, key::Symbol, fallback::Symbol=key) = Float64(get(entry, key, entry[fallback]))

function _argmax_by(entries, scorefn)
    best = first(entries)
    best_score = scorefn(best)
    for entry in entries[2:end]
        score = scorefn(entry)
        if score > best_score
            best = entry
            best_score = score
        end
    end
    return best
end

function _threshold_tradeoff_entry_3d(stats, best, target::Symbol)
    best_f1 = _metric_value(best, :source_f1, :f1)
    best_value = _metric_value(best, target, target == :source_recall ? :recall : target)
    metric_key = target == :source_recall ? :source_recall : :precision
    fallback_key = target == :source_recall ? :recall : :precision
    candidates = [entry for entry in stats if _metric_value(entry, metric_key, fallback_key) > best_value + 1e-9]
    for floor_fraction in (0.95, 0.90, 0.0)
        viable = [entry for entry in candidates if _metric_value(entry, :source_f1, :f1) >= floor_fraction * best_f1]
        isempty(viable) && continue
        if target == :source_recall
            return _argmax_by(viable, entry -> (
                _metric_value(entry, :source_recall, :recall),
                _metric_value(entry, :source_f1, :f1),
                _metric_value(entry, :precision),
            ))
        else
            return _argmax_by(viable, entry -> (
                _metric_value(entry, :precision),
                _metric_value(entry, :source_f1, :f1),
                _metric_value(entry, :source_recall, :recall),
            ))
        end
    end
    return best
end

function threshold_outline_entries_3d(stats)
    best = best_threshold_entry_3d(stats)
    recall = _threshold_tradeoff_entry_3d(stats, best, :source_recall)
    precision = _threshold_tradeoff_entry_3d(stats, best, :precision)
    return [
        (kind=:best_f1, label="best F1", color=:cyan, entry=best),
        (kind=:more_recall, label="more recall", color=:lime, entry=recall),
        (kind=:more_precision, label="more precision", color=:magenta, entry=precision),
    ]
end

function overlay_skull_2d!(ax, c, xvals, yvals; transpose_matrix=true)
    skull_mask = skull_mask_from_c_columnwise(c; mask_outside=false)
    any(skull_mask) || return nothing
    c_max = Float64(maximum(c[skull_mask]))
    overlay = fill(NaN, size(c))
    overlay[skull_mask] .= Float64.(c[skull_mask]) ./ c_max
    mat = transpose_matrix ? overlay' : overlay
    heatmap!(ax, xvals, yvals, mat; colormap=:grays, alpha=0.35, colorrange=(0, 1), nan_color=CairoMakie.RGBAf(0, 0, 0, 0))
    return nothing
end

function _c_slice_for_projection(c::AbstractArray{<:Real, 3}, projection::Symbol)
    if projection == :depth_y
        return dropdims(maximum(c; dims=3), dims=3)
    elseif projection == :depth_z
        return dropdims(maximum(c; dims=2), dims=2)
    else  # :y_z
        return dropdims(maximum(c; dims=1), dims=1)
    end
end

function overlay_skull_3d_projection!(ax, c::AbstractArray{<:Real, 3}, xvals, yvals, projection::Symbol)
    c2d = _c_slice_for_projection(c, projection)
    # :y_z projection is not transposed (matches _projection_heatmap_matrix_3d convention)
    overlay_skull_2d!(ax, c2d, xvals, yvals; transpose_matrix=(projection != :y_z))
    return nothing
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
    c=nothing,
)
    depth_mm = depth_coordinates(kgrid, cfg) .* 1e3
    lateral_mm = kgrid.y_vec .* 1e3
    ax = Axis(fig[row, 1]; title=title, xlabel="Lateral [mm]", ylabel="Depth [mm]", aspect=DataAspect())
    hm = heatmap!(ax, lateral_mm, depth_mm, Float64.(intensity ./ global_ref)'; colormap=:viridis, colorrange=(0, 1))
    !isnothing(c) && overlay_skull_2d!(ax, c, lateral_mm, depth_mm)
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
    gl = GridLayout(fig[row, col]; tellwidth=false, tellheight=true)
    Label(gl[1, 1:4], title; font="DejaVu Sans Mono", fontsize=13, halign=:left, tellwidth=false)
    headers = ["thr", "F1", "Prec", "Recall"]
    for (c, h) in enumerate(headers)
        Label(gl[2, c], h; font="DejaVu Sans Mono", fontsize=11, halign=:center)
    end
    for (r, entry) in enumerate(stats)
        vals = [
            @sprintf("%.2f", Float64(entry[:threshold_ratio])),
            @sprintf("%.3f", Float64(entry[:f1])),
            @sprintf("%.3f", Float64(entry[:precision])),
            @sprintf("%.3f", Float64(entry[:recall])),
        ]
        for (c, v) in enumerate(vals)
            Label(gl[2 + r, c], v; font="DejaVu Sans Mono", fontsize=11, halign=:center)
        end
    end
    colgap!(gl, 10)
    rowgap!(gl, 2)
end

function _project3d_values(intensity::AbstractArray{<:Real, 3}, projection::Symbol)
    values = Float64.(intensity)
    if projection == :depth_y
        return dropdims(maximum(values; dims=3), dims=3)
    elseif projection == :depth_z
        return dropdims(maximum(values; dims=2), dims=2)
    elseif projection == :y_z
        return dropdims(maximum(values; dims=1), dims=1)
    end
    error("Unknown 3D projection: $projection")
end

function _project3d_mask(mask::AbstractArray{Bool, 3}, projection::Symbol)
    if projection == :depth_y
        return dropdims(any(mask; dims=3), dims=3)
    elseif projection == :depth_z
        return dropdims(any(mask; dims=2), dims=2)
    elseif projection == :y_z
        return dropdims(any(mask; dims=1), dims=1)
    end
    error("Unknown 3D projection: $projection")
end

function _projection_axes_3d(grid, cfg::PAMConfig3D, projection::Symbol)
    depth_mm = depth_coordinates_3d(cfg) .* 1e3
    y_mm = collect(grid.y) .* 1e3
    z_mm = collect(grid.z) .* 1e3
    if projection == :depth_y
        return y_mm, depth_mm, "Y [mm]", "Depth [mm]"
    elseif projection == :depth_z
        return z_mm, depth_mm, "Z [mm]", "Depth [mm]"
    elseif projection == :y_z
        return y_mm, z_mm, "Y [mm]", "Z [mm]"
    end
    error("Unknown 3D projection: $projection")
end

function _projection_heatmap_matrix_3d(values::AbstractMatrix, projection::Symbol)
    projection == :y_z && return values
    return values'
end

function scatter_sources_3d_projection!(ax, sources, projection::Symbol; color=(:white, 0.75))
    truth = source_triples_mm(sources)
    if projection == :depth_y
        scatter!(ax, [t[2] for t in truth], [t[1] for t in truth]; color=color, marker=:x, markersize=13, strokewidth=2)
    elseif projection == :depth_z
        scatter!(ax, [t[3] for t in truth], [t[1] for t in truth]; color=color, marker=:x, markersize=13, strokewidth=2)
    elseif projection == :y_z
        scatter!(ax, [t[2] for t in truth], [t[3] for t in truth]; color=color, marker=:x, markersize=13, strokewidth=2)
    end
    return nothing
end

function add_projection_panel_3d!(
    fig,
    row,
    col,
    title,
    intensity,
    truth_mask,
    grid,
    cfg,
    sources;
    projection::Symbol,
    outline_entries,
    global_ref,
    c=nothing,
)
    xvals, yvals, xlabel, ylabel = _projection_axes_3d(grid, cfg, projection)
    proj = _project3d_values(intensity, projection)
    truth_proj = _project3d_mask(truth_mask, projection)
    ax = Axis(fig[row, col]; title=title, xlabel=xlabel, ylabel=ylabel, aspect=DataAspect())
    hm = heatmap!(
        ax,
        xvals,
        yvals,
        _projection_heatmap_matrix_3d(map_norm(proj, global_ref), projection);
        colormap=:viridis,
        colorrange=(0, 1),
    )
    !isnothing(c) && overlay_skull_3d_projection!(ax, c, xvals, yvals, projection)
    if any(truth_proj) && any(.!truth_proj)
        contour!(
            ax,
            xvals,
            yvals,
            _projection_heatmap_matrix_3d(Float64.(truth_proj), projection);
            levels=[0.5],
            color=(:white, 0.85),
            linewidth=2.4,
            linestyle=:dash,
        )
    end
    local_ref = max(maximum(Float64.(intensity)), eps(Float64))
    for outline in outline_entries
        ratio = Float64(outline.entry[:threshold_ratio])
        pred_proj = _project3d_mask(intensity .>= ratio * local_ref, projection)
        if any(pred_proj) && any(.!pred_proj)
            contour!(
                ax,
                xvals,
                yvals,
                _projection_heatmap_matrix_3d(Float64.(pred_proj), projection);
                levels=[0.5],
                color=outline.color,
                linewidth=2,
            )
        end
    end
    scatter_sources_3d_projection!(ax, sources, projection)
    return hm
end

function add_threshold_table_3d!(fig, row, col, title, stats; outline_entries=nothing)
    rows_data = isnothing(outline_entries) ? [(label="", entry=entry) for entry in stats] :
        [(label=outline.label, entry=outline.entry) for outline in outline_entries]
    gl = GridLayout(fig[row, col]; tellwidth=false, tellheight=true)
    Label(gl[1, 1:8], title; font="DejaVu Sans Mono", fontsize=13, halign=:left, tellwidth=false)
    headers = ["", "thr", "SrcF1", "Prec", "SrcRc", "VoxF1", "Vox"]
    for (c, h) in enumerate(headers)
        Label(gl[2, c], h; font="DejaVu Sans Mono", fontsize=11, halign=c == 1 ? :right : :center)
    end
    for (r, row_entry) in enumerate(rows_data)
        entry = row_entry.entry
        vals = [
            row_entry.label,
            @sprintf("%.2f",  Float64(entry[:threshold_ratio])),
            @sprintf("%.3f",  Float64(get(entry, :source_f1, entry[:f1]))),
            @sprintf("%.3f",  Float64(entry[:precision])),
            @sprintf("%.3f",  Float64(get(entry, :source_recall, entry[:recall]))),
            @sprintf("%.3f",  Float64(get(entry, :voxel_f1, entry[:f1]))),
            @sprintf("%d",    Int(entry[:predicted_voxels])),
        ]
        for (c, v) in enumerate(vals)
            Label(gl[2 + r, c], v; font="DejaVu Sans Mono", fontsize=11, halign=c == 1 ? :right : :center)
        end
    end
    colgap!(gl, 10)
    rowgap!(gl, 2)
end

function add_threshold_curve_panel_3d!(fig, row, col, title, stats; outline_entries)
    thresholds = [Float64(entry[:threshold_ratio]) for entry in stats]
    f1 = [Float64(get(entry, :source_f1, entry[:f1])) for entry in stats]
    precision = [Float64(entry[:precision]) for entry in stats]
    recall = [Float64(get(entry, :source_recall, entry[:recall])) for entry in stats]
    ax = Axis(fig[row, col]; title=title, xlabel="Threshold / max intensity", ylabel="Score")
    lines!(ax, thresholds, f1; color=:cyan, linewidth=2.5, label="source F1")
    lines!(ax, thresholds, precision; color=:magenta, linewidth=2.0, label="precision")
    lines!(ax, thresholds, recall; color=:lime, linewidth=2.0, label="source recall")
    for outline in outline_entries
        threshold = Float64(outline.entry[:threshold_ratio])
        lines!(ax, [threshold, threshold], [0.0, 1.0]; color=(outline.color, 0.45), linewidth=1.5, linestyle=:dash)
    end
    ylims!(ax, 0, 1)
    axislegend(ax; position=:rb, framevisible=false)
    return nothing
end

function save_threshold_boundary_detection(path, pam_geo, pam_hasa, kgrid, cfg, sources; threshold_ratios, truth_radius, truth_mask, truth_centerlines, frequencies, c=nothing)
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
        c=c,
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
        c=c,
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

function save_threshold_boundary_detection_3d(path, pam_geo, pam_hasa, grid, cfg, sources; threshold_ratios, truth_radius, c=nothing)
    truth_mask = pam_truth_mask_3d(sources, grid, cfg; radius=truth_radius)
    geo_stats = threshold_detection_stats_3d(pam_geo, grid, cfg, sources; threshold_ratios=threshold_ratios, truth_radius=truth_radius, truth_mask=truth_mask)
    hasa_stats = threshold_detection_stats_3d(pam_hasa, grid, cfg, sources; threshold_ratios=threshold_ratios, truth_radius=truth_radius, truth_mask=truth_mask)
    global_ref = max(maximum(Float64.(pam_geo)), maximum(Float64.(pam_hasa)), eps(Float64))
    best_geo = best_threshold_entry_3d(geo_stats)
    best_hasa = best_threshold_entry_3d(hasa_stats)
    geo_outlines = threshold_outline_entries_3d(geo_stats)
    hasa_outlines = threshold_outline_entries_3d(hasa_stats)

    fig = Figure(size=(1550, 1450))
    projections = (:depth_y, :depth_z, :y_z)
    titles = Dict(
        :depth_y => "Depth-Y max projection",
        :depth_z => "Depth-Z max projection",
        :y_z => "Y-Z max projection",
    )
    hm = nothing
    for (col, projection) in pairs(projections)
        hm = add_projection_panel_3d!(
            fig, 1, col, "Geometric: $(titles[projection])",
            pam_geo, truth_mask, grid, cfg, sources;
            projection=projection,
            outline_entries=geo_outlines,
            global_ref=global_ref,
            c=c,
        )
        add_projection_panel_3d!(
            fig, 2, col, "HASA: $(titles[projection])",
            pam_hasa, truth_mask, grid, cfg, sources;
            projection=projection,
            outline_entries=hasa_outlines,
            global_ref=global_ref,
            c=c,
        )
    end
    Colorbar(fig[1:2, 4], hm; label="Norm. PAM intensity")
    legend_specs = [
        (label="best F1", color=:cyan),
        (label="more recall", color=:lime),
        (label="more precision", color=:magenta),
    ]
    legend_elements = [LineElement(color=spec.color, linewidth=3) for spec in legend_specs]
    legend_labels = [spec.label for spec in legend_specs]
    Legend(fig[3, 1:2], legend_elements, legend_labels; orientation=:horizontal, tellheight=true, framevisible=false)
    Label(fig[3, 3], "Truth mask shown as dashed white contours; sources are x markers. Curves use the dense threshold search grid."; tellwidth=false, halign=:left)
    add_threshold_curve_panel_3d!(fig, 4, 1:2, "Geometric threshold response", geo_stats; outline_entries=geo_outlines)
    add_threshold_curve_panel_3d!(fig, 4, 3:4, "HASA threshold response", hasa_stats; outline_entries=hasa_outlines)
    add_threshold_table_3d!(fig, 5, 1:2, "Geometric selected thresholds", geo_stats; outline_entries=geo_outlines)
    add_threshold_table_3d!(fig, 5, 3:4, "HASA selected thresholds", hasa_stats; outline_entries=hasa_outlines)
    save(path, fig)
    return Dict(
        "threshold_ratios" => threshold_ratios,
        "selection_metric" => "source_f1",
        "source_detection_radius_m" => Float64(truth_radius),
        "best_geometric_threshold" => best_geo[:threshold_ratio],
        "best_geometric_metric" => string_key_dict(best_geo),
        "best_hasa_threshold" => best_hasa[:threshold_ratio],
        "best_hasa_metric" => string_key_dict(best_hasa),
        "geometric_selected_outlines" => [
            Dict("kind" => String(outline.kind), "label" => outline.label, "metric" => string_key_dict(outline.entry))
            for outline in geo_outlines
        ],
        "hasa_selected_outlines" => [
            Dict("kind" => String(outline.kind), "label" => outline.label, "metric" => string_key_dict(outline.entry))
            for outline in hasa_outlines
        ],
        "geometric" => [string_key_dict(stats) for stats in geo_stats],
        "hasa" => [string_key_dict(stats) for stats in hasa_stats],
    )
end

function _voxel_points_3d(mask::AbstractArray{Bool, 3}, grid, cfg::PAMConfig3D)
    depth_mm = depth_coordinates_3d(cfg) .* 1e3
    y_mm = collect(grid.y) .* 1e3
    z_mm = collect(grid.z) .* 1e3
    idxs = Tuple.(findall(mask))
    return (
        depth = [depth_mm[idx[1]] for idx in idxs],
        y = [y_mm[idx[2]] for idx in idxs],
        z = [z_mm[idx[3]] for idx in idxs],
        indices = idxs,
    )
end

function save_best_threshold_volume_3d(path, intensity, grid, cfg, sources; threshold::Real, truth_radius::Real)
    local_ref = max(maximum(Float64.(intensity)), eps(Float64))
    pred_mask = intensity .>= Float64(threshold) * local_ref
    truth_mask = pam_truth_mask_3d(sources, grid, cfg; radius=truth_radius)
    pred = _voxel_points_3d(pred_mask, grid, cfg)
    truth = _voxel_points_3d(truth_mask, grid, cfg)

    fig = Figure(size=(1100, 900))
    ax = Axis3(
        fig[1, 1];
        title="HASA 3D reconstructed region at best threshold $(round(Float64(threshold); digits=3))",
        xlabel="Y [mm]",
        ylabel="Z [mm]",
        zlabel="Depth [mm]",
        aspect=:data,
        azimuth=0.75pi,
        elevation=0.22pi,
    )

    if !isempty(truth.indices)
        scatter!(
            ax,
            truth.y,
            truth.z,
            truth.depth;
            markersize=10,
            color=(:white, 0.16),
            strokecolor=(:black, 0.25),
            strokewidth=0.4,
        )
    end

    if !isempty(pred.indices)
        pred_values = [Float64(intensity[idx...]) / local_ref for idx in pred.indices]
        sc = scatter!(
            ax,
            pred.y,
            pred.z,
            pred.depth;
            markersize=14,
            color=pred_values,
            colormap=:viridis,
            colorrange=(Float64(threshold), 1.0),
            strokewidth=0,
        )
        Colorbar(fig[1, 2], sc; label="Norm. HASA intensity")
    else
        Label(fig[1, 2], "No voxels at threshold."; tellheight=false)
    end

    truth_sources = source_triples_mm(sources)
    scatter!(
        ax,
        [t[2] for t in truth_sources],
        [t[3] for t in truth_sources],
        [t[1] for t in truth_sources];
        marker=:xcross,
        markersize=24,
        color=:red,
        strokewidth=2,
    )

    Label(
        fig[2, 1:2],
        "Colored voxels are the thresholded HASA reconstruction; translucent white voxels are the truth mask; red x markers are source locations.";
        tellwidth=false,
        halign=:left,
    )
    save(path, fig)
    return Dict(
        "threshold_ratio" => Float64(threshold),
        "predicted_voxels" => count(pred_mask),
        "truth_voxels" => count(truth_mask),
    )
end

function save_napari_npz_3d(out_dir, pam_geo, pam_hasa, c, rho, grid, cfg, sources; truth_radius)
    np = TranscranialFUS.PythonCall.pyimport("numpy")

    depth_mm = Float32.(depth_coordinates_3d(cfg) .* 1e3)
    y_mm     = Float32.(collect(grid.y) .* 1e3)
    z_mm     = Float32.(collect(grid.z) .* 1e3)

    ref = max(maximum(Float64.(pam_hasa)), eps(Float64))
    hasa_norm  = Float32.(Float64.(pam_hasa) ./ ref)
    geo_norm   = Float32.(Float64.(pam_geo)  ./ ref)
    c_vol      = Float32.(c)
    rho_vol    = Float32.(rho)
    truth_mask = Float32.(pam_truth_mask_3d(sources, grid, cfg; radius=truth_radius))

    triples = source_triples_mm(sources)
    src_depth = Float32[t[1] for t in triples]
    src_y     = Float32[t[2] for t in triples]
    src_z     = Float32[t[3] for t in triples]

    # voxel spacing in mm for napari scale parameter: (depth, y, z)
    scale = [Float64(cfg.dx * 1e3), Float64(cfg.dy * 1e3), Float64(cfg.dz * 1e3)]

    npz_path = joinpath(out_dir, "napari_data.npz")
    np.savez(
        npz_path,
        hasa        = hasa_norm,
        geometric   = geo_norm,
        sound_speed = c_vol,
        density     = rho_vol,
        truth_mask  = truth_mask,
        depth_mm    = depth_mm,
        y_mm        = y_mm,
        z_mm        = z_mm,
        src_depth_mm = src_depth,
        src_y_mm     = src_y,
        src_z_mm     = src_z,
        scale        = Float64.(scale),
    )

    py_script = """
import numpy as np, napari, sys

data = np.load(r\"$(replace(npz_path, "\\" => "\\\\"))\")
scale = tuple(data[\"scale\"])   # (depth_mm, y_mm, z_mm)

viewer = napari.Viewer(title=\"PAM 3D: $(basename(out_dir))\")
viewer.add_image(data[\"hasa\"],        name=\"HASA (norm)\",       scale=scale, colormap=\"inferno\",  opacity=0.9)
viewer.add_image(data[\"geometric\"],   name=\"Geometric (norm)\",  scale=scale, colormap=\"viridis\", opacity=0.5, visible=False)
viewer.add_image(data[\"sound_speed\"], name=\"Sound speed [m/s]\", scale=scale, colormap=\"gray\",    opacity=0.35, visible=False)
viewer.add_image(data[\"density\"],     name=\"Density [kg/m3]\",   scale=scale, colormap=\"gray\",    opacity=0.35, visible=False)
viewer.add_image(data[\"truth_mask\"],  name=\"Truth mask\",        scale=scale, colormap=\"green\",   opacity=0.25)

depth_idx = np.interp(data[\"src_depth_mm\"], data[\"depth_mm\"], np.arange(len(data[\"depth_mm\"])))
y_idx     = np.interp(data[\"src_y_mm\"],     data[\"y_mm\"],     np.arange(len(data[\"y_mm\"])))
z_idx     = np.interp(data[\"src_z_mm\"],     data[\"z_mm\"],     np.arange(len(data[\"z_mm\"])))
pts = np.stack([depth_idx, y_idx, z_idx], axis=1)
viewer.add_points(pts, name=\"Sources\", size=1.5, face_color=\"red\", symbol=\"cross\", scale=scale)

napari.run()
"""
    open(joinpath(out_dir, "view_pam.py"), "w") do io
        write(io, py_script)
    end

    println("Saved napari data → $npz_path")
    println("  Open with:  python $(joinpath(out_dir, "view_pam.py"))")
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
        "effective_window_duration_s" => get(info, :effective_window_duration_s, nothing),
        "effective_hop_s" => get(info, :effective_hop_s, nothing),
        "energy_threshold" => info[:energy_threshold],
        "used_window_ranges" => range_pairs(info[:used_window_ranges]),
        "skipped_window_ranges" => range_pairs(info[:skipped_window_ranges]),
        "accumulation" => haskey(info, :accumulation) ? String(info[:accumulation]) : nothing,
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

function source_summary(src::PointSource3D)
    return Dict(
        "kind" => "point3d",
        "depth_m" => src.depth,
        "lateral_y_m" => src.lateral_y,
        "lateral_z_m" => src.lateral_z,
        "frequency_hz" => src.frequency,
        "amplitude_pa" => src.amplitude,
        "phase_rad" => src.phase,
        "delay_s" => src.delay,
        "num_cycles" => src.num_cycles,
    )
end

function source_summary(src::BubbleCluster3D)
    return Dict(
        "kind" => "bubble3d",
        "depth_m" => src.depth,
        "lateral_y_m" => src.lateral_y,
        "lateral_z_m" => src.lateral_z,
        "fundamental_hz" => src.fundamental,
        "amplitude_pa" => src.amplitude,
        "n_bubbles" => src.n_bubbles,
        "harmonics" => src.harmonics,
        "harmonic_amplitudes" => src.harmonic_amplitudes,
        "harmonic_phases_rad" => src.harmonic_phases,
        "gate_duration_s" => src.gate_duration,
        "delay_s" => src.delay,
    )
end

opts, provided_keys = parse_cli(ARGS)
dimension = parse_dimension(opts["dimension"])
source_model = parse_source_model(opts["source-model"])
from_run_dir = strip(opts["from-run-dir"])
peak_method = Symbol(lowercase(strip(opts["peak-method"])))
peak_method in (:argmax, :clean) || error("--peak-method must be argmax or clean, got: $(opts["peak-method"])")
detection_truth_radius_m = parse(Float64, opts["vascular-radius-mm"]) * 1e-3
detection_threshold_ratio = parse(Float64, opts["detection-threshold-ratio"])
boundary_threshold_ratios = parse_threshold_ratios(opts["boundary-threshold-ratios"])
auto_threshold_search = parse_bool(opts["auto-threshold-search"])
threshold_score_ratios = auto_threshold_search ? parse_threshold_search_ratios(opts) : boundary_threshold_ratios

if dimension == 3
    isempty(from_run_dir) || error("--from-run-dir is not implemented for 3D PAM yet.")
    source_model in (:point, :squiggle) ||
        error("3D PAM CLI supports --source-model=point or --source-model=squiggle.")
    aberrator = parse_aberrator(opts["aberrator"])
    aberrator in (:none, :skull) || error("3D PAM CLI currently supports only --aberrator=none or --aberrator=skull.")

    dy_mm = isempty(strip(opts["dy-mm"])) ? parse(Float64, opts["dz-mm"]) : parse(Float64, opts["dy-mm"])
    transverse_y_mm = isempty(strip(opts["transverse-y-mm"])) ? parse(Float64, opts["transverse-mm"]) : parse(Float64, opts["transverse-y-mm"])
    transverse_z_mm = isempty(strip(opts["transverse-z-mm"])) ? parse(Float64, opts["transverse-mm"]) : parse(Float64, opts["transverse-z-mm"])
    receiver_aperture_y_spec = isempty(strip(opts["receiver-aperture-y-mm"])) ? opts["receiver-aperture-mm"] : opts["receiver-aperture-y-mm"]
    receiver_aperture_z_spec = isempty(strip(opts["receiver-aperture-z-mm"])) ? opts["receiver-aperture-mm"] : opts["receiver-aperture-z-mm"]

    cfg_base = PAMConfig3D(
        dx=parse(Float64, opts["dx-mm"]) * 1e-3,
        dy=dy_mm * 1e-3,
        dz=parse(Float64, opts["dz-mm"]) * 1e-3,
        axial_dim=parse(Float64, opts["axial-mm"]) * 1e-3,
        transverse_dim_y=transverse_y_mm * 1e-3,
        transverse_dim_z=transverse_z_mm * 1e-3,
        t_max=parse(Float64, opts["t-max-us"]) * 1e-6,
        dt=parse(Float64, opts["dt-ns"]) * 1e-9,
        zero_pad_factor=parse(Int, opts["zero-pad-factor"]),
        receiver_aperture_y=parse_receiver_aperture_mm(receiver_aperture_y_spec),
        receiver_aperture_z=parse_receiver_aperture_mm(receiver_aperture_z_spec),
        peak_suppression_radius=parse(Float64, opts["peak-suppression-radius-mm"]) * 1e-3,
        success_tolerance=parse(Float64, opts["success-tolerance-mm"]) * 1e-3,
        axial_gain_power=parse(Float64, opts["axial-gain-power"]),
    )

    sources, emission_meta = if source_model == :point
        parse_point_sources_3d(opts)
    else
        parse_squiggle_sources_3d(opts, cfg_base)
    end
    bottom_margin_m = parse(Float64, opts["bottom-margin-mm"]) * 1e-3
    cfg = fit_pam_config_3d(cfg_base, sources; min_bottom_margin=bottom_margin_m)

    out_dir = if haskey(opts, "out-dir") && !isempty(strip(opts["out-dir"]))
        opts["out-dir"]
    else
        default_output_dir(opts, sources, cfg, emission_meta)
    end
    mkpath(out_dir)

    c, rho, medium_info = make_pam_medium_3d(cfg;
        aberrator             = aberrator,
        ct_path               = opts["ct-path"],
        slice_index_z         = parse(Int, opts["slice-index"]),
        skull_to_transducer   = parse(Float64, opts["skull-transducer-distance-mm"]) * 1e-3,
        hu_bone_thr           = parse(Int, opts["hu-bone-thr"]),
    )
    recon_frequencies = if haskey(opts, "recon-frequencies-mhz") && !isempty(strip(opts["recon-frequencies-mhz"]))
        parse_float_list(opts["recon-frequencies-mhz"]) .* 1e6
    else
        default_recon_frequencies(sources)
    end
    reconstruction_mode = resolve_reconstruction_mode(opts["recon-mode"], source_model)
    recon_bandwidth_hz = parse(Float64, opts["recon-bandwidth-khz"]) * 1e3
    window_config = make_window_config(opts, reconstruction_mode)
    source_phase_mode = parse_source_phase_mode(opts["source-phase-mode"])
    rng_sim = Random.MersenneTwister(parse(Int, opts["random-seed"]) + 1)
    source_variability = parse_source_variability(opts)
    if source_model == :squiggle
        emission_meta["activity_model"] = Dict(
            "activity_mode" => String(source_phase_mode),
            "frequency_jitter_percent" => source_variability.frequency_jitter_fraction * 100.0,
        )
    end

    sim_mode = parse_sim_mode(opts["sim-mode"])
    sim_mode == :analytic && aberrator == :skull && error("--sim-mode=analytic is not compatible with --aberrator=skull; use --sim-mode=kwave.")
    results = run_pam_case_3d(
        c,
        rho,
        sources,
        cfg;
        frequencies=recon_frequencies,
        bandwidth=recon_bandwidth_hz,
        use_gpu=parse_bool(opts["use-gpu"]),
        reconstruction_axial_step=parse(Float64, opts["recon-step-um"]) * 1e-6,
        reconstruction_mode=reconstruction_mode,
        window_config=window_config,
        show_progress=parse_bool(opts["recon-progress"]),
        benchmark=parse_bool(opts["benchmark"]),
        window_batch=parse(Int, opts["window-batch"]),
        sim_mode=sim_mode,
        source_phase_mode=source_phase_mode,
        rng=rng_sim,
        source_variability=source_variability,
    )

    medium_summary = Dict{String, Any}()
    for (key, value) in medium_info
        key == :mask && continue
        medium_summary[String(key)] = value
    end

    activity_boundary_path = joinpath(out_dir, "activity_boundaries.png")
    activity_boundary_metrics = save_threshold_boundary_detection_3d(
        activity_boundary_path,
        results[:pam_geo],
        results[:pam_hasa],
        results[:kgrid],
        cfg,
        sources;
        threshold_ratios=threshold_score_ratios,
        truth_radius=detection_truth_radius_m,
        c=c,
    )
    activity_boundary_metrics["auto_threshold_search"] = auto_threshold_search
    activity_boundary_metrics["display_threshold_mode"] = "selected_best_recall_precision"
    best_volume_path = joinpath(out_dir, "best_threshold_3d.png")
    best_volume_metrics = save_best_threshold_volume_3d(
        best_volume_path,
        results[:pam_hasa],
        results[:kgrid],
        cfg,
        sources;
        threshold=activity_boundary_metrics["best_hasa_threshold"],
        truth_radius=detection_truth_radius_m,
    )

    summary = Dict(
        "out_dir" => out_dir,
        "dimension" => 3,
        "reconstruction_source" => Dict("mode" => aberrator == :none ? "analytic_3d_water" : "heterogeneous_3d"),
        "activity_boundary_figure" => activity_boundary_path,
        "activity_boundary_metrics" => activity_boundary_metrics,
        "best_threshold_3d_figure" => best_volume_path,
        "best_threshold_3d_metrics" => best_volume_metrics,
        "sources" => [source_summary(src) for src in sources],
        "clusters" => [source_summary(src) for src in sources],
        "emission_meta" => emission_meta,
        "config" => Dict(
            "dx" => cfg.dx,
            "dy" => cfg.dy,
            "dz" => cfg.dz,
            "axial_dim" => cfg.axial_dim,
            "transverse_dim_y" => cfg.transverse_dim_y,
            "transverse_dim_z" => cfg.transverse_dim_z,
            "receiver_aperture_y" => cfg.receiver_aperture_y,
            "receiver_aperture_z" => cfg.receiver_aperture_z,
            "t_max" => cfg.t_max,
            "dt" => cfg.dt,
            "c0" => cfg.c0,
            "rho0" => cfg.rho0,
            "zero_pad_factor" => cfg.zero_pad_factor,
            "peak_suppression_radius" => cfg.peak_suppression_radius,
            "success_tolerance" => cfg.success_tolerance,
            "axial_gain_power" => cfg.axial_gain_power,
            "bottom_margin" => bottom_margin_m,
        ),
        "medium" => medium_summary,
        "reconstruction_frequencies_hz" => recon_frequencies,
        "reconstruction_bandwidth_hz" => recon_bandwidth_hz,
        "reconstruction_mode" => String(results[:reconstruction_mode]),
        "reconstruction_progress" => parse_bool(opts["recon-progress"]),
        "source_phase_mode" => String(results[:source_phase_mode]),
        "n_frames" => Int(get(results, :n_frames, 1)),
        "source_variability" => Dict(
            "frequency_jitter_percent" => source_variability.frequency_jitter_fraction * 100.0,
        ),
        "threshold_search" => Dict(
            "auto" => auto_threshold_search,
            "min_ratio" => minimum(threshold_score_ratios),
            "max_ratio" => maximum(threshold_score_ratios),
            "step" => auto_threshold_search ? parse(Float64, opts["auto-threshold-step"]) : nothing,
            "count" => length(threshold_score_ratios),
            "selection_metric" => "source_f1",
            "display_threshold_mode" => "selected_best_recall_precision",
        ),
        "physical_source_count" => length(sources),
        "emission_event_count" => Int(get(results, :emission_event_count, length(sources))),
        "window_config" => string_key_dict(results[:window_config]),
        "window_info" => Dict(
            "geometric" => compact_window_info(results[:geo_info]),
            "hasa" => compact_window_info(results[:hasa_info]),
        ),
        "benchmark" => parse_bool(opts["benchmark"]),
        "gpu_timing" => Dict(
            "geometric" => get(results[:geo_info], :gpu_timing, nothing),
            "hasa" => get(results[:hasa_info], :gpu_timing, nothing),
        ),
        "reconstruction_axial_step_m" => results[:geo_info][:axial_step],
        "reference_sound_speed_m_per_s" => results[:geo_info][:reference_sound_speed],
        "analysis_mode" => String(results[:analysis_mode]),
        "simulation" => Dict(
            "receiver_row" => results[:simulation][:receiver_row],
            "receiver_cols_y" => [first(results[:simulation][:receiver_cols_y]), last(results[:simulation][:receiver_cols_y])],
            "receiver_cols_z" => [first(results[:simulation][:receiver_cols_z]), last(results[:simulation][:receiver_cols_z])],
            "source_indices" => [[row, col_y, col_z] for (row, col_y, col_z) in get(results[:simulation], :source_indices, NTuple{3, Int}[])],
        ),
        "geometric" => results[:stats_geo],
        "hasa" => results[:stats_hasa],
    )

    open(joinpath(out_dir, "summary.json"), "w") do io
        JSON3.pretty(io, summary)
    end

    clusters = sources
    @save joinpath(out_dir, "result.jld2") c rho cfg clusters results medium_info

    save_napari_npz_3d(
        out_dir,
        results[:pam_geo],
        results[:pam_hasa],
        c, rho,
        results[:kgrid],
        cfg,
        sources;
        truth_radius=detection_truth_radius_m,
    )

    println("Saved 3D PAM outputs to $out_dir")
    exit()
end

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
        benchmark=parse_bool(opts["benchmark"]),
        window_batch=parse(Int, opts["window-batch"]),
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
            "vascular-squiggle-amplitude-x-mm", "vascular-squiggle-wavelength-mm",
            "vascular-squiggle-slope", "squiggle-phase-x-deg",
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
        benchmark=parse_bool(opts["benchmark"]),
        window_batch=parse(Int, opts["window-batch"]),
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
    c=c,
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
    "benchmark" => parse_bool(opts["benchmark"]),
    "gpu_timing" => Dict(
        "geometric" => get(results[:geo_info], :gpu_timing, nothing),
        "hasa" => get(results[:hasa_info], :gpu_timing, nothing),
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
