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
        "clusters-mm" => "30:0",
        "fundamental-mhz" => "0.5",
        "amplitude-pa" => "1.0",
        "n-bubbles" => "10",
        "harmonics" => "2,3",
        "harmonic-amplitudes" => "1.0,0.6",
        "cavitation-model" => "harmonic-cos",
        "gate-us" => "50",
        "taper-ratio" => "0.25",
        "axial-mm" => "80",
        "transverse-mm" => "60",
        "dx-mm" => "0.2",
        "dz-mm" => "0.2",
        "receiver-aperture-mm" => "50",
        "t-max-us" => "80",
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
        "recon-bandwidth-khz" => "20",
        "recon-step-um" => "50",
        "recon-mode" => "auto",
        "recon-window-us" => "10",
        "recon-hop-us" => "5",
        "recon-window-taper" => "hann",
        "recon-min-window-energy-ratio" => "0.001",
        "phase-mode" => "geometric",
        "phase-jitter-rad" => "0.2",
        "random-seed" => "0",
        "source-phase-mode" => "coherent",
        "n-realizations" => "1",
        "transducer-mm" => "-30:0",
        "delays-us" => "0",
        "cluster-model" => "vascular",
        "vascular-topology" => "squiggle",
        "vascular-length-mm" => "12",
        "vascular-branch-levels" => "2",
        "vascular-branch-angle-deg" => "30",
        "vascular-branch-scale" => "0.65",
        "vascular-squiggle-amplitude-mm" => "1.5",
        "vascular-squiggle-wavelength-mm" => "8",
        "vascular-squiggle-slope" => "0.0",
        "vascular-bundle-count" => "3",
        "vascular-bundle-spacing-mm" => "2.0",
        "vascular-source-spacing-mm" => "0.8",
        "vascular-position-jitter-mm" => "0.15",
        "vascular-min-separation-mm" => "0.3",
        "vascular-max-sources-per-anchor" => "0",
        "vascular-radius-mm" => "1.0",
        "activity-mode" => "burst-train",
        "activity-frame-us" => "10",
        "activity-hop-us" => "5",
        "activity-phase-jitter-rad" => "0.3",
        "activity-amplitude-jitter" => "0.5",
        "activity-active-probability" => "1.0",
        "analysis-mode" => "auto",
        "detection-threshold-ratio" => "0.2",
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

function parse_aberrator(s::AbstractString)
    value = Symbol(lowercase(strip(s)))
    value in (:none, :lens, :skull) || error("Unknown aberrator: $s")
    return value
end

function parse_cluster_model(s::AbstractString)
    value = Symbol(lowercase(strip(s)))
    value in (:point, :vascular) || error("--cluster-model must be point or vascular, got: $s")
    return value
end

function parse_vascular_topology(s::AbstractString)
    value = Symbol(lowercase(strip(s)))
    value in (:squiggle, :bundle, :tree) || error("--vascular-topology must be squiggle, bundle, or tree, got: $s")
    return value
end

function parse_cavitation_model(s::AbstractString)
    raw = lowercase(strip(s))
    value = Symbol(replace(raw, "-" => "_"))
    value in (:harmonic_cos, :gaussian_pulse) || error("--cavitation-model must be harmonic-cos or gaussian-pulse, got: $s")
    return value
end

function parse_activity_mode(s::AbstractString)
    value = Symbol(replace(lowercase(strip(s)), "-" => "_"))
    value in (:static, :burst_train) || error("--activity-mode must be static or burst-train, got: $s")
    return value
end

function parse_source_phase_mode(s::AbstractString)
    value = Symbol(replace(lowercase(strip(s)), "-" => "_"))
    value in (
        :coherent, :random_static_phase, :random_phase_per_window,
        :random_phase_per_realization, :stochastic_broadband,
    ) || error(
        "--source-phase-mode must be one of: coherent, random_static_phase, " *
        "random_phase_per_window, random_phase_per_realization, stochastic_broadband, got: $s",
    )
    return value
end

function parse_analysis_mode(s::AbstractString, cluster_model::Symbol)
    value = Symbol(lowercase(strip(s)))
    value == :auto && return cluster_model == :vascular ? :detection : :localization
    value in (:localization, :detection) || error("--analysis-mode must be auto, localization, or detection, got: $s")
    return value
end

function resolve_reconstruction_mode(s::AbstractString, cluster_model::Symbol)
    return TranscranialFUS.pam_reconstruction_mode(s, cluster_model)
end

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

function effective_recon_bandwidth_hz!(opts, provided_keys::Set{String}, reconstruction_mode::Symbol)
    if reconstruction_mode == :windowed && !("recon-bandwidth-khz" in provided_keys)
        opts["recon-bandwidth-khz"] = "150"
    end
    return parse(Float64, opts["recon-bandwidth-khz"]) * 1e3
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

function expand_cluster_values(values::Vector{Float64}, n::Int, default::Float64)
    isempty(values) && return fill(default, n)
    length(values) == 1 && return fill(values[1], n)
    length(values) == n && return values
    error("Per-cluster parameter list must have length 1 or match the number of clusters ($n).")
end

function geometric_drive_phase(depth::Real, lateral::Real, tx_depth::Real, tx_lateral::Real, n::Int, f0::Real, c0::Real)
    d = hypot(depth - tx_depth, lateral - tx_lateral)
    return -2π * n * f0 * d / c0
end

function parse_clusters(opts, cfg::PAMConfig)
    coord_tokens = [strip(token) for token in split(opts["clusters-mm"], ",") if !isempty(strip(token))]
    1 <= length(coord_tokens) <= 10 || error("Provide between 1 and 10 cluster anchors via --clusters-mm=depth:lateral,...")

    cluster_model = parse_cluster_model(opts["cluster-model"])
    f0 = parse(Float64, opts["fundamental-mhz"]) * 1e6
    harmonics = parse_int_list(opts["harmonics"])
    isempty(harmonics) && error("--harmonics must be a non-empty integer list.")
    harmonic_amplitudes = parse_float_list(opts["harmonic-amplitudes"])
    length(harmonic_amplitudes) == length(harmonics) ||
        error("--harmonic-amplitudes must have the same length as --harmonics ($(length(harmonics))).")

    gate = parse(Float64, opts["gate-us"]) * 1e-6
    taper = parse(Float64, opts["taper-ratio"])
    per_bubble_amp = parse(Float64, opts["amplitude-pa"])
    cavitation = parse_cavitation_model(opts["cavitation-model"])
    activity_mode = cluster_model == :vascular ? parse_activity_mode(opts["activity-mode"]) : :static

    n_clusters = length(coord_tokens)
    n_bubbles_per = expand_cluster_values(parse_float_list(opts["n-bubbles"]), n_clusters, 10.0)
    delays_us = expand_cluster_values(parse_float_list(opts["delays-us"]), n_clusters, 0.0)

    tx_depth, tx_lateral = parse_transducer_mm(opts["transducer-mm"])
    phase_mode = lowercase(strip(opts["phase-mode"]))
    phase_mode in ("coherent", "geometric", "random", "random_static_phase", "jittered") ||
        error("Unknown phase-mode: $phase_mode (expected coherent|geometric|random|random_static_phase|jittered).")
    rng = Random.MersenneTwister(parse(Int, opts["random-seed"]))
    jitter_rad = parse(Float64, opts["phase-jitter-rad"])

    anchors = Tuple{Float64, Float64}[]
    for token in coord_tokens
        parts = split(token, ":"; limit=2)
        length(parts) == 2 || error("Each cluster anchor must be specified as depth_mm:lateral_mm, got: $token")
        push!(anchors, (parse(Float64, strip(parts[1])) * 1e-3, parse(Float64, strip(parts[2])) * 1e-3))
    end

    clusters = EmissionSource2D[]
    vascular_meta_by_anchor = Dict{String, Any}[]
    all_centerlines_m = Any[]

    if cluster_model == :vascular
        vascular_topology = parse_vascular_topology(opts["vascular-topology"])
        max_sources_per_anchor_raw = parse(Int, opts["vascular-max-sources-per-anchor"])
        max_sources_per_anchor = max_sources_per_anchor_raw <= 0 ? nothing : max_sources_per_anchor_raw
        for (idx, anchor) in pairs(anchors)
            anchor_clusters, anchor_meta = make_vascular_bubble_clusters(
                [anchor];
                topology=vascular_topology,
                root_length=parse(Float64, opts["vascular-length-mm"]) * 1e-3,
                branch_levels=parse(Int, opts["vascular-branch-levels"]),
                branch_angle=deg2rad(parse(Float64, opts["vascular-branch-angle-deg"])),
                branch_scale=parse(Float64, opts["vascular-branch-scale"]),
                squiggle_amplitude=parse(Float64, opts["vascular-squiggle-amplitude-mm"]) * 1e-3,
                squiggle_wavelength=parse(Float64, opts["vascular-squiggle-wavelength-mm"]) * 1e-3,
                squiggle_slope=parse(Float64, opts["vascular-squiggle-slope"]),
                bundle_count=parse(Int, opts["vascular-bundle-count"]),
                bundle_spacing=parse(Float64, opts["vascular-bundle-spacing-mm"]) * 1e-3,
                source_spacing=parse(Float64, opts["vascular-source-spacing-mm"]) * 1e-3,
                position_jitter=parse(Float64, opts["vascular-position-jitter-mm"]) * 1e-3,
                min_separation=parse(Float64, opts["vascular-min-separation-mm"]) * 1e-3,
                max_sources_per_anchor=max_sources_per_anchor,
                depth_bounds=(0.0, Inf),
                lateral_bounds=(-cfg.transverse_dim / 2, cfg.transverse_dim / 2),
                fundamental=f0,
                amplitude=per_bubble_amp,
                n_bubbles=n_bubbles_per[idx],
                harmonics=harmonics,
                harmonic_amplitudes=harmonic_amplitudes,
                cavitation_model=cavitation,
                gate_duration=gate,
                taper_ratio=taper,
                delay=delays_us[idx] * 1e-6,
                phase_mode=Symbol(phase_mode),
                phase_jitter=jitter_rad,
                transducer_depth=tx_depth,
                transducer_lateral=tx_lateral,
                c0=cfg.c0,
                rng=rng,
            )
            append!(clusters, anchor_clusters)
            anchor_centerlines = centerlines_m(anchor_meta[:centerlines])
            append!(all_centerlines_m, anchor_centerlines)
            anchor_meta_dict = Dict(
                "anchor_m" => collect(anchor),
                "source_count" => length(anchor_clusters),
                "segments_m" => [collect(segment) for segment in anchor_meta[:segments]],
                "centerlines_m" => anchor_centerlines,
            )
            push!(vascular_meta_by_anchor, anchor_meta_dict)
        end
    else
        for (idx, anchor) in pairs(anchors)
            depth_m, lateral_m = anchor

            phases = Vector{Float64}(undef, length(harmonics))
            for (h_idx, n) in pairs(harmonics)
                base = if phase_mode in ("geometric", "jittered")
                    geometric_drive_phase(depth_m, lateral_m, tx_depth, tx_lateral, n, f0, cfg.c0)
                elseif phase_mode == "random"
                    2π * rand(rng)
                else
                    0.0
                end
                if phase_mode == "jittered"
                    base += randn(rng) * jitter_rad
                end
                phases[h_idx] = base
            end

            kwargs = (
                depth=depth_m,
                lateral=lateral_m,
                fundamental=f0,
                amplitude=per_bubble_amp,
                n_bubbles=n_bubbles_per[idx],
                harmonics=copy(harmonics),
                harmonic_amplitudes=copy(harmonic_amplitudes),
                harmonic_phases=phases,
                gate_duration=gate,
                taper_ratio=taper,
                delay=delays_us[idx] * 1e-6,
            )
            push!(clusters, cavitation == :gaussian_pulse ? GaussianPulseCluster2D(; kwargs...) : BubbleCluster2D(; kwargs...))
        end
    end

    physical_source_count = length(clusters)
    activity_meta_sym = Dict{Symbol, Any}()
    if cluster_model == :vascular && activity_mode == :burst_train
        clusters, activity_meta_sym = make_burst_train_sources(
            clusters;
            activity_mode=:burst_train,
            frame_duration=parse(Float64, opts["activity-frame-us"]) * 1e-6,
            hop=parse(Float64, opts["activity-hop-us"]) * 1e-6,
            amplitude_jitter=parse(Float64, opts["activity-amplitude-jitter"]),
            phase_jitter=parse(Float64, opts["activity-phase-jitter-rad"]),
            active_probability=parse(Float64, opts["activity-active-probability"]),
            rng=rng,
        )
    else
        clusters, activity_meta_sym = make_burst_train_sources(clusters; activity_mode=:static, rng=rng)
    end
    activity_meta = Dict(String(key) => value isa Symbol ? String(value) : value for (key, value) in activity_meta_sym)

    meta = Dict{String, Any}(
        "cluster_model" => String(cluster_model),
        "anchor_clusters_m" => [collect(anchor) for anchor in anchors],
        "n_anchor_clusters" => length(anchors),
        "n_emission_sources" => length(clusters),
        "physical_source_count" => physical_source_count,
        "emission_event_count" => length(clusters),
        "activity_model" => activity_meta,
        "phase_mode" => phase_mode,
        "fundamental_hz" => f0,
        "harmonics" => harmonics,
        "harmonic_amplitudes" => harmonic_amplitudes,
        "cavitation_model" => String(cavitation),
        "gate_duration_s" => gate,
        "transducer_m" => (tx_depth, tx_lateral),
        "phase_jitter_rad" => jitter_rad,
        "random_seed" => parse(Int, opts["random-seed"]),
        "n_bubbles_per_cluster" => n_bubbles_per,
        "delays_s" => delays_us .* 1e-6,
    )
    if cluster_model == :vascular
        meta["vascular"] = Dict(
            "topology" => opts["vascular-topology"],
            "length_m" => parse(Float64, opts["vascular-length-mm"]) * 1e-3,
            "branch_levels" => parse(Int, opts["vascular-branch-levels"]),
            "branch_angle_rad" => deg2rad(parse(Float64, opts["vascular-branch-angle-deg"])),
            "branch_scale" => parse(Float64, opts["vascular-branch-scale"]),
            "squiggle_amplitude_m" => parse(Float64, opts["vascular-squiggle-amplitude-mm"]) * 1e-3,
            "squiggle_wavelength_m" => parse(Float64, opts["vascular-squiggle-wavelength-mm"]) * 1e-3,
            "squiggle_slope" => parse(Float64, opts["vascular-squiggle-slope"]),
            "bundle_count" => parse(Int, opts["vascular-bundle-count"]),
            "bundle_spacing_m" => parse(Float64, opts["vascular-bundle-spacing-mm"]) * 1e-3,
            "source_spacing_m" => parse(Float64, opts["vascular-source-spacing-mm"]) * 1e-3,
            "position_jitter_m" => parse(Float64, opts["vascular-position-jitter-mm"]) * 1e-3,
            "min_separation_m" => parse(Float64, opts["vascular-min-separation-mm"]) * 1e-3,
            "max_sources_per_anchor" => parse(Int, opts["vascular-max-sources-per-anchor"]),
            "truth_radius_m" => parse(Float64, opts["vascular-radius-mm"]) * 1e-3,
            "centerlines_m" => all_centerlines_m,
            "anchors" => vascular_meta_by_anchor,
        )
    end
    return clusters, meta
end

function default_output_dir(opts, clusters, cfg, cluster_meta)
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    cluster_model = lowercase(cluster_meta["cluster_model"])
    vascular = haskey(cluster_meta, "vascular") ? cluster_meta["vascular"] : Dict{String, Any}()
    topology = cluster_model == "vascular" ? lowercase(String(get(vascular, "topology", "tree"))) : ""
    parts = String[
        timestamp,
        "run_pam_clusters",
        lowercase(opts["aberrator"]),
        cluster_model,
    ]
    !isempty(topology) && push!(parts, topology)
    append!(parts, String[
        "$(cluster_meta["n_anchor_clusters"])anchors",
        "$(length(clusters))src",
        "f$(slug_value(parse(Float64, opts["fundamental-mhz"]); digits=2))mhz",
        "h$(replace(opts["harmonics"], "," => ""))",
        lowercase(opts["phase-mode"]),
        replace(lowercase(opts["source-phase-mode"]), "_" => ""),
        "ax$(slug_value(cfg.axial_dim * 1e3; digits=0))mm",
        "lat$(slug_value(cfg.transverse_dim * 1e3; digits=0))mm",
    ])
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
    error("--from-run-dir reuses the previous RF simulation, medium, clusters, and grid. Remove simulation-specific option(s): $formatted")
end

function default_simulation_info(cfg::PAMConfig)
    return Dict{Symbol, Any}(
        :receiver_row => receiver_row(cfg),
        :receiver_cols => receiver_col_range(cfg),
        :source_indices => Tuple{Int, Int}[],
    )
end

function default_cluster_recon_frequencies(clusters)
    freqs = Float64[]
    for cluster in clusters
        append!(freqs, emission_frequencies(cluster))
    end
    return sort(unique(freqs))
end

json3_to_any(x) = x
json3_to_any(x::JSON3.Object) = Dict(String(key) => json3_to_any(value) for (key, value) in pairs(x))
json3_to_any(x::JSON3.Array) = [json3_to_any(value) for value in x]

function centerlines_from_cluster_meta(cluster_meta)
    haskey(cluster_meta, "vascular") || return nothing
    vascular = cluster_meta["vascular"]
    haskey(vascular, "centerlines_m") || return nothing
    centerlines = Vector{Tuple{Float64, Float64}}[]
    for raw_line in vascular["centerlines_m"]
        line = Tuple{Float64, Float64}[]
        for point in raw_line
            length(point) >= 2 || continue
            push!(line, (Float64(point[1]), Float64(point[2])))
        end
        length(line) >= 2 && push!(centerlines, line)
    end
    return isempty(centerlines) ? nothing : centerlines
end

function detection_truth_mask_from_meta(cluster_meta, kgrid, cfg, radius::Real)
    centerlines = centerlines_from_cluster_meta(cluster_meta)
    isnothing(centerlines) && return nothing
    return pam_centerline_truth_mask(centerlines, kgrid, cfg; radius=radius)
end

function map_db(map::AbstractMatrix{<:Real}, ref::Real)
    safe_ref = max(Float64(ref), eps(Float64))
    return 10 .* log10.(max.(Float64.(map), eps(Float64)) ./ safe_ref)
end

function scatter_cluster_points!(ax, clusters; color=:red, marker=nothing, markersize=nothing, strokewidth=2)
    depths_mm = [cl.depth * 1e3 for cl in clusters]
    laterals_mm = [cl.lateral * 1e3 for cl in clusters]
    marker = isnothing(marker) ? (length(clusters) > 20 ? :circle : :x) : marker
    markersize = isnothing(markersize) ? (length(clusters) > 20 ? 4 : 14) : markersize
    scatter!(ax, laterals_mm, depths_mm; color=color, marker=marker, markersize=markersize, strokewidth=strokewidth)
end

function scatter_predicted_points!(ax, stats; color=:white, marker=:circle, markersize=12, strokewidth=2)
    haskey(stats, :predicted_mm) || return nothing
    pred = stats[:predicted_mm]
    scatter!(ax, last.(pred), first.(pred); color=color, marker=marker, markersize=markersize, strokecolor=color, strokewidth=strokewidth)
end

function pam_stats_title(label, stats)
    if haskey(stats, :mean_radial_error_mm)
        return "$label | err = $(round(stats[:mean_radial_error_mm]; digits=2)) mm"
    elseif haskey(stats, :precision)
        precision = round(stats[:precision]; digits=2)
        recall = round(stats[:recall]; digits=2)
        f1 = round(stats[:f1]; digits=2)
        return "$label | P=$(precision), R=$(recall), F1=$(f1)"
    end
    return label
end

function save_overview(path, c, rf, pam_geo, pam_hasa, kgrid, cfg, clusters, stats_geo, stats_hasa)
    depth_mm = depth_coordinates(kgrid, cfg) .* 1e3
    lateral_mm = kgrid.y_vec .* 1e3
    time_us = collect(0:(size(rf, 2) - 1)) .* cfg.dt .* 1e6
    map_ref = max(maximum(Float64.(pam_geo)), maximum(Float64.(pam_hasa)), eps(Float64))
    pam_geo_db = map_db(pam_geo, map_ref)
    pam_hasa_db = map_db(pam_hasa, map_ref)

    fig = Figure(size=(1500, 1000))
    ax_medium = Axis(fig[1, 1]; title="Simulation Medium", xlabel="Lateral [mm]", ylabel="Depth [mm]", aspect=DataAspect())
    hm_medium = heatmap!(ax_medium, lateral_mm, depth_mm, Float64.(c)'; colormap=:thermal)
    hlines!(ax_medium, [0.0]; color=:white, linestyle=:dash)
    scatter_cluster_points!(ax_medium, clusters)
    Colorbar(fig[1, 2], hm_medium; label="c [m/s]")

    ax_rf = Axis(fig[1, 3]; title="Recorded RF", xlabel="Time [μs]", ylabel="Lateral [mm]")
    hm_rf = heatmap!(ax_rf, time_us, lateral_mm, rf'; colormap=:balance)
    Colorbar(fig[1, 4], hm_rf; label="Pressure [Pa]")

    ax_geo = Axis(fig[2, 1]; title=pam_stats_title("Geometric ASA", stats_geo),
                  xlabel="Lateral [mm]", ylabel="Depth [mm]", aspect=DataAspect())
    hm_geo = heatmap!(ax_geo, lateral_mm, depth_mm, pam_geo_db'; colormap=:viridis, colorrange=(-30, 0))
    hlines!(ax_geo, [0.0]; color=:white, linestyle=:dash)
    scatter_cluster_points!(ax_geo, clusters)
    scatter_predicted_points!(ax_geo, stats_geo)

    ax_hasa = Axis(fig[2, 3]; title=pam_stats_title("HASA", stats_hasa),
                   xlabel="Lateral [mm]", ylabel="Depth [mm]", aspect=DataAspect())
    hm_hasa = heatmap!(ax_hasa, lateral_mm, depth_mm, pam_hasa_db'; colormap=:viridis, colorrange=(-30, 0))
    hlines!(ax_hasa, [0.0]; color=:white, linestyle=:dash)
    scatter_cluster_points!(ax_hasa, clusters)
    scatter_predicted_points!(ax_hasa, stats_hasa)
    Colorbar(fig[2, 4], hm_hasa; label="Intensity [dB]")

    save(path, fig)
end

format_sci(value::Real) = @sprintf("%.2e", Float64(value))

function string_key_dict(dict::AbstractDict)
    return Dict(String(key) => value isa Symbol ? String(value) : value for (key, value) in dict)
end

function overlay_medium_contour!(ax, c::AbstractMatrix{<:Real}, lateral_mm, depth_mm, cfg; tol::Real=5.0)
    medium = Float64.(abs.(Float64.(c) .- cfg.c0) .> Float64(tol))
    any(medium .> 0.0) || return nothing
    contour!(ax, lateral_mm, depth_mm, medium'; levels=[0.5], color=(:white, 0.45), linewidth=1.2)
    return nothing
end

function pam_heatmap_title(label::AbstractString, metrics)
    rel_peak = round(metrics[:relative_peak_intensity]; digits=2)
    peak = format_sci(metrics[:peak_intensity])
    total = format_sci(metrics[:integrated_intensity_m2])
    return "$label | rel peak=$rel_peak, peak=$peak, sum=$total"
end

function add_pam_heatmap_panel!(
    fig,
    row::Int,
    label::AbstractString,
    intensity::AbstractMatrix{<:Real},
    c,
    kgrid,
    cfg,
    clusters,
    global_ref::Real,
    metrics,
)
    depth_mm = depth_coordinates(kgrid, cfg) .* 1e3
    lateral_mm = kgrid.y_vec .* 1e3
    norm_intensity = clamp.(Float64.(intensity) ./ max(Float64(global_ref), eps(Float64)), 0.0, 1.0)

    ax = Axis(
        fig[row, 1];
        title=pam_heatmap_title(label, metrics),
        xlabel=row == 2 ? "Lateral distance [mm]" : "",
        ylabel="Axial distance [mm]",
        aspect=DataAspect(),
    )
    hm = heatmap!(ax, lateral_mm, depth_mm, norm_intensity'; colormap=:turbo, colorrange=(0, 1))
    overlay_medium_contour!(ax, c, lateral_mm, depth_mm, cfg)
    scatter_cluster_points!(ax, clusters; color=(:white, 0.75), marker=:circle, markersize=3, strokewidth=0)
    xlims!(ax, minimum(lateral_mm), maximum(lateral_mm))
    ylims!(ax, minimum(depth_mm), maximum(depth_mm))
    return hm
end

function save_pam_heatmap(path, c, pam_geo, pam_hasa, kgrid, cfg, clusters; threshold_ratio::Real=0.2)
    global_ref = max(maximum(Float64.(pam_geo)), maximum(Float64.(pam_hasa)), eps(Float64))
    geo_metrics = pam_intensity_metrics(
        pam_geo,
        kgrid,
        cfg;
        threshold_ratio=threshold_ratio,
        reference_intensity=global_ref,
    )
    hasa_metrics = pam_intensity_metrics(
        pam_hasa,
        kgrid,
        cfg;
        threshold_ratio=threshold_ratio,
        reference_intensity=global_ref,
    )

    fig = Figure(size=(900, 1100), fontsize=22)
    hm = add_pam_heatmap_panel!(fig, 1, "Uncorrected", pam_geo, c, kgrid, cfg, clusters, global_ref, geo_metrics)
    add_pam_heatmap_panel!(fig, 2, "Corrected", pam_hasa, c, kgrid, cfg, clusters, global_ref, hasa_metrics)
    Colorbar(fig[1:2, 2], hm; label="Norm. PAM intensity (shared max)")

    save(path, fig)
    return Dict(
        "global_reference_intensity" => global_ref,
        "geometric" => string_key_dict(geo_metrics),
        "hasa" => string_key_dict(hasa_metrics),
    )
end

function cluster_points_mm(clusters)
    return [cl.lateral * 1e3 for cl in clusters], [cl.depth * 1e3 for cl in clusters]
end

function mask_points_mm(mask::BitMatrix, kgrid, cfg; max_points::Int=2500)
    idxs = findall(mask)
    isempty(idxs) && return Float64[], Float64[]

    stride = max(1, ceil(Int, length(idxs) / max_points))
    sampled = idxs[1:stride:end]
    depth_mm = depth_coordinates(kgrid, cfg) .* 1e3
    lateral_mm = kgrid.y_vec .* 1e3
    return [lateral_mm[idx[2]] for idx in sampled], [depth_mm[idx[1]] for idx in sampled]
end

function medium_points_mm(c::AbstractMatrix{<:Real}, kgrid, cfg; tol::Real=5.0, max_points::Int=5000)
    medium_mask = BitMatrix(abs.(Float64.(c) .- cfg.c0) .> Float64(tol))
    medium_mask[1:receiver_row(cfg), :] .= false
    return mask_points_mm(medium_mask, kgrid, cfg; max_points=max_points)
end

function centerline_points_mm(centerlines)
    isnothing(centerlines) && return Float64[], Float64[]
    lateral_mm = Float64[]
    depth_mm = Float64[]
    for line in centerlines
        for (depth, lateral) in line
            push!(lateral_mm, lateral * 1e3)
            push!(depth_mm, depth * 1e3)
        end
    end
    return lateral_mm, depth_mm
end

function lines_centerlines!(ax, centerlines; color=(:black, 0.45), linewidth=2)
    isnothing(centerlines) && return nothing
    for line in centerlines
        length(line) >= 2 || continue
        lines!(
            ax,
            [point[2] * 1e3 for point in line],
            [point[1] * 1e3 for point in line];
            color=color,
            linewidth=linewidth,
        )
    end
    return nothing
end

function nearest_detection_errors_mm(mask::BitMatrix, kgrid, cfg, clusters)
    pred_lateral_mm, pred_depth_mm = mask_points_mm(mask, kgrid, cfg; max_points=typemax(Int))
    isempty(pred_lateral_mm) && return Float64[], Float64[]

    lateral_errors = Float64[]
    depth_errors = Float64[]
    for cl in clusters
        truth_lateral = cl.lateral * 1e3
        truth_depth = cl.depth * 1e3
        best_idx = argmin((pred_lateral_mm .- truth_lateral) .^ 2 .+ (pred_depth_mm .- truth_depth) .^ 2)
        push!(lateral_errors, pred_lateral_mm[best_idx] - truth_lateral)
        push!(depth_errors, pred_depth_mm[best_idx] - truth_depth)
    end
    return lateral_errors, depth_errors
end

function pam_panel_limits(clusters, datasets...; min_depth_max_mm::Real=80.0)
    truth_lat, truth_depth = cluster_points_mm(clusters)
    all_lat = copy(truth_lat)
    all_depth = copy(truth_depth)
    for (lat, depth) in datasets
        append!(all_lat, lat)
        append!(all_depth, depth)
    end

    x_pad = max(2.0, 0.12 * max(maximum(all_lat) - minimum(all_lat), 1.0))
    y_pad = max(2.0, 0.12 * max(maximum(all_depth) - minimum(all_depth), 1.0))
    return (
        floor(minimum(all_lat) - x_pad),
        ceil(maximum(all_lat) + x_pad),
        floor(minimum(all_depth) - y_pad),
        max(Float64(min_depth_max_mm), ceil(maximum(all_depth) + y_pad)),
    )
end

function paper_style_figure_size(limits; px_per_mm::Real=8.0)
    x_range_mm = max(Float64(limits[2] - limits[1]), 1.0)
    y_range_mm = max(Float64(limits[4] - limits[3]), 1.0)
    axis_width_px = clamp(Float64(px_per_mm) * x_range_mm, 420.0, 620.0)
    axis_height_px = axis_width_px * y_range_mm / x_range_mm
    figure_width_px = round(Int, axis_width_px + 170.0)
    figure_height_px = round(Int, 2.0 * axis_height_px + 210.0)
    return (figure_width_px, figure_height_px)
end

function add_paper_style_panel!(
    fig,
    row::Int,
    title::AbstractString,
    intensity,
    c,
    kgrid,
    cfg,
    clusters;
    color,
    threshold_ratio::Real,
    limits,
    show_xlabel::Bool,
    truth_centerlines=nothing,
)
    mask = threshold_pam_map(intensity, cfg; threshold_ratio=threshold_ratio)
    pred_lat, pred_depth = mask_points_mm(mask, kgrid, cfg)
    truth_lat, truth_depth = cluster_points_mm(clusters)
    medium_lat, medium_depth = medium_points_mm(c, kgrid, cfg)

    ax = Axis(
        fig[row, 1];
        xlabel=show_xlabel ? "Lateral Distance [mm]" : "",
        ylabel=row == 1 ? "Axial Distance [mm]" : "",
        yreversed=true,
        xgridvisible=false,
        ygridvisible=false,
        xticks=WilkinsonTicks(3),
        yticks=WilkinsonTicks(3),
        aspect=DataAspect(),
    )
    hidespines!(ax, :t, :r)
    xlims!(ax, limits[1], limits[2])
    ylims!(ax, limits[3], limits[4])

    scatter!(ax, medium_lat, medium_depth; color=(:gray70, 0.35), markersize=4, marker=:rect)
    lines_centerlines!(ax, truth_centerlines; color=(:gray20, 0.55), linewidth=2)
    scatter!(ax, truth_lat, truth_depth; color=(:gray45, 0.75), markersize=5, marker=:rect)
    scatter!(ax, pred_lat, pred_depth; color=(color, 0.85), markersize=6, marker=:circle)
    text!(ax, 0.92, 0.86; text=title, space=:relative, align=(:right, :center), color=color, fontsize=30)

    lat_err, depth_err = nearest_detection_errors_mm(mask, kgrid, cfg, clusters)
    inset = Axis(
        fig[row, 1];
        width=90,
        height=115,
        halign=:left,
        valign=:top,
        tellwidth=false,
        tellheight=false,
        yreversed=true,
        title=row == 1 ? "Error" : "",
        titlesize=20,
        xgridvisible=false,
        ygridvisible=false,
        backgroundcolor=(:white, 0.9),
    )
    hidespines!(inset, :t, :r)
    hidedecorations!(inset; grid=false)
    xlims!(inset, -10, 10)
    ylims!(inset, -10, 10)
    vlines!(inset, [0.0]; color=:gray65, linewidth=1)
    hlines!(inset, [0.0]; color=:gray65, linewidth=1)
    scatter!(inset, lat_err, depth_err; color=(color, 0.85), markersize=3.5)
    if row == 1
        text!(ax, 0.23, 0.69; text="1 cm", space=:relative, align=(:left, :center), color=:black, fontsize=24)
    end

    return pred_lat, pred_depth
end

function save_paper_style_detection(path, c, pam_geo, pam_hasa, kgrid, cfg, clusters; threshold_ratio::Real, truth_centerlines=nothing)
    geo_mask = threshold_pam_map(pam_geo, cfg; threshold_ratio=threshold_ratio)
    hasa_mask = threshold_pam_map(pam_hasa, cfg; threshold_ratio=threshold_ratio)
    geo_points = mask_points_mm(geo_mask, kgrid, cfg)
    hasa_points = mask_points_mm(hasa_mask, kgrid, cfg)
    centerline_points = centerline_points_mm(truth_centerlines)
    limits = pam_panel_limits(clusters, geo_points, hasa_points, centerline_points; min_depth_max_mm=80.0)

    fig = Figure(size=paper_style_figure_size(limits), fontsize=24)
    add_paper_style_panel!(
        fig,
        1,
        "Uncorrected",
        pam_geo,
        c,
        kgrid,
        cfg,
        clusters;
        color=RGBf(0.55, 0.0, 0.34),
        threshold_ratio=threshold_ratio,
        limits=limits,
        show_xlabel=false,
        truth_centerlines=truth_centerlines,
    )
    add_paper_style_panel!(
        fig,
        2,
        "Corrected",
        pam_hasa,
        c,
        kgrid,
        cfg,
        clusters;
        color=RGBf(1.0, 0.28, 0.04),
        threshold_ratio=threshold_ratio,
        limits=limits,
        show_xlabel=true,
        truth_centerlines=truth_centerlines,
    )
    rowgap!(fig.layout, 8)
    save(path, fig)
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

opts, provided_keys = parse_cli(ARGS)
from_run_dir = strip(opts["from-run-dir"])
peak_method = Symbol(lowercase(strip(opts["peak-method"])))
peak_method in (:argmax, :clean) || error("--peak-method must be argmax or clean, got: $(opts["peak-method"])")
detection_truth_radius_m = parse(Float64, opts["vascular-radius-mm"]) * 1e-3
detection_threshold_ratio = parse(Float64, opts["detection-threshold-ratio"])

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

    clusters, cluster_meta = parse_clusters(opts, cfg_base)

    aberrator = parse_aberrator(opts["aberrator"])
    bottom_margin_m = parse(Float64, opts["bottom-margin-mm"]) * 1e-3
    cfg = fit_pam_config(
        cfg_base,
        clusters;
        min_bottom_margin=bottom_margin_m,
        reference_depth=aberrator == :skull ? parse(Float64, opts["skull-transducer-distance-mm"]) * 1e-3 : nothing,
    )

    out_dir = if haskey(opts, "out-dir") && !isempty(strip(opts["out-dir"]))
        opts["out-dir"]
    else
        default_output_dir(opts, clusters, cfg, cluster_meta)
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
        default_cluster_recon_frequencies(clusters)
    end
    cluster_model = parse_cluster_model(opts["cluster-model"])
    reconstruction_mode = resolve_reconstruction_mode(opts["recon-mode"], cluster_model)
    recon_bandwidth_hz = effective_recon_bandwidth_hz!(opts, provided_keys, reconstruction_mode)
    window_config = make_window_config(opts, reconstruction_mode)
    analysis_mode = parse_analysis_mode(opts["analysis-mode"], cluster_model)
    truth_centerlines = centerlines_from_cluster_meta(cluster_meta)
    detection_truth_mask = detection_truth_mask_from_meta(cluster_meta, pam_grid(cfg), cfg, detection_truth_radius_m)

    source_phase_mode = parse_source_phase_mode(opts["source-phase-mode"])
    n_realizations = parse(Int, opts["n-realizations"])
    rng_sim = Random.MersenneTwister(parse(Int, opts["random-seed"]) + 1)

    results = run_pam_case(
        c,
        rho,
        clusters,
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
    )
    reconstruction_source = Dict("mode" => "simulation")
else
    reject_cached_simulation_options!(
        provided_keys,
        (
            "clusters-mm", "fundamental-mhz", "amplitude-pa", "n-bubbles",
            "harmonics", "harmonic-amplitudes", "cavitation-model", "gate-us", "taper-ratio",
            "axial-mm", "transverse-mm", "dx-mm", "dz-mm", "receiver-aperture-mm",
            "t-max-us", "dt-ns", "zero-pad-factor", "peak-suppression-radius-mm",
            "success-tolerance-mm", "aberrator", "ct-path", "slice-index",
            "skull-transducer-distance-mm", "bottom-margin-mm", "hu-bone-thr",
            "lens-depth-mm", "lens-lateral-mm", "lens-axial-radius-mm", "lens-lateral-radius-mm",
            "aberrator-c", "aberrator-rho", "use-gpu", "phase-mode", "phase-jitter-rad",
            "random-seed", "transducer-mm", "delays-us", "cluster-model",
            "vascular-topology", "vascular-length-mm", "vascular-branch-levels", "vascular-branch-angle-deg",
            "vascular-branch-scale", "vascular-squiggle-amplitude-mm", "vascular-squiggle-wavelength-mm",
            "vascular-squiggle-slope", "vascular-bundle-count", "vascular-bundle-spacing-mm",
            "vascular-source-spacing-mm", "vascular-position-jitter-mm",
            "vascular-min-separation-mm", "vascular-max-sources-per-anchor",
            "activity-mode", "activity-frame-us", "activity-hop-us",
            "activity-phase-jitter-rad", "activity-amplitude-jitter",
            "activity-active-probability",
            "source-phase-mode", "n-realizations",
        ),
    )
    cached_path = joinpath(from_run_dir, "result.jld2")
    isfile(cached_path) || error("--from-run-dir must contain result.jld2, missing: $cached_path")
    cached = load(cached_path)
    c = cached["c"]
    rho = haskey(cached, "rho") ? cached["rho"] : fill(Float32(cached["cfg"].rho0), size(c))
    cfg = cached["cfg"]
    clusters = cached["clusters"]
    cached_results = cached["results"]
    rf = cached_results[:rf]
    medium_info = haskey(cached, "medium_info") ? cached["medium_info"] : Dict{Symbol, Any}(:aberrator => :cached)
    bottom_margin_m = nothing
    cached_summary_path = joinpath(from_run_dir, "summary.json")
    cached_summary = isfile(cached_summary_path) ? JSON3.read(read(cached_summary_path, String)) : nothing
    cluster_meta = if !isnothing(cached_summary) && hasproperty(cached_summary, :emission_meta)
        Dict{String, Any}(json3_to_any(cached_summary.emission_meta))
    else
        harmonics = Int[]
        for cluster in clusters
            append!(harmonics, cluster.harmonics)
        end
        Dict{String, Any}(
            "source_count" => length(clusters),
            "fundamental_hz" => isempty(clusters) ? NaN : first(clusters).fundamental,
            "harmonics" => sort(unique(harmonics)),
            "cavitation_model" => isempty(clusters) ? "unknown" : String(cavitation_model(first(clusters))),
        )
    end
    get!(cluster_meta, "cluster_model", length(clusters) > 10 ? "vascular" : "point")
    cluster_meta["from_run_dir"] = abspath(from_run_dir)

    out_dir = if haskey(opts, "out-dir") && !isempty(strip(opts["out-dir"]))
        opts["out-dir"]
    else
        default_reconstruction_output_dir(from_run_dir)
    end
    mkpath(out_dir)

    recon_frequencies = if haskey(opts, "recon-frequencies-mhz") && !isempty(strip(opts["recon-frequencies-mhz"]))
        parse_float_list(opts["recon-frequencies-mhz"]) .* 1e6
    else
        default_cluster_recon_frequencies(clusters)
    end
    cached_model = Symbol(String(cluster_meta["cluster_model"]))
    reconstruction_mode = resolve_reconstruction_mode(opts["recon-mode"], cached_model)
    recon_bandwidth_hz = effective_recon_bandwidth_hz!(opts, provided_keys, reconstruction_mode)
    window_config = make_window_config(opts, reconstruction_mode)
    analysis_mode = parse_analysis_mode(opts["analysis-mode"], cached_model)
    simulation_info = haskey(cached_results, :simulation) ? cached_results[:simulation] : default_simulation_info(cfg)
    truth_centerlines = centerlines_from_cluster_meta(cluster_meta)
    detection_truth_mask = detection_truth_mask_from_meta(cluster_meta, pam_grid(cfg), cfg, detection_truth_radius_m)
    results = reconstruct_pam_case(
        rf,
        c,
        clusters,
        cfg;
        simulation_info=simulation_info,
        frequencies=recon_frequencies,
        bandwidth=recon_bandwidth_hz,
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
    results[:kgrid], cfg, clusters, results[:stats_geo], results[:stats_hasa],
)

heatmap_path = joinpath(out_dir, "pam_heatmap.png")
heatmap_metrics = save_pam_heatmap(
    heatmap_path,
    c,
    results[:pam_geo],
    results[:pam_hasa],
    results[:kgrid],
    cfg,
    clusters;
    threshold_ratio=detection_threshold_ratio,
)

paper_style_path = joinpath(out_dir, "paper_style.png")
save_paper_style_detection(
    paper_style_path,
    c,
    results[:pam_geo],
    results[:pam_hasa],
    results[:kgrid],
    cfg,
    clusters;
    threshold_ratio=detection_threshold_ratio,
    truth_centerlines=truth_centerlines,
)

summary = Dict(
    "out_dir" => out_dir,
    "reconstruction_source" => reconstruction_source,
    "paper_style_figure" => paper_style_path,
    "heatmap_figure" => heatmap_path,
    "pam_heatmap_metrics" => heatmap_metrics,
    "clusters" => [Dict(
        "depth_m" => cl.depth,
        "lateral_m" => cl.lateral,
        "fundamental_hz" => cl.fundamental,
        "amplitude_pa" => cl.amplitude,
        "n_bubbles" => cl.n_bubbles,
        "harmonics" => cl.harmonics,
        "harmonic_amplitudes" => cl.harmonic_amplitudes,
        "harmonic_phases_rad" => cl.harmonic_phases,
        "cavitation_model" => String(cavitation_model(cl)),
        "gate_duration_s" => cl.gate_duration,
        "delay_s" => cl.delay,
    ) for cl in clusters],
    "emission_meta" => cluster_meta,
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
    "source_phase_mode" => String(get(results, :source_phase_mode, :coherent)),
    "n_realizations" => Int(get(results, :n_realizations, 1)),
    "window_config" => string_key_dict(results[:window_config]),
    "window_info" => Dict(
        "geometric" => compact_window_info(results[:geo_info]),
        "hasa" => compact_window_info(results[:hasa_info]),
    ),
    "reconstruction_axial_step_m" => results[:geo_info][:axial_step],
    "reference_sound_speed_m_per_s" => results[:geo_info][:reference_sound_speed],
    "activity_model" => get(cluster_meta, "activity_model", Dict("activity_mode" => "static")),
    "physical_source_count" => get(cluster_meta, "physical_source_count", length(clusters)),
    "emission_event_count" => get(cluster_meta, "emission_event_count", length(clusters)),
    "analysis_mode" => String(analysis_mode),
    "detection_truth_radius_m" => detection_truth_radius_m,
    "detection_truth_mode" => isnothing(detection_truth_mask) ? "source_disks" : "centerline_tube",
    "detection_threshold_ratio" => detection_threshold_ratio,
    "peak_method" => String(peak_method),
    "clean_loop_gain" => parse(Float64, opts["clean-loop-gain"]),
    "clean_max_iter" => parse(Int, opts["clean-max-iter"]),
    "clean_threshold_ratio" => parse(Float64, opts["clean-threshold-ratio"]),
    "simulation" => Dict(
        "receiver_row" => results[:simulation][:receiver_row],
        "receiver_cols" => [first(results[:simulation][:receiver_cols]), last(results[:simulation][:receiver_cols])],
        "source_indices" => [[row, col] for (row, col) in results[:simulation][:source_indices]],
    ),
    "geometric" => results[:stats_geo],
    "hasa" => results[:stats_hasa],
)

open(joinpath(out_dir, "summary.json"), "w") do io
    JSON3.pretty(io, summary)
end

@save joinpath(out_dir, "result.jld2") c rho cfg clusters results medium_info

println("Saved PAM cluster outputs to $out_dir")
