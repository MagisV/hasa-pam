#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Dates
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
        "gate-us" => "50",
        "taper-ratio" => "0.25",
        "axial-mm" => "60",
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
        "phase-mode" => "geometric",
        "phase-jitter-rad" => "0.2",
        "random-seed" => "0",
        "transducer-mm" => "-30:0",
        "delays-us" => "0",
        "cluster-model" => "vascular",
        "vascular-length-mm" => "12",
        "vascular-branch-levels" => "2",
        "vascular-branch-angle-deg" => "30",
        "vascular-branch-scale" => "0.65",
        "vascular-source-spacing-mm" => "0.8",
        "vascular-position-jitter-mm" => "0.15",
        "vascular-min-separation-mm" => "0.3",
        "vascular-max-sources-per-anchor" => "0",
        "vascular-radius-mm" => "1.0",
        "analysis-mode" => "auto",
        "detection-threshold-ratio" => "0.2",
        "peak-method" => "argmax",
        "clean-loop-gain" => "0.1",
        "clean-max-iter" => "500",
        "clean-threshold-ratio" => "0.01",
    )

    for arg in args
        startswith(arg, "--") || error("Unsupported argument format: $arg")
        parts = split(arg[3:end], "="; limit=2)
        length(parts) == 2 || error("Arguments must use --name=value, got: $arg")
        opts[parts[1]] = parts[2]
    end
    return opts
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

function parse_analysis_mode(s::AbstractString, cluster_model::Symbol)
    value = Symbol(lowercase(strip(s)))
    value == :auto && return cluster_model == :vascular ? :detection : :localization
    value in (:localization, :detection) || error("--analysis-mode must be auto, localization, or detection, got: $s")
    return value
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

    n_clusters = length(coord_tokens)
    n_bubbles_per = expand_cluster_values(parse_float_list(opts["n-bubbles"]), n_clusters, 10.0)
    delays_us = expand_cluster_values(parse_float_list(opts["delays-us"]), n_clusters, 0.0)

    tx_depth, tx_lateral = parse_transducer_mm(opts["transducer-mm"])
    phase_mode = lowercase(strip(opts["phase-mode"]))
    phase_mode in ("coherent", "geometric", "random", "jittered") ||
        error("Unknown phase-mode: $phase_mode (expected coherent|geometric|random|jittered).")
    rng = Random.MersenneTwister(parse(Int, opts["random-seed"]))
    jitter_rad = parse(Float64, opts["phase-jitter-rad"])

    anchors = Tuple{Float64, Float64}[]
    for token in coord_tokens
        parts = split(token, ":"; limit=2)
        length(parts) == 2 || error("Each cluster anchor must be specified as depth_mm:lateral_mm, got: $token")
        push!(anchors, (parse(Float64, strip(parts[1])) * 1e-3, parse(Float64, strip(parts[2])) * 1e-3))
    end

    clusters = BubbleCluster2D[]
    vascular_meta_by_anchor = Dict{String, Any}[]

    if cluster_model == :vascular
        max_sources_per_anchor_raw = parse(Int, opts["vascular-max-sources-per-anchor"])
        max_sources_per_anchor = max_sources_per_anchor_raw <= 0 ? nothing : max_sources_per_anchor_raw
        for (idx, anchor) in pairs(anchors)
            anchor_clusters, anchor_meta = make_vascular_bubble_clusters(
                [anchor];
                root_length=parse(Float64, opts["vascular-length-mm"]) * 1e-3,
                branch_levels=parse(Int, opts["vascular-branch-levels"]),
                branch_angle=deg2rad(parse(Float64, opts["vascular-branch-angle-deg"])),
                branch_scale=parse(Float64, opts["vascular-branch-scale"]),
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
            push!(vascular_meta_by_anchor, Dict(
                "anchor_m" => collect(anchor),
                "source_count" => length(anchor_clusters),
                "segments_m" => [collect(segment) for segment in anchor_meta[:segments]],
            ))
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

            push!(clusters, BubbleCluster2D(
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
            ))
        end
    end

    meta = Dict{String, Any}(
        "cluster_model" => String(cluster_model),
        "anchor_clusters_m" => [collect(anchor) for anchor in anchors],
        "n_anchor_clusters" => length(anchors),
        "n_emission_sources" => length(clusters),
        "phase_mode" => phase_mode,
        "fundamental_hz" => f0,
        "harmonics" => harmonics,
        "harmonic_amplitudes" => harmonic_amplitudes,
        "gate_duration_s" => gate,
        "transducer_m" => (tx_depth, tx_lateral),
        "phase_jitter_rad" => jitter_rad,
        "random_seed" => parse(Int, opts["random-seed"]),
        "n_bubbles_per_cluster" => n_bubbles_per,
        "delays_s" => delays_us .* 1e-6,
    )
    if cluster_model == :vascular
        meta["vascular"] = Dict(
            "length_m" => parse(Float64, opts["vascular-length-mm"]) * 1e-3,
            "branch_levels" => parse(Int, opts["vascular-branch-levels"]),
            "branch_angle_rad" => deg2rad(parse(Float64, opts["vascular-branch-angle-deg"])),
            "branch_scale" => parse(Float64, opts["vascular-branch-scale"]),
            "source_spacing_m" => parse(Float64, opts["vascular-source-spacing-mm"]) * 1e-3,
            "position_jitter_m" => parse(Float64, opts["vascular-position-jitter-mm"]) * 1e-3,
            "min_separation_m" => parse(Float64, opts["vascular-min-separation-mm"]) * 1e-3,
            "max_sources_per_anchor" => parse(Int, opts["vascular-max-sources-per-anchor"]),
            "truth_radius_m" => parse(Float64, opts["vascular-radius-mm"]) * 1e-3,
            "anchors" => vascular_meta_by_anchor,
        )
    end
    return clusters, meta
end

function default_output_dir(opts, clusters, cfg, cluster_meta)
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    parts = String[
        timestamp,
        "run_pam_clusters",
        lowercase(opts["aberrator"]),
        lowercase(cluster_meta["cluster_model"]),
        "$(cluster_meta["n_anchor_clusters"])anchors",
        "$(length(clusters))src",
        "f$(slug_value(parse(Float64, opts["fundamental-mhz"]); digits=2))mhz",
        "h$(replace(opts["harmonics"], "," => ""))",
        lowercase(opts["phase-mode"]),
        "ax$(slug_value(cfg.axial_dim * 1e3; digits=0))mm",
        "lat$(slug_value(cfg.transverse_dim * 1e3; digits=0))mm",
    ]
    if lowercase(opts["aberrator"]) == "skull"
        insert!(parts, length(parts), "slice" * opts["slice-index"])
        insert!(parts, length(parts), "st$(slug_value(parse(Float64, opts["skull-transducer-distance-mm"]); digits=1))mm")
    end
    return joinpath(pwd(), "outputs", join(parts, "_"))
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

function pam_panel_limits(clusters, datasets...)
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
        ceil(maximum(all_depth) + y_pad),
    )
end

function add_paper_style_panel!(
    fig,
    row::Int,
    title::AbstractString,
    intensity,
    kgrid,
    cfg,
    clusters;
    color,
    threshold_ratio::Real,
    limits,
    show_xlabel::Bool,
)
    mask = threshold_pam_map(intensity, cfg; threshold_ratio=threshold_ratio)
    pred_lat, pred_depth = mask_points_mm(mask, kgrid, cfg)
    truth_lat, truth_depth = cluster_points_mm(clusters)

    ax = Axis(
        fig[row, 1];
        xlabel=show_xlabel ? "Lateral Distance [mm]" : "",
        ylabel=row == 1 ? "Axial Distance [mm]" : "",
        yreversed=true,
        xgridvisible=false,
        ygridvisible=false,
        xticks=WilkinsonTicks(3),
        yticks=WilkinsonTicks(3),
    )
    hidespines!(ax, :t, :r)
    xlims!(ax, limits[1], limits[2])
    ylims!(ax, limits[3], limits[4])

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

function save_paper_style_detection(path, pam_geo, pam_hasa, kgrid, cfg, clusters; threshold_ratio::Real)
    geo_mask = threshold_pam_map(pam_geo, cfg; threshold_ratio=threshold_ratio)
    hasa_mask = threshold_pam_map(pam_hasa, cfg; threshold_ratio=threshold_ratio)
    geo_points = mask_points_mm(geo_mask, kgrid, cfg)
    hasa_points = mask_points_mm(hasa_mask, kgrid, cfg)
    limits = pam_panel_limits(clusters, geo_points, hasa_points)

    fig = Figure(size=(760, 760), fontsize=24)
    add_paper_style_panel!(
        fig,
        1,
        "Uncorrected",
        pam_geo,
        kgrid,
        cfg,
        clusters;
        color=RGBf(0.55, 0.0, 0.34),
        threshold_ratio=threshold_ratio,
        limits=limits,
        show_xlabel=false,
    )
    add_paper_style_panel!(
        fig,
        2,
        "Corrected",
        pam_hasa,
        kgrid,
        cfg,
        clusters;
        color=RGBf(1.0, 0.28, 0.04),
        threshold_ratio=threshold_ratio,
        limits=limits,
        show_xlabel=true,
    )
    rowgap!(fig.layout, 8)
    save(path, fig)
end

opts = parse_cli(ARGS)

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
cfg = fit_pam_config(
    cfg_base,
    clusters;
    min_bottom_margin=parse(Float64, opts["bottom-margin-mm"]) * 1e-3,
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

medium_summary = Dict{String, Any}()
for (key, value) in medium_info
    key == :mask && continue
    medium_summary[String(key)] = value
end

recon_frequencies = if haskey(opts, "recon-frequencies-mhz") && !isempty(strip(opts["recon-frequencies-mhz"]))
    parse_float_list(opts["recon-frequencies-mhz"]) .* 1e6
else
    sort(unique(Float64[n * cluster_meta["fundamental_hz"] for n in cluster_meta["harmonics"]]))
end
recon_bandwidth_hz = parse(Float64, opts["recon-bandwidth-khz"]) * 1e3
analysis_mode = parse_analysis_mode(opts["analysis-mode"], parse_cluster_model(opts["cluster-model"]))
peak_method = Symbol(lowercase(strip(opts["peak-method"])))
peak_method in (:argmax, :clean) || error("--peak-method must be argmax or clean, got: $(opts["peak-method"])")

results = run_pam_case(
    c,
    rho,
    clusters,
    cfg;
    frequencies=recon_frequencies,
    bandwidth=recon_bandwidth_hz,
    use_gpu=parse_bool(opts["use-gpu"]),
    analysis_mode=analysis_mode,
    peak_method=peak_method,
    clean_loop_gain=parse(Float64, opts["clean-loop-gain"]),
    clean_max_iter=parse(Int, opts["clean-max-iter"]),
    clean_threshold_ratio=parse(Float64, opts["clean-threshold-ratio"]),
    detection_truth_radius=parse(Float64, opts["vascular-radius-mm"]) * 1e-3,
    detection_threshold_ratio=parse(Float64, opts["detection-threshold-ratio"]),
)

save_overview(
    joinpath(out_dir, "overview.png"),
    c, results[:rf], results[:pam_geo], results[:pam_hasa],
    results[:kgrid], cfg, clusters, results[:stats_geo], results[:stats_hasa],
)

paper_style_path = joinpath(out_dir, "paper_style.png")
save_paper_style_detection(
    paper_style_path,
    results[:pam_geo],
    results[:pam_hasa],
    results[:kgrid],
    cfg,
    clusters;
    threshold_ratio=parse(Float64, opts["detection-threshold-ratio"]),
)

summary = Dict(
    "out_dir" => out_dir,
    "paper_style_figure" => paper_style_path,
    "clusters" => [Dict(
        "depth_m" => cl.depth,
        "lateral_m" => cl.lateral,
        "fundamental_hz" => cl.fundamental,
        "amplitude_pa" => cl.amplitude,
        "n_bubbles" => cl.n_bubbles,
        "harmonics" => cl.harmonics,
        "harmonic_amplitudes" => cl.harmonic_amplitudes,
        "harmonic_phases_rad" => cl.harmonic_phases,
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
        "bottom_margin" => parse(Float64, opts["bottom-margin-mm"]) * 1e-3,
    ),
    "medium" => medium_summary,
    "reconstruction_frequencies_hz" => recon_frequencies,
    "reconstruction_bandwidth_hz" => recon_bandwidth_hz,
    "analysis_mode" => String(analysis_mode),
    "detection_truth_radius_m" => parse(Float64, opts["vascular-radius-mm"]) * 1e-3,
    "detection_threshold_ratio" => parse(Float64, opts["detection-threshold-ratio"]),
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
