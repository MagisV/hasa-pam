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
import TranscranialFUS: parse_cli, slug_value, parse_bool, parse_dimension, parse_float_list, parse_int_list,
    parse_threshold_ratios, parse_threshold_search_ratios, parse_source_model, parse_aberrator,
    parse_simulation_backend, parse_source_phase_mode, parse_source_variability,
    source_variability_from_summary, parse_analysis_mode, resolve_reconstruction_mode,
    make_window_config, parse_receiver_aperture_mm, parse_transducer_mm, parse_point_sources,
    parse_point_sources_3d, parse_squiggle_sources_3d, parse_network_sources_3d,
    parse_squiggle_sources, parse_sources, default_simulation_info, default_recon_frequencies,
    analytic_rf_for_point_sources_3d


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
    if occursin("squiggle", source_model) || occursin("network", source_model)
        count_key = haskey(emission_meta, "n_anchor_clusters") ? "n_anchor_clusters" : "n_network_centers"
        label = occursin("network", source_model) ? "centers" : "anchors"
        insert!(parts, 5, "$(emission_meta[count_key])$(label)")
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


function run_pam_case_3d(
    c::AbstractArray{<:Real, 3},
    rho::AbstractArray{<:Real, 3},
    sources::AbstractVector{<:EmissionSource3D},
    cfg::PAMConfig3D;
    frequencies::Union{Nothing, AbstractVector{<:Real}}=nothing,
    bandwidth::Real=0.0,
    kwave_use_gpu::Bool=true,
    recon_use_gpu::Bool=true,
    reconstruction_axial_step::Union{Nothing, Real}=nothing,
    reconstruction_mode::Symbol=:full,
    window_config::PAMWindowConfig=PAMWindowConfig(),
    show_progress::Bool=false,
    benchmark::Bool=false,
    window_batch::Int=1,
    simulation_backend::Symbol=:analytic,
    source_phase_mode::Symbol=:coherent,
    rng::Random.AbstractRNG=Random.default_rng(),
    source_variability::SourceVariabilityConfig=SourceVariabilityConfig(),
)
    recon_use_gpu || error("3D PAM reconstruction currently requires --recon-use-gpu=true.")
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
    end
    rf, grid, sim_info = if simulation_backend == :kwave
        simulate_point_sources_3d(c, rho, sim_sources, cfg; use_gpu=kwave_use_gpu)
    else
        analytic_rf_for_point_sources_3d(cfg, sim_sources)
    end
    recon_kwargs = (
        frequencies=recon_freqs,
        bandwidth=bandwidth,
        reference_sound_speed=TranscranialFUS._pam_reference_sound_speed(c, cfg, sources),
        axial_step=reconstruction_axial_step,
        use_gpu=recon_use_gpu,
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
        :kwave_use_gpu => kwave_use_gpu,
        :recon_use_gpu => recon_use_gpu,
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
        model == :squiggle3d && return :squiggle
        model == :network3d && return :network
        return model
    end
    if haskey(meta, "cluster_model")
        old = Symbol(String(meta["cluster_model"]))
        return old == :vascular ? :squiggle : old
    end
    return any(src -> src isa BubbleCluster2D, sources) ? :squiggle : :point
end

function centerlines_from_emission_meta(meta)
    key = haskey(meta, "squiggle") ? "squiggle" : (haskey(meta, "network") ? "network" : (haskey(meta, "vascular") ? "vascular" : ""))
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

function source_summary(src::BubbleCluster2D)
    return Dict(
        "kind" => "bubble_cluster",
        "depth_m" => src.depth,
        "lateral_m" => src.lateral,
        "fundamental_hz" => src.fundamental,
        "amplitude_pa" => src.amplitude,
        "harmonics" => src.harmonics,
        "harmonic_amplitudes" => src.harmonic_amplitudes,
        "harmonic_phases_rad" => src.harmonic_phases,
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
detection_truth_radius_m = parse(Float64, opts["vascular-radius-mm"]) * 1e-3
detection_threshold_ratio = parse(Float64, opts["detection-threshold-ratio"])
boundary_threshold_ratios = parse_threshold_ratios(opts["boundary-threshold-ratios"])
auto_threshold_search = parse_bool(opts["auto-threshold-search"])
threshold_score_ratios = auto_threshold_search ? parse_threshold_search_ratios(opts) : boundary_threshold_ratios

if dimension == 3
    isempty(from_run_dir) || error("--from-run-dir is not implemented for 3D PAM yet.")
    source_model in (:point, :squiggle, :network) ||
        error("3D PAM CLI supports --source-model=point, --source-model=squiggle, or --source-model=network.")
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
    elseif source_model == :network
        parse_network_sources_3d(opts, cfg_base)
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
    if source_model in (:squiggle, :network)
        emission_meta["activity_model"] = Dict(
            "activity_mode" => String(source_phase_mode),
            "frequency_jitter_percent" => source_variability.frequency_jitter_fraction * 100.0,
        )
    end

    simulation_backend = parse_simulation_backend(opts["simulation-backend"])
    simulation_backend == :analytic && aberrator == :skull &&
        error("--simulation-backend=analytic is not compatible with --aberrator=skull; use --simulation-backend=kwave.")
    results = run_pam_case_3d(
        c,
        rho,
        sources,
        cfg;
        frequencies=recon_frequencies,
        bandwidth=recon_bandwidth_hz,
        kwave_use_gpu=parse_bool(opts["kwave-use-gpu"]),
        recon_use_gpu=parse_bool(opts["recon-use-gpu"]),
        reconstruction_axial_step=parse(Float64, opts["recon-step-um"]) * 1e-6,
        reconstruction_mode=reconstruction_mode,
        window_config=window_config,
        show_progress=parse_bool(opts["recon-progress"]),
        benchmark=parse_bool(opts["benchmark"]),
        window_batch=parse(Int, opts["window-batch"]),
        simulation_backend=simulation_backend,
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
        "reconstruction_source" => Dict("mode" => String(simulation_backend)),
        "simulation_backend" => String(simulation_backend),
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
        use_gpu=parse_bool(opts["recon-use-gpu"]),
        kwave_use_gpu=parse_bool(opts["kwave-use-gpu"]),
        reconstruction_axial_step=parse(Float64, opts["recon-step-um"]) * 1e-6,
        analysis_mode=analysis_mode,
        detection_truth_radius=detection_truth_radius_m,
        detection_threshold_ratio=detection_threshold_ratio,
        detection_truth_mask=detection_truth_mask,
        reconstruction_mode=reconstruction_mode,
        window_config=window_config,
        source_phase_mode=source_phase_mode,
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
            "num-cycles", "harmonics", "harmonic-amplitudes",
            "gate-us", "taper-ratio", "axial-mm", "transverse-mm", "dx-mm", "dz-mm",
            "receiver-aperture-mm", "t-max-us", "dt-ns", "zero-pad-factor",
            "peak-suppression-radius-mm", "success-tolerance-mm", "aberrator", "ct-path",
            "slice-index", "skull-transducer-distance-mm", "bottom-margin-mm", "hu-bone-thr",
            "simulation-backend", "phase-mode", "phase-jitter-rad", "random-seed",
            "transducer-mm", "delays-us", "vascular-length-mm", "vascular-squiggle-amplitude-mm",
            "vascular-squiggle-amplitude-x-mm", "vascular-squiggle-wavelength-mm",
            "vascular-squiggle-slope", "squiggle-phase-x-deg",
            "vascular-source-spacing-mm", "vascular-position-jitter-mm",
            "vascular-min-separation-mm", "vascular-max-sources-per-anchor",
            "network-axial-radius-mm", "network-lateral-y-radius-mm",
            "network-lateral-z-radius-mm", "network-root-count", "network-generations",
            "network-branch-length-mm", "network-branch-step-mm", "network-branch-angle-deg",
            "network-tortuosity", "network-orientation", "network-density-sigma-mm", "network-density-axial-sigma-mm",
            "network-density-lateral-y-sigma-mm", "network-density-lateral-z-sigma-mm",
            "network-max-sources-per-center",
            "source-phase-mode", "frequency-jitter-percent",
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
        use_gpu=parse_bool(opts["recon-use-gpu"]),
        reconstruction_axial_step=parse(Float64, opts["recon-step-um"]) * 1e-6,
        analysis_mode=analysis_mode,
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
