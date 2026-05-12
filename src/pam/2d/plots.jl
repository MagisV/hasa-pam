"""
    map_db(map, ref)

Convert a nonnegative PAM intensity map to decibels relative to `ref`.
"""
function map_db(map::AbstractMatrix{<:Real}, ref::Real)
    safe_ref = max(Float64(ref), eps(Float64))
    return 10 .* log10.(max.(Float64.(map), eps(Float64)) ./ safe_ref)
end

"""
    map_norm(map, ref)

Normalize a PAM intensity map by `ref` with an epsilon-protected denominator.
"""
function map_norm(map::AbstractMatrix{<:Real}, ref::Real)
    safe_ref = max(Float64(ref), eps(Float64))
    return Float64.(map) ./ safe_ref
end

"""
    source_pairs_mm(sources)

Return `(depth_mm, lateral_mm)` coordinate pairs for 2D sources.
"""
function source_pairs_mm(sources)
    return [(src.depth * 1e3, src.lateral * 1e3) for src in sources]
end

"""
    scatter_sources!(ax, sources; kwargs...)

Add 2D source markers, in millimeters, to a Makie axis.
"""
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

"""
    summary_line(stats)

Format localization or detection statistics as one compact plot label.
"""
function summary_line(stats)
    if haskey(stats, :mean_radial_error_mm)
        return "mean err=$(round(stats[:mean_radial_error_mm]; digits=2)) mm, success=$(get(stats, :num_success, 0))/$(get(stats, :num_truth_sources, 0))"
    elseif haskey(stats, :f1)
        return "F1=$(round(stats[:f1]; digits=3)), precision=$(round(stats[:precision]; digits=3)), recall=$(round(stats[:recall]; digits=3))"
    end
    return string(stats)
end

"""
    overlay_skull_2d!(ax, c, xvals, yvals; transpose_matrix=true)

Overlay a semi-transparent skull mask inferred from a 2D sound-speed map.
"""
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

"""
    lines_centerlines!(ax, centerlines; kwargs...)

Draw 2D centerline polylines, converting coordinates from meters to millimeters.
"""
function lines_centerlines!(ax, centerlines; color=(:black, 0.45), linewidth=2)
    isnothing(centerlines) && return nothing
    for line in centerlines
        length(line) >= 2 || continue
        lines!(ax, [point[2] * 1e3 for point in line], [point[1] * 1e3 for point in line]; color=color, linewidth=linewidth)
    end
    return nothing
end

"""
    save_overview(path, c, rf, pam_geo, pam_hasa, kgrid, cfg, sources, stats_geo, stats_hasa)

Save the standard 2D PAM overview figure to `path`.
"""
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

"""
    add_threshold_panel!(fig, row, title, intensity, kgrid, cfg, sources; kwargs...)

Add one 2D threshold-contour panel to a Makie figure and return its heatmap.
"""
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

"""
    add_threshold_table!(fig, row, col, title, stats)

Add a compact threshold metric table to a Makie figure.
"""
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

"""
    save_threshold_boundary_detection(path, pam_geo, pam_hasa, kgrid, cfg, sources; kwargs...)

Save the 2D threshold-boundary detection comparison figure and return its
serializable metric summary.
"""
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
