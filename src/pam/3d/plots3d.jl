function source_triples_mm(sources::AbstractVector{<:EmissionSource3D})
    return [(src.depth * 1e3, src.lateral_y * 1e3, src.lateral_z * 1e3) for src in sources]
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
    # :y_z projection is not transposed (matches _projection_heatmap_matrix_3d convention).
    overlay_skull_2d!(ax, c2d, xvals, yvals; transpose_matrix=(projection != :y_z))
    return nothing
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
    np = PythonCall.pyimport("numpy")

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

    println("Saved napari data -> $npz_path")
    println("  Open with:  python $(joinpath(out_dir, "view_pam.py"))")
end
