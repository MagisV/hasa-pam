function analyse_focus_2d(
    p::AbstractMatrix{<:Real},
    kgrid::KGrid2D,
    cfg::SimulationConfig;
    thr_db::Real=3.0,
    exclude_mask::Union{Nothing, AbstractMatrix{Bool}}=nothing,
    exclude_skull::Bool=false,
)
    x_vec = kgrid.x_vec
    y_vec = kgrid.y_vec
    trans_idx = cfg.trans_index

    row_tgt = target_index(cfg)
    x_focus_global = x_vec[row_tgt]
    y_focus_global = y_vec[fld(length(y_vec), 2) + 1] + cfg.x_focus

    global_rows = 1:(trans_idx - 1)
    p_roi = Float64.(p[global_rows, :])
    if exclude_skull && !isnothing(exclude_mask)
        p_roi = ifelse.(exclude_mask[global_rows, :], -Inf, p_roi)
    end

    idx_local = Tuple(argmax(p_roi))
    idx_peak = (first(global_rows) + idx_local[1] - 1, idx_local[2])
    p_peak = Float64(p[idx_peak...])

    x_pk = x_vec[idx_peak[1]]
    y_pk = y_vec[idx_peak[2]]
    idx_peak_global = Tuple(argmax(p))
    is_global_peak = idx_peak == idx_peak_global

    d_axial_mm = (x_focus_global - x_pk) * 1e3
    d_lateral_mm = (y_focus_global - y_pk) * 1e3
    error_mm = hypot(d_axial_mm, d_lateral_mm)

    thr = p_peak / (10^(thr_db / 20))
    mask = p_roi .>= thr
    d_a = kgrid.dx * kgrid.dy
    area_mm2 = count(mask) * d_a * 1e6

    rows_masked = findall(vec(any(mask; dims=2)))
    axial_len_mm = isempty(rows_masked) ? 0.0 :
        (maximum(x_vec[first(global_rows) .+ rows_masked .- 1]) - minimum(x_vec[first(global_rows) .+ rows_masked .- 1])) * 1e3

    lat_cols = findall(mask[idx_local[1], :])
    lat_diam_mm = isempty(lat_cols) ? 0.0 : (maximum(y_vec[lat_cols]) - minimum(y_vec[lat_cols])) * 1e3

    return Dict{Symbol, Any}(
        :p_peak => p_peak,
        :error_mm => error_mm,
        :d_axial_mm => d_axial_mm,
        :d_lateral_mm => d_lateral_mm,
        :focal_area_mm2 => area_mm2,
        :axial_len_mm => axial_len_mm,
        :lat_diam_mm => lat_diam_mm,
        :peak_mm => (x_pk * 1e3, y_pk * 1e3),
        :is_global_peak => is_global_peak,
    )
end

function run_focus_case(
    hu_vol::AbstractArray{<:Real, 3},
    cfg::SimulationConfig,
    medium_type::MediumType,
    est_type::Est,
    sweep_settings::SweepSettings;
    slice_index::Union{Nothing, Integer}=nothing,
    hu_bone_thr::Integer=200,
    animation_settings::Union{Nothing, AnimationSettings}=nothing,
    return_c::Bool=false,
)
    c, rho, info = make_medium(
        hu_vol,
        cfg,
        medium_type;
        slice_index=slice_index,
        hu_bone_thr=hu_bone_thr,
    )

    cfg.trans_index = info[:z_trans_idx]
    pressure, hasa_info, kgrid = focus(c, rho, est_type, cfg, sweep_settings, animation_settings)

    stats = nothing
    if pressure !== nothing && ndims(pressure) == 2
        exclude_mask = nothing
        if sweep_settings.exclude_skull
            exclude_mask = skull_mask_from_c_columnwise(
                c;
                c_water=cfg.c0,
                mask_outside=sweep_settings.mask_outside,
            )
        end
        stats = analyse_focus_2d(
            pressure,
            kgrid,
            cfg;
            exclude_mask=exclude_mask,
            exclude_skull=sweep_settings.exclude_skull,
        )
    end

    return stats, pressure, return_c ? c : nothing, kgrid, cfg, hasa_info
end
