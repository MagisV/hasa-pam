function hu_to_rho_c(
    hu;
    hu_bone_thr::Real=200,
    rho_water::Real=1000.0,
    rho_bone::Real=2100.0,
    c_water::Real=1500.0,
    c_bone::Real=2500.0,
)
    hu_clipped = clamp.(Float32.(hu), -1000.0f0, 3000.0f0)
    mask_bone = hu_clipped .>= Float32(hu_bone_thr)
    mask_fluid = .!mask_bone

    rho = similar(hu_clipped)
    c = similar(hu_clipped)

    if any(mask_bone)
        h_bone_max = Float32(quantile(vec(hu_clipped[mask_bone]), 0.995))
        denom = max(h_bone_max, 1.0f0)
        psi = clamp.((h_bone_max .- hu_clipped[mask_bone]) ./ denom, 0.0f0, 1.0f0)
        rho[mask_bone] .= Float32(rho_water) .+ Float32(rho_bone - rho_water) .* (1.0f0 .- psi)
        c[mask_bone] .= Float32(c_water) .+ Float32(c_bone - c_water) .* (1.0f0 .- psi)
    end

    rho[mask_fluid] .= Float32(rho_water)
    c[mask_fluid] .= Float32(c_water)
    return rho, c
end

function find_skull_boundaries(
    hu_slice::AbstractMatrix{<:Real};
    hu_bone_thr::Real=200,
    num_cols::Integer=10,
    expand_if_empty::Bool=true,
)
    rows, cols = size(hu_slice)
    band_cols = min(Int(num_cols), cols)
    mid = fld(cols, 2) + 1
    half = fld(band_cols, 2)
    start_col = max(1, mid - half)
    end_col = min(cols, start_col + band_cols - 1)
    start_col = max(1, end_col - band_cols + 1)

    rows_with_bone = Int[]
    while true
        sub = hu_slice[:, start_col:end_col] .>= hu_bone_thr
        rows_with_bone = findall(vec(any(sub; dims=2)))
        if !isempty(rows_with_bone) || !expand_if_empty
            break
        end
        start_col == 1 && end_col == cols && break
        start_col = max(1, start_col - 2)
        end_col = min(cols, end_col + 2)
    end

    isempty(rows_with_bone) && error("No bone detected in the inspected columns with the given threshold.")
    return first(rows_with_bone), last(rows_with_bone)
end

function _binary_dilation(mask::BitMatrix, structure::BitMatrix)
    rows, cols = size(mask)
    srows, scols = size(structure)
    rc = fld(srows, 2) + 1
    cc = fld(scols, 2) + 1
    out = falses(rows, cols)
    @inbounds for i in 1:rows
        for j in 1:cols
            acc = false
            for si in 1:srows
                for sj in 1:scols
                    structure[si, sj] || continue
                    ii = i + si - rc
                    jj = j + sj - cc
                    if 1 <= ii <= rows && 1 <= jj <= cols && mask[ii, jj]
                        acc = true
                        break
                    end
                end
                acc && break
            end
            out[i, j] = acc
        end
    end
    return out
end

function _binary_erosion(mask::BitMatrix, structure::BitMatrix)
    rows, cols = size(mask)
    srows, scols = size(structure)
    rc = fld(srows, 2) + 1
    cc = fld(scols, 2) + 1
    out = falses(rows, cols)
    @inbounds for i in 1:rows
        for j in 1:cols
            acc = true
            for si in 1:srows
                for sj in 1:scols
                    structure[si, sj] || continue
                    ii = i + si - rc
                    jj = j + sj - cc
                    if !(1 <= ii <= rows && 1 <= jj <= cols && mask[ii, jj])
                        acc = false
                        break
                    end
                end
                acc || break
            end
            out[i, j] = acc
        end
    end
    return out
end

function _binary_closing(mask::BitMatrix; iterations::Integer=1)
    out = copy(mask)
    structure = trues(3, 3)
    for _ in 1:Int(iterations)
        out = _binary_dilation(out, structure)
        out = _binary_erosion(out, structure)
    end
    return out
end

function skull_mask_from_c_columnwise(
    c::AbstractMatrix{<:Real};
    c_water::Real=1500.0,
    tol::Real=5.0,
    min_thick_rows::Integer=2,
    dilate_rows::Integer=1,
    close_iters::Integer=1,
    mask_outside::Bool=true,
)
    nx, ny = size(c)
    diff_mask = abs.(Float64.(c) .- Float64(c_water)) .> Float64(tol)

    first_hit = Vector{Int}(undef, ny)
    last_hit = Vector{Int}(undef, ny)
    has_any = vec(any(diff_mask; dims=1))
    for j in 1:ny
        if has_any[j]
            col_rows = findall(diff_mask[:, j])
            first_hit[j] = first(col_rows)
            last_hit[j] = last(col_rows)
        else
            first_hit[j] = 0
            last_hit[j] = 0
        end
    end

    skull_mask = falses(nx, ny)
    @inbounds for j in 1:ny
        has_any[j] || continue
        i0 = first_hit[j]
        i1 = last_hit[j]
        if i1 - i0 + 1 >= min_thick_rows
            skull_mask[i0:i1, j] .= true
        end
    end

    if close_iters > 0
        skull_mask = _binary_closing(skull_mask; iterations=close_iters)
    end
    if dilate_rows > 0
        structure = trues(2 * dilate_rows + 1, 1)
        skull_mask = _binary_dilation(skull_mask, structure)
    end

    exclude_mask = copy(skull_mask)
    if mask_outside
        rows = reshape(1:nx, :, 1)
        col_any = vec(any(exclude_mask; dims=1))
        last_after = zeros(Int, ny)
        @inbounds for j in 1:ny
            if col_any[j]
                last_after[j] = last(findall(exclude_mask[:, j]))
            end
        end
        exclude_mask .|= (rows .>= reshape(last_after, 1, :)) .& reshape(col_any, 1, :)
    end
    return exclude_mask
end

function _adjust_lateral_size(hu_slice::AbstractMatrix{<:Real}, target_cols::Int)
    if size(hu_slice, 2) > target_cols
        extra = size(hu_slice, 2) - target_cols
        left = fld(extra, 2)
        return hu_slice[:, (left + 1):(left + target_cols)]
    elseif size(hu_slice, 2) < target_cols
        extra = target_cols - size(hu_slice, 2)
        left = fld(extra, 2)
        right = extra - left
        out = fill(Float32(-1000), size(hu_slice, 1), target_cols)
        out[:, (left + 1):(left + size(hu_slice, 2))] .= Float32.(hu_slice)
        return out
    end
    return Float32.(hu_slice)
end

function make_medium_fixed_distance_from_skull(
    hu_vol::AbstractArray{<:Real, 3},
    cfg::SimulationConfig,
    medium_type::MediumType;
    slice_index::Union{Nothing, Integer}=nothing,
    hu_bone_thr::Integer=200,
    min_water_gap::Real=2e-3,
)
    nx_cfg = Nx(cfg)
    ny_cfg = Nz(cfg)
    if medium_type == WATER
        c = fill(Float32(cfg.c0), nx_cfg, ny_cfg)
        rho = fill(Float32(cfg.rho0), nx_cfg, ny_cfg)
        cfg.trans_index = nx_cfg - cfg.PML_GUARD
        return c, rho, Dict(
            :target_idx => nothing,
            :z_trans_idx => cfg.trans_index,
            :inner_row => nothing,
            :outer_row => nothing,
        )
    end

    slice0 = isnothing(slice_index) ? fld(size(hu_vol, 1), 2) : Int(slice_index)
    0 <= slice0 < size(hu_vol, 1) || error("slice_index=$slice0 is out of bounds for $(size(hu_vol, 1)) slices.")
    hu_slice = reverse(Float32.(hu_vol[slice0 + 1, :, :]); dims=1)
    hu_slice = _adjust_lateral_size(hu_slice, ny_cfg)

    inner_row_rel, outer_row_rel = find_skull_boundaries(
        hu_slice;
        hu_bone_thr=hu_bone_thr,
        num_cols=10,
        expand_if_empty=true,
    )

    focus_depth = cfg.focus_depth_from_inner_skull
    isnothing(focus_depth) && error("focus_depth_from_inner_skull must be set for make_medium_fixed_distance_from_skull.")
    n_offset = round(Int, focus_depth / cfg.dx)
    n_focus = round(Int, cfg.z_focus / cfg.dx)

    target_idx = inner_row_rel - n_offset
    target_idx >= 1 || error("Target offset puts the target above the top of the slice.")
    z_trans_idx = target_idx + n_focus
    inner_row = inner_row_rel
    outer_row = outer_row_rel

    min_water_rows = max(1, round(Int, min_water_gap / cfg.dx))
    if (z_trans_idx - 1 - outer_row) < min_water_rows
        add = min_water_rows - (z_trans_idx - 1 - outer_row)
        z_trans_idx += add
        target_idx += add
        inner_row += add
        outer_row += add
    end

    nx_needed = z_trans_idx + cfg.PML_GUARD
    nx_top_needed = size(hu_slice, 1)
    nx_local = max(nx_needed, nx_top_needed)

    c = fill(Float32(cfg.c0), nx_local, ny_cfg)
    rho = fill(Float32(cfg.rho0), nx_local, ny_cfg)

    rho_slice, c_slice = hu_to_rho_c(
        hu_slice;
        hu_bone_thr=hu_bone_thr,
        rho_water=cfg.rho0,
        rho_bone=2100.0,
        c_water=cfg.c0,
        c_bone=2500.0,
    )

    z_offset = inner_row - inner_row_rel
    c[(z_offset + 1):(z_offset + size(hu_slice, 1)), :] .= c_slice
    rho[(z_offset + 1):(z_offset + size(hu_slice, 1)), :] .= rho_slice

    height_up_rows = round(Int, cfg.axial_padding * cfg.z_focus / cfg.dx)
    desired_top = max(1, z_trans_idx - height_up_rows)
    c = c[desired_top:end, :]
    rho = rho[desired_top:end, :]

    shift = desired_top - 1
    z_trans_idx -= shift
    target_idx -= shift
    inner_row -= shift
    outer_row -= shift

    cfg.axial_dim = size(c, 1) * cfg.dx
    cfg.trans_index = z_trans_idx
    return c, rho, Dict(
        :target_idx => target_idx,
        :z_trans_idx => z_trans_idx,
        :inner_row => inner_row,
        :outer_row => outer_row,
    )
end

function make_medium_fixed_transducer(
    hu_vol::AbstractArray{<:Real, 3},
    cfg::SimulationConfig,
    medium_type::MediumType;
    slice_index::Union{Nothing, Integer}=nothing,
    hu_bone_thr::Integer=200,
)
    nx_cfg = Nx(cfg)
    ny_cfg = Nz(cfg)
    c = fill(Float32(cfg.c0), nx_cfg, ny_cfg)
    rho = fill(Float32(cfg.rho0), nx_cfg, ny_cfg)

    if medium_type == WATER
        return c, rho, Dict(
            :z_trans_idx => nx_cfg - cfg.PML_GUARD,
            :outer_row => nothing,
            :inner_row => nothing,
            :slice_top => nothing,
            :slice_bottom => nothing,
            :inner_row_rel => nothing,
            :outer_row_rel => nothing,
        )
    end

    slice0 = isnothing(slice_index) ? fld(size(hu_vol, 1), 2) : Int(slice_index)
    0 <= slice0 < size(hu_vol, 1) || error("slice_index=$slice0 is out of bounds for $(size(hu_vol, 1)) slices.")
    hu_slice = reverse(Float32.(hu_vol[slice0 + 1, :, :]); dims=1)
    hu_slice = _adjust_lateral_size(hu_slice, ny_cfg)

    inner_row_rel, outer_row_rel = find_skull_boundaries(
        hu_slice;
        hu_bone_thr=hu_bone_thr,
        num_cols=10,
        expand_if_empty=true,
    )

    skull_to_trans_rows = round(Int, cfg.trans_skull_dist / cfg.dx)
    outer_row_target = nx_cfg - cfg.PML_GUARD - skull_to_trans_rows
    shift = outer_row_target - outer_row_rel
    if shift > 0
        padded = fill(Float32(-1000), size(hu_slice, 1) + shift, size(hu_slice, 2))
        padded[(shift + 1):end, :] .= hu_slice
        hu_slice = padded
    elseif shift < 0
        hu_slice = hu_slice[(1 - shift):end, :]
    end

    desired_rows = outer_row_target
    if size(hu_slice, 1) > desired_rows
        hu_slice = hu_slice[1:desired_rows, :]
    elseif size(hu_slice, 1) < desired_rows
        padded = fill(Float32(-1000), desired_rows, size(hu_slice, 2))
        padded[1:size(hu_slice, 1), :] .= hu_slice
        hu_slice = padded
    end

    inner_row_rel, outer_row_rel = find_skull_boundaries(
        hu_slice;
        hu_bone_thr=hu_bone_thr,
        num_cols=10,
        expand_if_empty=true,
    )
    outer_row_rel == outer_row_target || error("Failed to align the skull to the target outer row.")

    rho_slice, c_slice = hu_to_rho_c(
        hu_slice;
        hu_bone_thr=hu_bone_thr,
        rho_water=cfg.rho0,
        rho_bone=2100.0,
        c_water=cfg.c0,
        c_bone=2500.0,
    )

    c[1:desired_rows, :] .= c_slice
    rho[1:desired_rows, :] .= rho_slice
    return c, rho, Dict(
        :z_trans_idx => nx_cfg - cfg.PML_GUARD,
        :outer_row => outer_row_target,
        :inner_row => inner_row_rel,
        :slice_top => 1,
        :slice_bottom => desired_rows,
        :inner_row_rel => inner_row_rel,
        :outer_row_rel => outer_row_rel,
    )
end

function make_medium(
    hu_vol::AbstractArray{<:Real, 3},
    cfg::SimulationConfig,
    medium_type::MediumType;
    slice_index::Union{Nothing, Integer}=nothing,
    hu_bone_thr::Integer=200,
)
    if !isnothing(cfg.focus_depth_from_inner_skull)
        return make_medium_fixed_distance_from_skull(
            hu_vol,
            cfg,
            medium_type;
            slice_index=slice_index,
            hu_bone_thr=hu_bone_thr,
        )
    end
    return make_medium_fixed_transducer(
        hu_vol,
        cfg,
        medium_type;
        slice_index=slice_index,
        hu_bone_thr=hu_bone_thr,
    )
end
