function _resample_pam_slice(
    slice::AbstractMatrix{<:Real},
    spacing_row_mm::Float64,
    spacing_col_mm::Float64,
    new_row_mm::Float64,
    new_col_mm::Float64,
)
    Interpolations = _require_pkg(:Interpolations)
    out_rows = round(Int, size(slice, 1) * spacing_row_mm / new_row_mm)
    out_cols = round(Int, size(slice, 2) * spacing_col_mm / new_col_mm)
    row_coords = 1 .+ (0:(out_rows - 1)) .* (new_row_mm / spacing_row_mm)
    col_coords = 1 .+ (0:(out_cols - 1)) .* (new_col_mm / spacing_col_mm)
    itp = Interpolations.extrapolate(
        Interpolations.interpolate(
            Float32.(slice),
            Interpolations.BSpline(Interpolations.Linear()),
        ),
        Interpolations.Flat(),
    )

    out = Matrix{Float32}(undef, out_rows, out_cols)
    @inbounds for row in 1:out_rows
        row_coord = row_coords[row]
        for col in 1:out_cols
            out[row, col] = Float32(itp(row_coord, col_coords[col]))
        end
    end
    return out
end

function _load_pam_ct(
    hu_vol::Union{Nothing, AbstractArray{<:Real, 3}},
    spacing_m::Union{Nothing, NTuple{3, <:Real}},
    ct_path::AbstractString,
)
    if isnothing(hu_vol)
        isnothing(spacing_m) || error("Pass both hu_vol and spacing_m, or neither.")
        return load_default_ct(ct_path=ct_path)
    end
    isnothing(spacing_m) && error("spacing_m is required when supplying hu_vol for PAM skull medium construction.")
    return hu_vol, spacing_m
end

function make_pam_medium(
    cfg::PAMConfig;
    aberrator::Symbol=:none,
    lens_center_depth::Real=20e-3,
    lens_center_lateral::Real=0.0,
    lens_axial_radius::Real=4e-3,
    lens_lateral_radius::Real=12e-3,
    c_aberrator::Real=1700.0,
    rho_aberrator::Real=1150.0,
    hu_vol::Union{Nothing, AbstractArray{<:Real, 3}}=nothing,
    spacing_m::Union{Nothing, NTuple{3, <:Real}}=nothing,
    ct_path::AbstractString=DEFAULT_CT_PATH,
    slice_index::Integer=250,
    skull_to_transducer::Real=30e-3,
    hu_bone_thr::Integer=200,
)
    kgrid = pam_grid(cfg)
    c = fill(Float32(cfg.c0), kgrid.Nx, kgrid.Ny)
    rho = fill(Float32(cfg.rho0), kgrid.Nx, kgrid.Ny)

    if aberrator == :none
        return c, rho, Dict{Symbol, Any}(:aberrator => :none)
    elseif aberrator == :skull
        hu_local, spacing_local = _load_pam_ct(hu_vol, spacing_m, ct_path)
        slice0 = Int(slice_index)
        0 <= slice0 < size(hu_local, 1) || error("slice_index=$slice0 is out of bounds for $(size(hu_local, 1)) CT slices.")

        hu_slice = Float32.(hu_local[slice0 + 1, :, :])
        row_spacing_mm = Float64(spacing_local[2]) * 1e3
        col_spacing_mm = Float64(spacing_local[1]) * 1e3
        target_row_mm = cfg.dx * 1e3
        target_col_mm = cfg.dz * 1e3
        if !isapprox(row_spacing_mm, target_row_mm; atol=1e-9) || !isapprox(col_spacing_mm, target_col_mm; atol=1e-9)
            hu_slice = _resample_pam_slice(
                hu_slice,
                row_spacing_mm,
                col_spacing_mm,
                target_row_mm,
                target_col_mm,
            )
        end
        hu_slice = _adjust_lateral_size(hu_slice, kgrid.Ny)

        outer_row_rel, inner_row_rel = find_skull_boundaries(
            hu_slice;
            hu_bone_thr=hu_bone_thr,
            num_cols=10,
            expand_if_empty=true,
        )

        outer_row_target = receiver_row(cfg) + round(Int, Float64(skull_to_transducer) / cfg.dx)
        shift = outer_row_target - outer_row_rel
        if shift > 0
            padded = fill(Float32(-1000), size(hu_slice, 1) + shift, size(hu_slice, 2))
            padded[(shift + 1):end, :] .= hu_slice
            hu_slice = padded
        elseif shift < 0
            crop_start = 1 - shift
            crop_start <= size(hu_slice, 1) || error("Skull alignment would crop away the entire CT slice.")
            hu_slice = hu_slice[crop_start:end, :]
        end

        desired_rows = kgrid.Nx
        if size(hu_slice, 1) > desired_rows
            hu_slice = hu_slice[1:desired_rows, :]
        elseif size(hu_slice, 1) < desired_rows
            padded = fill(Float32(-1000), desired_rows, size(hu_slice, 2))
            padded[1:size(hu_slice, 1), :] .= hu_slice
            hu_slice = padded
        end

        outer_row_rel, inner_row_rel = find_skull_boundaries(
            hu_slice;
            hu_bone_thr=hu_bone_thr,
            num_cols=10,
            expand_if_empty=true,
        )
        outer_row_rel == outer_row_target || error("Failed to align the skull to the requested PAM outer row.")

        rho_slice, c_slice = hu_to_rho_c(
            hu_slice;
            hu_bone_thr=hu_bone_thr,
            rho_water=cfg.rho0,
            rho_bone=2100.0,
            c_water=cfg.c0,
            c_bone=2500.0,
        )
        c .= c_slice
        rho .= rho_slice
        return c, rho, Dict{Symbol, Any}(
            :aberrator => :skull,
            :slice_index => slice0,
            :outer_row => outer_row_rel,
            :inner_row => inner_row_rel,
            :receiver_row => receiver_row(cfg),
            :skull_to_transducer => Float64(skull_to_transducer),
            :hu_bone_thr => Int(hu_bone_thr),
            :ct_path => ct_path,
        )
    elseif aberrator != :lens
        error("Unknown PAM medium aberrator: $aberrator")
    end

    depth = depth_coordinates(kgrid, cfg)
    lateral = kgrid.y_vec
    mask = falses(kgrid.Nx, kgrid.Ny)
    @inbounds for i in 1:kgrid.Nx, j in 1:kgrid.Ny
        value = ((depth[i] - lens_center_depth) / lens_axial_radius)^2 +
                ((lateral[j] - lens_center_lateral) / lens_lateral_radius)^2
        if value <= 1.0
            mask[i, j] = true
            c[i, j] = Float32(c_aberrator)
            rho[i, j] = Float32(rho_aberrator)
        end
    end

    return c, rho, Dict{Symbol, Any}(
        :aberrator => :lens,
        :mask => mask,
        :lens_center_depth => Float64(lens_center_depth),
        :lens_center_lateral => Float64(lens_center_lateral),
        :lens_axial_radius => Float64(lens_axial_radius),
        :lens_lateral_radius => Float64(lens_lateral_radius),
        :c_aberrator => Float64(c_aberrator),
        :rho_aberrator => Float64(rho_aberrator),
    )
end

