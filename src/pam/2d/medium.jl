"""
    _resample_pam_slice(slice, spacing_row_mm, spacing_col_mm, new_row_mm, new_col_mm)

Resample a 2D CT slice from millimeter pixel spacing to PAM grid spacing.
"""
function _resample_pam_slice(
    slice::AbstractMatrix{<:Real},
    spacing_row_mm::Float64,
    spacing_col_mm::Float64,
    new_row_mm::Float64,
    new_col_mm::Float64,
)
    out_rows = round(Int, size(slice, 1) * spacing_row_mm / new_row_mm)
    out_cols = round(Int, size(slice, 2) * spacing_col_mm / new_col_mm)
    row_coords = 1 .+ (0:(out_rows - 1)) .* (new_row_mm / spacing_row_mm)
    col_coords = 1 .+ (0:(out_cols - 1)) .* (new_col_mm / spacing_col_mm)
    itp = extrapolate(interpolate(Float32.(slice), BSpline(Linear())), Flat())

    out = Matrix{Float32}(undef, out_rows, out_cols)
    @inbounds for row in 1:out_rows
        row_coord = row_coords[row]
        for col in 1:out_cols
            out[row, col] = Float32(itp(row_coord, col_coords[col]))
        end
    end
    return out
end

"""
    _load_pam_ct(hu_vol, spacing_m, ct_path)

Return CT Hounsfield data and voxel spacing, loading from `ct_path` when no
preloaded volume is supplied.
"""
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

"""
    make_pam_medium(cfg; aberrator=:none, kwargs...)

Build 2D sound-speed and density maps for a PAM configuration.

Returns `(c, rho, info)`, where `c` is in m/s, `rho` is in kg/m^3, and `info`
describes the homogeneous or skull-backed medium construction.
"""
function make_pam_medium(
    cfg::PAMConfig;
    aberrator::Symbol=:none,
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
    end

    error("Unknown PAM medium aberrator: $aberrator")
end
