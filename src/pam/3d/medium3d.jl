"""
    _resample_pam_volume_z(hu_3d, spacing_z_m, target_dz_m, target_nz)

Linearly resample a PAM CT volume along Z from CT slice spacing to grid spacing.
"""
function _resample_pam_volume_z(
    hu_3d::AbstractArray{Float32, 3},
    spacing_z_m::Float64,
    target_dz_m::Float64,
    target_nz::Int,
)
    nz_ct = size(hu_3d, 3)
    nz_ct == 1 && return repeat(hu_3d; outer=(1, 1, target_nz))

    # source z-coordinates in metres
    src_z = [0.0 + (k - 1) * spacing_z_m for k in 1:nz_ct]
    # target z-coordinates
    tgt_z = [(k - 1) * target_dz_m for k in 1:target_nz]

    nx, ny = size(hu_3d, 1), size(hu_3d, 2)
    out = Array{Float32}(undef, nx, ny, target_nz)
    @inbounds for it in 1:target_nz
        tz = tgt_z[it]
        # find bracket
        k1 = searchsortedlast(src_z, tz)
        k1 = clamp(k1, 1, nz_ct - 1)
        k2 = k1 + 1
        denom = src_z[k2] - src_z[k1]
        t = denom > 0.0 ? Float32((tz - src_z[k1]) / denom) : 0.0f0
        t = clamp(t, 0.0f0, 1.0f0)
        @inbounds for j in 1:ny, i in 1:nx
            out[i, j, it] = hu_3d[i, j, k1] * (1.0f0 - t) + hu_3d[i, j, k2] * t
        end
    end
    return out
end

"""
    make_pam_medium_3d(cfg; aberrator=:none, kwargs...)

Build 3D sound-speed and density volumes for a PAM configuration.

Returns `(c, rho, info)`, where `c` is in m/s, `rho` is in kg/m^3, and `info`
describes the homogeneous, water, or skull-backed medium construction.
"""
function make_pam_medium_3d(
    cfg::PAMConfig3D;
    aberrator::Symbol = :none,
    hu_vol::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
    spacing_m::Union{Nothing, NTuple{3, <:Real}} = nothing,
    ct_path::AbstractString = DEFAULT_CT_PATH,
    slice_index_z::Integer = 250,
    skull_to_transducer::Real = 30e-3,
    hu_bone_thr::Integer = 200,
)
    nx = pam_Nx(cfg)
    ny = pam_Ny(cfg)
    nz = pam_Nz(cfg)
    c   = fill(Float32(cfg.c0),   nx, ny, nz)
    rho = fill(Float32(cfg.rho0), nx, ny, nz)

    if aberrator == :none || aberrator == :water
        return c, rho, Dict{Symbol, Any}(:aberrator => aberrator)
    end

    aberrator == :skull || error("Aberrator type :$aberrator not supported for 3D PAM medium (only :none, :water, :skull).")

    # ── 1. Load CT volume ────────────────────────────────────────────────────
    hu_local, spacing_local = _load_pam_ct(hu_vol, spacing_m, ct_path)
    # spacing_local = (dx_ct, dy_ct, dz_ct) in metres; CT array is (nz_ct, ny_ct, nx_ct)
    spacing_x_mm = Float64(spacing_local[1]) * 1e3   # in-plane X (cols in CT slice)
    spacing_y_mm = Float64(spacing_local[2]) * 1e3   # in-plane Y (rows in CT slice)
    spacing_z_m  = Float64(spacing_local[3])          # between CT slices (metres)
    nz_ct_total  = size(hu_local, 1)

    # ── 2. Extract Z-slab centred on slice_index_z ───────────────────────────
    # Physical Z extent needed for the PAM grid
    phys_z_m   = nz * cfg.dz
    # Number of CT slices to cover this extent (before resampling)
    nz_ct_need = ceil(Int, phys_z_m / spacing_z_m) + 2   # +2 for interpolation guard
    nz_ct_need = min(nz_ct_need, nz_ct_total)

    z0 = Int(slice_index_z)
    0 <= z0 < nz_ct_total || error("slice_index_z=$z0 out of bounds for $nz_ct_total CT slices.")
    half = fld(nz_ct_need, 2)
    z_start = clamp(z0 - half, 0, nz_ct_total - nz_ct_need)
    z_end   = z_start + nz_ct_need - 1
    # hu_slab: (nz_ct_need, ny_ct, nx_ct)
    hu_slab = Float32.(hu_local[(z_start + 1):(z_end + 1), :, :])

    # ── 3. Resample each CT Z-slice (X-Y plane) to PAM grid spacing ──────────
    target_row_mm = cfg.dx * 1e3
    target_col_mm = cfg.dy * 1e3
    # Sample one slice to get output size
    sample_resampled = _resample_pam_slice(
        hu_slab[1, :, :], spacing_y_mm, spacing_x_mm, target_row_mm, target_col_mm,
    )
    nr_res, nc_res = size(sample_resampled)

    hu_3d = Array{Float32}(undef, nr_res, nc_res, nz_ct_need)
    hu_3d[:, :, 1] .= sample_resampled
    for k in 2:nz_ct_need
        hu_3d[:, :, k] = _resample_pam_slice(
            hu_slab[k, :, :], spacing_y_mm, spacing_x_mm, target_row_mm, target_col_mm,
        )
    end

    # ── 4. Resample Z axis from CT spacing to cfg.dz ─────────────────────────
    hu_3d = _resample_pam_volume_z(hu_3d, spacing_z_m, cfg.dz, nz)

    # ── 5. Adjust lateral Y size to match pam_Ny ─────────────────────────────
    if size(hu_3d, 2) != ny
        hu_3d_adj = Array{Float32}(undef, size(hu_3d, 1), ny, nz)
        for k in 1:nz
            hu_3d_adj[:, :, k] = _adjust_lateral_size(hu_3d[:, :, k], ny)
        end
        hu_3d = hu_3d_adj
    end

    # ── 6. Skull alignment in depth (X axis) ─────────────────────────────────
    # Use the central Z-plane as reference for boundary detection
    ref_z = fld(nz, 2) + 1
    outer_row_rel, _ = find_skull_boundaries(
        hu_3d[:, :, ref_z];
        hu_bone_thr = hu_bone_thr,
        num_cols    = 10,
        expand_if_empty = true,
    )
    outer_row_target = receiver_row(cfg) + round(Int, Float64(skull_to_transducer) / cfg.dx)
    shift = outer_row_target - outer_row_rel

    nr_cur = size(hu_3d, 1)
    if shift > 0
        pad = fill(Float32(-1000), shift, ny, nz)
        hu_3d = cat(pad, hu_3d; dims=1)
    elseif shift < 0
        crop_start = 1 - shift
        crop_start <= nr_cur || error("Skull alignment would crop away the entire CT slab.")
        hu_3d = hu_3d[crop_start:end, :, :]
    end

    # Trim or pad depth to nx
    nr_cur = size(hu_3d, 1)
    if nr_cur > nx
        hu_3d = hu_3d[1:nx, :, :]
    elseif nr_cur < nx
        pad = fill(Float32(-1000), nx - nr_cur, ny, nz)
        hu_3d = cat(hu_3d, pad; dims=1)
    end

    # ── 7. Verify alignment ───────────────────────────────────────────────────
    outer_row_rel2, inner_row_rel2 = find_skull_boundaries(
        hu_3d[:, :, ref_z];
        hu_bone_thr = hu_bone_thr,
        num_cols    = 10,
        expand_if_empty = true,
    )
    outer_row_rel2 == outer_row_target || error("Failed to align the skull to the requested PAM outer row.")

    # ── 8. HU → ρ, c slice-by-slice over Z ───────────────────────────────────
    for k in 1:nz
        rho_k, c_k = hu_to_rho_c(
            hu_3d[:, :, k];
            hu_bone_thr = hu_bone_thr,
            rho_water   = cfg.rho0,
            rho_bone    = 2100.0,
            c_water     = cfg.c0,
            c_bone      = 2500.0,
        )
        c[:, :, k]   .= c_k
        rho[:, :, k] .= rho_k
    end

    return c, rho, Dict{Symbol, Any}(
        :aberrator          => :skull,
        :slice_index_z      => Int(slice_index_z),
        :outer_row          => outer_row_rel2,
        :inner_row          => inner_row_rel2,
        :receiver_row       => receiver_row(cfg),
        :skull_to_transducer => Float64(skull_to_transducer),
        :hu_bone_thr        => Int(hu_bone_thr),
        :ct_path            => ct_path,
    )
end
