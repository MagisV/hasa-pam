function find_pam_peaks_3d(
    intensity::AbstractArray{<:Real, 3},
    ::NamedTuple,
    cfg::PAMConfig3D;
    n_peaks::Integer,
    suppression_radius::Real=cfg.peak_suppression_radius,
)
    work = copy(Float64.(intensity))
    row_start = receiver_row(cfg) + 1
    pml_guard = _pam_pml_guard_3d(cfg)
    row_stop  = max(row_start, size(work, 1) - pml_guard)
    row_start <= row_stop || error("No valid reconstruction rows remain after excluding the receiver row and PML.")
    work[1:(row_start - 1), :, :] .= -Inf
    if row_stop < size(work, 1)
        work[(row_stop + 1):end, :, :] .= -Inf
    end
    # Suppress the outermost lateral cell on each edge (aperture boundary fold-back).
    work[:, 1, :] .= -Inf
    work[:, end, :] .= -Inf
    work[:, :, 1] .= -Inf
    work[:, :, end] .= -Inf

    rad_rows = max(1, round(Int, suppression_radius / cfg.dx))
    rad_y    = max(1, round(Int, suppression_radius / cfg.dy))
    rad_z    = max(1, round(Int, suppression_radius / cfg.dz))
    peaks = NTuple{3, Int}[]

    for _ in 1:Int(n_peaks)
        idx = Tuple(argmax(work))
        isfinite(work[idx...]) || break
        push!(peaks, idx)
        r0, y0, z0 = idx
        work[max(1, r0 - rad_rows):min(size(work, 1), r0 + rad_rows),
             max(1, y0 - rad_y):min(size(work, 2), y0 + rad_y),
             max(1, z0 - rad_z):min(size(work, 3), z0 + rad_z)] .= -Inf
    end

    return peaks
end

function pam_truth_mask_3d(
    sources::AbstractVector{<:EmissionSource3D},
    grid::NamedTuple,
    cfg::PAMConfig3D;
    radius::Real=cfg.success_tolerance,
)
    radius_m  = Float64(radius)
    radius_m >= 0 || error("truth-mask radius must be non-negative.")
    nx, ny, nz = pam_Nx(cfg), pam_Ny(cfg), pam_Nz(cfg)
    mask = falses(nx, ny, nz)
    x    = collect(grid.x)
    y    = collect(grid.y)
    z    = collect(grid.z)
    r0   = receiver_row(cfg)
    radius2 = radius_m^2
    row_r = ceil(Int, radius_m / cfg.dx)
    col_r_y = ceil(Int, radius_m / cfg.dy)
    col_r_z = ceil(Int, radius_m / cfg.dz)

    for src in sources
        depth = src.depth
        src_x = x[r0] + depth
        row0  = r0 + round(Int, depth / cfg.dx)
        col0_y = argmin(abs.(y .- src.lateral_y))
        col0_z = argmin(abs.(z .- src.lateral_z))
        for row in max(r0 + 1, row0 - row_r):min(nx, row0 + row_r)
            dx2 = (x[row] - src_x)^2
            for iy in max(1, col0_y - col_r_y):min(ny, col0_y + col_r_y)
                dy2 = (y[iy] - src.lateral_y)^2
                for iz in max(1, col0_z - col_r_z):min(nz, col0_z + col_r_z)
                    dz2 = (z[iz] - src.lateral_z)^2
                    if dx2 + dy2 + dz2 <= radius2
                        mask[row, iy, iz] = true
                    end
                end
            end
        end
    end
    return mask
end

function analyse_pam_3d(
    intensity::AbstractArray{<:Real, 3},
    grid::NamedTuple,
    cfg::PAMConfig3D,
    sources::AbstractVector{<:EmissionSource3D};
    n_peaks::Union{Nothing, Integer}=nothing,
    success_tolerance::Real=cfg.success_tolerance,
    suppression_radius::Real=cfg.peak_suppression_radius,
)
    n_truth = length(sources)
    n_truth > 0 || error("At least one emission source is required for 3D PAM analysis.")
    n_find = isnothing(n_peaks) ? n_truth : Int(n_peaks)
    peaks = find_pam_peaks_3d(intensity, grid, cfg; n_peaks=n_find, suppression_radius=suppression_radius)
    length(peaks) == n_truth || error("Expected to recover $n_truth peaks, found $(length(peaks)).")

    x = collect(grid.x)
    y = collect(grid.y)
    z = collect(grid.z)
    rr = receiver_row(cfg)

    truth_mm = [(src.depth * 1e3, src.lateral_y * 1e3, src.lateral_z * 1e3) for src in sources]
    pred_mm  = [((x[idx[1]] - x[rr]) * 1e3, y[idx[2]] * 1e3, z[idx[3]] * 1e3) for idx in peaks]

    cost = Matrix{Float64}(undef, n_truth, n_truth)
    for i in 1:n_truth, j in 1:n_truth
        d_ax  = truth_mm[i][1] - pred_mm[j][1]
        d_lat_y = truth_mm[i][2] - pred_mm[j][2]
        d_lat_z = truth_mm[i][3] - pred_mm[j][3]
        cost[i, j] = sqrt(d_ax^2 + d_lat_y^2 + d_lat_z^2)
    end
    assignment, _ = _best_assignment(cost)

    matched_pred_mm  = [pred_mm[assignment[i]]  for i in 1:n_truth]
    matched_indices  = [peaks[assignment[i]]     for i in 1:n_truth]
    axial_errors_mm  = [truth_mm[i][1] - matched_pred_mm[i][1] for i in 1:n_truth]
    lateral_y_errors_mm = [truth_mm[i][2] - matched_pred_mm[i][2] for i in 1:n_truth]
    lateral_z_errors_mm = [truth_mm[i][3] - matched_pred_mm[i][3] for i in 1:n_truth]
    radial_errors_mm = [sqrt(axial_errors_mm[i]^2 + lateral_y_errors_mm[i]^2 + lateral_z_errors_mm[i]^2) for i in 1:n_truth]

    raw_peak_intensities = [Float64(intensity[idx...]) for idx in matched_indices]
    max_intensity = max(maximum(Float64.(intensity)), eps(Float64))
    norm_peak_intensities = raw_peak_intensities ./ max_intensity

    tol_mm = Float64(success_tolerance) * 1e3
    successes = radial_errors_mm .<= tol_mm
    num_success = count(identity, successes)

    return Dict{Symbol, Any}(
        :truth_mm => truth_mm,
        :predicted_mm => matched_pred_mm,
        :peak_indices => matched_indices,
        :axial_errors_mm => axial_errors_mm,
        :lateral_y_errors_mm => lateral_y_errors_mm,
        :lateral_z_errors_mm => lateral_z_errors_mm,
        :radial_errors_mm => radial_errors_mm,
        :mean_axial_error_mm => mean(abs.(axial_errors_mm)),
        :mean_lateral_y_error_mm => mean(abs.(lateral_y_errors_mm)),
        :mean_lateral_z_error_mm => mean(abs.(lateral_z_errors_mm)),
        :mean_radial_error_mm => mean(radial_errors_mm),
        :max_radial_error_mm => maximum(radial_errors_mm),
        :success_tolerance_mm => tol_mm,
        :success_rate => num_success / n_truth,
        :num_success => num_success,
        :raw_peak_intensities => raw_peak_intensities,
        :norm_peak_intensities => norm_peak_intensities,
        :mean_norm_peak_intensity => mean(norm_peak_intensities),
    )
end
