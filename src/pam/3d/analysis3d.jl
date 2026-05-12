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

function source_detection_stats_3d(pred, grid, cfg::PAMConfig3D, sources; radius::Real)
    radius_m = Float64(radius)
    radius_m >= 0 || error("source detection radius must be non-negative.")
    isempty(sources) && return Dict{Symbol, Any}(
        :source_recall => 0.0,
        :detected_source_count => 0,
        :num_truth_sources => 0,
        :mean_detected_source_distance_mm => nothing,
        :max_detected_source_distance_mm => nothing,
    )
    x = collect(grid.x)
    y = collect(grid.y)
    z = collect(grid.z)
    r0 = receiver_row(cfg)
    row_r = ceil(Int, radius_m / cfg.dx)
    col_r_y = ceil(Int, radius_m / cfg.dy)
    col_r_z = ceil(Int, radius_m / cfg.dz)
    radius2 = radius_m^2

    detected = 0
    distances_mm = Float64[]
    for src in sources
        src_x = x[r0] + src.depth
        row0 = r0 + round(Int, src.depth / cfg.dx)
        col0_y = argmin(abs.(y .- src.lateral_y))
        col0_z = argmin(abs.(z .- src.lateral_z))
        best_d2 = Inf
        for row in max(r0 + 1, row0 - row_r):min(size(pred, 1), row0 + row_r)
            dx2 = (x[row] - src_x)^2
            for iy in max(1, col0_y - col_r_y):min(size(pred, 2), col0_y + col_r_y)
                dy2 = (y[iy] - src.lateral_y)^2
                for iz in max(1, col0_z - col_r_z):min(size(pred, 3), col0_z + col_r_z)
                    pred[row, iy, iz] || continue
                    d2 = dx2 + dy2 + (z[iz] - src.lateral_z)^2
                    if d2 <= radius2 && d2 < best_d2
                        best_d2 = d2
                    end
                end
            end
        end
        if isfinite(best_d2)
            detected += 1
            push!(distances_mm, sqrt(best_d2) * 1e3)
        end
    end

    return Dict{Symbol, Any}(
        :source_recall => detected / length(sources),
        :detected_source_count => detected,
        :num_truth_sources => length(sources),
        :mean_detected_source_distance_mm => isempty(distances_mm) ? nothing : mean(distances_mm),
        :max_detected_source_distance_mm => isempty(distances_mm) ? nothing : maximum(distances_mm),
    )
end

function threshold_detection_stats_3d(intensity, grid, cfg, sources; threshold_ratios, truth_radius, truth_mask)
    truth = isnothing(truth_mask) ? pam_truth_mask_3d(sources, grid, cfg; radius=truth_radius) : truth_mask
    local_ref = max(maximum(Float64.(intensity)), eps(Float64))
    return [
        begin
            pred = intensity .>= ratio * local_ref
            tp = count(pred .& truth)
            fp = count(pred .& .!truth)
            fn = count(.!pred .& truth)
            precision = tp + fp == 0 ? 0.0 : tp / (tp + fp)
            recall = tp + fn == 0 ? 0.0 : tp / (tp + fn)
            f1 = precision + recall == 0 ? 0.0 : 2 * precision * recall / (precision + recall)
            source_stats = source_detection_stats_3d(pred, grid, cfg, sources; radius=truth_radius)
            source_recall = Float64(source_stats[:source_recall])
            source_f1 = precision + source_recall == 0 ? 0.0 : 2 * precision * source_recall / (precision + source_recall)
            merge(Dict(
                :threshold_ratio => ratio,
                :f1 => source_f1,
                :source_f1 => source_f1,
                :voxel_f1 => f1,
                :precision => precision,
                :recall => source_recall,
                :voxel_recall => recall,
                :true_positive_voxels => tp,
                :false_positive_voxels => fp,
                :false_negative_voxels => fn,
                :predicted_voxels => count(pred),
                :truth_voxels => count(truth),
            ), source_stats)
        end
        for ratio in threshold_ratios
    ]
end

function best_threshold_entry_3d(stats)
    isempty(stats) && error("No 3D threshold stats available.")
    best = first(stats)
    for entry in stats[2:end]
        score_metric = haskey(entry, :source_f1) ? :source_f1 : :f1
        best_metric = haskey(best, :source_f1) ? :source_f1 : :f1
        score = (Float64(entry[score_metric]), Float64(entry[:precision]), Float64(entry[:threshold_ratio]))
        best_score = (Float64(best[best_metric]), Float64(best[:precision]), Float64(best[:threshold_ratio]))
        if score > best_score
            best = entry
        end
    end
    return best
end

_metric_value(entry, key::Symbol, fallback::Symbol=key) = Float64(get(entry, key, entry[fallback]))

function _argmax_by(entries, scorefn)
    best = first(entries)
    best_score = scorefn(best)
    for entry in entries[2:end]
        score = scorefn(entry)
        if score > best_score
            best = entry
            best_score = score
        end
    end
    return best
end

function _threshold_tradeoff_entry_3d(stats, best, target::Symbol)
    best_f1 = _metric_value(best, :source_f1, :f1)
    best_value = _metric_value(best, target, target == :source_recall ? :recall : target)
    metric_key = target == :source_recall ? :source_recall : :precision
    fallback_key = target == :source_recall ? :recall : :precision
    candidates = [entry for entry in stats if _metric_value(entry, metric_key, fallback_key) > best_value + 1e-9]
    for floor_fraction in (0.95, 0.90, 0.0)
        viable = [entry for entry in candidates if _metric_value(entry, :source_f1, :f1) >= floor_fraction * best_f1]
        isempty(viable) && continue
        if target == :source_recall
            return _argmax_by(viable, entry -> (
                _metric_value(entry, :source_recall, :recall),
                _metric_value(entry, :source_f1, :f1),
                _metric_value(entry, :precision),
            ))
        else
            return _argmax_by(viable, entry -> (
                _metric_value(entry, :precision),
                _metric_value(entry, :source_f1, :f1),
                _metric_value(entry, :source_recall, :recall),
            ))
        end
    end
    return best
end

function threshold_outline_entries_3d(stats)
    best = best_threshold_entry_3d(stats)
    recall = _threshold_tradeoff_entry_3d(stats, best, :source_recall)
    precision = _threshold_tradeoff_entry_3d(stats, best, :precision)
    return [
        (kind=:best_f1, label="best F1", color=:cyan, entry=best),
        (kind=:more_recall, label="more recall", color=:lime, entry=recall),
        (kind=:more_precision, label="more precision", color=:magenta, entry=precision),
    ]
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
