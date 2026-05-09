function _connected_component(mask::BitMatrix, seed::Tuple{Int, Int})
    mask[seed...] || return Tuple{Int, Int}[]
    rows, cols = size(mask)
    visited = falses(rows, cols)
    queue = [seed]
    visited[seed...] = true
    component = Tuple{Int, Int}[]

    while !isempty(queue)
        current = popfirst!(queue)
        push!(component, current)
        i, j = current
        for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
            ii = i + di
            jj = j + dj
            if 1 <= ii <= rows && 1 <= jj <= cols && mask[ii, jj] && !visited[ii, jj]
                visited[ii, jj] = true
                push!(queue, (ii, jj))
            end
        end
    end

    return component
end

function _peak_fwhm_mm(intensity::AbstractMatrix{<:Real}, kgrid::KGrid2D, cfg::PAMConfig, idx::Tuple{Int, Int})
    peak = Float64(intensity[idx...])
    peak <= 0.0 && return 0.0, 0.0

    mask = Float64.(intensity) .>= peak / 2
    component = _connected_component(mask, idx)
    isempty(component) && return 0.0, 0.0

    depth = depth_coordinates(kgrid, cfg)
    lateral = kgrid.y_vec
    rows = first.(component)
    cols = last.(component)
    axial_fwhm = (maximum(depth[rows]) - minimum(depth[rows])) * 1e3
    lateral_fwhm = (maximum(lateral[cols]) - minimum(lateral[cols])) * 1e3
    return axial_fwhm, lateral_fwhm
end

function _best_assignment(cost::AbstractMatrix{<:Real})
    n_rows, n_cols = size(cost)
    n_rows == n_cols || error("Cost matrix must be square for assignment.")

    best_cost = Ref(Inf)
    best_perm = collect(1:n_cols)
    used = falses(n_cols)
    current = Vector{Int}(undef, n_rows)

    function recurse(row::Int, running::Float64)
        if row > n_rows
            if running < best_cost[]
                best_cost[] = running
                best_perm .= current
            end
            return
        end

        for col in 1:n_cols
            used[col] && continue
            new_cost = running + Float64(cost[row, col])
            new_cost < best_cost[] || continue
            used[col] = true
            current[row] = col
            recurse(row + 1, new_cost)
            used[col] = false
        end
    end

    recurse(1, 0.0)
    return best_perm, best_cost[]
end

function find_pam_peaks(
    intensity::AbstractMatrix{<:Real},
    kgrid::KGrid2D,
    cfg::PAMConfig;
    n_peaks::Integer,
    suppression_radius::Real=cfg.peak_suppression_radius,
)
    work = copy(Float64.(intensity))
    row_start = receiver_row(cfg) + 1
    row_stop = size(work, 1)
    row_start <= row_stop || error("No valid reconstruction rows remain after excluding the receiver row and PML.")
    work[1:(row_start - 1), :] .= -Inf
    if row_stop < size(work, 1)
        work[(row_stop + 1):end, :] .= -Inf
    end

    rad_rows = max(1, round(Int, suppression_radius / cfg.dx))
    rad_cols = max(1, round(Int, suppression_radius / cfg.dz))
    peaks = Tuple{Int, Int}[]

    for _ in 1:Int(n_peaks)
        idx = Tuple(argmax(work))
        isfinite(work[idx...]) || break
        push!(peaks, idx)
        r0, c0 = idx
        r1 = max(1, r0 - rad_rows)
        r2 = min(size(work, 1), r0 + rad_rows)
        c1 = max(1, c0 - rad_cols)
        c2 = min(size(work, 2), c0 + rad_cols)
        work[r1:r2, c1:c2] .= -Inf
    end

    return peaks
end

function _default_psf_widths(
    cfg::PAMConfig,
    kgrid::KGrid2D,
    frequencies::Union{Nothing, AbstractVector{<:Real}};
    characteristic_depth::Real=30e-3,
)
    aperture = something(cfg.receiver_aperture, kgrid.Ny * cfg.dz)
    depth = max(Float64(characteristic_depth), 2 * cfg.dx)

    freqs = isnothing(frequencies) || isempty(frequencies) ? nothing : Float64.(frequencies)
    if isnothing(freqs)
        # Fallback: assume one wavelength worth of structure
        lambda = cfg.c0 / 5e5
        lateral = lambda * depth / aperture
        axial = 2 * lambda * (depth / aperture)^2
        return max(axial, 2 * cfg.dx), max(lateral, 2 * cfg.dz)
    end

    f_max = maximum(freqs)
    f_min = minimum(freqs)
    lambda_min = cfg.c0 / f_max
    bw = f_max - f_min

    lateral = lambda_min * depth / aperture
    axial = if bw > 0
        cfg.c0 / (2 * bw)
    else
        2 * lambda_min * (depth / aperture)^2
    end
    return max(axial, 2 * cfg.dx), max(lateral, 2 * cfg.dz)
end

"""
    find_pam_peaks_clean(intensity, kgrid, cfg; n_peaks, frequencies=nothing,
                         psf_axial_fwhm=nothing, psf_lateral_fwhm=nothing,
                         loop_gain=0.1, max_iter=500, threshold_ratio=1e-2,
                         suppression_radius=nothing)

Iterative CLEAN (Högbom) peak detector for PAM intensity maps. Each iteration
finds the brightest residual pixel, adds `loop_gain * peak` to the accumulator,
and subtracts a scaled Gaussian PSF from the residual. The `n_peaks` brightest
maxima in the accumulator are returned. If `suppression_radius` is not given,
it defaults to the lateral PSF FWHM, which lets sources as close as one PSF
width apart be resolved distinctly.
"""
function find_pam_peaks_clean(
    intensity::AbstractMatrix{<:Real},
    kgrid::KGrid2D,
    cfg::PAMConfig;
    n_peaks::Integer,
    frequencies::Union{Nothing, AbstractVector{<:Real}}=nothing,
    psf_axial_fwhm::Union{Nothing, Real}=nothing,
    psf_lateral_fwhm::Union{Nothing, Real}=nothing,
    loop_gain::Real=0.1,
    max_iter::Integer=500,
    threshold_ratio::Real=1e-2,
    suppression_radius::Union{Nothing, Real}=nothing,
)
    0 < loop_gain <= 1 || error("loop_gain must lie in (0, 1].")
    n_peaks > 0 || error("n_peaks must be positive.")

    residual = copy(Float64.(intensity))
    row_start = receiver_row(cfg) + 1
    row_stop = size(residual, 1)
    row_start <= row_stop || error("No valid reconstruction rows remain.")
    residual[1:(row_start - 1), :] .= -Inf
    if row_stop < size(residual, 1)
        residual[(row_stop + 1):end, :] .= -Inf
    end

    ax_fwhm, lat_fwhm = if isnothing(psf_axial_fwhm) || isnothing(psf_lateral_fwhm)
        _default_psf_widths(cfg, kgrid, frequencies)
    else
        Float64(psf_axial_fwhm), Float64(psf_lateral_fwhm)
    end
    ax_fwhm = something(psf_axial_fwhm, ax_fwhm)
    lat_fwhm = something(psf_lateral_fwhm, lat_fwhm)

    σ_ax_cells = max(1.0, Float64(ax_fwhm) / (cfg.dx * 2.3548))
    σ_lat_cells = max(1.0, Float64(lat_fwhm) / (cfg.dz * 2.3548))
    half_ax = max(1, ceil(Int, 3 * σ_ax_cells))
    half_lat = max(1, ceil(Int, 3 * σ_lat_cells))

    finite_mask = isfinite.(residual)
    any(finite_mask) || return Tuple{Int, Int}[]
    peak_init = maximum(residual[finite_mask])
    peak_init > 0 || return Tuple{Int, Int}[]
    threshold = peak_init * Float64(threshold_ratio)

    accum = zeros(Float64, size(residual))
    nx, ny = size(residual)

    for _ in 1:Int(max_iter)
        idx = Tuple(argmax(residual))
        pv = residual[idx...]
        (!isfinite(pv) || pv < threshold) && break

        scale = Float64(loop_gain) * pv
        r0, c0 = idx
        accum[r0, c0] += scale

        r1 = max(1, r0 - half_ax)
        r2 = min(nx, r0 + half_ax)
        c1 = max(1, c0 - half_lat)
        c2 = min(ny, c0 + half_lat)
        @inbounds for r in r1:r2
            dr = (r - r0) / σ_ax_cells
            for c in c1:c2
                dc = (c - c0) / σ_lat_cells
                weight = exp(-0.5 * (dr^2 + dc^2))
                residual[r, c] -= scale * weight
            end
        end
    end

    sup_radius = isnothing(suppression_radius) ? Float64(lat_fwhm) : Float64(suppression_radius)
    return find_pam_peaks(accum, kgrid, cfg; n_peaks=n_peaks, suppression_radius=sup_radius)
end

function pam_truth_mask(
    sources::AbstractVector{<:EmissionSource2D},
    kgrid::KGrid2D,
    cfg::PAMConfig;
    radius::Real=cfg.success_tolerance,
)
    radius_m = Float64(radius)
    radius_m >= 0 || error("truth-mask radius must be non-negative.")

    mask = falses(kgrid.Nx, kgrid.Ny)
    depth = depth_coordinates(kgrid, cfg)
    lateral = kgrid.y_vec
    radius2 = radius_m^2
    row_radius = ceil(Int, radius_m / cfg.dx)
    col_radius = ceil(Int, radius_m / cfg.dz)

    for src in sources
        row0, col0 = source_grid_index(src, cfg, kgrid)
        row_start = max(receiver_row(cfg) + 1, row0 - row_radius)
        row_stop = min(kgrid.Nx, row0 + row_radius)
        col_start = max(1, col0 - col_radius)
        col_stop = min(kgrid.Ny, col0 + col_radius)
        @inbounds for row in row_start:row_stop
            dd = depth[row] - src.depth
            for col in col_start:col_stop
                dl = lateral[col] - src.lateral
                if dd^2 + dl^2 <= radius2
                    mask[row, col] = true
                end
            end
        end
    end

    return mask
end

function _mark_centerline_segment!(
    mask::BitMatrix,
    depth::AbstractVector{<:Real},
    lateral::AbstractVector{<:Real},
    cfg::PAMConfig,
    d0::Real,
    l0::Real,
    d1::Real,
    l1::Real,
    radius_m::Real,
)
    dd = Float64(d1) - Float64(d0)
    dl = Float64(l1) - Float64(l0)
    seg_len2 = dd^2 + dl^2
    radius2 = Float64(radius_m)^2
    row_pad = ceil(Int, Float64(radius_m) / cfg.dx) + 1
    col_pad = ceil(Int, Float64(radius_m) / cfg.dz) + 1
    row_min = clamp(searchsortedfirst(depth, min(Float64(d0), Float64(d1)) - Float64(radius_m)) - row_pad, receiver_row(cfg) + 1, length(depth))
    row_max = clamp(searchsortedlast(depth, max(Float64(d0), Float64(d1)) + Float64(radius_m)) + row_pad, receiver_row(cfg) + 1, length(depth))
    col_min = clamp(searchsortedfirst(lateral, min(Float64(l0), Float64(l1)) - Float64(radius_m)) - col_pad, 1, length(lateral))
    col_max = clamp(searchsortedlast(lateral, max(Float64(l0), Float64(l1)) + Float64(radius_m)) + col_pad, 1, length(lateral))

    @inbounds for row in row_min:row_max
        pd = Float64(depth[row])
        for col in col_min:col_max
            pl = Float64(lateral[col])
            t = seg_len2 <= eps(Float64) ? 0.0 : clamp(((pd - Float64(d0)) * dd + (pl - Float64(l0)) * dl) / seg_len2, 0.0, 1.0)
            nearest_d = Float64(d0) + t * dd
            nearest_l = Float64(l0) + t * dl
            if (pd - nearest_d)^2 + (pl - nearest_l)^2 <= radius2
                mask[row, col] = true
            end
        end
    end
    return mask
end

function pam_centerline_truth_mask(
    centerlines::AbstractVector,
    kgrid::KGrid2D,
    cfg::PAMConfig;
    radius::Real=cfg.success_tolerance,
)
    radius_m = Float64(radius)
    radius_m >= 0 || error("centerline truth-mask radius must be non-negative.")
    mask = falses(kgrid.Nx, kgrid.Ny)
    depth = depth_coordinates(kgrid, cfg)
    lateral = kgrid.y_vec

    for centerline in centerlines
        length(centerline) >= 2 || continue
        for idx in 1:(length(centerline) - 1)
            d0, l0 = centerline[idx]
            d1, l1 = centerline[idx + 1]
            _mark_centerline_segment!(mask, depth, lateral, cfg, d0, l0, d1, l1, radius_m)
        end
    end
    return mask
end

_source_activity_weight(src::PointSource2D) = abs(src.amplitude)
_source_activity_weight(src::BubbleCluster2D) = abs(src.amplitude * src.n_bubbles)
_source_activity_weight(src::GaussianPulseCluster2D) = abs(src.amplitude * src.n_bubbles)

"""
    pam_source_map(sources, kgrid, cfg; weights=:amplitude)

Rasterize source positions onto the PAM grid. Multiple sources in the same
cell are accumulated. `weights=:amplitude` uses source activity amplitudes,
while `weights=:uniform` assigns each source equal weight.
"""
function pam_source_map(
    sources::AbstractVector{<:EmissionSource2D},
    kgrid::KGrid2D,
    cfg::PAMConfig;
    weights::Symbol=:amplitude,
)
    weights in (:amplitude, :uniform) ||
        error("Unknown source-map weights: $weights (expected :amplitude or :uniform).")
    source_map = zeros(Float64, kgrid.Nx, kgrid.Ny)
    for src in sources
        row, col = source_grid_index(src, cfg, kgrid)
        source_map[row, col] += weights == :uniform ? 1.0 : _source_activity_weight(src)
    end
    source_map[1:receiver_row(cfg), :] .= 0.0
    return source_map
end

function _gaussian_kernel_cells(σ::Real)
    sigma = Float64(σ)
    sigma > 0 || return [1.0]
    half_width = max(1, ceil(Int, 3 * sigma))
    kernel = [exp(-0.5 * (offset / sigma)^2) for offset in -half_width:half_width]
    kernel ./= sum(kernel)
    return kernel
end

function _convolve_axis_zero(a::AbstractMatrix{<:Real}, kernel::AbstractVector{<:Real}, axis::Int)
    axis in (1, 2) || error("axis must be 1 or 2.")
    out = zeros(Float64, size(a))
    rows, cols = size(a)
    center = fld(length(kernel), 2) + 1
    @inbounds if axis == 1
        for row in 1:rows, col in 1:cols
            acc = 0.0
            for (kidx, kval) in pairs(kernel)
                rr = row + kidx - center
                1 <= rr <= rows || continue
                acc += Float64(kval) * Float64(a[rr, col])
            end
            out[row, col] = acc
        end
    else
        for row in 1:rows, col in 1:cols
            acc = 0.0
            for (kidx, kval) in pairs(kernel)
                cc = col + kidx - center
                1 <= cc <= cols || continue
                acc += Float64(kval) * Float64(a[row, cc])
            end
            out[row, col] = acc
        end
    end
    return out
end

"""
    pam_psf_blur(map, kgrid, cfg; frequencies=nothing,
                 psf_axial_fwhm=nothing, psf_lateral_fwhm=nothing)

Blur a ground-truth activity map by a Gaussian approximation to the PAM point
spread function. If PSF FWHM values are omitted, a diffraction/bandwidth-based
default is estimated from the reconstruction frequencies.
"""
function pam_psf_blur(
    truth_map::AbstractMatrix{<:Real},
    kgrid::KGrid2D,
    cfg::PAMConfig;
    frequencies::Union{Nothing, AbstractVector{<:Real}}=nothing,
    psf_axial_fwhm::Union{Nothing, Real}=nothing,
    psf_lateral_fwhm::Union{Nothing, Real}=nothing,
)
    size(truth_map) == (kgrid.Nx, kgrid.Ny) ||
        error("truth_map size $(size(truth_map)) does not match kgrid size ($(kgrid.Nx), $(kgrid.Ny)).")
    ax_fwhm, lat_fwhm = if isnothing(psf_axial_fwhm) || isnothing(psf_lateral_fwhm)
        _default_psf_widths(cfg, kgrid, frequencies)
    else
        Float64(psf_axial_fwhm), Float64(psf_lateral_fwhm)
    end
    ax_fwhm = Float64(something(psf_axial_fwhm, ax_fwhm))
    lat_fwhm = Float64(something(psf_lateral_fwhm, lat_fwhm))
    ax_fwhm >= 0 || error("psf_axial_fwhm must be non-negative.")
    lat_fwhm >= 0 || error("psf_lateral_fwhm must be non-negative.")

    work = max.(Float64.(truth_map), 0.0)
    work[1:receiver_row(cfg), :] .= 0.0
    σ_ax = ax_fwhm / (2.3548 * cfg.dx)
    σ_lat = lat_fwhm / (2.3548 * cfg.dz)
    blurred = _convolve_axis_zero(work, _gaussian_kernel_cells(σ_ax), 1)
    blurred = _convolve_axis_zero(blurred, _gaussian_kernel_cells(σ_lat), 2)
    blurred[1:receiver_row(cfg), :] .= 0.0
    return blurred
end

function pam_psf_blurred_truth_map(
    sources::AbstractVector{<:EmissionSource2D},
    kgrid::KGrid2D,
    cfg::PAMConfig;
    frequencies::Union{Nothing, AbstractVector{<:Real}}=nothing,
    psf_axial_fwhm::Union{Nothing, Real}=nothing,
    psf_lateral_fwhm::Union{Nothing, Real}=nothing,
    weights::Symbol=:amplitude,
)
    source_map = pam_source_map(sources, kgrid, cfg; weights=weights)
    return pam_psf_blur(
        source_map,
        kgrid,
        cfg;
        frequencies=frequencies,
        psf_axial_fwhm=psf_axial_fwhm,
        psf_lateral_fwhm=psf_lateral_fwhm,
    )
end

function threshold_pam_map(
    intensity::AbstractMatrix{<:Real},
    cfg::PAMConfig;
    threshold_ratio::Real=0.2,
)
    ratio = Float64(threshold_ratio)
    ratio > 0 || error("threshold_ratio must be positive.")
    work = Float64.(intensity)
    ref = maximum(work)
    ref > 0 || return falses(size(work))

    mask = work .>= (ratio * ref)
    mask[1:receiver_row(cfg), :] .= false
    return mask
end

function pam_intensity_metrics(
    intensity::AbstractMatrix{<:Real},
    kgrid::KGrid2D,
    cfg::PAMConfig;
    threshold_ratio::Real=0.2,
    reference_intensity::Union{Nothing, Real}=nothing,
)
    size(intensity) == (kgrid.Nx, kgrid.Ny) ||
        error("PAM intensity size $(size(intensity)) does not match kgrid size ($(kgrid.Nx), $(kgrid.Ny)).")
    ratio = Float64(threshold_ratio)
    ratio > 0 || error("threshold_ratio must be positive.")

    work = Float64.(intensity)
    peak = maximum(work)
    ref = isnothing(reference_intensity) ? peak : Float64(reference_intensity)
    ref = max(ref, eps(Float64))
    local_threshold = ratio * max(peak, eps(Float64))
    shared_threshold = ratio * ref

    active = work .>= local_threshold
    active[1:receiver_row(cfg), :] .= false
    shared_active = work .>= shared_threshold
    shared_active[1:receiver_row(cfg), :] .= false

    depth = depth_coordinates(kgrid, cfg)
    lateral = kgrid.y_vec
    active_idxs = findall(active)
    centroid_depth_mm = NaN
    centroid_lateral_mm = NaN
    if !isempty(active_idxs)
        weight_sum = sum(work[idx] for idx in active_idxs)
        if weight_sum > 0
            centroid_depth_mm = sum(work[idx] * depth[idx[1]] for idx in active_idxs) / weight_sum * 1e3
            centroid_lateral_mm = sum(work[idx] * lateral[idx[2]] for idx in active_idxs) / weight_sum * 1e3
        end
    end

    pixel_area_mm2 = cfg.dx * cfg.dz * 1e6
    max_idx = Tuple(argmax(work))
    return Dict{Symbol, Any}(
        :peak_intensity => peak,
        :relative_peak_intensity => peak / ref,
        :integrated_intensity_m2 => sum(work) * cfg.dx * cfg.dz,
        :threshold_ratio => ratio,
        :active_area_mm2 => count(active) * pixel_area_mm2,
        :shared_scale_active_area_mm2 => count(shared_active) * pixel_area_mm2,
        :centroid_depth_mm => centroid_depth_mm,
        :centroid_lateral_mm => centroid_lateral_mm,
        :peak_depth_mm => depth[max_idx[1]] * 1e3,
        :peak_lateral_mm => lateral[max_idx[2]] * 1e3,
    )
end

function _component_overlap_counts(mask::BitMatrix, reference::BitMatrix)
    size(mask) == size(reference) || error("Component masks must have the same size.")
    rows, cols = size(mask)
    visited = falses(rows, cols)
    total = 0
    overlapping = 0

    for row in 1:rows, col in 1:cols
        (mask[row, col] && !visited[row, col]) || continue
        total += 1
        touches_reference = false
        queue = [(row, col)]
        visited[row, col] = true

        while !isempty(queue)
            current = popfirst!(queue)
            i, j = current
            touches_reference |= reference[i, j]
            for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
                ii = i + di
                jj = j + dj
                if 1 <= ii <= rows && 1 <= jj <= cols && mask[ii, jj] && !visited[ii, jj]
                    visited[ii, jj] = true
                    push!(queue, (ii, jj))
                end
            end
        end

        overlapping += touches_reference ? 1 : 0
    end

    return total, overlapping, total - overlapping
end

_safe_fraction(num::Real, den::Real) = den > 0 ? Float64(num) / Float64(den) : 0.0

function _valid_reconstruction_mask(kgrid::KGrid2D, cfg::PAMConfig)
    valid = trues(kgrid.Nx, kgrid.Ny)
    valid[1:receiver_row(cfg), :] .= false
    return valid
end

function _weighted_centroid_spread_mm(
    weights::AbstractMatrix{<:Real},
    kgrid::KGrid2D,
    cfg::PAMConfig;
    valid_mask::Union{Nothing, AbstractMatrix{Bool}}=nothing,
)
    size(weights) == (kgrid.Nx, kgrid.Ny) ||
        error("weights size $(size(weights)) does not match kgrid size ($(kgrid.Nx), $(kgrid.Ny)).")
    valid = isnothing(valid_mask) ? _valid_reconstruction_mask(kgrid, cfg) : BitMatrix(valid_mask)
    size(valid) == (kgrid.Nx, kgrid.Ny) ||
        error("valid_mask size $(size(valid)) does not match kgrid size ($(kgrid.Nx), $(kgrid.Ny)).")

    work = max.(Float64.(weights), 0.0)
    work[.!valid] .= 0.0
    total = sum(work)
    total > 0 || return NaN, NaN, NaN, NaN

    depth = depth_coordinates(kgrid, cfg)
    lateral = kgrid.y_vec
    centroid_depth = 0.0
    centroid_lateral = 0.0
    @inbounds for row in 1:kgrid.Nx, col in 1:kgrid.Ny
        w = work[row, col]
        w > 0 || continue
        centroid_depth += w * depth[row]
        centroid_lateral += w * lateral[col]
    end
    centroid_depth /= total
    centroid_lateral /= total

    axial_var = 0.0
    lateral_var = 0.0
    @inbounds for row in 1:kgrid.Nx, col in 1:kgrid.Ny
        w = work[row, col]
        w > 0 || continue
        axial_var += w * (depth[row] - centroid_depth)^2
        lateral_var += w * (lateral[col] - centroid_lateral)^2
    end
    axial_spread_mm = sqrt(axial_var / total) * 1e3
    lateral_spread_mm = sqrt(lateral_var / total) * 1e3
    return centroid_depth * 1e3, centroid_lateral * 1e3, axial_spread_mm, lateral_spread_mm
end

function _unit_sum_map(a::AbstractMatrix{<:Real}, valid_mask::AbstractMatrix{Bool})
    size(a) == size(valid_mask) || error("map and valid_mask must have the same size.")
    out = max.(Float64.(a), 0.0)
    out[.!valid_mask] .= 0.0
    total = sum(out)
    total > 0 && (out ./= total)
    return out
end

function _pearson_correlation(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    length(a) == length(b) || error("correlation vectors must have the same length.")
    isempty(a) && return NaN
    ma = mean(a)
    mb = mean(b)
    da = Float64.(a) .- ma
    db = Float64.(b) .- mb
    den = sqrt(sum(abs2, da) * sum(abs2, db))
    den > 0 || return NaN
    return sum(da .* db) / den
end

function _global_ssim_like(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    length(a) == length(b) || error("SSIM vectors must have the same length.")
    isempty(a) && return NaN
    af = Float64.(a)
    bf = Float64.(b)
    μa = mean(af)
    μb = mean(bf)
    σa2 = mean((af .- μa) .^ 2)
    σb2 = mean((bf .- μb) .^ 2)
    σab = mean((af .- μa) .* (bf .- μb))
    dynamic_range = max(maximum(af), maximum(bf)) - min(minimum(af), minimum(bf))
    dynamic_range = max(dynamic_range, eps(Float64))
    c1 = (0.01 * dynamic_range)^2
    c2 = (0.03 * dynamic_range)^2
    return ((2 * μa * μb + c1) * (2 * σab + c2)) /
           ((μa^2 + μb^2 + c1) * (σa2 + σb2 + c2))
end

function _psf_target_similarity_metrics(
    intensity::AbstractMatrix{<:Real},
    target::AbstractMatrix{<:Real},
    kgrid::KGrid2D,
    cfg::PAMConfig,
)
    size(intensity) == size(target) == (kgrid.Nx, kgrid.Ny) ||
        error("intensity, target, and kgrid sizes must agree.")
    valid = _valid_reconstruction_mask(kgrid, cfg)
    pred = _unit_sum_map(intensity, valid)
    truth = _unit_sum_map(target, valid)
    pred_vec = pred[valid]
    truth_vec = truth[valid]
    target_norm = sqrt(sum(abs2, truth_vec))
    normalized_l2 = target_norm > 0 ? sqrt(sum(abs2, pred_vec .- truth_vec)) / target_norm : NaN
    return Dict{Symbol, Any}(
        :correlation => _pearson_correlation(pred_vec, truth_vec),
        :ssim_like => _global_ssim_like(pred_vec, truth_vec),
        :normalized_l2_error => normalized_l2,
        :target_integral => sum(max.(Float64.(target), 0.0)),
    )
end

function analyse_pam_detection_2d(
    intensity::AbstractMatrix{<:Real},
    kgrid::KGrid2D,
    cfg::PAMConfig,
    sources::AbstractVector{<:EmissionSource2D};
    truth_radius::Real=cfg.success_tolerance,
    threshold_ratio::Real=0.2,
    truth_mask::Union{Nothing, AbstractMatrix{Bool}}=nothing,
    frequencies::Union{Nothing, AbstractVector{<:Real}}=nothing,
    psf_axial_fwhm::Union{Nothing, Real}=nothing,
    psf_lateral_fwhm::Union{Nothing, Real}=nothing,
)
    isempty(sources) && error("At least one emission source is required for PAM detection analysis.")
    truth = if isnothing(truth_mask)
        pam_truth_mask(sources, kgrid, cfg; radius=truth_radius)
    else
        size(truth_mask) == (kgrid.Nx, kgrid.Ny) ||
            error("truth_mask size $(size(truth_mask)) does not match kgrid size ($(kgrid.Nx), $(kgrid.Ny)).")
        BitMatrix(truth_mask)
    end
    predicted = threshold_pam_map(intensity, cfg; threshold_ratio=threshold_ratio)
    valid = _valid_reconstruction_mask(kgrid, cfg)

    tp = count(predicted .& truth)
    fp = count(predicted .& (.!truth))
    fn = count((.!predicted) .& truth)
    tn = length(predicted) - tp - fp - fn

    precision = _safe_fraction(tp, tp + fp)
    recall = _safe_fraction(tp, tp + fn)
    f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0
    dice = _safe_fraction(2 * tp, 2 * tp + fp + fn)

    prediction_components, matched_prediction_components, spurious_prediction_components =
        _component_overlap_counts(predicted, truth)
    truth_components, recovered_truth_components, missed_truth_components =
        _component_overlap_counts(truth, predicted)

    pixel_area_mm2 = cfg.dx * cfg.dz * 1e6
    max_idx = Tuple(argmax(Float64.(intensity)))
    depth = depth_coordinates(kgrid, cfg)
    lateral = kgrid.y_vec
    max_intensity = maximum(Float64.(intensity))
    work = max.(Float64.(intensity), 0.0)
    work[.!valid] .= 0.0
    total_energy = sum(work)
    energy_inside = sum(work[truth])
    energy_outside = max(total_energy - energy_inside, 0.0)
    energy_inside_predicted = sum(work[predicted])
    energy_outside_predicted = max(total_energy - energy_inside_predicted, 0.0)
    centroid_depth_mm, centroid_lateral_mm, axial_spread_mm, lateral_spread_mm =
        _weighted_centroid_spread_mm(work, kgrid, cfg; valid_mask=valid)

    target_base = if isnothing(truth_mask)
        pam_source_map(sources, kgrid, cfg; weights=:amplitude)
    else
        Float64.(truth)
    end
    psf_target = pam_psf_blur(
        target_base,
        kgrid,
        cfg;
        frequencies=frequencies,
        psf_axial_fwhm=psf_axial_fwhm,
        psf_lateral_fwhm=psf_lateral_fwhm,
    )
    target_centroid_depth_mm, target_centroid_lateral_mm, target_axial_spread_mm, target_lateral_spread_mm =
        _weighted_centroid_spread_mm(psf_target, kgrid, cfg; valid_mask=valid)
    centroid_error_mm = if isfinite(centroid_depth_mm) && isfinite(target_centroid_depth_mm)
        hypot(centroid_depth_mm - target_centroid_depth_mm, centroid_lateral_mm - target_centroid_lateral_mm)
    else
        NaN
    end
    psf_similarity = _psf_target_similarity_metrics(intensity, psf_target, kgrid, cfg)

    return Dict{Symbol, Any}(
        :truth_mm => [(src.depth * 1e3, src.lateral * 1e3) for src in sources],
        :num_truth_sources => length(sources),
        :truth_radius_mm => Float64(truth_radius) * 1e3,
        :truth_mask_mode => isnothing(truth_mask) ? :source_disks : :provided,
        :psf_target_mode => isnothing(truth_mask) ? :source_map : :provided_mask,
        :psf_axial_fwhm_mm => Float64(something(psf_axial_fwhm, _default_psf_widths(cfg, kgrid, frequencies)[1])) * 1e3,
        :psf_lateral_fwhm_mm => Float64(something(psf_lateral_fwhm, _default_psf_widths(cfg, kgrid, frequencies)[2])) * 1e3,
        :threshold_ratio => Float64(threshold_ratio),
        :threshold_db => 10 * log10(Float64(threshold_ratio)),
        :true_positive_pixels => tp,
        :false_positive_pixels => fp,
        :false_negative_pixels => fn,
        :true_negative_pixels => tn,
        :precision => precision,
        :recall => recall,
        :f1 => f1,
        :dice => dice,
        :truth_area_mm2 => count(truth) * pixel_area_mm2,
        :predicted_area_mm2 => count(predicted) * pixel_area_mm2,
        :overlap_area_mm2 => tp * pixel_area_mm2,
        :false_positive_area_mm2 => fp * pixel_area_mm2,
        :false_negative_area_mm2 => fn * pixel_area_mm2,
        :prediction_components => prediction_components,
        :matched_prediction_components => matched_prediction_components,
        :spurious_prediction_components => spurious_prediction_components,
        :truth_components => truth_components,
        :recovered_truth_components => recovered_truth_components,
        :missed_truth_components => missed_truth_components,
        :energy_inside_mask => energy_inside,
        :energy_outside_mask => energy_outside,
        :energy_total => total_energy,
        :energy_fraction_inside_mask => _safe_fraction(energy_inside, total_energy),
        :energy_fraction_outside_mask => _safe_fraction(energy_outside, total_energy),
        :energy_inside_predicted_mask => energy_inside_predicted,
        :energy_outside_predicted_mask => energy_outside_predicted,
        :energy_fraction_inside_predicted_mask => _safe_fraction(energy_inside_predicted, total_energy),
        :energy_fraction_outside_predicted_mask => _safe_fraction(energy_outside_predicted, total_energy),
        :centroid_depth_mm => centroid_depth_mm,
        :centroid_lateral_mm => centroid_lateral_mm,
        :target_centroid_depth_mm => target_centroid_depth_mm,
        :target_centroid_lateral_mm => target_centroid_lateral_mm,
        :centroid_error_mm => centroid_error_mm,
        :axial_spread_mm => axial_spread_mm,
        :lateral_spread_mm => lateral_spread_mm,
        :target_axial_spread_mm => target_axial_spread_mm,
        :target_lateral_spread_mm => target_lateral_spread_mm,
        :psf_target_correlation => psf_similarity[:correlation],
        :psf_target_ssim_like => psf_similarity[:ssim_like],
        :psf_target_normalized_l2_error => psf_similarity[:normalized_l2_error],
        :psf_target_integral => psf_similarity[:target_integral],
        :peak_mm => (depth[max_idx[1]] * 1e3, lateral[max_idx[2]] * 1e3),
        :peak_intensity => Float64(intensity[max_idx...]),
        :max_intensity => max_intensity,
    )
end

function analyse_pam_2d(
    intensity::AbstractMatrix{<:Real},
    kgrid::KGrid2D,
    cfg::PAMConfig,
    sources::AbstractVector{<:EmissionSource2D};
    n_peaks::Union{Nothing, Integer}=nothing,
    success_tolerance::Real=cfg.success_tolerance,
    suppression_radius::Real=cfg.peak_suppression_radius,
    peak_method::Symbol=:argmax,
    frequencies::Union{Nothing, AbstractVector{<:Real}}=nothing,
    clean_loop_gain::Real=0.1,
    clean_max_iter::Integer=500,
    clean_threshold_ratio::Real=1e-2,
    clean_psf_axial_fwhm::Union{Nothing, Real}=nothing,
    clean_psf_lateral_fwhm::Union{Nothing, Real}=nothing,
)
    n_truth = length(sources)
    n_truth > 0 || error("At least one emission source is required for PAM analysis.")
    n_find = isnothing(n_peaks) ? n_truth : Int(n_peaks)
    peaks = if peak_method == :clean
        find_pam_peaks_clean(
            intensity, kgrid, cfg;
            n_peaks=n_find,
            frequencies=frequencies,
            psf_axial_fwhm=clean_psf_axial_fwhm,
            psf_lateral_fwhm=clean_psf_lateral_fwhm,
            loop_gain=clean_loop_gain,
            max_iter=clean_max_iter,
            threshold_ratio=clean_threshold_ratio,
            suppression_radius=nothing,
        )
    elseif peak_method == :argmax
        find_pam_peaks(intensity, kgrid, cfg; n_peaks=n_find, suppression_radius=suppression_radius)
    else
        error("Unknown peak_method: $peak_method (expected :argmax or :clean).")
    end
    length(peaks) == n_truth || error("Expected to recover $n_truth peaks, found $(length(peaks)).")

    depth = depth_coordinates(kgrid, cfg)
    lateral = kgrid.y_vec
    truth_mm = [(src.depth * 1e3, src.lateral * 1e3) for src in sources]
    pred_mm = [(depth[idx[1]] * 1e3, lateral[idx[2]] * 1e3) for idx in peaks]

    cost = Matrix{Float64}(undef, n_truth, n_truth)
    for i in 1:n_truth, j in 1:n_truth
        d_ax = truth_mm[i][1] - pred_mm[j][1]
        d_lat = truth_mm[i][2] - pred_mm[j][2]
        cost[i, j] = hypot(d_ax, d_lat)
    end
    assignment, _ = _best_assignment(cost)

    matched_pred_mm = [pred_mm[assignment[i]] for i in 1:n_truth]
    matched_indices = [peaks[assignment[i]] for i in 1:n_truth]
    axial_errors_mm = [truth_mm[i][1] - matched_pred_mm[i][1] for i in 1:n_truth]
    lateral_errors_mm = [truth_mm[i][2] - matched_pred_mm[i][2] for i in 1:n_truth]
    radial_errors_mm = [hypot(axial_errors_mm[i], lateral_errors_mm[i]) for i in 1:n_truth]

    raw_peak_intensities = [Float64(intensity[idx...]) for idx in matched_indices]
    max_intensity = max(maximum(Float64.(intensity)), eps(Float64))
    norm_peak_intensities = raw_peak_intensities ./ max_intensity

    axial_fwhm_mm = Float64[]
    lateral_fwhm_mm = Float64[]
    for idx in matched_indices
        axial_fwhm, lateral_fwhm = _peak_fwhm_mm(intensity, kgrid, cfg, idx)
        push!(axial_fwhm_mm, axial_fwhm)
        push!(lateral_fwhm_mm, lateral_fwhm)
    end

    tol_mm = Float64(success_tolerance) * 1e3
    successes = radial_errors_mm .<= tol_mm

    num_success = count(identity, successes)
    return Dict{Symbol, Any}(
        :truth_mm => truth_mm,
        :predicted_mm => matched_pred_mm,
        :peak_indices => matched_indices,
        :axial_errors_mm => axial_errors_mm,
        :lateral_errors_mm => lateral_errors_mm,
        :radial_errors_mm => radial_errors_mm,
        :mean_axial_error_mm => mean(abs.(axial_errors_mm)),
        :mean_lateral_error_mm => mean(abs.(lateral_errors_mm)),
        :mean_radial_error_mm => mean(radial_errors_mm),
        :max_radial_error_mm => maximum(radial_errors_mm),
        :success_tolerance_mm => tol_mm,
        :success_rate => num_success / n_truth,
        :num_success => num_success,
        :raw_peak_intensities => raw_peak_intensities,
        :norm_peak_intensities => norm_peak_intensities,
        :mean_norm_peak_intensity => mean(norm_peak_intensities),
        :axial_fwhm_mm => axial_fwhm_mm,
        :lateral_fwhm_mm => lateral_fwhm_mm,
        :mean_axial_fwhm_mm => mean(axial_fwhm_mm),
        :mean_lateral_fwhm_mm => mean(lateral_fwhm_mm),
    )
end

