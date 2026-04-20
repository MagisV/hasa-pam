abstract type EmissionSource2D end

Base.@kwdef struct PointSource2D <: EmissionSource2D
    depth::Float64
    lateral::Float64
    frequency::Float64 = 8e5
    amplitude::Float64 = 1.0
    phase::Float64 = 0.0
    delay::Float64 = 0.0
    num_cycles::Int = 5
end

Base.@kwdef struct BubbleCluster2D <: EmissionSource2D
    depth::Float64
    lateral::Float64
    fundamental::Float64 = 5e5
    amplitude::Float64 = 1.0
    n_bubbles::Float64 = 1.0
    harmonics::Vector{Int} = [2, 3]
    harmonic_amplitudes::Vector{Float64} = [1.0, 0.6]
    harmonic_phases::Vector{Float64} = [0.0, 0.0]
    gate_duration::Float64 = 50e-6
    taper_ratio::Float64 = 0.25
    delay::Float64 = 0.0
end

_emission_frequencies(src::PointSource2D) = Float64[src.frequency]
_emission_frequencies(src::BubbleCluster2D) = Float64[n * src.fundamental for n in src.harmonics]

Base.@kwdef struct PAMConfig
    dx::Float64 = 0.2e-3
    dz::Float64 = 0.2e-3
    axial_dim::Float64 = 90e-3
    transverse_dim::Float64 = 60e-3
    receiver_aperture::Union{Nothing, Float64} = 50e-3
    receiver_row::Union{Nothing, Int} = nothing
    t_max::Float64 = 80e-6
    dt::Float64 = 40e-9
    c0::Float64 = 1500.0
    rho0::Float64 = 1000.0
    PML_GUARD::Int = 20
    zero_pad_factor::Int = 4
    tukey_ratio::Float64 = 0.25
    peak_suppression_radius::Float64 = 2e-3
    success_tolerance::Float64 = 1e-3
end

function _default_pam_pml_guard(dx::Real)
    # Keep the default physical guard close to 4 mm across PAM resolutions.
    return max(4, round(Int, 4e-3 / Float64(dx)))
end

function _pam_pml_guard(cfg::PAMConfig)
    # `PML_GUARD=20` is the historical default. Interpret it as a placeholder and
    # scale the actual guard with `dx` so coarse grids don't silently lose most
    # of the reconstruction depth.
    if cfg.PML_GUARD == 20
        return _default_pam_pml_guard(cfg.dx)
    end
    return cfg.PML_GUARD
end

function fit_pam_config(
    cfg::PAMConfig,
    sources::AbstractVector{PointSource2D};
    min_bottom_margin::Real=10e-3,
    reference_depth::Union{Nothing, Real}=nothing,
)
    deepest_source_depth = isempty(sources) ? 0.0 : maximum(src.depth for src in sources)
    deepest_required = max(
        deepest_source_depth,
        isnothing(reference_depth) ? 0.0 : Float64(reference_depth),
    ) + Float64(min_bottom_margin)
    required_rows = receiver_row(cfg) + round(Int, deepest_required / cfg.dx)
    target_rows = max(pam_Nx(cfg), required_rows)
    target_axial_dim = target_rows * cfg.dx

    target_rows == pam_Nx(cfg) && return cfg
    return PAMConfig(
        dx=cfg.dx,
        dz=cfg.dz,
        axial_dim=target_axial_dim,
        transverse_dim=cfg.transverse_dim,
        receiver_aperture=cfg.receiver_aperture,
        receiver_row=cfg.receiver_row,
        t_max=cfg.t_max,
        dt=cfg.dt,
        c0=cfg.c0,
        rho0=cfg.rho0,
        PML_GUARD=cfg.PML_GUARD,
        zero_pad_factor=cfg.zero_pad_factor,
        tukey_ratio=cfg.tukey_ratio,
        peak_suppression_radius=cfg.peak_suppression_radius,
        success_tolerance=cfg.success_tolerance,
    )
end

pam_Nx(cfg::PAMConfig) = round(Int, cfg.axial_dim / cfg.dx)
pam_Ny(cfg::PAMConfig) = round(Int, cfg.transverse_dim / cfg.dz)
pam_Nt(cfg::PAMConfig) = round(Int, cfg.t_max / cfg.dt)
receiver_row(cfg::PAMConfig) = something(cfg.receiver_row, 1)

function receiver_col_range(cfg::PAMConfig)
    ny = pam_Ny(cfg)
    if isnothing(cfg.receiver_aperture)
        return 1:ny
    end
    n_active = clamp(round(Int, cfg.receiver_aperture / cfg.dz), 1, ny)
    mid = fld(ny, 2) + 1
    half = fld(n_active, 2)
    start_col = mid - half
    end_col = start_col + n_active - 1
    return start_col:end_col
end

function pam_grid(cfg::PAMConfig; Nt::Union{Nothing, Integer}=nothing)
    nt = isnothing(Nt) ? pam_Nt(cfg) : Int(Nt)
    return KGrid2D(pam_Nx(cfg), pam_Ny(cfg), cfg.dx, cfg.dz; dt=cfg.dt, Nt=nt)
end

function depth_coordinates(kgrid::KGrid2D, cfg::PAMConfig)
    return kgrid.x_vec .- kgrid.x_vec[receiver_row(cfg)]
end

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

function _tukey_window(n::Int, ratio::Real)
    n <= 0 && return Float64[]
    n == 1 && return ones(Float64, 1)
    r = clamp(Float64(ratio), 0.0, 1.0)
    r == 0.0 && return ones(Float64, n)

    x = collect(range(0.0, 1.0; length=n))
    w = ones(Float64, n)
    if r == 1.0
        return 0.5 .* (1 .- cos.(2π .* x))
    end

    left_edge = r / 2
    right_edge = 1 - left_edge
    @inbounds for idx in eachindex(x)
        xi = x[idx]
        if xi < left_edge
            w[idx] = 0.5 * (1 + cos((2π / r) * (xi - left_edge)))
        elseif xi > right_edge
            w[idx] = 0.5 * (1 + cos((2π / r) * (xi - right_edge)))
        end
    end
    return w
end

function _tone_burst_signal(nt::Int, dt::Real, src::PointSource2D; taper_ratio::Real=0.25)
    signal = zeros(Float64, nt)
    duration = src.num_cycles / src.frequency
    samples = collect(0:(nt - 1))
    t = samples .* Float64(dt) .- src.delay
    active = findall((t .>= 0.0) .& (t .<= duration))
    isempty(active) && return signal

    envelope = _tukey_window(length(active), taper_ratio)
    signal[active] .= src.amplitude .* envelope .* sin.(2π .* src.frequency .* t[active] .+ src.phase)
    return signal
end

function _cluster_emission_signal(nt::Int, dt::Real, src::BubbleCluster2D)
    length(src.harmonics) == length(src.harmonic_amplitudes) ||
        error("BubbleCluster2D: harmonics and harmonic_amplitudes must have equal length.")
    length(src.harmonics) == length(src.harmonic_phases) ||
        error("BubbleCluster2D: harmonics and harmonic_phases must have equal length.")

    signal = zeros(Float64, nt)
    samples = collect(0:(nt - 1))
    t = samples .* Float64(dt) .- src.delay
    active = findall((t .>= 0.0) .& (t .<= src.gate_duration))
    isempty(active) && return signal

    envelope = _tukey_window(length(active), src.taper_ratio)
    total_amp = src.amplitude * src.n_bubbles
    t_active = t[active]

    accumulator = zeros(Float64, length(active))
    @inbounds for i in eachindex(src.harmonics)
        n = src.harmonics[i]
        αn = src.harmonic_amplitudes[i]
        φn = src.harmonic_phases[i]
        accumulator .+= αn .* cos.(2π .* n .* src.fundamental .* t_active .+ φn)
    end
    signal[active] .= total_amp .* envelope .* accumulator
    return signal
end

_source_signal(nt::Int, dt::Real, src::PointSource2D) = _tone_burst_signal(nt, dt, src)
_source_signal(nt::Int, dt::Real, src::BubbleCluster2D) = _cluster_emission_signal(nt, dt, src)

function source_grid_index(src::EmissionSource2D, cfg::PAMConfig, kgrid::KGrid2D)
    src.depth >= 0.0 || error("Source depth must be >= 0.")
    row = receiver_row(cfg) + round(Int, src.depth / cfg.dx)
    col = argmin(abs.(kgrid.y_vec .- src.lateral))
    1 <= row <= kgrid.Nx || error("Source depth $(src.depth) m lies outside the computational grid.")
    return row, col
end

function _zero_pad_receiver_rf(rf::AbstractMatrix, target_ny::Int)
    ny, nt = size(rf)
    target_ny >= ny || error("target_ny must be >= current Ny.")
    extra = target_ny - ny
    left = fld(extra, 2)
    range = (left + 1):(left + ny)
    out = zeros(promote_type(Float64, eltype(rf)), target_ny, nt)
    out[range, :] .= rf
    return out, range
end

function _edge_pad_lateral(a::AbstractMatrix{<:Real}, target_ny::Int)
    nx, ny = size(a)
    target_ny >= ny || error("target_ny must be >= current Ny.")
    extra = target_ny - ny
    left = fld(extra, 2)
    range = (left + 1):(left + ny)

    out = Matrix{Float64}(undef, nx, target_ny)
    out[:, range] .= Float64.(a)
    if left > 0
        out[:, 1:left] .= reshape(Float64.(a[:, 1]), :, 1)
    end
    right = target_ny - last(range)
    if right > 0
        out[:, (last(range) + 1):end] .= reshape(Float64.(a[:, end]), :, 1)
    end
    return out, range
end

function _fft_wavenumbers(n::Int, spacing::Real)
    dk = 2π / Float64(spacing)
    start_val = -fld(n, 2)
    end_val = ceil(Int, n / 2) - 1
    return collect(start_val:end_val) .* dk ./ n
end

function _select_frequency_bins(
    rf::AbstractMatrix{<:Real},
    dt::Real,
    frequencies;
    bandwidth::Real=0.0,
)
    nt = size(rf, 2)
    freq_axis = collect(0:(nt - 1)) ./ (nt * Float64(dt))
    pos_bins = 2:(fld(nt, 2) + 1)  # positive frequencies, excluding DC

    if isnothing(frequencies)
        spectrum = fft(rf, 2)
        mean_mag = vec(mean(abs.(spectrum[:, pos_bins]); dims=1))
        idx = argmax(mean_mag)
        return [freq_axis[pos_bins[idx]]], [pos_bins[idx]]
    end

    bins = Int[]
    resolved_freqs = Float64[]
    half_bw = Float64(bandwidth) / 2
    for freq in frequencies
        f = Float64(freq)
        if half_bw > 0
            for bin in pos_bins
                fb = freq_axis[bin]
                if fb >= f - half_bw && fb <= f + half_bw && bin ∉ bins
                    push!(bins, bin)
                    push!(resolved_freqs, fb)
                end
            end
        else
            idx = argmin(abs.(freq_axis[pos_bins] .- f))
            bin = pos_bins[idx]
            if bin ∉ bins
                push!(bins, bin)
                push!(resolved_freqs, freq_axis[bin])
            end
        end
    end
    return resolved_freqs, bins
end

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

function analyse_pam_2d(
    intensity::AbstractMatrix{<:Real},
    kgrid::KGrid2D,
    cfg::PAMConfig,
    sources::AbstractVector{<:EmissionSource2D};
    n_peaks::Union{Nothing, Integer}=nothing,
    success_tolerance::Real=cfg.success_tolerance,
    suppression_radius::Real=cfg.peak_suppression_radius,
)
    n_truth = length(sources)
    n_truth > 0 || error("At least one emission source is required for PAM analysis.")
    n_find = isnothing(n_peaks) ? n_truth : Int(n_peaks)
    peaks = find_pam_peaks(intensity, kgrid, cfg; n_peaks=n_find, suppression_radius=suppression_radius)
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

function reconstruct_pam(
    rf::AbstractMatrix{<:Real},
    c::AbstractMatrix{<:Real},
    cfg::PAMConfig;
    frequencies::Union{Nothing, AbstractVector{<:Real}}=nothing,
    bandwidth::Real=0.0,
    corrected::Bool=true,
)
    nx, ny = size(c)
    size(rf, 1) == ny || error("RF data must have size (Ny, Nt); expected Ny=$ny, got $(size(rf, 1)).")
    nt = size(rf, 2)
    kgrid = KGrid2D(nx, ny, cfg.dx, cfg.dz; dt=cfg.dt, Nt=nt)
    rr = receiver_row(cfg)
    rr <= nx || error("Receiver row lies outside the computational grid.")

    selected_freqs, selected_bins = _select_frequency_bins(rf, cfg.dt, frequencies; bandwidth=bandwidth)
    rf_fft = fft(Float64.(rf), 2)
    padded_ny = cfg.zero_pad_factor > 1 ? cfg.zero_pad_factor * ny : ny
    _, crop_range = _zero_pad_receiver_rf(rf, padded_ny)
    c_padded, _ = _edge_pad_lateral(c, padded_ny)
    intensity_padded = zeros(Float64, nx, padded_ny)
    row_stop = nx
    row_stop > rr || error("No valid reconstruction rows remain below the receiver row.")

    for (freq, bin) in zip(selected_freqs, selected_bins)
        p0 = rf_fft[:, bin]
        p0_padded, _ = _zero_pad_receiver_rf(reshape(p0, ny, 1), padded_ny)
        p0_vec = vec(p0_padded[:, 1])

        c0 = mean(c_padded)
        k0 = 2π * freq / c0
        k = _fft_wavenumbers(padded_ny, cfg.dz)
        kz = sqrt.(complex.(k0^2 .- k .^ 2, 0.0))
        propagator = exp.(1im .* kz .* cfg.dx)

        real_inds = findall(real.(kz ./ k0) .> 0.0)
        weighting = zeros(Float64, padded_ny)
        weighting[real_inds] .= _tukey_window(length(real_inds), cfg.tukey_ratio)

        current = _fftshift(fft(p0_vec))
        current .*= weighting

        mu = (c0 ./ c_padded) .^ 2
        lambda = (k0^2) .* (1 .- mu)

        correction = zeros(ComplexF64, padded_ny)
        for idx in real_inds
            abs(kz[idx]) > sqrt(eps(Float64)) || continue
            correction[idx] = propagator[idx] * cfg.dx / (2im * kz[idx])
        end

        for row in (rr + 1):row_stop
            if corrected
                p_space = ifft(_ifftshift(current))
                conv_term = _fftshift(fft(lambda[row, :] .* p_space))
                next = current .* propagator
                next .+= correction .* conv_term
            else
                next = current .* propagator
            end
            next[setdiff(eachindex(next), real_inds)] .= 0.0
            next .*= weighting
            current = next

            p_row = ifft(_ifftshift(current))
            intensity_padded[row, :] .+= abs2.(p_row)
        end
    end

    intensity = intensity_padded[:, crop_range]
    info = Dict{Symbol, Any}(
        :frequencies => selected_freqs,
        :frequency_bins => selected_bins,
        :bandwidth => Float64(bandwidth),
        :corrected => corrected,
        :receiver_row => rr,
        :crop_range => crop_range,
    )
    return intensity, kgrid, info
end

function _default_recon_frequencies(sources::AbstractVector{<:EmissionSource2D})
    all_freqs = Float64[]
    for src in sources
        append!(all_freqs, _emission_frequencies(src))
    end
    return sort(unique(all_freqs))
end

function run_pam_case(
    c::AbstractMatrix{<:Real},
    rho::AbstractMatrix{<:Real},
    sources::AbstractVector{<:EmissionSource2D},
    cfg::PAMConfig;
    frequencies::Union{Nothing, AbstractVector{<:Real}}=nothing,
    bandwidth::Real=0.0,
    use_gpu::Bool=false,
)
    recon_freqs = isnothing(frequencies) ? _default_recon_frequencies(sources) : Float64.(frequencies)
    rf, kgrid, sim_info = simulate_point_sources(c, rho, sources, cfg; use_gpu=use_gpu)
    pam_geo, _, geo_info = reconstruct_pam(rf, c, cfg; frequencies=recon_freqs, bandwidth=bandwidth, corrected=false)
    pam_hasa, _, hasa_info = reconstruct_pam(rf, c, cfg; frequencies=recon_freqs, bandwidth=bandwidth, corrected=true)

    stats_geo = analyse_pam_2d(pam_geo, kgrid, cfg, sources)
    stats_hasa = analyse_pam_2d(pam_hasa, kgrid, cfg, sources)

    return Dict{Symbol, Any}(
        :rf => rf,
        :kgrid => kgrid,
        :simulation => sim_info,
        :pam_geo => pam_geo,
        :pam_hasa => pam_hasa,
        :geo_info => geo_info,
        :hasa_info => hasa_info,
        :stats_geo => stats_geo,
        :stats_hasa => stats_hasa,
        :reconstruction_frequencies => recon_freqs,
    )
end

function _pam_mm_key(depth_mm::Real, lateral_mm::Real)
    return (round(Float64(depth_mm); digits=6), round(Float64(lateral_mm); digits=6))
end

function _pam_mm_key(src::PointSource2D)
    return _pam_mm_key(src.depth * 1e3, src.lateral * 1e3)
end

function _resolve_pam_sweep_targets(
    preset::Union{Symbol, AbstractString};
    axial_targets_mm::Union{Nothing, AbstractVector{<:Real}}=nothing,
    lateral_targets_mm::Union{Nothing, AbstractVector{<:Real}}=nothing,
)
    explicit_targets = !isnothing(axial_targets_mm) || !isnothing(lateral_targets_mm)
    if explicit_targets
        isnothing(axial_targets_mm) && error("Custom PAM sweep requires explicit axial target positions.")
        isnothing(lateral_targets_mm) && error("Custom PAM sweep requires explicit lateral target positions.")
        axial = sort(unique(Float64.(axial_targets_mm)))
        lateral = sort(unique(Float64.(lateral_targets_mm)))
        isempty(axial) && error("At least one axial target is required for a PAM sweep.")
        isempty(lateral) && error("At least one lateral target is required for a PAM sweep.")
        return :custom, axial, lateral
    end

    mode = preset isa Symbol ? preset : Symbol(lowercase(strip(preset)))
    if mode == :paper
        return :paper, [30.0, 40.0, 50.0, 60.0, 70.0, 80.0], [-20.0, -10.0, 0.0, 10.0, 20.0]
    elseif mode == :quick
        return :quick, [40.0, 60.0, 80.0], [-10.0, 0.0, 10.0]
    elseif mode == :custom
        error("Custom PAM sweep requires both --axial-targets-mm and --lateral-targets-mm.")
    end
    error("Unknown PAM sweep preset: $preset")
end

function _default_pam_sweep_examples(targets::AbstractVector{PointSource2D})
    isempty(targets) && error("At least one target is required to choose PAM sweep examples.")

    depth_values = sort(unique(Float64[src.depth * 1e3 for src in targets]))
    num_examples = min(3, length(depth_values))
    selected_depth_indices = unique(round.(Int, collect(range(1, length(depth_values); length=num_examples))))

    examples = Tuple{Float64, Float64}[]
    for depth_idx in selected_depth_indices
        depth_mm = depth_values[depth_idx]
        candidates = [src for src in targets if isapprox(src.depth * 1e3, depth_mm; atol=1e-6)]
        isempty(candidates) && continue
        best = candidates[argmin(abs.([src.lateral for src in candidates]))]
        push!(examples, _pam_mm_key(best))
    end
    return examples
end

function _normalize_pam_sweep_examples(
    targets::AbstractVector{PointSource2D},
    example_targets_mm::Union{Nothing, AbstractVector{<:Tuple{<:Real, <:Real}}},
)
    if isnothing(example_targets_mm)
        return _default_pam_sweep_examples(targets)
    end

    1 <= length(example_targets_mm) <= 3 || error("Provide between 1 and 3 PAM sweep example targets.")
    available = Set(_pam_mm_key(src) for src in targets)
    examples = Tuple{Float64, Float64}[]
    for target in example_targets_mm
        key = _pam_mm_key(target[1], target[2])
        key in available || error("Example target $(target[1]) mm, $(target[2]) mm is not part of the PAM sweep.")
        push!(examples, key)
    end
    return sort(unique(examples))
end

function _pam_skull_cavity_start_rows(
    c::AbstractMatrix{<:Real};
    c_water::Real=1500.0,
    tol::Real=5.0,
    min_thick_rows::Integer=2,
)
    skull_mask = skull_mask_from_c_columnwise(
        c;
        c_water=c_water,
        tol=tol,
        min_thick_rows=min_thick_rows,
        dilate_rows=1,
        close_iters=1,
        mask_outside=false,
    )

    ny = size(c, 2)
    start_rows = zeros(Int, ny)
    has_skull = falses(ny)
    for col in 1:ny
        rows = findall(skull_mask[:, col])
        isempty(rows) && continue
        has_skull[col] = true
        start_rows[col] = last(rows) + 1
    end
    return start_rows, has_skull
end

function _filter_pam_targets_in_skull_cavity(
    c::AbstractMatrix{<:Real},
    cfg::PAMConfig,
    targets::AbstractVector{PointSource2D};
    min_margin::Real=1e-3,
    c_water::Real=cfg.c0,
    tol::Real=5.0,
    min_thick_rows::Integer=2,
)
    kgrid = pam_grid(cfg)
    cavity_start_rows, has_skull = _pam_skull_cavity_start_rows(
        c;
        c_water=c_water,
        tol=tol,
        min_thick_rows=min_thick_rows,
    )
    margin_rows = max(0, ceil(Int, Float64(min_margin) / cfg.dx))

    valid_targets = PointSource2D[]
    dropped_targets = Dict{Symbol, Any}[]

    for src in targets
        row, col = source_grid_index(src, cfg, kgrid)
        truth_mm = (src.depth * 1e3, src.lateral * 1e3)
        if !has_skull[col]
            push!(dropped_targets, Dict{Symbol, Any}(
                :truth_mm => truth_mm,
                :row => row,
                :col => col,
                :reason => :no_skull_above,
            ))
            continue
        end

        required_row = cavity_start_rows[col] + margin_rows
        if row < required_row
            push!(dropped_targets, Dict{Symbol, Any}(
                :truth_mm => truth_mm,
                :row => row,
                :col => col,
                :required_row => required_row,
                :reason => :too_shallow_for_cavity,
            ))
            continue
        end

        if abs(Float64(c[row, col]) - Float64(c_water)) > Float64(tol)
            push!(dropped_targets, Dict{Symbol, Any}(
                :truth_mm => truth_mm,
                :row => row,
                :col => col,
                :reason => :non_fluid_target_cell,
            ))
            continue
        end

        push!(valid_targets, src)
    end

    return valid_targets, dropped_targets, cavity_start_rows
end

function run_pam_sweep(
    c::AbstractMatrix{<:Real},
    rho::AbstractMatrix{<:Real},
    targets::AbstractVector{PointSource2D},
    cfg::PAMConfig;
    frequencies::Union{Nothing, AbstractVector{<:Real}}=nothing,
    example_targets_mm::Union{Nothing, AbstractVector{<:Tuple{<:Real, <:Real}}}=nothing,
    use_gpu::Bool=false,
    runner::Function=run_pam_case,
    case_callback::Union{Nothing, Function}=nothing,
)
    isempty(targets) && error("At least one PAM sweep target is required.")

    sorted_targets = sort(collect(targets); by=src -> _pam_mm_key(src))
    axial_targets_mm = sort(unique(Float64[src.depth * 1e3 for src in sorted_targets]))
    lateral_targets_mm = sort(unique(Float64[src.lateral * 1e3 for src in sorted_targets]))
    axial_index = Dict(_pam_mm_key(depth_mm, 0.0)[1] => idx for (idx, depth_mm) in pairs(axial_targets_mm))
    lateral_index = Dict(_pam_mm_key(0.0, lateral_mm)[2] => idx for (idx, lateral_mm) in pairs(lateral_targets_mm))

    geo_error_mm = fill(NaN, length(axial_targets_mm), length(lateral_targets_mm))
    hasa_error_mm = similar(geo_error_mm)
    geo_peak_intensity = similar(geo_error_mm)
    hasa_peak_intensity = similar(geo_error_mm)

    example_keys = Set(_normalize_pam_sweep_examples(sorted_targets, example_targets_mm))
    cases = Dict{Symbol, Any}[]
    example_cases = Dict{Symbol, Any}[]

    for src in sorted_targets
        results = runner(
            c,
            rho,
            PointSource2D[src],
            cfg;
            frequencies=frequencies,
            use_gpu=use_gpu,
        )
        stats_geo = results[:stats_geo]
        stats_hasa = results[:stats_hasa]

        target_key = _pam_mm_key(src)
        row = axial_index[target_key[1]]
        col = lateral_index[target_key[2]]
        geo_error_mm[row, col] = Float64(stats_geo[:mean_radial_error_mm])
        hasa_error_mm[row, col] = Float64(stats_hasa[:mean_radial_error_mm])
        geo_peak_intensity[row, col] = Float64(stats_geo[:mean_norm_peak_intensity])
        hasa_peak_intensity[row, col] = Float64(stats_hasa[:mean_norm_peak_intensity])

        case_result = Dict{Symbol, Any}(
            :source => src,
            :truth_mm => (src.depth * 1e3, src.lateral * 1e3),
            :stats_geo => stats_geo,
            :stats_hasa => stats_hasa,
            :geo_predicted_mm => only(stats_geo[:predicted_mm]),
            :hasa_predicted_mm => only(stats_hasa[:predicted_mm]),
            :reconstruction_frequencies => results[:reconstruction_frequencies],
            :simulation => results[:simulation],
        )
        push!(cases, case_result)

        if !isnothing(case_callback)
            case_callback(case_result, results)
        end

        if target_key in example_keys
            example_result = copy(case_result)
            example_result[:rf] = results[:rf]
            example_result[:pam_geo] = results[:pam_geo]
            example_result[:pam_hasa] = results[:pam_hasa]
            example_result[:kgrid] = results[:kgrid]
            push!(example_cases, example_result)
        end
    end

    sort!(example_cases; by=case -> _pam_mm_key(case[:truth_mm]...))
    return Dict{Symbol, Any}(
        :cases => cases,
        :axial_targets_mm => axial_targets_mm,
        :lateral_targets_mm => lateral_targets_mm,
        :geo_error_mm => geo_error_mm,
        :hasa_error_mm => hasa_error_mm,
        :geo_peak_intensity => geo_peak_intensity,
        :hasa_peak_intensity => hasa_peak_intensity,
        :example_cases => example_cases,
        :example_targets_mm => [case[:truth_mm] for case in example_cases],
    )
end
