"""
    _pam_reference_sound_speed(c, cfg, sources; margin=10e-3)

Estimate a representative 3D sound speed in m/s over the receiver-to-source
depth range used by reconstruction.
"""
function _pam_reference_sound_speed(
    c::AbstractArray{<:Real, 3},
    cfg::PAMConfig3D,
    sources::AbstractVector{<:EmissionSource3D};
    margin::Real=10e-3,
)
    isempty(sources) && return mean(Float64.(c))
    row_start = clamp(receiver_row(cfg), 1, size(c, 1))
    deepest_source_depth = maximum(src.depth for src in sources)
    row_stop = row_start + ceil(Int, (deepest_source_depth + Float64(margin)) / cfg.dx)
    row_stop = clamp(row_stop, row_start, size(c, 1))
    return mean(Float64.(view(c, row_start:row_stop, :, :)))
end

"""
    _zero_pad_receiver_rf_3d(rf, target_ny, target_nz)

Center-pad 3D receiver RF data laterally and return Y/Z crop ranges.
"""
function _zero_pad_receiver_rf_3d(rf::AbstractArray{<:Real, 3}, target_ny::Int, target_nz::Int)
    ny, nz, nt = size(rf)
    target_ny >= ny || error("target_ny must be >= current Ny.")
    target_nz >= nz || error("target_nz must be >= current Nz.")
    extra_y = target_ny - ny
    extra_z = target_nz - nz
    left_y  = fld(extra_y, 2)
    left_z  = fld(extra_z, 2)
    range_y = (left_y + 1):(left_y + ny)
    range_z = (left_z + 1):(left_z + nz)
    out = zeros(promote_type(Float64, eltype(rf)), target_ny, target_nz, nt)
    out[range_y, range_z, :] .= rf
    return out, range_y, range_z
end

"""
    _edge_pad_lateral_3d(a, target_ny, target_nz)

Pad a 3D field laterally by repeating edge planes and return crop ranges.
"""
function _edge_pad_lateral_3d(a::AbstractArray{<:Real, 3}, target_ny::Int, target_nz::Int)
    nx, ny, nz = size(a)
    target_ny >= ny || error("target_ny must be >= current Ny.")
    target_nz >= nz || error("target_nz must be >= current Nz.")
    extra_y = target_ny - ny
    extra_z = target_nz - nz
    left_y  = fld(extra_y, 2)
    left_z  = fld(extra_z, 2)
    range_y = (left_y + 1):(left_y + ny)
    range_z = (left_z + 1):(left_z + nz)

    out = Array{Float64, 3}(undef, nx, target_ny, target_nz)
    out[:, range_y, range_z] .= Float64.(a)

    if left_y > 0
        out[:, 1:left_y, range_z] .= Float64.(a[:, 1:1, :])
    end
    right_y = target_ny - last(range_y)
    if right_y > 0
        out[:, (last(range_y) + 1):end, range_z] .= Float64.(a[:, end:end, :])
    end
    if left_z > 0
        out[:, :, 1:left_z] .= out[:, :, (left_z + 1):(left_z + 1)]
    end
    right_z = target_nz - last(range_z)
    if right_z > 0
        out[:, :, (last(range_z) + 1):end] .= out[:, :, last(range_z):last(range_z)]
    end
    return out, range_y, range_z
end

"""
    _ifftshift_2d!(a)

In-place 2D inverse FFT shift used for lateral spectral operators.
"""
function _ifftshift_2d!(a::AbstractMatrix)
    ny, nz = size(a)
    sy = -fld(ny, 2)
    sz = -fld(nz, 2)
    circshift!(similar(a), a, (sy, sz))
end

"""
    _ifftshift_2d(a)

Return a 2D inverse FFT shift of a lateral spectral operator.
"""
function _ifftshift_2d(a::AbstractMatrix)
    ny, nz = size(a)
    return circshift(a, (-fld(ny, 2), -fld(nz, 2)))
end

"""
    _tukey_radial(k_radii, k_max, ratio)

Return a radial Tukey taper for 3D lateral wavenumber magnitudes.
"""
function _tukey_radial(k_radii::AbstractMatrix, k_max::Real, ratio::Real)
    w = ones(Float64, size(k_radii))
    k_max > 0 || return w
    ratio_f = clamp(Float64(ratio), 0.0, 1.0)
    transition_start = k_max * (1 - ratio_f)
    for i in eachindex(k_radii)
        r = k_radii[i]
        if r > k_max
            w[i] = 0.0
        elseif r >= transition_start && ratio_f > 0
            t = (r - transition_start) / (k_max - transition_start)
            w[i] = 0.5 * (1 + cos(π * t))
        end
    end
    return w
end

"""
    _select_frequency_bins_3d(rf, dt, frequencies; bandwidth=0.0)

Resolve reconstruction frequencies in Hz and FFT bin indices for 3D RF data.
"""
function _select_frequency_bins_3d(
    rf::AbstractArray{<:Real, 3},
    dt::Real,
    frequencies;
    bandwidth::Real=0.0,
)
    nt = size(rf, 3)
    return _select_frequency_bins(reshape(rf, :, nt), dt, frequencies; bandwidth=bandwidth)
end

"""
    _apply_axial_gain_3d!(intensity, cfg)

Apply the configured depth-dependent axial gain to a 3D PAM intensity volume.
"""
function _apply_axial_gain_3d!(intensity::AbstractArray{<:Real, 3}, cfg::PAMConfig3D)
    power = Float64(cfg.axial_gain_power)
    power == 0.0 && return intensity
    rr = receiver_row(cfg)
    for row in axes(intensity, 1)
        depth_cells = max(row - rr, 1)
        intensity[row, :, :] .*= depth_cells ^ power
    end
    return intensity
end

"""
    PAMCUDASetup3D

Precomputed 3D CUDA arrays and reconstruction metadata shared across one or
more PAM GPU window batches.
"""
struct PAMCUDASetup3D
    eta_yznx_d         # (padded_ny, padded_nz, nx) Float32 — speed-contrast field
    prop_d1            # (padded_ny, padded_nz, nfreq) ComplexF32 — propagator, un-tiled
    corr_d1            # (padded_ny, padded_nz, nfreq) ComplexF32 — correction, un-tiled
    weight_d1          # (padded_ny, padded_nz, nfreq) Float32   — 2D Tukey taper, un-tiled
    prop_n_weight_d1   # (padded_ny, padded_nz, nfreq) ComplexF32
    k0_sq_d1           # (1, 1, nfreq) Float32
    selected_freqs::Vector{Float64}
    selected_bins::Vector{Int}
    crop_range_y::UnitRange{Int}
    crop_range_z::UnitRange{Int}
    nfreq::Int
    padded_ny::Int
    padded_nz::Int
    nx::Int
    rr::Int
    row_stop::Int
    axial_substeps::Int
    setup_s::Float64
end

"""
    _pam_cuda_setup_3d(c_padded, cfg, selected_freqs, selected_bins, crop_range_y, crop_range_z, nx, padded_ny, padded_nz, rr, row_stop, c0, effective_axial_step, axial_substeps)

Precompute 3D CUDA propagation operators for the selected frequencies.
"""
function _pam_cuda_setup_3d(
    c_padded::AbstractArray{<:Real, 3},
    cfg::PAMConfig3D,
    selected_freqs::AbstractVector{<:Real},
    selected_bins::AbstractVector{<:Integer},
    crop_range_y::UnitRange{Int},
    crop_range_z::UnitRange{Int},
    nx::Int,
    padded_ny::Int,
    padded_nz::Int,
    rr::Int,
    row_stop::Int,
    c0::Float64,
    effective_axial_step::Float64,
    axial_substeps::Int,
)
    _assert_pam_cuda_available()
    t0_setup = time()
    T  = _PAM_CUDA_PRECISION
    CT = _PAM_CUDA_COMPLEX
    nfreq = length(selected_freqs)

    eta_cpu = Float32.(permutedims(1 .- (c0 ./ c_padded) .^ 2, (2, 3, 1)))  # (padded_ny, padded_nz, nx)
    eta_yznx_d = CUDA.CuArray(eta_cpu)

    k_y = _fft_wavenumbers(padded_ny, cfg.dy)
    k_z = _fft_wavenumbers(padded_nz, cfg.dz)
    KY  = reshape(k_y, :, 1) .* ones(1, padded_nz)
    KZ  = ones(padded_ny, 1) .* reshape(k_z, 1, :)
    k_lat2 = KY .^ 2 .+ KZ .^ 2
    k_radii = sqrt.(k_lat2)

    prop_cpu          = zeros(ComplexF64, padded_ny, padded_nz, nfreq)
    corr_cpu          = zeros(ComplexF64, padded_ny, padded_nz, nfreq)
    weight_cpu        = zeros(Float64,    padded_ny, padded_nz, nfreq)
    prop_n_weight_cpu = zeros(ComplexF64, padded_ny, padded_nz, nfreq)
    k0_sq_cpu         = zeros(Float64, nfreq)

    for (f, freq) in enumerate(selected_freqs)
        k0 = 2π * Float64(freq) / c0
        k_axial = sqrt.(complex.(k0^2 .- k_lat2, 0.0))
        propagating = real.(k_axial ./ k0) .> 0.0
        propagator = exp.(1im .* k_axial .* effective_axial_step) .* propagating

        k_max_prop = maximum(k_radii[propagating]; init=0.0)
        weighting = _tukey_radial(k_radii, k_max_prop, cfg.tukey_ratio) .* propagating

        correction = zeros(ComplexF64, padded_ny, padded_nz)
        for j in eachindex(k_axial)
            propagating[j] || continue
            abs(k_axial[j]) > sqrt(eps(Float64)) || continue
            correction[j] = propagator[j] * effective_axial_step / (2im * k_axial[j])
        end

        prop_shifted          = _ifftshift_2d(propagating .* propagator)
        corr_shifted          = _ifftshift_2d(propagating .* correction)
        weight_shifted        = _ifftshift_2d(weighting)
        prop_n_shifted        = _ifftshift_2d(propagating .* propagator .^ axial_substeps .* weighting)

        prop_cpu[:, :, f]          .= prop_shifted
        corr_cpu[:, :, f]          .= corr_shifted
        weight_cpu[:, :, f]        .= weight_shifted
        prop_n_weight_cpu[:, :, f] .= prop_n_shifted
        k0_sq_cpu[f]                = k0^2
    end

    prop_d1          = CUDA.CuArray(CT.(prop_cpu))
    corr_d1          = CUDA.CuArray(CT.(corr_cpu))
    weight_d1        = CUDA.CuArray(T.(weight_cpu))
    prop_n_weight_d1 = CUDA.CuArray(CT.(prop_n_weight_cpu))
    k0_sq_d1         = reshape(CUDA.CuArray(T.(k0_sq_cpu)), 1, 1, nfreq)

    return PAMCUDASetup3D(
        eta_yznx_d, prop_d1, corr_d1, weight_d1, prop_n_weight_d1, k0_sq_d1,
        Vector{Float64}(selected_freqs), Vector{Int}(selected_bins),
        crop_range_y, crop_range_z,
        nfreq, padded_ny, padded_nz, nx, rr, row_stop, axial_substeps,
        time() - t0_setup,
    )
end

"""
    _accum_abs2_sum_batched_3d!(intensity, src, nfreq_per_w, row, crop_y0, crop_z0)

CUDA kernel that accumulates batched 3D frequency-plane `abs2` sums into one
axial row of the cropped intensity volume.
"""
function _accum_abs2_sum_batched_3d!(intensity, src, nfreq_per_w, row, crop_y0, crop_z0)
    i = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x
    ny = size(intensity, 1)
    nz = size(intensity, 2)
    i > ny * nz && return
    iy = (i - 1) % ny + 1
    iz = (i - 1) ÷ ny + 1
    src_y = crop_y0 + iy - 1
    src_z = crop_z0 + iz - 1
    W  = size(src, 4)
    @inbounds for w in 1:W
        acc = zero(real(eltype(src)))
        for j in 1:nfreq_per_w
            v = src[src_y, src_z, j, w]
            acc += real(v)^2 + imag(v)^2
        end
        intensity[iy, iz, w, row] += acc
    end
    return
end

"""
    _reconstruct_pam_cuda_3d(setup, rf_windows, t0_windows, corrected, recon_label, show_progress, benchmark)

Run one 3D CUDA reconstruction batch and return raw intensity volumes plus
timing metadata.
"""
function _reconstruct_pam_cuda_3d(
    setup::PAMCUDASetup3D,
    rf_windows::AbstractVector{<:AbstractArray{<:Real, 3}},
    t0_windows::AbstractVector{<:Real},
    corrected::Bool,
    recon_label::AbstractString,
    show_progress::Bool,
    benchmark::Bool,
)
    fn_start = time()
    W           = length(rf_windows)
    nfreq       = setup.nfreq
    padded_ny   = setup.padded_ny
    padded_nz   = setup.padded_nz
    nx          = setup.nx
    rr          = setup.rr
    row_stop    = setup.row_stop
    axial_substeps = setup.axial_substeps
    crop_range_y   = setup.crop_range_y
    crop_range_z   = setup.crop_range_z

    T  = _PAM_CUDA_PRECISION
    CT = _PAM_CUDA_COMPLEX

    let dev = CUDA.device()
        println(
            "[ PAM 3D ] $recon_label: GPU $(CUDA.name(dev)) (device $(CUDA.deviceid(dev))), " *
            "$(_PAM_CUDA_PRECISION) arithmetic, $nfreq freq bins × $W windows batched",
        )
        flush(stdout)
    end

    # Keep the window axis explicit so the frequency-domain operators broadcast
    # over windows instead of being physically repeated W times.
    weight_d        = reshape(setup.weight_d1,        padded_ny, padded_nz, nfreq, 1)
    prop_n_weight_d = corrected ? nothing : reshape(setup.prop_n_weight_d1, padded_ny, padded_nz, nfreq, 1)
    prop_weight_d   = corrected ? reshape(setup.prop_d1 .* setup.weight_d1, padded_ny, padded_nz, nfreq, 1) : nothing
    corr_weight_d   = corrected ? reshape(setup.corr_d1 .* setup.weight_d1, padded_ny, padded_nz, nfreq, 1) : nothing
    k0_sq_d         = corrected ? reshape(setup.k0_sq_d1, 1, 1, nfreq, 1) : nothing

    # Intensity accumulator: (ny, nz, W, nx). The zero-padded halo is only
    # needed while propagating; storing it for every row/window is too large.
    crop_ny = length(crop_range_y)
    crop_nz = length(crop_range_z)
    crop_y0 = first(crop_range_y)
    crop_z0 = first(crop_range_z)
    padded_ny_nz = padded_ny * padded_nz
    crop_ny_nz = crop_ny * crop_nz
    intensity_yzWx_d = CUDA.zeros(T, crop_ny, crop_nz, W, nx)

    # Build batched initial conditions: (padded_ny, padded_nz, nfreq, W).
    p0_d = CUDA.zeros(CT, padded_ny, padded_nz, nfreq, W)
    for (w, (rf_w, t0_w)) in enumerate(zip(rf_windows, t0_windows))
        rf_fft_w   = fft(CUDA.CuArray(T.(rf_w)), 3)  # time FFT → (ny, nz, nt)
        for (f, (freq, bin)) in enumerate(zip(setup.selected_freqs, setup.selected_bins))
            phase = CT(cis(-T(2π) * T(freq) * T(t0_w)))
            p0_d[crop_range_y, crop_range_z, f, w] .= rf_fft_w[:, :, bin] .* phase
        end
        rf_fft_w = nothing
        GC.gc(false)
        CUDA.reclaim()
    end

    plan_fwd = plan_fft!(p0_d, (1, 2))
    plan_bwd = plan_ifft!(similar(p0_d), (1, 2))
    plan_fwd * p0_d
    current_d = p0_d
    current_d .*= weight_d
    next_d    = corrected ? similar(current_d) : nothing
    tmp_d     = corrected ? similar(current_d) : nothing
    p_space_d = corrected ? similar(current_d) : nothing
    p_row_d   = similar(current_d)

    t_batch_setup = time() - fn_start
    t_fft_s = 0.0
    t_ew_s  = 0.0

    march_wall_start = time()

    nthreads = min(crop_ny_nz, 512)
    nblocks  = cld(crop_ny_nz, nthreads)

    eta_yznx_d = setup.eta_yznx_d

    if benchmark
        for row in (rr + 1):row_stop
            if corrected
                for _ in 1:(axial_substeps - 1)
                    t_fft_s += CUDA.@elapsed (p_space_d .= current_d; plan_bwd * p_space_d)
                    t_ew_s  += CUDA.@elapsed (tmp_d .= k0_sq_d .* eta_yznx_d[:, :, row:row] .* p_space_d)
                    t_fft_s += CUDA.@elapsed (plan_fwd * tmp_d)
                    t_ew_s  += CUDA.@elapsed (next_d .= current_d .* prop_weight_d .+ corr_weight_d .* tmp_d)
                    current_d, next_d = next_d, current_d
                end
                t_fft_s += CUDA.@elapsed (p_space_d .= current_d; plan_bwd * p_space_d)
                t_ew_s  += CUDA.@elapsed (tmp_d .= k0_sq_d .* eta_yznx_d[:, :, row:row] .* p_space_d)
                t_fft_s += CUDA.@elapsed (plan_fwd * tmp_d)
                t_ew_s  += CUDA.@elapsed (current_d .= current_d .* prop_weight_d .+ corr_weight_d .* tmp_d)
            else
                t_ew_s  += CUDA.@elapsed (current_d .*= prop_n_weight_d)
            end
            t_ew_s  += CUDA.@elapsed (p_row_d .= current_d)
            # Inverse FFT on the full padded plane first; only crop when
            # accumulating intensity so lateral wraparound stays suppressed.
            t_fft_s += CUDA.@elapsed (plan_bwd * p_row_d)
            t_ew_s  += CUDA.@elapsed CUDA.@cuda threads=nthreads blocks=nblocks _accum_abs2_sum_batched_3d!(
                intensity_yzWx_d, p_row_d, nfreq, row, crop_y0, crop_z0,
            )
        end
        march_gpu_s = t_fft_s + t_ew_s
    else
        march_gpu_s = CUDA.@elapsed for row in (rr + 1):row_stop
            if corrected
                for _ in 1:(axial_substeps - 1)
                    p_space_d .= current_d
                    plan_bwd * p_space_d
                    tmp_d .= k0_sq_d .* eta_yznx_d[:, :, row:row] .* p_space_d
                    plan_fwd * tmp_d
                    next_d .= current_d .* prop_weight_d .+ corr_weight_d .* tmp_d
                    current_d, next_d = next_d, current_d
                end
                p_space_d .= current_d
                plan_bwd * p_space_d
                tmp_d .= k0_sq_d .* eta_yznx_d[:, :, row:row] .* p_space_d
                plan_fwd * tmp_d
                current_d .= current_d .* prop_weight_d .+ corr_weight_d .* tmp_d
            else
                current_d .*= prop_n_weight_d
            end
            p_row_d .= current_d
            # Inverse FFT on the full padded plane first; only crop when
            # accumulating intensity so lateral wraparound stays suppressed.
            plan_bwd * p_row_d
            CUDA.@cuda threads=nthreads blocks=nblocks _accum_abs2_sum_batched_3d!(
                intensity_yzWx_d, p_row_d, nfreq, row, crop_y0, crop_z0,
            )
        end
    end
    march_wall_s = time() - march_wall_start

    passes_substep  = 13
    passes_last_row = corrected ? 19 : 9
    nfreq_total = nfreq * W
    bytes_march = Int64(row_stop - rr) * (
        (corrected ? Int64(axial_substeps - 1) * passes_substep : 0) * padded_ny_nz * nfreq_total * sizeof(CT) +
        passes_last_row * padded_ny_nz * nfreq_total * sizeof(CT)
    )
    bandwidth_GBps = bytes_march / march_gpu_s / 1e9

    t_download = @elapsed begin
        # intensity_yzWx_d: (ny, nz, W, nx)
        # permute → (ny, nz, nx, W), then slice per w → (ny, nz, nx) → (nx, ny, nz)
        raw_all = permutedims(Array(intensity_yzWx_d), (1, 2, 4, 3))  # (ny, nz, nx, W)
    end
    raws = [Float64.(permutedims(raw_all[:, :, :, w], (3, 1, 2))) for w in 1:W]  # each: (nx, ny, nz)

    _pam_progress(
        show_progress,
        "PAM 3D $recon_label $nfreq freq × $W win batch: march $(round(march_gpu_s * 1e3; digits=1)) ms GPU / " *
        "$(round(march_wall_s * 1e3; digits=1)) ms wall, " *
        "BW ~$(round(bandwidth_GBps; digits=0)) GB/s" *
        (benchmark ? ", FFT $(round(100 * t_fft_s / march_gpu_s; digits=1))% / EW $(round(100 * t_ew_s / march_gpu_s; digits=1))%" : ""),
    )

    timing = Dict{Symbol, Any}(
        :setup_s         => setup.setup_s + t_batch_setup,
        :operator_setup_s => setup.setup_s,
        :batch_setup_s   => t_batch_setup,
        :march_gpu_s     => march_gpu_s,
        :march_wall_s    => march_wall_s,
        :download_s      => t_download,
        :bandwidth_GBps  => bandwidth_GBps,
        :fft_s           => benchmark ? t_fft_s : nothing,
        :elementwise_s   => benchmark ? t_ew_s  : nothing,
        :nrows           => row_stop - rr,
        :nfreq           => nfreq,
        :nwindows        => W,
        :padded_ny       => padded_ny,
        :padded_nz       => padded_nz,
        :axial_substeps  => axial_substeps,
        :bytes_march_est => bytes_march,
    )
    prop_n_weight_d = nothing
    prop_weight_d = nothing
    corr_weight_d = nothing
    k0_sq_d = nothing
    weight_d = nothing
    p0_d = nothing
    current_d = nothing
    next_d = nothing
    tmp_d = nothing
    p_space_d = nothing
    p_row_d = nothing
    intensity_yzWx_d = nothing
    plan_fwd = nothing
    plan_bwd = nothing
    GC.gc(false)
    CUDA.reclaim()
    return raws, timing
end

"""
    _reconstruct_pam_cpu_3d(c_padded, rf, cfg, selected_freqs, selected_bins, crop_range_y, crop_range_z, padded_ny, padded_nz, rr, row_stop, c0, effective_axial_step, axial_substeps, corrected, t0, recon_label, show_progress)

Run the CPU fallback for a single 3D PAM reconstruction window.
"""
function _reconstruct_pam_cpu_3d(
    c_padded::AbstractArray{<:Real, 3},
    rf::AbstractArray{<:Real, 3},
    cfg::PAMConfig3D,
    selected_freqs::AbstractVector{<:Real},
    selected_bins::AbstractVector{<:Integer},
    crop_range_y::UnitRange{Int},
    crop_range_z::UnitRange{Int},
    padded_ny::Int,
    padded_nz::Int,
    rr::Int,
    row_stop::Int,
    c0::Float64,
    effective_axial_step::Float64,
    axial_substeps::Int,
    corrected::Bool,
    t0::Float64,
    recon_label::AbstractString,
    show_progress::Bool,
)
    t_setup_start = time()
    nx      = size(c_padded, 1)
    crop_ny = length(crop_range_y)
    crop_nz = length(crop_range_z)
    crop_y0 = first(crop_range_y)
    crop_z0 = first(crop_range_z)
    nfreq   = length(selected_freqs)

    # Sound-speed contrast field: (padded_ny, padded_nz, nx), matching GPU eta layout
    eta_spatial = permutedims(Float64.(1.0 .- (c0 ./ c_padded) .^ 2), (2, 3, 1))

    k_y    = _fft_wavenumbers(padded_ny, cfg.dy)
    k_z    = _fft_wavenumbers(padded_nz, cfg.dz)
    k_lat2 = reshape(k_y, :, 1) .^ 2 .+ reshape(k_z, 1, :) .^ 2
    k_radii = sqrt.(k_lat2)

    # Pre-plan FFTW once for the padded lateral plane; MEASURE finds the optimal kernel
    dummy     = zeros(ComplexF64, padded_ny, padded_nz)
    plan_fwd  = plan_fft(dummy,  (1, 2); flags=FFTW.MEASURE)
    plan_bwd  = plan_ifft(dummy, (1, 2); flags=FFTW.MEASURE)

    rf_fft    = fft(Float64.(rf), 3)  # time FFT: (ny, nz, nt)
    t_setup_s = time() - t_setup_start

    intensity = zeros(Float64, nx, crop_ny, crop_nz)

    println("[ PAM 3D ] $recon_label: CPU (single-threaded FFTW, MEASURE), $nfreq freq bins")
    flush(stdout)

    march_wall_start = time()

    for (freq, bin) in zip(selected_freqs, selected_bins)
        k0    = 2π * freq / c0
        k0_sq = k0^2

        k_axial    = sqrt.(complex.(k0^2 .- k_lat2, 0.0))
        propagating = real.(k_axial ./ k0) .> 0.0
        propagator  = exp.(1im .* k_axial .* effective_axial_step) .* propagating

        k_max_prop  = maximum(k_radii[propagating]; init=0.0)
        weighting   = _tukey_radial(k_radii, k_max_prop, cfg.tukey_ratio) .* propagating

        correction = zeros(ComplexF64, padded_ny, padded_nz)
        if corrected
            for j in eachindex(k_axial)
                propagating[j] || continue
                abs(k_axial[j]) > sqrt(eps(Float64)) || continue
                correction[j] = propagator[j] * effective_axial_step / (2im * k_axial[j])
            end
        end

        # Apply ifftshift to match GPU FFT convention
        prop_shifted   = _ifftshift_2d(propagating .* propagator)
        weight_shifted = _ifftshift_2d(weighting)
        prop_n_weight  = _ifftshift_2d(propagating .* propagator .^ axial_substeps .* weighting)
        prop_w         = corrected ? prop_shifted .* weight_shifted : nothing
        corr_w         = corrected ? _ifftshift_2d(propagating .* correction) .* weight_shifted : nothing

        # Initial condition: place RF frequency slice into padded array
        phase = cis(-2π * freq * t0)
        p0    = zeros(ComplexF64, padded_ny, padded_nz)
        p0[crop_range_y, crop_range_z] .= rf_fft[:, :, bin] .* phase
        current  = plan_fwd * p0
        current .*= weight_shifted

        for row in (rr + 1):row_stop
            if corrected
                eta_slice = @view eta_spatial[:, :, row]
                for _ in 1:(axial_substeps - 1)
                    p_spatial = plan_bwd * current
                    lp_fft    = plan_fwd * (k0_sq .* eta_slice .* p_spatial)
                    current   = current .* prop_w .+ corr_w .* lp_fft
                end
                p_spatial = plan_bwd * current
                lp_fft    = plan_fwd * (k0_sq .* eta_slice .* p_spatial)
                current   = current .* prop_w .+ corr_w .* lp_fft
            else
                current .*= prop_n_weight
            end
            p_row = plan_bwd * current
            for iz in 1:crop_nz, iy in 1:crop_ny
                v = p_row[crop_y0 + iy - 1, crop_z0 + iz - 1]
                intensity[row, iy, iz] += real(v)^2 + imag(v)^2
            end
        end
    end

    march_wall_s = time() - march_wall_start

    _pam_progress(
        show_progress,
        "PAM 3D $recon_label CPU: march $(round(march_wall_s; digits=1)) s, $nfreq freq bins",
    )

    timing = Dict{Symbol, Any}(
        :setup_s          => t_setup_s,
        :operator_setup_s => t_setup_s,
        :batch_setup_s    => 0.0,
        :march_wall_s     => march_wall_s,
        :march_cpu_s      => march_wall_s,
        :march_gpu_s      => nothing,
        :download_s       => 0.0,
        :bandwidth_GBps   => nothing,
        :fft_s            => nothing,
        :elementwise_s    => nothing,
        :nrows            => row_stop - rr,
        :nfreq            => nfreq,
        :nwindows         => 1,
        :padded_ny        => padded_ny,
        :padded_nz        => padded_nz,
        :axial_substeps   => axial_substeps,
        :bytes_march_est  => nothing,
    )
    return [intensity], timing
end

"""
    reconstruct_pam_3d(rf, c, cfg; kwargs...)

Reconstruct a 3D PAM intensity volume from receiver RF data and a sound-speed
volume.

`rf` has shape `(Ny, Nz, Nt)`, `c` is in m/s, frequencies are in Hz, and
`time_origin` is in seconds. Returns `(intensity, grid, info)`.
"""
function reconstruct_pam_3d(
    rf::AbstractArray{<:Real, 3},
    c::AbstractArray{<:Real, 3},
    cfg::PAMConfig3D;
    frequencies::Union{Nothing, AbstractVector{<:Real}}=nothing,
    bandwidth::Real=0.0,
    corrected::Bool=true,
    reference_sound_speed::Union{Nothing, Real}=nothing,
    axial_step::Union{Nothing, Real}=nothing,
    time_origin::Real=0.0,
    use_gpu::Bool=false,
    show_progress::Bool=false,
    benchmark::Bool=false,
    window_batch::Int=1,
)
    total_start = time()
    _ = window_batch  # single-window entry point; batching is in reconstruct_pam_windowed_3d
    nx, ny, nz = size(c)
    size(rf, 1) == ny && size(rf, 2) == nz ||
        error("RF data must have size (ny, nz, nt); expected ($ny, $nz, ·), got ($(size(rf,1)), $(size(rf,2)), ·).")
    rr = receiver_row(cfg)
    rr <= nx || error("Receiver row lies outside the computational grid.")

    padded_ny = cfg.zero_pad_factor > 1 ? cfg.zero_pad_factor * ny : ny
    padded_nz = cfg.zero_pad_factor > 1 ? cfg.zero_pad_factor * nz : nz

    selected_freqs, selected_bins = _select_frequency_bins_3d(rf, cfg.dt, frequencies; bandwidth=bandwidth)

    _, crop_range_y, crop_range_z = _zero_pad_receiver_rf_3d(rf, padded_ny, padded_nz)
    c_padded, _, _ = _edge_pad_lateral_3d(c, padded_ny, padded_nz)
    c0 = isnothing(reference_sound_speed) ? mean(c_padded) : Float64(reference_sound_speed)
    c0 > 0 || error("reference_sound_speed must be positive.")
    target_axial_step = isnothing(axial_step) ? cfg.dx : Float64(axial_step)
    axial_substeps    = _pam_axial_substeps(cfg.dx, target_axial_step)
    effective_axial_step = cfg.dx / axial_substeps
    row_stop = nx
    row_stop > rr || error("No valid reconstruction rows remain below the receiver row.")
    t0 = Float64(time_origin)
    recon_label = corrected ? "HASA" : "geometric ASA"

    _pam_progress(
        show_progress,
        "PAM 3D $recon_label: $(length(selected_freqs)) frequency bins, " *
        "grid=$(nx)×$(ny)×$(nz), padded=($(padded_ny)×$(padded_nz)), substeps=$axial_substeps",
    )

    raws, gpu_timing = if use_gpu
        setup = _pam_cuda_setup_3d(
            c_padded, cfg, selected_freqs, selected_bins,
            crop_range_y, crop_range_z,
            nx, padded_ny, padded_nz, rr, row_stop,
            c0, effective_axial_step, axial_substeps,
        )
        _reconstruct_pam_cuda_3d(
            setup, [rf], [t0], corrected, recon_label, show_progress, benchmark,
        )
    else
        _reconstruct_pam_cpu_3d(
            c_padded, rf, cfg,
            selected_freqs, selected_bins,
            crop_range_y, crop_range_z,
            padded_ny, padded_nz, rr, row_stop,
            c0, effective_axial_step, axial_substeps,
            corrected, t0, recon_label, show_progress,
        )
    end
    intensity = raws[1]  # (nx, ny, nz)
    _apply_axial_gain_3d!(intensity, cfg)

    grid = pam_grid_3d(cfg)
    info = Dict{Symbol, Any}(
        :frequencies => selected_freqs,
        :frequency_bins => selected_bins,
        :bandwidth => Float64(bandwidth),
        :corrected => corrected,
        :receiver_row => rr,
        :crop_range_y => crop_range_y,
        :crop_range_z => crop_range_z,
        :reference_sound_speed => c0,
        :axial_step => effective_axial_step,
        :axial_substeps_per_cell => axial_substeps,
        :time_origin => t0,
        :use_gpu => use_gpu,
        :backend => use_gpu ? :cuda : :cpu,
        :gpu_precision => use_gpu ? _PAM_CUDA_PRECISION : nothing,
        :axial_gain_power => cfg.axial_gain_power,
        :show_progress => show_progress,
        :benchmark => benchmark,
        :gpu_timing => gpu_timing,
        :grid => grid,
    )

    _pam_progress(
        show_progress,
        "PAM 3D $recon_label total elapsed $(_format_elapsed(time() - total_start))",
    )
    return intensity, grid, info
end

"""
    reconstruct_pam_windowed_3d(rf, c, cfg; kwargs...)

Run CUDA-backed 3D PAM reconstruction over temporal RF windows and average
qualifying window intensities.
"""
function reconstruct_pam_windowed_3d(
    rf::AbstractArray{<:Real, 3},
    c::AbstractArray{<:Real, 3},
    cfg::PAMConfig3D;
    frequencies::Union{Nothing, AbstractVector{<:Real}}=nothing,
    bandwidth::Real=0.0,
    corrected::Bool=true,
    reference_sound_speed::Union{Nothing, Real}=nothing,
    axial_step::Union{Nothing, Real}=nothing,
    window_config::PAMWindowConfig=PAMWindowConfig(enabled=true),
    use_gpu::Bool=false,
    show_progress::Bool=false,
    benchmark::Bool=false,
    window_batch::Int=1,
)
    use_gpu || error("CPU path not implemented for 3D PAM; use use_gpu=true.")
    total_start = time()
    nx, ny, nz = size(c)
    size(rf, 1) == ny && size(rf, 2) == nz ||
        error("RF data must have size (ny, nz, nt); expected ($ny, $nz, ·).")
    nt = size(rf, 3)
    config = _validate_window_config(window_config)
    _assert_pam_cuda_available()

    ranges, window_samples, hop_samples = _pam_window_ranges(nt, cfg.dt, config)
    energies = [sum(abs2, @view rf[:, :, range]) for range in ranges]
    max_energy = isempty(energies) ? 0.0 : maximum(energies)
    threshold  = max_energy * config.min_energy_ratio
    recon_label = corrected ? "HASA" : "geometric ASA"

    padded_ny = cfg.zero_pad_factor > 1 ? cfg.zero_pad_factor * ny : ny
    padded_nz = cfg.zero_pad_factor > 1 ? cfg.zero_pad_factor * nz : nz
    b_extra_y = padded_ny - ny;  b_left_y = fld(b_extra_y, 2)
    b_extra_z = padded_nz - nz;  b_left_z = fld(b_extra_z, 2)
    b_crop_y  = (b_left_y + 1):(b_left_y + ny)
    b_crop_z  = (b_left_z + 1):(b_left_z + nz)
    c_padded, _, _ = _edge_pad_lateral_3d(c, padded_ny, padded_nz)
    b_c0 = isnothing(reference_sound_speed) ? mean(c_padded) : Float64(reference_sound_speed)
    b_target_step = isnothing(axial_step) ? cfg.dx : Float64(axial_step)
    b_substeps    = _pam_axial_substeps(cfg.dx, b_target_step)
    b_eff_step    = cfg.dx / b_substeps
    b_rr          = receiver_row(cfg)

    intensity = zeros(Float64, nx, ny, nz)
    used_ranges = UnitRange{Int}[]
    skipped_ranges = UnitRange{Int}[]
    window_infos = Dict{Symbol, Any}[]
    used_energy = Float64[]
    skipped_energy = Float64[]

    qual_ranges   = UnitRange{Int}[]
    qual_energies = Float64[]
    for (range, energy) in zip(ranges, energies)
        if energy < threshold || energy <= 0
            push!(skipped_ranges, range)
            push!(skipped_energy, Float64(energy))
        else
            push!(qual_ranges, range)
            push!(qual_energies, Float64(energy))
        end
    end

    if isempty(qual_ranges)
        sel_freqs = Float64[]
        sel_bins = Int[]
        gpu_setup = nothing
    else
        sel_freqs, sel_bins = _select_frequency_bins_3d(
            @view(rf[:, :, first(qual_ranges)]),
            cfg.dt,
            frequencies;
            bandwidth=bandwidth,
        )
        gpu_setup = _pam_cuda_setup_3d(
            c_padded, cfg, sel_freqs, sel_bins,
            b_crop_y, b_crop_z,
            nx, padded_ny, padded_nz, b_rr, nx,
            b_c0, b_eff_step, b_substeps,
        )
    end

    n_qual    = length(qual_ranges)
    batch_idx = 0
    wi        = 1
    while wi <= n_qual
        batch_end = min(wi + window_batch - 1, n_qual)
        W_actual  = batch_end - wi + 1
        batch_idx += 1

        b_ranges   = qual_ranges[wi:batch_end]
        b_energies = qual_energies[wi:batch_end]
        b_t0s      = [Float64((first(r) - 1) * cfg.dt) for r in b_ranges]
        b_rf_wins  = [
            Float64.(@view rf[:, :, r]) .* reshape(_pam_temporal_taper(length(r), config.taper), 1, 1, :)
            for r in b_ranges
        ]

        batch_start_t = time()
        b_raws, b_timing = _reconstruct_pam_cuda_3d(
            gpu_setup, b_rf_wins, b_t0s, corrected, recon_label, show_progress, benchmark,
        )

        for (raw_w, rng, en) in zip(b_raws, b_ranges, b_energies)
            intensity .+= raw_w
            push!(used_ranges, rng)
            push!(used_energy, en)
        end
        b_raws = nothing
        GC.gc(false)
        CUDA.reclaim()

        batch_info = Dict{Symbol, Any}(
            :frequencies    => sel_freqs,
            :frequency_bins => sel_bins,
            :corrected      => corrected,
            :receiver_row   => b_rr,
            :axial_step     => b_eff_step,
            :backend        => :cuda,
            :gpu_precision  => _PAM_CUDA_PRECISION,
            :use_gpu        => true,
            :gpu_timing     => b_timing,
        )
        push!(window_infos, batch_info)

        _pam_progress(
            show_progress,
            "PAM 3D $recon_label batch $batch_idx ($W_actual windows), elapsed $(_format_elapsed(time() - batch_start_t))",
        )
        wi = batch_end + 1
    end

    used_count = length(used_ranges)
    if used_count > 0
        intensity ./= used_count
    end
    _apply_axial_gain_3d!(intensity, cfg)
    first_info = isempty(window_infos) ? Dict{Symbol, Any}() : first(window_infos)

    agg_gpu_timing = let timings = filter(!isnothing, [get(wi, :gpu_timing, nothing) for wi in window_infos])
        if isempty(timings)
            nothing
        else
            sum_bytes = sum(t[:bytes_march_est] for t in timings)
            sum_march = sum(t[:march_gpu_s] for t in timings)
            operator_setup_s = get(first(timings), :operator_setup_s, 0.0)
            batch_setup_s = if all(haskey(t, :batch_setup_s) for t in timings)
                sum(t[:batch_setup_s] for t in timings)
            else
                sum(t[:setup_s] for t in timings)
            end
            Dict{Symbol, Any}(
                :setup_s         => operator_setup_s + batch_setup_s,
                :operator_setup_s => operator_setup_s,
                :batch_setup_s   => batch_setup_s,
                :march_gpu_s     => sum_march,
                :march_wall_s    => sum(t[:march_wall_s]  for t in timings),
                :download_s      => sum(t[:download_s]    for t in timings),
                :bandwidth_GBps  => sum_march > 0 ? sum_bytes / sum_march / 1e9 : 0.0,
                :fft_s           => all(!isnothing(t[:fft_s]) for t in timings) ? sum(t[:fft_s] for t in timings) : nothing,
                :elementwise_s   => all(!isnothing(t[:elementwise_s]) for t in timings) ? sum(t[:elementwise_s] for t in timings) : nothing,
                :nrows           => sum(t[:nrows]         for t in timings),
                :nfreq           => first(timings)[:nfreq],
                :padded_ny       => first(timings)[:padded_ny],
                :padded_nz       => first(timings)[:padded_nz],
                :axial_substeps  => first(timings)[:axial_substeps],
                :bytes_march_est => sum_bytes,
                :window_count    => length(timings),
            )
        end
    end

    grid = pam_grid_3d(cfg)
    info = Dict{Symbol, Any}(
        :frequencies    => get(first_info, :frequencies, Float64[]),
        :frequency_bins => get(first_info, :frequency_bins, Int[]),
        :bandwidth      => Float64(bandwidth),
        :corrected      => corrected,
        :receiver_row   => receiver_row(cfg),
        :reference_sound_speed => b_c0,
        :axial_step     => b_eff_step,
        :window_config  => _window_config_info(config),
        :use_gpu        => use_gpu,
        :backend        => :cuda,
        :gpu_precision  => _PAM_CUDA_PRECISION,
        :axial_gain_power => cfg.axial_gain_power,
        :window_samples => window_samples,
        :hop_samples    => hop_samples,
        :total_window_count   => length(ranges),
        :used_window_count    => used_count,
        :skipped_window_count => length(skipped_ranges),
        :energy_threshold     => threshold,
        :used_window_ranges   => used_ranges,
        :skipped_window_ranges => skipped_ranges,
        :gpu_timing     => agg_gpu_timing,
        :grid           => grid,
    )
    _pam_progress(
        show_progress,
        "PAM 3D $recon_label windowed complete: used=$used_count, skipped=$(length(skipped_ranges)), " *
        "total elapsed $(_format_elapsed(time() - total_start))",
    )
    return intensity, grid, info
end
