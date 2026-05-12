"""
    _zero_pad_receiver_rf(rf, target_ny)

Center-pad RF receiver data laterally to `target_ny` rows and return the crop range.
"""
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

"""
    _edge_pad_lateral(a, target_ny)

Pad a 2D field laterally by repeating edge columns and return the crop range.
"""
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

"""
    _format_elapsed(seconds)

Format an elapsed duration in seconds for progress output.
"""
function _format_elapsed(seconds::Real)
    s = Float64(seconds)
    if s < 1e-3
        return "$(round(s * 1e6; digits=1)) us"
    elseif s < 1
        return "$(round(s * 1e3; digits=1)) ms"
    elseif s < 60
        return "$(round(s; digits=2)) s"
    end
    minutes = floor(Int, s / 60)
    rem_s = s - 60 * minutes
    return "$(minutes)m $(round(rem_s; digits=1))s"
end

"""
    _format_frequency_mhz(freq)

Format a frequency in Hz as a MHz label.
"""
function _format_frequency_mhz(freq::Real)
    return "$(round(Float64(freq) / 1e6; digits=4)) MHz"
end

"""
    _format_frequency_list(freqs; max_items=8)

Format a frequency list in Hz for concise progress output.
"""
function _format_frequency_list(freqs::AbstractVector{<:Real}; max_items::Int=8)
    isempty(freqs) && return "none"
    labels = [_format_frequency_mhz(freq) for freq in freqs]
    length(labels) <= max_items && return join(labels, ", ")
    head_count = max(1, max_items - 1)
    return join(vcat(labels[1:head_count], ["...", labels[end]]), ", ")
end

"""
    _pam_progress(show, msg)

Print a PAM progress message to stderr when `show` is true.
"""
function _pam_progress(show::Bool, msg::AbstractString)
    show || return nothing
    println(stderr, msg)
    flush(stderr)
    return nothing
end

"""
    _fft_wavenumbers(n, spacing)

Return centered FFT wavenumbers for `n` samples with physical spacing in meters.
"""
function _fft_wavenumbers(n::Int, spacing::Real)
    dk = 2π / Float64(spacing)
    start_val = -fld(n, 2)
    end_val = ceil(Int, n / 2) - 1
    return collect(start_val:end_val) .* dk ./ n
end

"""
    _select_frequency_bins(rf, dt, frequencies; bandwidth=0.0)

Resolve reconstruction frequencies in Hz and FFT bin indices for RF data.
"""
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

const _PAM_CUDA_PRECISION = Float32
const _PAM_CUDA_COMPLEX = ComplexF32

"""
    _pam_cuda_functional()

Return whether CUDA.jl currently reports a functional CUDA backend for PAM.
"""
function _pam_cuda_functional()
    try
        return CUDA.functional()
    catch
        return false
    end
end

"""
    _assert_pam_cuda_available()

Throw a user-facing error unless CUDA is functional for PAM reconstruction.
"""
function _assert_pam_cuda_available()
    _pam_cuda_functional() && return nothing
    error(
        "PAM CUDA reconstruction requested with use_gpu=true, but CUDA.jl " *
        "does not see a functional NVIDIA CUDA GPU. Configure CUDA.jl on " *
        "a machine with an NVIDIA GPU, or run with use_gpu=false.",
    )
end

"""
    _accum_abs2_sum!(dst, src)

CUDA kernel that accumulates `abs2` over frequency columns into one lateral row.
"""
function _accum_abs2_sum!(dst, src)
    i = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x
    i > size(src, 1) && return
    acc = zero(real(eltype(src)))
    @inbounds for j in 1:size(src, 2)
        v = src[i, j]
        acc += real(v)^2 + imag(v)^2
    end
    @inbounds dst[i] += acc
    return
end

"""
    _accum_abs2_sum_batched!(dst, src, nfreq_per_w)

CUDA kernel that accumulates batched `abs2` sums from `(padded_ny, nfreq * W)`
frequency data into `(padded_ny, W)` intensity rows.
"""
function _accum_abs2_sum_batched!(dst, src, nfreq_per_w)
    i = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x
    i > size(src, 1) && return
    W = size(dst, 2)
    @inbounds for w in 1:W
        acc = zero(real(eltype(src)))
        base = (w - 1) * nfreq_per_w
        for j in 1:nfreq_per_w
            v = src[i, base + j]
            acc += real(v)^2 + imag(v)^2
        end
        dst[i, w] += acc
    end
    return
end

"""
    PAMCUDASetup

Precomputed 2D CUDA arrays and reconstruction metadata shared across one or
more PAM GPU window batches.
"""
struct PAMCUDASetup
    eta_yx_d          # (padded_ny, nx) Float32 — speed-contrast field
    prop_d1           # (padded_ny, nfreq) ComplexF32 — propagator, un-tiled
    corr_d1           # (padded_ny, nfreq) ComplexF32 — correction term, un-tiled
    weight_d1         # (padded_ny, nfreq) Float32    — Tukey taper, un-tiled
    prop_n_weight_d1  # (padded_ny, nfreq) ComplexF32 — prop^n*weight, un-tiled
    k0_sq_d1          # (1, nfreq) Float32            — k0² per frequency
    selected_freqs::Vector{Float64}
    selected_bins::Vector{Int}
    crop_range::UnitRange{Int}
    nfreq::Int
    padded_ny::Int
    nx::Int
    rr::Int
    row_stop::Int
    axial_substeps::Int
    setup_s::Float64
end

"""
    _pam_cuda_setup(c_padded, cfg, selected_freqs, selected_bins, crop_range, nx, padded_ny, rr, row_stop, c0, effective_axial_step, axial_substeps)

Precompute 2D CUDA propagation operators for the selected frequencies.
"""
function _pam_cuda_setup(
    c_padded::AbstractMatrix{<:Real},
    cfg::PAMConfig,
    selected_freqs::AbstractVector{<:Real},
    selected_bins::AbstractVector{<:Integer},
    crop_range::UnitRange{Int},
    nx::Int,
    padded_ny::Int,
    rr::Int,
    row_stop::Int,
    c0::Float64,
    effective_axial_step::Float64,
    axial_substeps::Int,
)
    _assert_pam_cuda_available()
    t0 = time()
    T  = _PAM_CUDA_PRECISION
    CT = _PAM_CUDA_COMPLEX
    nfreq = length(selected_freqs)

    eta_yx_d = CUDA.CuArray(T.(permutedims(1 .- (c0 ./ c_padded) .^ 2)))

    k = _fft_wavenumbers(padded_ny, cfg.dz)
    prop_cpu          = zeros(ComplexF64, padded_ny, nfreq)
    corr_cpu          = zeros(ComplexF64, padded_ny, nfreq)
    weight_cpu        = zeros(Float64,    padded_ny, nfreq)
    prop_n_weight_cpu = zeros(ComplexF64, padded_ny, nfreq)
    k0_sq_cpu         = zeros(Float64, nfreq)

    for (f, freq) in enumerate(selected_freqs)
        k0 = 2 * pi * Float64(freq) / c0
        kz = sqrt.(complex.(k0^2 .- k .^ 2, 0.0))
        propagator = exp.(1im .* kz .* effective_axial_step)

        real_inds = findall(real.(kz ./ k0) .> 0.0)
        propagating = falses(padded_ny)
        propagating[real_inds] .= true
        weighting = zeros(Float64, padded_ny)
        weighting[real_inds] .= _tukey_window(length(real_inds), cfg.tukey_ratio)

        correction = zeros(ComplexF64, padded_ny)
        for idx in real_inds
            abs(kz[idx]) > sqrt(eps(Float64)) || continue
            correction[idx] = propagator[idx] * effective_axial_step / (2im * kz[idx])
        end

        prop_cpu[:, f]          .= _ifftshift(propagating .* propagator)
        corr_cpu[:, f]          .= _ifftshift(propagating .* correction)
        weight_cpu[:, f]        .= _ifftshift(weighting)
        prop_n_weight_cpu[:, f] .= _ifftshift(propagating .* propagator .^ axial_substeps .* weighting)
        k0_sq_cpu[f]             = k0^2
    end

    prop_d1          = CUDA.CuArray(CT.(prop_cpu))
    corr_d1          = CUDA.CuArray(CT.(corr_cpu))
    weight_d1        = CUDA.CuArray(T.(weight_cpu))
    prop_n_weight_d1 = CUDA.CuArray(CT.(prop_n_weight_cpu))
    k0_sq_d1         = reshape(CUDA.CuArray(T.(k0_sq_cpu)), 1, nfreq)

    return PAMCUDASetup(
        eta_yx_d, prop_d1, corr_d1, weight_d1, prop_n_weight_d1, k0_sq_d1,
        Vector{Float64}(selected_freqs), Vector{Int}(selected_bins),
        crop_range, nfreq, padded_ny, nx, rr, row_stop, axial_substeps,
        time() - t0,
    )
end

"""
    _reconstruct_pam_cuda(setup, rf_windows, t0_windows, corrected, recon_label, show_progress, benchmark)

Run one 2D CUDA reconstruction batch and return raw intensity maps plus timing
metadata.
"""
function _reconstruct_pam_cuda(
    setup::PAMCUDASetup,
    rf_windows::AbstractVector{<:AbstractMatrix{<:Real}},
    t0_windows::AbstractVector{<:Real},
    corrected::Bool,
    recon_label::AbstractString,
    show_progress::Bool,
    benchmark::Bool,
)
    fn_start = time()
    W       = length(rf_windows)
    nfreq   = setup.nfreq
    nfreq_W = nfreq * W
    padded_ny     = setup.padded_ny
    nx            = setup.nx
    rr            = setup.rr
    row_stop      = setup.row_stop
    axial_substeps = setup.axial_substeps
    crop_range    = setup.crop_range

    T  = _PAM_CUDA_PRECISION
    CT = _PAM_CUDA_COMPLEX

    let dev = CUDA.device()
        println(
            "[ PAM ] $recon_label: GPU $(CUDA.name(dev)) (device $(CUDA.deviceid(dev))), " *
            "$(_PAM_CUDA_PRECISION) arithmetic, $nfreq freq bins × $W windows batched",
        )
        flush(stdout)
    end

    # Tile the un-tiled base operators W times along the frequency dimension.
    prop_d          = repeat(setup.prop_d1,          1, W)
    corr_d          = repeat(setup.corr_d1,          1, W)
    weight_d        = repeat(setup.weight_d1,        1, W)
    prop_weight_d   = prop_d .* weight_d
    corr_weight_d   = corr_d .* weight_d
    prop_n_weight_d = repeat(setup.prop_n_weight_d1, 1, W)
    k0_sq_d         = reshape(repeat(setup.k0_sq_d1, 1, W), 1, nfreq_W)

    # Intensity accumulator: (padded_ny, W, nx) — view(:, :, row) is contiguous.
    intensity_yWx_d = CUDA.zeros(T, padded_ny, W, nx)

    # Build batched initial conditions: (padded_ny, nfreq*W).
    p0_d = CUDA.zeros(CT, padded_ny, nfreq_W)
    for (w, (rf_w, t0_w)) in enumerate(zip(rf_windows, t0_windows))
        rf_fft_w   = fft(CUDA.CuArray(T.(rf_w)), 2)
        col_offset = (w - 1) * nfreq
        for (f, (freq, bin)) in enumerate(zip(setup.selected_freqs, setup.selected_bins))
            phase = CT(cis(-T(2 * pi) * T(freq) * T(t0_w)))
            p0_d[crop_range, col_offset + f] .= rf_fft_w[:, bin] .* phase
        end
    end

    plan_fwd = plan_fft!(p0_d, 1)
    plan_bwd = plan_ifft!(similar(p0_d), 1)
    plan_fwd * p0_d
    current_d = p0_d
    current_d .*= weight_d
    next_d    = similar(current_d)
    tmp_d     = similar(current_d)
    p_space_d = similar(current_d)
    p_row_d   = similar(current_d)

    t_batch_setup = time() - fn_start
    t_fft_s = 0.0
    t_ew_s  = 0.0
    march_wall_start = time()

    nthreads = min(padded_ny, 512)
    nblocks  = cld(padded_ny, nthreads)

    eta_yx_d = setup.eta_yx_d

    if benchmark
        for row in (rr + 1):row_stop
            if corrected
                for _ in 1:(axial_substeps - 1)
                    t_fft_s += CUDA.@elapsed (p_space_d .= current_d; plan_bwd * p_space_d)
                    t_ew_s  += CUDA.@elapsed (tmp_d .= k0_sq_d .* eta_yx_d[:, row:row] .* p_space_d)
                    t_fft_s += CUDA.@elapsed (plan_fwd * tmp_d)
                    t_ew_s  += CUDA.@elapsed (next_d .= current_d .* prop_d .+ corr_d .* tmp_d)
                    current_d, next_d = next_d, current_d
                end
                t_fft_s += CUDA.@elapsed (p_space_d .= current_d; plan_bwd * p_space_d)
                t_ew_s  += CUDA.@elapsed (tmp_d .= k0_sq_d .* eta_yx_d[:, row:row] .* p_space_d)
                t_fft_s += CUDA.@elapsed (plan_fwd * tmp_d)
                t_ew_s  += CUDA.@elapsed (current_d .= current_d .* prop_weight_d .+ corr_weight_d .* tmp_d)
            else
                t_ew_s  += CUDA.@elapsed (current_d .*= prop_n_weight_d)
            end
            t_ew_s  += CUDA.@elapsed (p_row_d .= current_d)
            t_fft_s += CUDA.@elapsed (plan_bwd * p_row_d)
            t_ew_s  += CUDA.@elapsed CUDA.@cuda threads=nthreads blocks=nblocks _accum_abs2_sum_batched!(
                view(intensity_yWx_d, :, :, row), p_row_d, nfreq,
            )
        end
        march_gpu_s = t_fft_s + t_ew_s
    else
        march_gpu_s = CUDA.@elapsed for row in (rr + 1):row_stop
            if corrected
                for _ in 1:(axial_substeps - 1)
                    p_space_d .= current_d
                    plan_bwd * p_space_d
                    tmp_d .= k0_sq_d .* eta_yx_d[:, row:row] .* p_space_d
                    plan_fwd * tmp_d
                    next_d .= current_d .* prop_d .+ corr_d .* tmp_d
                    current_d, next_d = next_d, current_d
                end
                p_space_d .= current_d
                plan_bwd * p_space_d
                tmp_d .= k0_sq_d .* eta_yx_d[:, row:row] .* p_space_d
                plan_fwd * tmp_d
                current_d .= current_d .* prop_weight_d .+ corr_weight_d .* tmp_d
            else
                current_d .*= prop_n_weight_d
            end
            p_row_d .= current_d
            plan_bwd * p_row_d
            CUDA.@cuda threads=nthreads blocks=nblocks _accum_abs2_sum_batched!(
                view(intensity_yWx_d, :, :, row), p_row_d, nfreq,
            )
        end
    end
    march_wall_s = time() - march_wall_start

    passes_substep  = 13
    passes_last_row = corrected ? 19 : 9
    bytes_march = Int64(row_stop - rr) * (
        (corrected ? Int64(axial_substeps - 1) * passes_substep : 0) * padded_ny * nfreq_W * sizeof(CT) +
        passes_last_row * padded_ny * nfreq_W * sizeof(CT)
    )
    bandwidth_GBps = bytes_march / march_gpu_s / 1e9

    t_download = @elapsed begin
        raw_all = permutedims(Array(intensity_yWx_d), (1, 3, 2))  # (padded_ny,W,nx)→(padded_ny,nx,W)
    end
    raws = [Float64.(permutedims(raw_all[:, :, w])) for w in 1:W]

    _pam_progress(
        show_progress,
        "PAM $recon_label $nfreq freq × $W win batch: march $(round(march_gpu_s * 1e3; digits=1)) ms GPU / " *
        "$(round(march_wall_s * 1e3; digits=1)) ms wall, " *
        "BW ~$(round(bandwidth_GBps; digits=0)) GB/s" *
        (benchmark ? ", FFT $(round(100 * t_fft_s / march_gpu_s; digits=1))% / EW $(round(100 * t_ew_s / march_gpu_s; digits=1))%" : ""),
    )

    timing = Dict{Symbol, Any}(
        :setup_s         => setup.setup_s + t_batch_setup,
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
        :axial_substeps  => axial_substeps,
        :bytes_march_est => bytes_march,
    )
    return raws, timing
end

"""
    reconstruct_pam(rf, c, cfg; kwargs...)

Reconstruct a 2D PAM intensity image from receiver RF data and a sound-speed map.

`rf` has shape `(Ny, Nt)`, `c` is in m/s, frequencies are in Hz, and
`time_origin` is in seconds. Returns `(intensity, kgrid, info)`.
"""
function reconstruct_pam(
    rf::AbstractMatrix{<:Real},
    c::AbstractMatrix{<:Real},
    cfg::PAMConfig;
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
    nx, ny = size(c)
    size(rf, 1) == ny || error("RF data must have size (Ny, Nt); expected Ny=$ny, got $(size(rf, 1)).")
    nt = size(rf, 2)
    kgrid = KGrid2D(nx, ny, cfg.dx, cfg.dz; dt=cfg.dt, Nt=nt)
    rr = receiver_row(cfg)
    rr <= nx || error("Receiver row lies outside the computational grid.")

    selected_freqs, selected_bins = _select_frequency_bins(rf, cfg.dt, frequencies; bandwidth=bandwidth)
    padded_ny = cfg.zero_pad_factor > 1 ? cfg.zero_pad_factor * ny : ny
    _, crop_range = _zero_pad_receiver_rf(rf, padded_ny)
    c_padded, _ = _edge_pad_lateral(c, padded_ny)
    c0 = isnothing(reference_sound_speed) ? mean(c_padded) : Float64(reference_sound_speed)
    c0 > 0 || error("reference_sound_speed must be positive.")
    target_axial_step = isnothing(axial_step) ? cfg.dx : Float64(axial_step)
    0 < target_axial_step <= cfg.dx || error("axial_step must lie in (0, cfg.dx].")
    axial_substeps = _pam_axial_substeps(cfg.dx, target_axial_step)
    effective_axial_step = cfg.dx / axial_substeps
    row_stop = nx
    row_stop > rr || error("No valid reconstruction rows remain below the receiver row.")
    t0 = Float64(time_origin)
    recon_label = corrected ? "HASA" : "geometric ASA"

    _pam_progress(
        show_progress,
        "PAM $recon_label reconstruction: $(length(selected_freqs)) frequency bins ($(_format_frequency_list(selected_freqs))), " *
        "grid=$(nx)x$(ny), padded_ny=$padded_ny, axial_substeps=$axial_substeps",
    )

    gpu_timing = nothing
    intensity_padded = if use_gpu
        setup = _pam_cuda_setup(
            c_padded, cfg, selected_freqs, selected_bins,
            crop_range, nx, padded_ny, rr, row_stop,
            c0, effective_axial_step, axial_substeps,
        )
        raws, timing = _reconstruct_pam_cuda(
            setup, [rf], [t0], corrected, recon_label, show_progress, benchmark,
        )
        gpu_timing = timing
        raws[1]
    else
        rf_fft = fft(Float64.(rf), 2)
        out = zeros(Float64, nx, padded_ny)

        for (freq_idx, (freq, bin)) in enumerate(zip(selected_freqs, selected_bins))
            freq_start = time()
            p0 = rf_fft[:, bin] .* exp(-1im * 2π * freq * t0)
            p0_padded, _ = _zero_pad_receiver_rf(reshape(p0, ny, 1), padded_ny)
            p0_vec = vec(p0_padded[:, 1])

            k0 = 2π * freq / c0
            k = _fft_wavenumbers(padded_ny, cfg.dz)
            kz = sqrt.(complex.(k0^2 .- k .^ 2, 0.0))
            propagator = exp.(1im .* kz .* effective_axial_step)

            real_inds = findall(real.(kz ./ k0) .> 0.0)
            propagating = falses(padded_ny)
            propagating[real_inds] .= true
            weighting = zeros(Float64, padded_ny)
            weighting[real_inds] .= _tukey_window(length(real_inds), cfg.tukey_ratio)

            mu = (c0 ./ c_padded) .^ 2
            lambda = (k0^2) .* (1 .- mu)

            correction = zeros(ComplexF64, padded_ny)
            for idx in real_inds
                abs(kz[idx]) > sqrt(eps(Float64)) || continue
                correction[idx] = propagator[idx] * effective_axial_step / (2im * kz[idx])
            end

            # Permute centered-order arrays to FFT order once per frequency bin so the
            # propagation loop needs no per-step fftshift/ifftshift operations.
            propagator = _ifftshift(propagator)
            weighting = _ifftshift(weighting)
            correction = _ifftshift(correction)
            evanescent_inds = findall(_ifftshift(.!propagating))

            current = fft(p0_vec)
            current .*= weighting

            for row in (rr + 1):row_stop
                for _ in 1:axial_substeps
                    if corrected
                        p_space = ifft(current)
                        conv_term = fft(lambda[row, :] .* p_space)
                        next = current .* propagator
                        next .+= correction .* conv_term
                    else
                        next = current .* propagator
                    end
                    next[evanescent_inds] .= 0.0
                    current = next
                end
                # Taper once per reconstruction row to suppress long-range numerical
                # growth without making damping depend on substep count.
                current .*= weighting

                p_row = ifft(current)
                out[row, :] .+= abs2.(p_row)
            end

            _pam_progress(
                show_progress,
                "PAM $recon_label frequency $freq_idx/$(length(selected_freqs)) " *
                "($(_format_frequency_mhz(freq)), bin $bin) elapsed $(_format_elapsed(time() - freq_start))",
            )
        end
        out
    end

    _pam_progress(
        show_progress,
        "PAM $recon_label reconstruction total elapsed $(_format_elapsed(time() - total_start))",
    )

    intensity = intensity_padded[:, crop_range]
    info = Dict{Symbol, Any}(
        :frequencies => selected_freqs,
        :frequency_bins => selected_bins,
        :bandwidth => Float64(bandwidth),
        :corrected => corrected,
        :receiver_row => rr,
        :crop_range => crop_range,
        :reference_sound_speed => c0,
        :axial_step => effective_axial_step,
        :axial_substeps_per_cell => axial_substeps,
        :time_origin => t0,
        :use_gpu => use_gpu,
        :backend => use_gpu ? :cuda : :cpu,
        :gpu_precision => use_gpu ? _PAM_CUDA_PRECISION : nothing,
        :show_progress => show_progress,
        :benchmark => benchmark,
        :gpu_timing => gpu_timing,
    )
    return intensity, kgrid, info
end

"""
    reconstruct_pam_windowed(rf, c, cfg; kwargs...)

Run 2D PAM reconstruction over temporal RF windows and average qualifying
window intensities.
"""
function reconstruct_pam_windowed(
    rf::AbstractMatrix{<:Real},
    c::AbstractMatrix{<:Real},
    cfg::PAMConfig;
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
    total_start = time()
    nx, ny = size(c)
    size(rf, 1) == ny || error("RF data must have size (Ny, Nt); expected Ny=$ny, got $(size(rf, 1)).")
    nt = size(rf, 2)
    kgrid = KGrid2D(nx, ny, cfg.dx, cfg.dz; dt=cfg.dt, Nt=nt)
    config = _validate_window_config(window_config)
    if use_gpu
        _assert_pam_cuda_available()
    end

    ranges, window_samples, hop_samples = _pam_window_ranges(nt, cfg.dt, config)
    energies = [sum(abs2, @view rf[:, range]) for range in ranges]
    max_energy = isempty(energies) ? 0.0 : maximum(energies)
    threshold = max_energy * config.min_energy_ratio
    recon_label = corrected ? "HASA" : "geometric ASA"

    _pam_progress(
        show_progress,
        "PAM $recon_label windowed reconstruction: $(length(ranges)) windows, " *
        "window_samples=$window_samples, hop_samples=$hop_samples, energy_threshold=$threshold",
    )

    intensity = zeros(Float64, nx, ny)
    used_ranges = UnitRange{Int}[]
    skipped_ranges = UnitRange{Int}[]
    window_infos = Dict{Symbol, Any}[]
    used_energy = Float64[]
    skipped_energy = Float64[]

    if use_gpu && window_batch > 1
        # Bin indices depend on the FFT length (window_samples), not the full RF length.
        ref_win_rf = zeros(Float64, ny, window_samples)
        sel_freqs, sel_bins = _select_frequency_bins(ref_win_rf, cfg.dt, frequencies; bandwidth=bandwidth)
        b_padded_ny = cfg.zero_pad_factor > 1 ? cfg.zero_pad_factor * ny : ny
        b_extra = b_padded_ny - ny
        b_left  = fld(b_extra, 2)
        b_crop  = (b_left + 1):(b_left + ny)
        b_c_padded, _ = _edge_pad_lateral(c, b_padded_ny)
        b_c0 = isnothing(reference_sound_speed) ? mean(b_c_padded) : Float64(reference_sound_speed)
        b_target_step = isnothing(axial_step) ? cfg.dx : Float64(axial_step)
        b_substeps    = _pam_axial_substeps(cfg.dx, b_target_step)
        b_eff_step    = cfg.dx / b_substeps
        b_rr          = receiver_row(cfg)

        # Build GPU setup once — eta_yx_d + spectral operators shared across all batches.
        gpu_setup = _pam_cuda_setup(
            b_c_padded, cfg, sel_freqs, sel_bins,
            b_crop, nx, b_padded_ny, b_rr, nx,
            b_c0, b_eff_step, b_substeps,
        )

        # Partition windows into qualifying (batched) and skipped.
        qual_ranges  = UnitRange{Int}[]
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

        n_qual = length(qual_ranges)
        batch_idx = 0
        wi = 1
        while wi <= n_qual
            batch_end = min(wi + window_batch - 1, n_qual)
            W_actual  = batch_end - wi + 1
            batch_idx += 1

            b_ranges   = qual_ranges[wi:batch_end]
            b_energies = qual_energies[wi:batch_end]
            b_t0s      = [Float64((first(r) - 1) * cfg.dt) for r in b_ranges]
            b_rf_wins  = [
                Float64.(@view rf[:, r]) .* reshape(_pam_temporal_taper(length(r), config.taper), 1, :)
                for r in b_ranges
            ]

            batch_start_t = time()
            b_raws, b_timing = _reconstruct_pam_cuda(
                gpu_setup, b_rf_wins, b_t0s, corrected, recon_label, show_progress, benchmark,
            )

            for (raw_w, rng, en) in zip(b_raws, b_ranges, b_energies)
                intensity .+= raw_w[:, b_crop]
                push!(used_ranges, rng)
                push!(used_energy, en)
            end

            batch_info = Dict{Symbol, Any}(
                :frequencies    => sel_freqs,
                :frequency_bins => sel_bins,
                :bandwidth      => Float64(bandwidth),
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
                "PAM $recon_label batch $batch_idx ($W_actual windows), " *
                "samples=$(first(b_ranges[1])):$(last(b_ranges[end])), " *
                "elapsed $(_format_elapsed(time() - batch_start_t))",
            )
            wi = batch_end + 1
        end
    else
        for (window_idx, (range, energy)) in enumerate(zip(ranges, energies))
            if energy < threshold || energy <= 0
                push!(skipped_ranges, range)
                push!(skipped_energy, Float64(energy))
                _pam_progress(
                    show_progress,
                    "PAM $recon_label window $window_idx/$(length(ranges)) skipped, " *
                    "samples=$(first(range)):$(last(range)), energy=$energy",
                )
                continue
            end

            window_start = time()
            taper = _pam_temporal_taper(length(range), config.taper)
            rf_window = Float64.(@view rf[:, range]) .* reshape(taper, 1, :)
            window_intensity, _, window_info = reconstruct_pam(
                rf_window,
                c,
                cfg;
                frequencies=frequencies,
                bandwidth=bandwidth,
                corrected=corrected,
                reference_sound_speed=reference_sound_speed,
                axial_step=axial_step,
                time_origin=(first(range) - 1) * cfg.dt,
                use_gpu=use_gpu,
                show_progress=false,
                benchmark=benchmark,
            )
            intensity .+= window_intensity
            push!(used_ranges, range)
            push!(used_energy, Float64(energy))
            push!(window_infos, window_info)
            _pam_progress(
                show_progress,
                "PAM $recon_label window $window_idx/$(length(ranges)), " *
                "$(length(window_info[:frequency_bins])) bins, samples=$(first(range)):$(last(range)), " *
                "elapsed $(_format_elapsed(time() - window_start))",
            )
        end
    end

    used_count = length(used_ranges)
    if used_count > 0
        intensity ./= used_count
    end
    first_info = isempty(window_infos) ? Dict{Symbol, Any}() : first(window_infos)

    # Aggregate gpu_timing across all used windows.
    agg_gpu_timing = let timings = filter(!isnothing, [get(wi, :gpu_timing, nothing) for wi in window_infos])
        if isempty(timings)
            nothing
        else
            sum_bytes = sum(t[:bytes_march_est] for t in timings)
            sum_march = sum(t[:march_gpu_s] for t in timings)
            Dict{Symbol, Any}(
                :setup_s         => sum(t[:setup_s]       for t in timings),
                :march_gpu_s     => sum_march,
                :march_wall_s    => sum(t[:march_wall_s]   for t in timings),
                :download_s      => sum(t[:download_s]     for t in timings),
                :bandwidth_GBps  => sum_march > 0 ? sum_bytes / sum_march / 1e9 : 0.0,
                :fft_s           => all(!isnothing(t[:fft_s]) for t in timings) ? sum(t[:fft_s] for t in timings) : nothing,
                :elementwise_s   => all(!isnothing(t[:elementwise_s]) for t in timings) ? sum(t[:elementwise_s] for t in timings) : nothing,
                :nrows           => sum(t[:nrows]          for t in timings),
                :nfreq           => first(timings)[:nfreq],
                :padded_ny       => first(timings)[:padded_ny],
                :axial_substeps  => first(timings)[:axial_substeps],
                :bytes_march_est => sum_bytes,
                :window_count    => length(timings),
            )
        end
    end

    info = Dict{Symbol, Any}(
        :frequencies => get(first_info, :frequencies, Float64[]),
        :frequency_bins => get(first_info, :frequency_bins, Int[]),
        :bandwidth => Float64(bandwidth),
        :corrected => corrected,
        :receiver_row => receiver_row(cfg),
        :reference_sound_speed => isnothing(reference_sound_speed) ? mean(Float64.(c)) : Float64(reference_sound_speed),
        :axial_step => get(first_info, :axial_step, isnothing(axial_step) ? cfg.dx : Float64(axial_step)),
        :window_config => _window_config_info(config),
        :use_gpu => use_gpu,
        :backend => get(first_info, :backend, use_gpu ? :cuda : :cpu),
        :gpu_precision => get(first_info, :gpu_precision, use_gpu ? _PAM_CUDA_PRECISION : nothing),
        :show_progress => show_progress,
        :window_samples => window_samples,
        :hop_samples => hop_samples,
        :effective_window_duration_s => window_samples * cfg.dt,
        :effective_hop_s => hop_samples * cfg.dt,
        :total_window_count => length(ranges),
        :used_window_count => used_count,
        :skipped_window_count => length(skipped_ranges),
        :energy_threshold => threshold,
        :window_energies => energies,
        :used_window_energies => used_energy,
        :skipped_window_energies => skipped_energy,
        :used_window_ranges => used_ranges,
        :skipped_window_ranges => skipped_ranges,
        :window_infos => window_infos,
        :accumulation => :intensity,
        :gpu_timing => agg_gpu_timing,
    )
    _pam_progress(
        show_progress,
        "PAM $recon_label windowed reconstruction complete: used=$used_count, " *
        "skipped=$(length(skipped_ranges)), total elapsed $(_format_elapsed(time() - total_start))",
    )
    return intensity, kgrid, info
end
