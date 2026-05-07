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

function _ifftshift_2d!(a::AbstractMatrix)
    ny, nz = size(a)
    sy = -fld(ny, 2)
    sz = -fld(nz, 2)
    circshift!(similar(a), a, (sy, sz))
end

function _ifftshift_2d(a::AbstractMatrix)
    ny, nz = size(a)
    return circshift(a, (-fld(ny, 2), -fld(nz, 2)))
end

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

function _select_frequency_bins_3d(
    rf::AbstractArray{<:Real, 3},
    dt::Real,
    frequencies;
    bandwidth::Real=0.0,
)
    nt = size(rf, 3)
    return _select_frequency_bins(reshape(rf, :, nt), dt, frequencies; bandwidth=bandwidth)
end

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

# Each thread handles one (iy, iz) lateral index.
# intensity: (padded_ny, padded_nz, W, nx); src: (padded_ny, padded_nz, nfreq*W)
# row: axial row index (1-based) into intensity
function _accum_abs2_sum_batched_3d!(intensity, src, nfreq_per_w, row)
    i = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x
    pny = size(src, 1)
    pnz = size(src, 2)
    i > pny * pnz && return
    iy = (i - 1) % pny + 1
    iz = (i - 1) ÷ pny + 1
    W  = size(intensity, 3)
    @inbounds for w in 1:W
        acc = zero(real(eltype(src)))
        base = (w - 1) * nfreq_per_w
        for j in 1:nfreq_per_w
            v = src[iy, iz, base + j]
            acc += real(v)^2 + imag(v)^2
        end
        intensity[iy, iz, w, row] += acc
    end
    return
end

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
    nfreq_W     = nfreq * W
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

    # Tile un-tiled base operators W times along the frequency dimension.
    prop_d          = repeat(setup.prop_d1,          1, 1, W)
    corr_d          = repeat(setup.corr_d1,          1, 1, W)
    weight_d        = repeat(setup.weight_d1,        1, 1, W)
    prop_weight_d   = prop_d .* weight_d
    corr_weight_d   = corr_d .* weight_d
    prop_n_weight_d = repeat(setup.prop_n_weight_d1, 1, 1, W)
    k0_sq_d         = reshape(repeat(reshape(setup.k0_sq_d1, 1, 1, nfreq), 1, 1, W), 1, 1, nfreq_W)

    # Intensity accumulator: (padded_ny, padded_nz, W, nx)
    ny_nz = padded_ny * padded_nz
    intensity_yzWx_d = CUDA.zeros(T, padded_ny, padded_nz, W, nx)

    # Build batched initial conditions: (padded_ny, padded_nz, nfreq*W).
    p0_d = CUDA.zeros(CT, padded_ny, padded_nz, nfreq_W)
    for (w, (rf_w, t0_w)) in enumerate(zip(rf_windows, t0_windows))
        rf_fft_w   = fft(CUDA.CuArray(T.(rf_w)), 3)  # time FFT → (ny, nz, nt)
        col_offset = (w - 1) * nfreq
        for (f, (freq, bin)) in enumerate(zip(setup.selected_freqs, setup.selected_bins))
            phase = CT(cis(-T(2π) * T(freq) * T(t0_w)))
            p0_d[crop_range_y, crop_range_z, col_offset + f] .= rf_fft_w[:, :, bin] .* phase
        end
    end

    plan_fwd = plan_fft!(p0_d, (1, 2))
    plan_bwd = plan_ifft!(similar(p0_d), (1, 2))
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

    nthreads = min(ny_nz, 512)
    nblocks  = cld(ny_nz, nthreads)

    eta_yznx_d = setup.eta_yznx_d

    if benchmark
        for row in (rr + 1):row_stop
            if corrected
                for _ in 1:(axial_substeps - 1)
                    t_fft_s += CUDA.@elapsed (p_space_d .= current_d; plan_bwd * p_space_d)
                    t_ew_s  += CUDA.@elapsed (tmp_d .= k0_sq_d .* eta_yznx_d[:, :, row:row] .* p_space_d)
                    t_fft_s += CUDA.@elapsed (plan_fwd * tmp_d)
                    t_ew_s  += CUDA.@elapsed (next_d .= current_d .* prop_d .+ corr_d .* tmp_d)
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
            t_fft_s += CUDA.@elapsed (plan_bwd * p_row_d)
            t_ew_s  += CUDA.@elapsed CUDA.@cuda threads=nthreads blocks=nblocks _accum_abs2_sum_batched_3d!(
                intensity_yzWx_d, p_row_d, nfreq, row,
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
                    next_d .= current_d .* prop_d .+ corr_d .* tmp_d
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
            plan_bwd * p_row_d
            CUDA.@cuda threads=nthreads blocks=nblocks _accum_abs2_sum_batched_3d!(
                intensity_yzWx_d, p_row_d, nfreq, row,
            )
        end
    end
    march_wall_s = time() - march_wall_start

    passes_substep  = 13
    passes_last_row = corrected ? 19 : 9
    bytes_march = Int64(row_stop - rr) * (
        (corrected ? Int64(axial_substeps - 1) * passes_substep : 0) * ny_nz * nfreq_W * sizeof(CT) +
        passes_last_row * ny_nz * nfreq_W * sizeof(CT)
    )
    bandwidth_GBps = bytes_march / march_gpu_s / 1e9

    t_download = @elapsed begin
        # intensity_yzWx_d: (padded_ny, padded_nz, W, nx)
        # permute → (padded_ny, padded_nz, nx, W), then slice per w → (padded_ny, padded_nz, nx) → (nx, ny, nz)
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
    return raws, timing
end

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
    nt = size(rf, 3)
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

    use_gpu || error("CPU path not implemented for 3D PAM; use use_gpu=true.")

    setup = _pam_cuda_setup_3d(
        c_padded, cfg, selected_freqs, selected_bins,
        crop_range_y, crop_range_z,
        nx, padded_ny, padded_nz, rr, row_stop,
        c0, effective_axial_step, axial_substeps,
    )
    raws, gpu_timing = _reconstruct_pam_cuda_3d(
        setup, [rf], [t0], corrected, recon_label, show_progress, benchmark,
    )
    intensity_padded = raws[1]  # (nx, padded_ny, padded_nz)
    intensity = intensity_padded[:, crop_range_y, crop_range_z]
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
        :backend => :cuda,
        :gpu_precision => _PAM_CUDA_PRECISION,
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
            intensity .+= raw_w[:, b_crop_y, b_crop_z]
            push!(used_ranges, rng)
            push!(used_energy, en)
        end

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
            Dict{Symbol, Any}(
                :setup_s         => sum(t[:setup_s]      for t in timings),
                :march_gpu_s     => sum_march,
                :march_wall_s    => sum(t[:march_wall_s]  for t in timings),
                :download_s      => sum(t[:download_s]    for t in timings),
                :bandwidth_GBps  => sum_march > 0 ? sum_bytes / sum_march / 1e9 : 0.0,
                :nrows           => sum(t[:nrows]         for t in timings),
                :nfreq           => first(timings)[:nfreq],
                :padded_ny       => first(timings)[:padded_ny],
                :padded_nz       => first(timings)[:padded_nz],
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
