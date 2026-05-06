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

function _format_frequency_mhz(freq::Real)
    return "$(round(Float64(freq) / 1e6; digits=4)) MHz"
end

function _format_frequency_list(freqs::AbstractVector{<:Real}; max_items::Int=8)
    isempty(freqs) && return "none"
    labels = [_format_frequency_mhz(freq) for freq in freqs]
    length(labels) <= max_items && return join(labels, ", ")
    head_count = max(1, max_items - 1)
    return join(vcat(labels[1:head_count], ["...", labels[end]]), ", ")
end

function _pam_progress(show::Bool, msg::AbstractString)
    show || return nothing
    println(stderr, msg)
    flush(stderr)
    return nothing
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

const _PAM_CUDA_PRECISION = Float32
const _PAM_CUDA_COMPLEX = ComplexF32

function _pam_cuda_functional()
    try
        return CUDA.functional()
    catch
        return false
    end
end

function _assert_pam_cuda_available()
    _pam_cuda_functional() && return nothing
    error(
        "PAM CUDA reconstruction requested with use_gpu=true, but CUDA.jl " *
        "does not see a functional NVIDIA CUDA GPU. Configure CUDA.jl on " *
        "a machine with an NVIDIA GPU, or run with use_gpu=false.",
    )
end

function _reconstruct_pam_cuda(
    rf::AbstractMatrix{<:Real},
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
    t0::Float64,
    corrected::Bool,
    recon_label::AbstractString,
    show_progress::Bool,
)
    _assert_pam_cuda_available()
    let dev = CUDA.device()
        println("[ PAM ] $recon_label: GPU $(CUDA.name(dev)) (device $(CUDA.deviceid(dev))), $(_PAM_CUDA_PRECISION) arithmetic, $(length(selected_freqs)) freq bins")
        flush(stdout)
    end

    T = _PAM_CUDA_PRECISION
    CT = _PAM_CUDA_COMPLEX
    rf_d = CUDA.CuArray(T.(rf))
    rf_fft_d = fft(rf_d, 2)
    eta_yx_d = CUDA.CuArray(T.(permutedims(1 .- (c0 ./ c_padded) .^ 2)))
    intensity_yx_d = CUDA.zeros(T, padded_ny, nx)

    p0_d = CUDA.zeros(CT, padded_ny)
    next_d = similar(p0_d)
    tmp_d = similar(p0_d)
    k = _fft_wavenumbers(padded_ny, cfg.dz)

    for (freq_idx, (freq, bin)) in enumerate(zip(selected_freqs, selected_bins))
        freq_start = time()
        fill!(p0_d, zero(CT))
        phase = CT(cis(-T(2 * pi) * T(freq) * T(t0)))
        p0_d[crop_range] .= rf_fft_d[:, bin] .* phase

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

        # Permute centered-order arrays to FFT order once per frequency bin so the
        # propagation loop needs no per-step fftshift/ifftshift (gather) operations.
        propagator_d = CUDA.CuArray(CT.(_ifftshift(propagator)))
        correction_d = CUDA.CuArray(CT.(_ifftshift(correction)))
        propagating_d = CUDA.CuArray(T.(_ifftshift(propagating)))
        weighting_d = CUDA.CuArray(T.(_ifftshift(weighting)))

        current_d = fft(p0_d)
        current_d .*= weighting_d

        for row in (rr + 1):row_stop
            for _ in 1:axial_substeps
                if corrected
                    p_space_d = ifft(current_d)
                    eta_row_d = @view eta_yx_d[:, row]
                    tmp_d .= T(k0^2) .* eta_row_d .* p_space_d
                    conv_term_d = fft(tmp_d)
                    next_d .= current_d .* propagator_d
                    next_d .+= correction_d .* conv_term_d
                else
                    next_d .= current_d .* propagator_d
                end
                next_d .*= propagating_d
                current_d, next_d = next_d, current_d
            end
            # Keep the same shifted spectral taper schedule as the CPU path.
            current_d .*= weighting_d

            p_row_d = ifft(current_d)
            intensity_yx_d[:, row] .+= abs2.(p_row_d)
        end

        if show_progress
            CUDA.synchronize()
        end
        _pam_progress(
            show_progress,
            "PAM $recon_label frequency $freq_idx/$(length(selected_freqs)) " *
            "($(_format_frequency_mhz(freq)), bin $bin) elapsed $(_format_elapsed(time() - freq_start))",
        )
    end

    CUDA.synchronize()
    return Float64.(permutedims(Array(intensity_yx_d)))
end

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

    intensity_padded = if use_gpu
        _reconstruct_pam_cuda(
            rf,
            c_padded,
            cfg,
            selected_freqs,
            selected_bins,
            crop_range,
            nx,
            padded_ny,
            rr,
            row_stop,
            c0,
            effective_axial_step,
            axial_substeps,
            t0,
            corrected,
            recon_label,
            show_progress,
        )
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
    )
    return intensity, kgrid, info
end

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

    used_count = length(used_ranges)
    if used_count > 0
        intensity ./= used_count
    end
    first_info = isempty(window_infos) ? Dict{Symbol, Any}() : first(window_infos)
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
    )
    _pam_progress(
        show_progress,
        "PAM $recon_label windowed reconstruction complete: used=$used_count, " *
        "skipped=$(length(skipped_ranges)), total elapsed $(_format_elapsed(time() - total_start))",
    )
    return intensity, kgrid, info
end
