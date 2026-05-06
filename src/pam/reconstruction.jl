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
    c0 = isnothing(reference_sound_speed) ? mean(c_padded) : Float64(reference_sound_speed)
    c0 > 0 || error("reference_sound_speed must be positive.")
    target_axial_step = isnothing(axial_step) ? cfg.dx : Float64(axial_step)
    0 < target_axial_step <= cfg.dx || error("axial_step must lie in (0, cfg.dx].")
    axial_substeps = _pam_axial_substeps(cfg.dx, target_axial_step)
    effective_axial_step = cfg.dx / axial_substeps
    intensity_padded = zeros(Float64, nx, padded_ny)
    row_stop = nx
    row_stop > rr || error("No valid reconstruction rows remain below the receiver row.")
    t0 = Float64(time_origin)

    for (freq, bin) in zip(selected_freqs, selected_bins)
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
        evanescent_inds = findall(x -> !x, propagating)
        weighting = zeros(Float64, padded_ny)
        weighting[real_inds] .= _tukey_window(length(real_inds), cfg.tukey_ratio)

        current = _fftshift(fft(p0_vec))
        current .*= weighting

        mu = (c0 ./ c_padded) .^ 2
        lambda = (k0^2) .* (1 .- mu)

        correction = zeros(ComplexF64, padded_ny)
        for idx in real_inds
            abs(kz[idx]) > sqrt(eps(Float64)) || continue
            correction[idx] = propagator[idx] * effective_axial_step / (2im * kz[idx])
        end

        for row in (rr + 1):row_stop
            for _ in 1:axial_substeps
                if corrected
                    p_space = ifft(_ifftshift(current))
                    conv_term = _fftshift(fft(lambda[row, :] .* p_space))
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
        :reference_sound_speed => c0,
        :axial_step => effective_axial_step,
        :axial_substeps_per_cell => axial_substeps,
        :time_origin => t0,
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
)
    nx, ny = size(c)
    size(rf, 1) == ny || error("RF data must have size (Ny, Nt); expected Ny=$ny, got $(size(rf, 1)).")
    nt = size(rf, 2)
    kgrid = KGrid2D(nx, ny, cfg.dx, cfg.dz; dt=cfg.dt, Nt=nt)
    config = _validate_window_config(window_config)

    ranges, window_samples, hop_samples = _pam_window_ranges(nt, cfg.dt, config)
    energies = [sum(abs2, @view rf[:, range]) for range in ranges]
    max_energy = isempty(energies) ? 0.0 : maximum(energies)
    threshold = max_energy * config.min_energy_ratio

    intensity = zeros(Float64, nx, ny)
    used_ranges = UnitRange{Int}[]
    skipped_ranges = UnitRange{Int}[]
    window_infos = Dict{Symbol, Any}[]
    used_energy = Float64[]
    skipped_energy = Float64[]

    for (range, energy) in zip(ranges, energies)
        if energy < threshold || energy <= 0
            push!(skipped_ranges, range)
            push!(skipped_energy, Float64(energy))
            continue
        end

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
        )
        intensity .+= window_intensity
        push!(used_ranges, range)
        push!(used_energy, Float64(energy))
        push!(window_infos, window_info)
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
    return intensity, kgrid, info
end

