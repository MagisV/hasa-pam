#!/usr/bin/env julia

using CairoMakie
using DICOM
using FFTW
using Interpolations
using Printf
using Random
using Statistics
using TranscranialFUS

const OUT_DIR = normpath(joinpath(@__DIR__, "..", "..", "figures", "hasa_pam_explainer"))
const WAVEFRONT_CT_PATH = "/Users/vm/INI_code/Ultrasound/DIRU_20240404_human_skull_phase_correction_1_2_(skull_Normal)/DICOM/PAT_0000/STD_0000/SER_0002/OBJ_0001"
const WAVEFRONT_SLICE_IDX0 = 250
const WAVEFRONT_LX = 0.100
const WAVEFRONT_LY = 0.056
const HU_THR = 200
const C_WATER = 1500.0
const C_BONE = 4500.0
const RHO_WATER = 1000.0
const RHO_BONE = 2800.0
const RECON_AXIAL_STEP = 50e-6
const PEAK_TOLERANCE_MM = 2.0
const BUBBLE_FUNDAMENTAL = 0.5e6
const BUBBLE_HARMONICS = [2, 3, 4]
const BUBBLE_HARMONIC_AMPLITUDES = [1.0, 0.6, 0.3]
const INTENSITY_OVERLAY_CUTOFF = 0.003
const INTENSITY_COLORMAP = [
    RGBf(0.30, 0.05, 0.45),
    RGBf(0.62, 0.12, 0.50),
    RGBf(0.92, 0.33, 0.25),
    RGBf(1.00, 0.68, 0.12),
    RGBf(1.00, 0.96, 0.45),
]

const THEME = Theme(
    fontsize = 20,
    Axis = (
        titlefont = :bold,
        titlesize = 22,
        xlabelsize = 17,
        ylabelsize = 17,
        xticklabelsize = 13,
        yticklabelsize = 13,
    ),
    Label = (fontsize = 18,),
)

mm(x) = Float64.(x) .* 1e3
mhz(x) = Float64.(x) ./ 1e6
to_float(x) = x isa AbstractString ? parse(Float64, x) : Float64(x)

function fftshift1(a)
    return circshift(a, fld(length(a), 2))
end

function safe_log10(a)
    b = Float64.(abs.(a))
    ref = max(maximum(b), eps(Float64))
    return log10.(b ./ ref .+ 1e-6)
end

function tukey_taper(n::Int, ratio::Real)
    n <= 0 && return Float64[]
    α = clamp(Float64(ratio), 0.0, 1.0)
    α == 0.0 && return ones(Float64, n)
    α == 1.0 && return [0.5 * (1 - cos(2π * (i - 1) / max(n - 1, 1))) for i in 1:n]
    taper = ones(Float64, n)
    edge = α * (n - 1) / 2
    @inbounds for i in 1:n
        x = i - 1
        if x < edge
            taper[i] = 0.5 * (1 + cos(π * (2x / (α * (n - 1)) - 1)))
        elseif x > (n - 1) * (1 - α / 2)
            taper[i] = 0.5 * (1 + cos(π * (2x / (α * (n - 1)) - 2 / α + 1)))
        end
    end
    return taper
end

function save_svg(fig, basename, generated)
    mkpath(OUT_DIR)
    path = joinpath(OUT_DIR, "$basename.svg")
    save(path, fig)
    push!(generated, path)
    return generated
end

function cleanup_output_dir()
    mkpath(OUT_DIR)
    for file in readdir(OUT_DIR; join=true)
        if isfile(file)
            rm(file; force=true)
        end
    end
    return nothing
end

function make_cfg()
    return PAMConfig(
        dx = 0.3e-3,
        dz = 0.3e-3,
        axial_dim = WAVEFRONT_LY,
        transverse_dim = WAVEFRONT_LX,
        receiver_aperture = 50e-3,
        receiver_row = 1,
        t_max = 60e-6,
        dt = 20e-9,
        c0 = C_WATER,
        rho0 = RHO_WATER,
        PML_GUARD = 20,
        zero_pad_factor = 4,
        tukey_ratio = 0.25,
        peak_suppression_radius = 2e-3,
        success_tolerance = 1e-3,
    )
end

function resample_matrix_linear(a::AbstractMatrix{<:Real}, out_rows::Int, out_cols::Int)
    itp = extrapolate(interpolate(Float32.(a), BSpline(Linear())), Flat())
    row_coords = range(1, size(a, 1); length=out_rows)
    col_coords = range(1, size(a, 2); length=out_cols)
    out = Matrix{Float32}(undef, out_rows, out_cols)
    @inbounds for row in 1:out_rows, col in 1:out_cols
        out[row, col] = Float32(itp(row_coords[row], col_coords[col]))
    end
    return out
end

function wavefront_hu_to_medium(hu)
    hu_clipped = clamp.(Float32.(hu), -1000.0f0, 3000.0f0)
    c = fill(Float32(C_WATER), size(hu_clipped))
    rho = fill(Float32(RHO_WATER), size(hu_clipped))
    mask = hu_clipped .>= Float32(HU_THR)
    if any(mask)
        bone = hu_clipped[mask]
        h_max = Float32(quantile(vec(bone), 0.995))
        psi = clamp.((h_max .- bone) ./ max(h_max, 1.0f0), 0.0f0, 1.0f0)
        c[mask] .= Float32(C_WATER) .+ Float32(C_BONE - C_WATER) .* (1.0f0 .- psi)
        rho[mask] .= Float32(RHO_WATER) .+ Float32(RHO_BONE - RHO_WATER) .* (1.0f0 .- psi)
    end
    return Float64.(c), Float64.(rho)
end

function load_wavefront_medium_2d(cfg::PAMConfig)
    isdir(WAVEFRONT_CT_PATH) || error("CT folder not found: $WAVEFRONT_CT_PATH")
    entries = Tuple{Float64, String}[]
    for path in filter(p -> isfile(p) && !startswith(basename(p), "."), readdir(WAVEFRONT_CT_PATH; join=true))
        meta = dcm_parse(path)
        pos = meta[(0x0020, 0x0032)]
        push!(entries, (to_float(pos[3]), path))
    end
    sort!(entries; by=first)
    target_path = entries[WAVEFRONT_SLICE_IDX0 + 1][2]
    meta = dcm_parse(target_path)
    raw = Float32.(meta[(0x7fe0, 0x0010)])
    slope = haskey(meta, (0x0028, 0x1053)) ? to_float(meta[(0x0028, 0x1053)]) : 1.0
    intercept = haskey(meta, (0x0028, 0x1052)) ? to_float(meta[(0x0028, 0x1052)]) : -1024.0
    hu = raw .* Float32(slope) .+ Float32(intercept)

    spacing = meta[(0x0028, 0x0030)]
    dx_ct = to_float(spacing[2]) * 1e-3
    n_rows, n_cols = size(hu)
    bone_per_row = vec(sum(hu .> HU_THR; dims=2))
    significant = findall(bone_per_row .> 30)
    isempty(significant) && error("No skull found in CT slice $WAVEFRONT_SLICE_IDX0")
    y_top = first(significant)
    if y_top > n_rows ÷ 2
        hu = reverse(hu; dims=1)
        bone_per_row = vec(sum(hu .> HU_THR; dims=2))
        significant = findall(bone_per_row .> 30)
        y_top = first(significant)
    end

    y_top0 = y_top - 1
    x_ctr0 = n_cols ÷ 2
    n_above_ct = round(Int, 0.020 / dx_ct)
    ny_ct = round(Int, WAVEFRONT_LY / dx_ct)
    nx_ct = round(Int, WAVEFRONT_LX / dx_ct)
    y0 = y_top0 - n_above_ct
    y1 = y0 + ny_ct
    x0 = x_ctr0 - nx_ct ÷ 2
    x1 = x0 + nx_ct

    pad_top = max(0, -y0)
    pad_bottom = max(0, y1 - n_rows)
    pad_left = max(0, -x0)
    pad_right = max(0, x1 - n_cols)
    y0c = max(0, y0)
    y1c = min(n_rows, y1)
    x0c = max(0, x0)
    x1c = min(n_cols, x1)
    region = hu[(y0c + 1):y1c, (x0c + 1):x1c]
    if pad_top + pad_bottom + pad_left + pad_right > 0
        padded = zeros(Float32, size(region, 1) + pad_top + pad_bottom, size(region, 2) + pad_left + pad_right)
        padded[(pad_top + 1):(pad_top + size(region, 1)), (pad_left + 1):(pad_left + size(region, 2))] .= region
        region = padded
    end

    hu_grid = resample_matrix_linear(region, pam_Nx(cfg), pam_Ny(cfg))
    c, rho = wavefront_hu_to_medium(hu_grid)
    return c, rho, target_path
end

function make_sources()
    depth = 44.0e-3
    lateral = 0.0
    return [
        BubbleCluster2D(
            depth=depth,
            lateral=lateral,
            fundamental=BUBBLE_FUNDAMENTAL,
            amplitude=1.0,
            harmonics=copy(BUBBLE_HARMONICS),
            harmonic_amplitudes=copy(BUBBLE_HARMONIC_AMPLITUDES),
            harmonic_phases=zeros(Float64, length(BUBBLE_HARMONICS)),
            gate_duration=40e-6,
            taper_ratio=0.25,
            delay=0.4e-6,
        )
    ]
end

selected_frequency_targets() = Float64.(BUBBLE_HARMONICS) .* BUBBLE_FUNDAMENTAL

function depth_coordinates_mm(kgrid, cfg)
    return mm(depth_coordinates(kgrid, cfg))
end

function padded_lateral_mm(kgrid, cfg, crop_range, padded_ny)
    return mm(range(first(kgrid.y_vec) - (first(crop_range) - 1) * cfg.dz; step=cfg.dz, length=padded_ny))
end

function source_time_origin(sources)
    weights = [max(abs(src.amplitude), eps(Float64)) for src in sources]
    centers = [
        src isa PointSource2D ? src.delay + 0.5 * src.num_cycles / src.frequency :
        src isa BubbleCluster2D ? src.delay + 0.5 * src.gate_duration :
        error("Unsupported source type $(typeof(src))")
        for src in sources
    ]
    return sum(weights .* centers) / sum(weights)
end

function reconstruct_selected(rf, c, cfg, selected_targets; corrected::Bool, time_origin::Real, reference_sound_speed::Real)
    return reconstruct_pam(
        rf,
        c,
        cfg;
        frequencies=selected_targets,
        corrected=corrected,
        reference_sound_speed=reference_sound_speed,
        axial_step=RECON_AXIAL_STEP,
        time_origin=time_origin,
        use_gpu=false,
        show_progress=false,
    )
end

function reconstruction_metrics(intensity_corr, intensity_base, kgrid, cfg, sources)
    src = source_location(sources)
    corr_peak = peak_location(intensity_corr, kgrid, cfg)
    base_peak = peak_location(intensity_base, kgrid, cfg)
    corr_error = location_error_mm(corr_peak, src)
    base_error = location_error_mm(base_peak, src)
    return (source=src, corrected_peak=corr_peak, uncorrected_peak=base_peak, corrected_error_mm=corr_error, uncorrected_error_mm=base_error)
end

function choose_reconstruction_reference(rf, c, cfg, selected_targets, sources)
    nominal = source_time_origin(sources)
    normal_run_c0 = TranscranialFUS._pam_reference_sound_speed(c, cfg, sources)
    c0_candidates = unique(vcat([cfg.c0, normal_run_c0], collect(1700.0:50.0:2400.0)))
    sort!(c0_candidates)
    t0_candidates = unique([0.0, nominal])
    sort!(t0_candidates)

    best = nothing
    println(@sprintf("Nominal source pulse center: %.3f us", nominal * 1e6))
    println(@sprintf("Normal-run reference sound speed estimate: %.1f m/s", normal_run_c0))
    for reference_c0 in c0_candidates, t0 in t0_candidates
        intensity, kgrid, _ = reconstruct_selected(
            rf,
            c,
            cfg,
            selected_targets;
            corrected=true,
            time_origin=t0,
            reference_sound_speed=reference_c0,
        )
        peak = peak_location(intensity, kgrid, cfg)
        err = location_error_mm(peak, source_location(sources))
        if isnothing(best) || err < best.error_mm
            best = (
                time_origin=t0,
                reference_sound_speed=reference_c0,
                intensity=intensity,
                kgrid=kgrid,
                peak=peak,
                error_mm=err,
            )
        end
    end
    println(@sprintf(
        "Selected reconstruction reference: c0 = %.1f m/s, time origin = %.3f us (corrected peak scan error %.2f mm)",
        best.reference_sound_speed,
        best.time_origin * 1e6,
        best.error_mm,
    ))
    return best
end

function collect_algorithm_arrays(rf, c, cfg, selected_targets; time_origin::Real=0.0, reference_sound_speed::Real=cfg.c0)
    selected_freqs, selected_bins = TranscranialFUS._select_frequency_bins(rf, cfg.dt, selected_targets; bandwidth=0.0)
    padded_ny = cfg.zero_pad_factor * pam_Ny(cfg)
    _, crop_range = TranscranialFUS._zero_pad_receiver_rf(rf, padded_ny)
    c_padded, _ = TranscranialFUS._edge_pad_lateral(c, padded_ny)
    c0 = Float64(reference_sound_speed)
    rf_fft = fft(Float64.(rf), 2)

    mid_f = cld(length(selected_freqs), 2)
    freq = selected_freqs[mid_f]
    bin = selected_bins[mid_f]
    k0 = 2π * freq / c0
    k = TranscranialFUS._fft_wavenumbers(padded_ny, cfg.dz)
    kz = sqrt.(complex.(k0^2 .- k .^ 2, 0.0))
    propagator = exp.(1im .* kz .* RECON_AXIAL_STEP)
    real_inds = findall(real.(kz ./ k0) .> 0.0)
    propagating = falses(padded_ny)
    propagating[real_inds] .= true
    weight = zeros(Float64, padded_ny)
    weight[real_inds] .= tukey_taper(length(real_inds), cfg.tukey_ratio)
    corr = zeros(ComplexF64, padded_ny)
    for idx in real_inds
        abs(kz[idx]) > sqrt(eps(Float64)) || continue
        corr[idx] = propagator[idx] * RECON_AXIAL_STEP / (2im * kz[idx])
    end
    prop = TranscranialFUS._ifftshift(propagating .* propagator)
    corr = TranscranialFUS._ifftshift(propagating .* corr)
    weight = TranscranialFUS._ifftshift(weight)

    p0_vec = zeros(ComplexF64, padded_ny)
    p0_vec[crop_range] .= rf_fft[:, bin] .* exp(-1im * 2π * freq * Float64(time_origin))
    P0 = fft(p0_vec)
    current = P0 .* weight
    row = receiver_row(cfg) + round(Int, 24.0e-3 / cfg.dx)
    row = clamp(row, receiver_row(cfg) + 1, pam_Nx(cfg))
    eta = 1 .- (c0 ./ c_padded[row, :]) .^ 2
    p = ifft(current)
    correction_source = k0^2 .* eta .* p
    P_next = current .* prop .+ corr .* fft(correction_source)

    return (
        selected_freqs=selected_freqs,
        selected_bins=selected_bins,
        mid_f=mid_f,
        freq=freq,
        bin=bin,
        rf_fft=rf_fft,
        p0=p0_vec,
        P0=P0,
        current=current,
        p=p,
        correction_source=correction_source,
        P_next=P_next,
        eta=eta,
        prop=prop,
        corr=corr,
        weight=weight,
        k=k,
        k0=k0,
        padded_ny=padded_ny,
        crop_range=crop_range,
        c_padded=c_padded,
        row=row,
    )
end

function collect_marching_animation_data(rf, c, cfg, selected_targets; corrected::Bool, time_origin::Real=0.0, reference_sound_speed::Real=cfg.c0, frame_stride::Int=4)
    selected_freqs, selected_bins = TranscranialFUS._select_frequency_bins(rf, cfg.dt, selected_targets; bandwidth=0.0)
    padded_ny = cfg.zero_pad_factor * pam_Ny(cfg)
    _, crop_range = TranscranialFUS._zero_pad_receiver_rf(rf, padded_ny)
    c_padded, _ = TranscranialFUS._edge_pad_lateral(c, padded_ny)
    c0 = Float64(reference_sound_speed)
    rf_fft = fft(Float64.(rf), 2)
    rows = collect((receiver_row(cfg) + 1):pam_Nx(cfg))
    axial_substeps = TranscranialFUS._pam_axial_substeps(cfg.dx, RECON_AXIAL_STEP)
    step = cfg.dx / axial_substeps
    k = TranscranialFUS._fft_wavenumbers(padded_ny, cfg.dz)

    intensity_frames = Matrix{Float64}[]
    eta_frames = Vector{Float64}[]
    row_depths = Float64[]
    freq_indices = Int[]
    freq_values = Float64[]
    cumulative = zeros(Float64, pam_Nx(cfg), pam_Ny(cfg))
    lateral_mm = mm(pam_grid(cfg).y_vec)
    completed_peak_lats = Float64[]
    completed_peak_depths = Float64[]
    peak_lats_frames = Vector{Float64}[]
    peak_depths_frames = Vector{Float64}[]

    for (fidx, (freq, bin)) in enumerate(zip(selected_freqs, selected_bins))
        k0 = 2π * freq / c0
        kz = sqrt.(complex.(k0^2 .- k .^ 2, 0.0))
        propagator = exp.(1im .* kz .* step)
        real_inds = findall(real.(kz ./ k0) .> 0.0)
        propagating = falses(padded_ny)
        propagating[real_inds] .= true
        weighting = zeros(Float64, padded_ny)
        weighting[real_inds] .= tukey_taper(length(real_inds), cfg.tukey_ratio)
        lambda = (k0^2) .* (1 .- (c0 ./ c_padded) .^ 2)
        correction = zeros(ComplexF64, padded_ny)
        for idx in real_inds
            abs(kz[idx]) > sqrt(eps(Float64)) || continue
            correction[idx] = propagator[idx] * step / (2im * kz[idx])
        end
        propagator = TranscranialFUS._ifftshift(propagator)
        weighting = TranscranialFUS._ifftshift(weighting)
        correction = TranscranialFUS._ifftshift(correction)
        evanescent_inds = findall(TranscranialFUS._ifftshift(.!propagating))

        p0 = zeros(ComplexF64, padded_ny)
        p0[crop_range] .= rf_fft[:, bin] .* exp(-1im * 2π * freq * Float64(time_origin))
        current = fft(p0)
        current .*= weighting

        for (ridx, row) in enumerate(rows)
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
            current .*= weighting
            p_row = ifft(current)
            cumulative[row, :] .+= abs2.(p_row[crop_range])

            if ridx == length(rows)
                peak_row, peak_col = Tuple(argmax(cumulative))
                empty!(completed_peak_lats)
                empty!(completed_peak_depths)
                push!(completed_peak_lats, lateral_mm[peak_col])
                push!(completed_peak_depths, (peak_row - receiver_row(cfg)) * cfg.dx * 1e3)
            end

            if ridx == 1 || ridx == length(rows) || mod(ridx - 1, frame_stride) == 0
                push!(intensity_frames, copy(cumulative))
                push!(eta_frames, copy(1 .- (c0 ./ c_padded[row, :]) .^ 2))
                push!(row_depths, (row - receiver_row(cfg)) * cfg.dx * 1e3)
                push!(freq_indices, fidx)
                push!(freq_values, freq)
                push!(peak_lats_frames, copy(completed_peak_lats))
                push!(peak_depths_frames, copy(completed_peak_depths))
            end
        end
    end

    return (
        intensity_frames=intensity_frames,
        eta_frames=eta_frames,
        row_depths=row_depths,
        freq_indices=freq_indices,
        freq_values=freq_values,
        peak_lats_frames=peak_lats_frames,
        peak_depths_frames=peak_depths_frames,
        selected_freqs=selected_freqs,
        nfreq=length(selected_freqs),
        padded_ny=padded_ny,
        crop_range=crop_range,
        corrected=corrected,
    )
end

function peak_location(intensity, kgrid, cfg)
    row, col = Tuple(argmax(intensity))
    return (
        row=row,
        col=col,
        depth_mm=(row - receiver_row(cfg)) * cfg.dx * 1e3,
        lateral_mm=kgrid.y_vec[col] * 1e3,
        value=maximum(intensity),
    )
end

function source_location(sources)
    src = first(sources)
    return (depth_mm=src.depth * 1e3, lateral_mm=src.lateral * 1e3)
end

function location_error_mm(peak, source)
    return hypot(peak.depth_mm - source.depth_mm, peak.lateral_mm - source.lateral_mm)
end

function validate_reconstruction(intensity_corr, intensity_base, kgrid, cfg, sources)
    metrics = reconstruction_metrics(intensity_corr, intensity_base, kgrid, cfg, sources)
    src = metrics.source
    corr_peak = metrics.corrected_peak
    base_peak = metrics.uncorrected_peak
    corr_error = metrics.corrected_error_mm
    base_error = metrics.uncorrected_error_mm
    println(@sprintf("Source location: depth %.2f mm, lateral %.2f mm", src.depth_mm, src.lateral_mm))
    println(@sprintf("Corrected peak: depth %.2f mm, lateral %.2f mm, error %.2f mm", corr_peak.depth_mm, corr_peak.lateral_mm, corr_error))
    println(@sprintf("Uncorrected peak: depth %.2f mm, lateral %.2f mm, error %.2f mm", base_peak.depth_mm, base_peak.lateral_mm, base_error))
    corr_error <= PEAK_TOLERANCE_MM || error(@sprintf(
        "Corrected reconstruction peak error %.2f mm exceeds %.2f mm.",
        corr_error,
        PEAK_TOLERANCE_MM,
    ))
    return metrics
end

function mark_source_and_peak!(ax, sources, peak)
    src = source_location(sources)
    scatter!(ax, [src.lateral_mm], [src.depth_mm]; marker=:star5, color=:red, markersize=24)
    scatter!(ax, [peak.lateral_mm], [peak.depth_mm]; marker=:xcross, color=:white, markersize=24, strokewidth=3)
    return nothing
end

function figure_step_geometry(c, cfg, kgrid, sources, generated)
    # Panel 1: physical setup used by k-Wave and by the 2D PAM grid.
    depth_mm = depth_coordinates_mm(kgrid, cfg)
    y_mm = mm(kgrid.y_vec)
    fig = Figure(size=(1150, 900), backgroundcolor=RGBf(0.985, 0.985, 0.975))
    ax = Axis(fig[1, 1], title="Geometry: receiver, source, and c(x,y)", xlabel="Lateral [mm]", ylabel="Depth [mm]", aspect=DataAspect(), yreversed=true)
    hm = heatmap!(ax, y_mm, depth_mm, c'; colormap=:turbo)
    lines!(ax, [first(y_mm), last(y_mm)], [0, 0]; color=:white, linewidth=5)
    src = source_location(sources)
    scatter!(ax, [src.lateral_mm], [src.depth_mm]; marker=:star5, color=:red, markersize=34)
    text!(ax, first(y_mm) + 2, 1.5; text="receiver plane", color=:white, fontsize=18)
    Colorbar(fig[1, 2], hm, label="c [m/s]")
    return save_svg(fig, "step01_geometry", generated)
end

function figure_step_rf_data(rf, cfg, kgrid, generated)
    # Panel 2: receiver pressure recording, rf(y,t), returned by k-Wave.
    y_mm = mm(kgrid.y_vec)
    t_us = collect(0:(size(rf, 2) - 1)) .* cfg.dt .* 1e6
    fig = Figure(size=(1150, 850), backgroundcolor=RGBf(0.985, 0.985, 0.975))
    ax = Axis(fig[1, 1], title="RF data: rf(y,t)", xlabel="Time [us]", ylabel="Lateral [mm]")
    hm = heatmap!(ax, t_us, y_mm, rf'; colormap=:balance)
    Colorbar(fig[1, 2], hm, label="rf")
    return save_svg(fig, "step02_rf_data", generated)
end

function figure_step_temporal_fft(rf, cfg, arrays, generated)
    # Panel 3: temporal FFT over t, with reconstruction bins selected from rf.
    spectrum = vec(mean(abs.(arrays.rf_fft); dims=1))
    pos = 2:(fld(length(spectrum), 2) + 1)
    fig = Figure(size=(1150, 850), backgroundcolor=RGBf(0.985, 0.985, 0.975))
    ax = Axis(fig[1, 1], title="Temporal FFT: selected_freqs and selected_bins", xlabel="Frequency [MHz]", ylabel="Mean |FFT(rf)|")
    lines!(ax, mhz((collect(pos) .- 1) ./ (size(rf, 2) * cfg.dt)), spectrum[pos]; color=:black, linewidth=3)
    scatter!(ax, mhz(arrays.selected_freqs), spectrum[arrays.selected_bins]; color=:red, markersize=18)
    return save_svg(fig, "step03_temporal_fft", generated)
end

function figure_step_receiver_frequency_plane(kgrid, arrays, generated)
    # Panel 4: one receiver-plane frequency slice p0(y,f).
    y_mm = mm(kgrid.y_vec)
    fig = Figure(size=(1050, 850), backgroundcolor=RGBf(0.985, 0.985, 0.975))
    ax = Axis(fig[1, 1], title=@sprintf("Receiver frequency data: p0(y,f), f = %.2f MHz", arrays.freq / 1e6), xlabel="Lateral [mm]", ylabel="|p0|")
    lines!(ax, y_mm, abs.(arrays.rf_fft[:, arrays.bin]); color=RGBf(0.2, 0.25, 0.35), linewidth=3)
    return save_svg(fig, "step04_receiver_frequency_plane", generated)
end

function figure_step_lateral_spectrum(cfg, arrays, generated)
    # Panel 5: lateral FFT of p0, showing the angular spectrum P0(ky).
    k_plot = fftshift1(arrays.k) ./ 1e3
    P_plot = fftshift1(safe_log10(arrays.P0))
    fig = Figure(size=(1050, 850), backgroundcolor=RGBf(0.985, 0.985, 0.975))
    ax = Axis(fig[1, 1], title="Lateral angular spectrum: |P0(ky)|", xlabel="ky [rad/mm]", ylabel="log |P0|")
    lines!(ax, k_plot, P_plot; color=:black, linewidth=3)
    vlines!(ax, [-arrays.k0 / 1e3, arrays.k0 / 1e3]; color=:red, linewidth=2, linestyle=:dash)
    return save_svg(fig, "step05_lateral_angular_spectrum", generated)
end

function figure_step_eta(c, cfg, kgrid, generated; reference_sound_speed::Real)
    # Panel 6: heterogeneous contrast used in the corrected HASA update.
    depth_mm = depth_coordinates_mm(kgrid, cfg)
    y_mm = mm(kgrid.y_vec)
    eta = 1 .- (Float64(reference_sound_speed) ./ c) .^ 2
    fig = Figure(size=(1150, 900), backgroundcolor=RGBf(0.985, 0.985, 0.975))
    ax = Axis(fig[1, 1], title="Heterogeneity field: eta(x,y) = 1 - (c0/c)^2", xlabel="Lateral [mm]", ylabel="Depth [mm]", aspect=DataAspect(), yreversed=true)
    hm = heatmap!(ax, y_mm, depth_mm, eta'; colormap=:coolwarm)
    Colorbar(fig[1, 2], hm, label="eta")
    return save_svg(fig, "step06_eta_field", generated)
end

function figure_step_hasa_update(cfg, kgrid, arrays, generated)
    # Panel 7: one axial HASA update, split into p, correction source, and P_next.
    y_pad_mm = padded_lateral_mm(kgrid, cfg, arrays.crop_range, arrays.padded_ny)
    fig = Figure(size=(1500, 720), backgroundcolor=RGBf(0.985, 0.985, 0.975))
    Label(fig[1, 1:3], "One corrected HASA marching update", fontsize=30, font=:bold, halign=:left)
    labels = ["p = IFFT(current)", "k0^2 * eta * p", "P_next = current * prop + corr * FFT(...)"]
    values = [abs.(arrays.p), abs.(arrays.correction_source), fftshift1(safe_log10(arrays.P_next))]
    xs = [y_pad_mm, y_pad_mm, fftshift1(arrays.k) ./ 1e3]
    xlabels = ["Lateral [mm]", "Lateral [mm]", "ky [rad/mm]"]
    for idx in 1:3
        ax = Axis(fig[2, idx], title=labels[idx], xlabel=xlabels[idx], ylabel=idx == 1 ? "Magnitude" : "")
        lines!(ax, xs[idx], values[idx]; color=idx == 2 ? :orange : :black, linewidth=3)
    end
    return save_svg(fig, "step07_hasa_marching_update", generated)
end

function figure_step_final_pam(intensity_corr, intensity_base, cfg, kgrid, sources, metrics, generated)
    # Panel 8: final intensity after summing abs2 pressure over selected frequencies.
    y_mm = mm(kgrid.y_vec)
    depth_mm = depth_coordinates_mm(kgrid, cfg)
    global_ref = max(maximum(intensity_corr), maximum(intensity_base), eps(Float64))
    fig = Figure(size=(1500, 850), backgroundcolor=RGBf(0.985, 0.985, 0.975))
    Label(fig[1, 1:2], "Final PAM: I(x,y) = sum_f |p_f(x,y)|^2", fontsize=30, font=:bold, halign=:left)
    panels = [
        ("corrected", intensity_corr, metrics.corrected_peak),
        ("uncorrected", intensity_base, metrics.uncorrected_peak),
    ]
    hm = nothing
    for (idx, (name, intensity, peak)) in enumerate(panels)
        ax = Axis(fig[2, idx], title=name, xlabel="Lateral [mm]", ylabel=idx == 1 ? "Depth [mm]" : "", aspect=DataAspect(), yreversed=true)
        hm = heatmap!(ax, y_mm, depth_mm, (intensity ./ global_ref)'; colormap=:viridis, colorrange=(0, 1))
        mark_source_and_peak!(ax, sources, peak)
    end
    Colorbar(fig[2, 3], hm, label="normalized intensity")
    return save_svg(fig, "step08_final_pam", generated)
end

function figure_workflow(rf, c, cfg, kgrid, sources, arrays, intensity_corr, intensity_base, metrics, generated; reference_sound_speed::Real)
    # Compact overview built from the same per-step arrays as the individual SVGs.
    depth_mm = depth_coordinates_mm(kgrid, cfg)
    y_mm = mm(kgrid.y_vec)
    t_us = collect(0:(size(rf, 2) - 1)) .* cfg.dt .* 1e6
    fig = Figure(size=(2600, 1450), backgroundcolor=RGBf(0.985, 0.985, 0.975))
    gl = fig[1, 1] = GridLayout()
    Label(gl[1, 1:4], "2D HASA-PAM workflow", fontsize=34, font=:bold, halign=:left)

    ax1 = Axis(gl[2, 1], title="1  Geometry: c(x,y)", xlabel="Lateral [mm]", ylabel="Depth [mm]", aspect=DataAspect(), yreversed=true)
    hm1 = heatmap!(ax1, y_mm, depth_mm, c'; colormap=:turbo)
    lines!(ax1, [first(y_mm), last(y_mm)], [0, 0]; color=:white, linewidth=4)
    src = source_location(sources)
    scatter!(ax1, [src.lateral_mm], [src.depth_mm]; marker=:star5, color=:red, markersize=28)
    Colorbar(gl[2, 1, Right()], hm1, label="c [m/s]", width=12)

    ax2 = Axis(gl[2, 2], title="2  RF data: rf(y,t)", xlabel="Time [us]", ylabel="Lateral [mm]")
    hm2 = heatmap!(ax2, t_us, y_mm, rf'; colormap=:balance)
    Colorbar(gl[2, 2, Right()], hm2, label="rf", width=12)

    spectrum = vec(mean(abs.(arrays.rf_fft); dims=1))
    pos = 2:(fld(length(spectrum), 2) + 1)
    ax3 = Axis(gl[2, 3], title="3  selected_freqs", xlabel="Frequency [MHz]", ylabel="Mean |FFT(rf)|")
    lines!(ax3, mhz((collect(pos) .- 1) ./ (size(rf, 2) * cfg.dt)), spectrum[pos]; color=:black, linewidth=2)
    scatter!(ax3, mhz(arrays.selected_freqs), spectrum[arrays.selected_bins]; color=:red, markersize=13)

    ax4 = Axis(gl[2, 4], title=@sprintf("4  p0(y,f), f = %.2f MHz", arrays.freq / 1e6), xlabel="Lateral [mm]", ylabel="|p0|")
    lines!(ax4, y_mm, abs.(arrays.rf_fft[:, arrays.bin]); color=:black, linewidth=2)

    ax5 = Axis(gl[3, 1], title="5  |P0(ky)|, zero_pad_factor = 4", xlabel="ky [rad/mm]", ylabel="log |P0|")
    lines!(ax5, fftshift1(arrays.k) ./ 1e3, fftshift1(safe_log10(arrays.P0)); color=:black, linewidth=2)
    vlines!(ax5, [-arrays.k0 / 1e3, arrays.k0 / 1e3]; color=:red, linewidth=2, linestyle=:dash)

    eta = 1 .- (Float64(reference_sound_speed) ./ c) .^ 2
    ax6 = Axis(gl[3, 2], title="6  eta(x,y)", xlabel="Lateral [mm]", ylabel="Depth [mm]", aspect=DataAspect(), yreversed=true)
    hm6 = heatmap!(ax6, y_mm, depth_mm, eta'; colormap=:coolwarm)
    Colorbar(gl[3, 2, Right()], hm6, label="eta", width=12)

    step_gl = GridLayout(gl[3, 3])
    Label(step_gl[1, 1:3], "7  One HASA marching step", fontsize=20, font=:bold)
    y_pad_mm = padded_lateral_mm(kgrid, cfg, arrays.crop_range, arrays.padded_ny)
    for (idx, (label, values)) in enumerate([
        ("p", abs.(arrays.p)),
        ("k0^2 eta p", abs.(arrays.correction_source)),
        ("P_next", fftshift1(safe_log10(arrays.P_next))),
    ])
        ax = Axis(step_gl[2, idx], title=label, xlabel=idx == 2 ? "Lateral / ky" : "", titlesize=14, xticklabelsize=10, yticklabelsize=10)
        xvals = idx == 3 ? fftshift1(arrays.k) ./ 1e3 : y_pad_mm
        lines!(ax, xvals, values; color=idx == 2 ? :orange : :black, linewidth=2)
    end
    colgap!(step_gl, 8)

    final_gl = GridLayout(gl[3, 4])
    Label(final_gl[1, 1:2], "8  Final intensity", fontsize=20, font=:bold)
    global_ref = max(maximum(intensity_corr), maximum(intensity_base), eps(Float64))
    for (idx, (name, intensity, peak)) in enumerate([
        ("corrected", intensity_corr, metrics.corrected_peak),
        ("uncorrected", intensity_base, metrics.uncorrected_peak),
    ])
        ax = Axis(final_gl[2, idx], title=name, xlabel="Lateral [mm]", ylabel=idx == 1 ? "Depth [mm]" : "", aspect=DataAspect(), yreversed=true, xticklabelsize=10, yticklabelsize=10, titlesize=14)
        heatmap!(ax, y_mm, depth_mm, (intensity ./ global_ref)'; colormap=:viridis, colorrange=(0, 1))
        mark_source_and_peak!(ax, sources, peak)
    end
    colgap!(final_gl, 8)

    colgap!(gl, 28)
    rowgap!(gl, 18)
    return save_svg(fig, "hasa_pam_workflow", generated)
end

function figure_individual_steps(rf, c, cfg, kgrid, sources, arrays, intensity_corr, intensity_base, metrics, generated; reference_sound_speed::Real)
    figure_step_geometry(c, cfg, kgrid, sources, generated)
    figure_step_rf_data(rf, cfg, kgrid, generated)
    figure_step_temporal_fft(rf, cfg, arrays, generated)
    figure_step_receiver_frequency_plane(kgrid, arrays, generated)
    figure_step_lateral_spectrum(cfg, arrays, generated)
    figure_step_eta(c, cfg, kgrid, generated; reference_sound_speed=reference_sound_speed)
    figure_step_hasa_update(cfg, kgrid, arrays, generated)
    figure_step_final_pam(intensity_corr, intensity_base, cfg, kgrid, sources, metrics, generated)
    return generated
end

function frequency_label(freqs, current_idx)
    labels = String[]
    for (idx, freq) in enumerate(freqs)
        push!(labels, idx == current_idx ? @sprintf("[%.2f]", freq / 1e6) : @sprintf("%.2f", freq / 1e6))
    end
    return "reconstructed frequencies [MHz]: " * join(labels, "  ")
end

function normalized_overlay(frame_data, frame_idx::Int, global_ref::Real)
    normalized = frame_data.intensity_frames[frame_idx] ./ Float64(global_ref)
    return map(x -> x < INTENSITY_OVERLAY_CUTOFF ? NaN : x, normalized')
end

function map_panel_width(depth_mm, y_mm, plot_height::Int)
    depth_span = maximum(depth_mm) - minimum(depth_mm)
    lateral_span = maximum(y_mm) - minimum(y_mm)
    return round(Int, plot_height * lateral_span / depth_span)
end

function animation_marching_comparison(rf, c, cfg, kgrid, sources, selected_targets, generated; filename::String, time_origin::Real, reference_sound_speed::Real)
    # CPU-style march: outer loop over frequency, inner loop over axial rows.
    corr_data = collect_marching_animation_data(
        rf,
        c,
        cfg,
        selected_targets;
        corrected=true,
        time_origin=time_origin,
        reference_sound_speed=reference_sound_speed,
    )
    base_data = collect_marching_animation_data(
        rf,
        c,
        cfg,
        selected_targets;
        corrected=false,
        time_origin=time_origin,
        reference_sound_speed=reference_sound_speed,
    )
    nframes = min(length(corr_data.row_depths), length(base_data.row_depths))
    global_ref = max(
        maximum(map(maximum, corr_data.intensity_frames)),
        maximum(map(maximum, base_data.intensity_frames)),
        eps(Float64),
    )
    depth_mm = depth_coordinates_mm(kgrid, cfg)
    y_mm = mm(kgrid.y_vec)
    y_pad_mm = padded_lateral_mm(kgrid, cfg, corr_data.crop_range, corr_data.padded_ny)
    eta_min = minimum(minimum.(corr_data.eta_frames))
    eta_max = maximum(maximum.(corr_data.eta_frames))
    frame = Observable(1)
    row_depth = @lift(corr_data.row_depths[$frame])
    corr_overlay = @lift(normalized_overlay(corr_data, $frame, global_ref))
    base_overlay = @lift(normalized_overlay(base_data, $frame, global_ref))
    eta = @lift(corr_data.eta_frames[$frame])
    corr_peak_lats = @lift(corr_data.peak_lats_frames[$frame])
    corr_peak_depths = @lift(corr_data.peak_depths_frames[$frame])
    base_peak_lats = @lift(base_data.peak_lats_frames[$frame])
    base_peak_depths = @lift(base_data.peak_depths_frames[$frame])
    title = @lift(@sprintf(
        "CPU march: frequency %d/%d, f = %.2f MHz, depth %.1f mm",
        corr_data.freq_indices[$frame],
        corr_data.nfreq,
        corr_data.freq_values[$frame] / 1e6,
        corr_data.row_depths[$frame],
    ))
    freq_text = @lift(frequency_label(corr_data.selected_freqs, corr_data.freq_indices[$frame]))

    map_height = 360
    row_gap = 28
    stack_height = 2 * map_height + row_gap
    map_width = map_panel_width(depth_mm, y_mm, map_height)
    eta_width = 520
    fig = Figure(size=(1850, 1450), backgroundcolor=RGBf(0.985, 0.985, 0.975))
    Label(fig[1, 1], "Corrected vs uncorrected CPU marching", fontsize=34, font=:bold, halign=:left)
    Label(fig[2, 1], title, fontsize=20, halign=:left)
    Label(fig[3, 1], freq_text, fontsize=18, halign=:left, font="DejaVu Sans Mono")

    plots_gl = GridLayout(fig[4, 1])
    src = source_location(sources)

    ax_eta = Axis(
        plots_gl[2, 1],
        title="Correction term eta(row)",
        xlabel="Lateral [mm]",
        ylabel="eta",
        width=eta_width,
        height=map_height,
        titlesize=24,
        titlegap=12,
    )
    lines!(ax_eta, y_pad_mm, eta; color=RGBf(0.7, 0.18, 0.10), linewidth=3)
    ylims!(ax_eta, eta_min - 0.05 * max(eta_max - eta_min, eps(Float64)), eta_max + 0.05 * max(eta_max - eta_min, eps(Float64)))

    ax_corr = Axis(
        plots_gl[2, 2],
        title="Corrected HASA",
        xlabel="Lateral [mm]",
        ylabel="Depth [mm]",
        width=map_width,
        height=map_height,
        aspect=DataAspect(),
        titlesize=24,
        titlegap=12,
        yreversed=true,
    )
    hm_c = heatmap!(ax_corr, y_mm, depth_mm, c'; colormap=:grays, colorrange=(C_WATER, C_BONE))
    hm_i = heatmap!(
        ax_corr,
        y_mm,
        depth_mm,
        corr_overlay;
        colormap=INTENSITY_COLORMAP,
        colorrange=(0, 1),
        nan_color=RGBAf(0, 0, 0, 0),
        alpha=0.82,
    )
    hlines!(ax_corr, row_depth; color=:white, linewidth=4)
    scatter!(ax_corr, [src.lateral_mm], [src.depth_mm]; marker=:star5, color=:cyan, markersize=28)
    scatter!(ax_corr, corr_peak_lats, corr_peak_depths; marker=:xcross, color=:white, markersize=28, strokewidth=4)

    ax_base = Axis(
        plots_gl[1, 2],
        title="Uncorrected ASA",
        xlabel="Lateral [mm]",
        ylabel="Depth [mm]",
        width=map_width,
        height=map_height,
        aspect=DataAspect(),
        titlesize=24,
        titlegap=12,
        yreversed=true,
    )
    heatmap!(ax_base, y_mm, depth_mm, c'; colormap=:grays, colorrange=(C_WATER, C_BONE))
    heatmap!(
        ax_base,
        y_mm,
        depth_mm,
        base_overlay;
        colormap=INTENSITY_COLORMAP,
        colorrange=(0, 1),
        nan_color=RGBAf(0, 0, 0, 0),
        alpha=0.82,
    )
    hlines!(ax_base, row_depth; color=:white, linewidth=4)
    scatter!(ax_base, [src.lateral_mm], [src.depth_mm]; marker=:star5, color=:cyan, markersize=28)
    scatter!(ax_base, base_peak_lats, base_peak_depths; marker=:xcross, color=:white, markersize=28, strokewidth=4)

    Colorbar(plots_gl[1:2, 3], hm_c, label="c [m/s]", width=18, height=stack_height)
    Colorbar(plots_gl[1:2, 4], hm_i, label="normalized intensity", width=18, height=stack_height)
    colgap!(plots_gl, 30)
    rowgap!(plots_gl, row_gap)
    rowsize!(plots_gl, 1, Fixed(map_height))
    rowsize!(plots_gl, 2, Fixed(map_height))

    path = joinpath(OUT_DIR, filename)
    CairoMakie.record(fig, path, 1:nframes; framerate=12) do idx
        frame[] = idx
    end
    push!(generated, path)
    return generated
end

function main()
    Random.seed!(20260513)
    set_theme!(THEME)
    cleanup_output_dir()
    generated = String[]

    cfg = make_cfg()
    c, rho, ct_path = load_wavefront_medium_2d(cfg)
    sources = make_sources()
    selected_targets = selected_frequency_targets()
    println("Using CT slice: $ct_path")
    println("Running k-Wave 2D RF simulation...")
    rf, kgrid, _ = simulate_point_sources(c, rho, sources, cfg; use_gpu=false)

    best = choose_reconstruction_reference(rf, c, cfg, selected_targets, sources)
    time_origin = best.time_origin
    reference_c0 = best.reference_sound_speed
    intensity_corr = best.intensity
    kgrid = best.kgrid
    intensity_base, _, _ = reconstruct_selected(
        rf,
        c,
        cfg,
        selected_targets;
        corrected=false,
        time_origin=time_origin,
        reference_sound_speed=reference_c0,
    )
    arrays = collect_algorithm_arrays(rf, c, cfg, selected_targets; time_origin=time_origin, reference_sound_speed=reference_c0)
    metrics = validate_reconstruction(intensity_corr, intensity_base, kgrid, cfg, sources)

    figure_individual_steps(rf, c, cfg, kgrid, sources, arrays, intensity_corr, intensity_base, metrics, generated; reference_sound_speed=reference_c0)
    figure_workflow(rf, c, cfg, kgrid, sources, arrays, intensity_corr, intensity_base, metrics, generated; reference_sound_speed=reference_c0)
    animation_marching_comparison(
        rf,
        c,
        cfg,
        kgrid,
        sources,
        selected_targets,
        generated;
        filename="hasa_pam_marching_corrected_vs_uncorrected.mp4",
        time_origin=time_origin,
        reference_sound_speed=reference_c0,
    )

    println(@sprintf("Reconstruction reference sound speed: %.1f m/s", reference_c0))
    println(@sprintf("Reconstruction time origin: %.3f us", time_origin * 1e6))
    println("Reconstructed frequencies:")
    for (idx, freq) in enumerate(arrays.selected_freqs)
        println(@sprintf("  %d. %.4f MHz (bin %d)", idx, freq / 1e6, arrays.selected_bins[idx]))
    end
    println("Generated HASA-PAM explainer outputs:")
    for path in generated
        println(path)
    end
end

main()
