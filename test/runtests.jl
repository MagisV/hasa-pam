using Test
using Random
using Statistics
using FFTW
using TranscranialFUS

function pam_gpu_maps_approx(a, b; rtol::Real=1e-2)
    scale = max(maximum(abs.(a)), maximum(abs.(b)), 1.0)
    return isapprox(a, b; rtol=rtol, atol=1e-5 * scale)
end

function synthetic_hu_volume(nslices::Int=5, rows::Int=80, cols::Int=60)
    hu = fill(Float32(-1000), nslices, rows, cols)
    for z in 1:nslices
        hu[z, 24:30, 20:40] .= 1200
    end
    return hu
end

function analytic_rf_for_point_source(cfg::PAMConfig, src::PointSource2D)
    kgrid = pam_grid(cfg)
    nt = pam_Nt(cfg)
    rf = zeros(Float64, kgrid.Ny, nt)
    duration = src.num_cycles / src.frequency
    base_t = collect(0:(nt - 1)) .* cfg.dt

    for j in 1:kgrid.Ny
        distance = hypot(src.depth, kgrid.y_vec[j] - src.lateral)
        arrival = src.delay + distance / cfg.c0
        local_t = base_t .- arrival
        active = findall((local_t .>= 0.0) .& (local_t .<= duration))
        isempty(active) && continue
        env = TranscranialFUS._tukey_window(length(active), 0.25)
        rf[j, active] .= (src.amplitude / sqrt(max(distance, cfg.dx))) .* env .* sin.(2π .* src.frequency .* local_t[active] .+ src.phase)
    end
    return rf, kgrid
end

function fake_pam_sweep_runner(c, rho, sources, cfg; frequencies=nothing, use_gpu=false)
    src = only(sources)
    kgrid = pam_grid(cfg)
    depth_mm = src.depth * 1e3
    lateral_mm = src.lateral * 1e3
    geo_error_mm = depth_mm / 100 + abs(lateral_mm) / 10
    hasa_error_mm = geo_error_mm / 2

    stats_geo = Dict{Symbol, Any}(
        :predicted_mm => [(depth_mm, lateral_mm + geo_error_mm)],
        :mean_radial_error_mm => geo_error_mm,
        :mean_norm_peak_intensity => 0.6,
    )
    stats_hasa = Dict{Symbol, Any}(
        :predicted_mm => [(depth_mm, lateral_mm + hasa_error_mm)],
        :mean_radial_error_mm => hasa_error_mm,
        :mean_norm_peak_intensity => 0.8,
    )

    return Dict{Symbol, Any}(
        :rf => zeros(Float64, pam_Ny(cfg), pam_Nt(cfg)),
        :kgrid => kgrid,
        :simulation => Dict{Symbol, Any}(:receiver_row => receiver_row(cfg)),
        :pam_geo => fill(geo_error_mm, pam_Nx(cfg), pam_Ny(cfg)),
        :pam_hasa => fill(hasa_error_mm, pam_Nx(cfg), pam_Ny(cfg)),
        :stats_geo => stats_geo,
        :stats_hasa => stats_hasa,
        :reconstruction_frequencies => isnothing(frequencies) ? [src.frequency] : collect(Float64.(frequencies)),
    )
end

function capture_stderr_result(f::Function)
    mktemp() do _, io
        result = redirect_stderr(io) do
            f()
        end
        flush(io)
        seekstart(io)
        return result, read(io, String)
    end
end

@testset "HU conversion" begin
    hu = Float32[-1000 0 300 1200]
    rho, c = hu_to_rho_c(hu)
    @test rho[1] ≈ 1000
    @test c[2] ≈ 1500
    @test all(rho[3:4] .>= 1000)
    @test all(c[3:4] .>= 1500)
end

@testset "Skull boundary detection" begin
    hu_slice = fill(Float32(-1000), 40, 20)
    hu_slice[12:18, 8:13] .= 900
    inner_row, outer_row = find_skull_boundaries(hu_slice)
    @test inner_row == 12
    @test outer_row == 18
end

@testset "Skull mask construction" begin
    c = fill(1500.0f0, 30, 10)
    c[8:12, :] .= 2200
    c[10, 4] = 1500
    mask = skull_mask_from_c_columnwise(c; mask_outside=true)
    @test !mask[8, 1]
    @test mask[8, 4]
    @test mask[20, 4]
    @test !mask[3, 4]
end

@testset "Medium construction" begin
    hu_vol = synthetic_hu_volume()
    cfg = SimulationConfig(z_focus=0.05, dx=1e-3, dz=1e-3, transverse_dim=0.06, trans_aperture=0.03)
    c, rho, info = make_medium_fixed_transducer(hu_vol, cfg, SKULL_IN_WATER; slice_index=2)
    @test size(c) == (Nx(cfg), Nz(cfg))
    @test size(rho) == size(c)
    @test info[:z_trans_idx] == Nx(cfg) - cfg.PML_GUARD
    @test info[:outer_row] < info[:z_trans_idx]

    cfg2 = SimulationConfig(
        z_focus=0.03,
        dx=1e-3,
        dz=1e-3,
        transverse_dim=0.06,
        trans_aperture=0.03,
        focus_depth_from_inner_skull=0.005,
    )
    c2, rho2, info2 = make_medium_fixed_distance_from_skull(hu_vol, cfg2, SKULL_IN_WATER; slice_index=2)
    @test size(c2, 2) == Nz(cfg2)
    @test info2[:z_trans_idx] - info2[:target_idx] == Nx_hasa(cfg2)
    @test info2[:z_trans_idx] - info2[:outer_row] >= 2
    @test size(rho2) == size(c2)
end

@testset "Geometric delay symmetry" begin
    cfg = SimulationConfig(z_focus=0.03, dx=1e-3, dz=1e-3, transverse_dim=0.06, trans_aperture=0.021, axial_padding=2.0)
    cfg.trans_index = Nx(cfg) - cfg.PML_GUARD
    c = fill(Float32(cfg.c0), Nx(cfg), Nz(cfg))
    rho = fill(Float32(cfg.rho0), Nx(cfg), Nz(cfg))
    _, hasa_info, _ = focus(
        c,
        rho,
        GEOMETRIC,
        cfg,
        SweepSettings();
        animation_settings=AnimationSettings(run_kwave=false, Nt=200),
    )
    tau = hasa_info[:tau]
    @test tau ≈ reverse(tau) atol=1e-9
    @test all(hasa_info[:amplitudes] .≈ 1.0)
end

@testset "Placement resolution" begin
    mode, depth = resolve_placement_mode(:auto, SKULL_IN_WATER)
    @test mode == :fixed_focus_depth
    @test depth ≈ 30e-3

    mode, depth = resolve_placement_mode(:auto, WATER)
    @test mode == :fixed_transducer
    @test isnothing(depth)

    mode, depth = resolve_placement_mode(:fixed_transducer, SKULL_IN_WATER)
    @test mode == :fixed_transducer
    @test isnothing(depth)

    mode, depth = resolve_placement_mode(:fixed_focus_depth, SKULL_IN_WATER)
    @test mode == :fixed_focus_depth
    @test depth ≈ 30e-3

    mode, depth = resolve_placement_mode(:fixed_focus_depth, SKULL_IN_WATER; focus_depth_from_inner_skull=20e-3)
    @test mode == :fixed_focus_depth
    @test depth ≈ 20e-3

    mode, depth = resolve_placement_mode(:auto, SKULL_IN_WATER; focus_depth_from_inner_skull=25e-3)
    @test mode == :fixed_focus_depth
    @test depth ≈ 25e-3

    @test_throws ErrorException resolve_placement_mode(:fixed_transducer, SKULL_IN_WATER; focus_depth_from_inner_skull=20e-3)
    @test_throws ErrorException resolve_placement_mode(:fixed_focus_depth, WATER)
    @test_throws ErrorException resolve_placement_mode("bad_mode", WATER)
end

@testset "Phase unwrapping" begin
    truth = collect(range(0.0, 4π; length=17))
    wrapped = mod.(truth .+ π, 2π) .- π
    @test TranscranialFUS._unwrap_phase(wrapped) ≈ truth atol=1e-10
end

@testset "Focus analysis" begin
    cfg = SimulationConfig(z_focus=0.02, x_focus=0.0, dx=0.5e-3, dz=0.5e-3, transverse_dim=0.03, trans_aperture=0.01, axial_padding=2.5)
    cfg.trans_index = Nx(cfg) - cfg.PML_GUARD
    kgrid = KGrid2D(Nx(cfg), Nz(cfg), cfg.dx, cfg.dz; dt=cfg.dt, Nt=Nt(cfg))
    row_tgt = target_index(cfg)
    col_tgt = fld(length(kgrid.y_vec), 2) + 1

    p = Array{Float64}(undef, kgrid.Nx, kgrid.Ny)
    σ = 1.5e-3
    for i in 1:kgrid.Nx, j in 1:kgrid.Ny
        p[i, j] = exp(-((kgrid.x_vec[i] - kgrid.x_vec[row_tgt])^2 + (kgrid.y_vec[j] - kgrid.y_vec[col_tgt])^2) / (2σ^2))
    end

    stats = analyse_focus_2d(p, kgrid, cfg)
    @test stats[:error_mm] < 0.6
    @test stats[:p_peak] ≈ maximum(p)
    @test stats[:focal_area_mm2] > 0
end

@testset "PAM reconstruction in water" begin
    cfg = PAMConfig(
        dx=0.5e-3,
        dz=0.5e-3,
        axial_dim=0.04,
        transverse_dim=0.03,
        receiver_aperture=nothing,
        t_max=40e-6,
        dt=50e-9,
        PML_GUARD=5,
        zero_pad_factor=2,
        peak_suppression_radius=1e-3,
        success_tolerance=1.0e-3,
    )
    source = PointSource2D(depth=0.015, lateral=0.003, frequency=0.4e6, amplitude=1.0, num_cycles=5)
    c, _, _ = make_pam_medium(cfg; aberrator=:none)
    rf, kgrid = analytic_rf_for_point_source(cfg, source)
    cuda_ok = TranscranialFUS._pam_cuda_functional()

    quiet_result, quiet_progress = capture_stderr_result() do
        reconstruct_pam(rf, c, cfg; frequencies=[source.frequency], corrected=false)
    end
    intensity, _, info = quiet_result
    @test quiet_progress == ""
    stats = analyse_pam_2d(intensity, kgrid, cfg, [source])
    hasa_result, hasa_progress = capture_stderr_result() do
        reconstruct_pam(
            rf,
            c,
            cfg;
            frequencies=[source.frequency],
            corrected=true,
            use_gpu=cuda_ok,
            show_progress=true,
        )
    end
    intensity_hasa, _, info_hasa = hasa_result
    stats_hasa = analyse_pam_2d(intensity_hasa, kgrid, cfg, [source])
    one_window = PAMWindowConfig(
        enabled=true,
        window_duration=pam_Nt(cfg) * cfg.dt,
        hop=pam_Nt(cfg) * cfg.dt,
        taper=:none,
        min_energy_ratio=0.0,
    )
    windowed_result, windowed_progress = capture_stderr_result() do
        reconstruct_pam_windowed(
            rf,
            c,
            cfg;
            frequencies=[source.frequency],
            corrected=false,
            window_config=one_window,
            use_gpu=cuda_ok,
            show_progress=true,
        )
    end
    intensity_windowed, _, info_windowed = windowed_result
    cropped_range = 101:500
    cropped_origin = (first(cropped_range) - 1) * cfg.dt
    intensity_cropped, _, info_cropped = reconstruct_pam(
        rf[:, cropped_range],
        c,
        cfg;
        frequencies=[source.frequency],
        use_gpu=cuda_ok,
        corrected=false,
        time_origin=cropped_origin,
    )
    stats_cropped = analyse_pam_2d(intensity_cropped, kgrid, cfg, [source])

    @test info[:corrected] == false
    @test info[:show_progress] == false
    @test stats[:mean_radial_error_mm] < 1.0
    @test stats[:success_rate] == 1.0
    @test stats[:mean_norm_peak_intensity] > 0.5
    @test info_hasa[:corrected] == true
    @test info_hasa[:use_gpu] == cuda_ok
    @test info_hasa[:backend] == (cuda_ok ? :cuda : :cpu)
    @test info_hasa[:gpu_precision] == (cuda_ok ? Float32 : nothing)
    @test info_hasa[:show_progress] == true
    @test occursin("PAM HASA reconstruction", hasa_progress)
    @test occursin("frequency 1/", hasa_progress)
    @test occursin("total elapsed", hasa_progress)
    @test stats_hasa[:mean_radial_error_mm] < 1.0
    @test stats_hasa[:success_rate] == 1.0
    if cuda_ok
        @test pam_gpu_maps_approx(intensity_windowed, intensity)
        intensity_hasa_cpu, _, _ = reconstruct_pam(
            rf,
            c,
            cfg;
            frequencies=[source.frequency],
            corrected=true,
            use_gpu=false,
        )
        @test pam_gpu_maps_approx(intensity_hasa, intensity_hasa_cpu)

        c_lens = copy(c)
        c_lens[20:30, 25:35] .= 1700.0
        intensity_lens_cpu, _, _ = reconstruct_pam(
            rf,
            c_lens,
            cfg;
            frequencies=[source.frequency],
            corrected=true,
            use_gpu=false,
        )
        intensity_lens_gpu, _, info_lens_gpu = reconstruct_pam(
            rf,
            c_lens,
            cfg;
            frequencies=[source.frequency],
            corrected=true,
            use_gpu=true,
        )
        @test info_lens_gpu[:backend] == :cuda
        @test pam_gpu_maps_approx(intensity_lens_gpu, intensity_lens_cpu)
    else
        @test intensity_windowed ≈ intensity
        err = try
            reconstruct_pam(rf, c, cfg; frequencies=[source.frequency], corrected=false, use_gpu=true)
            nothing
        catch err
            err
        end
        @test err isa ErrorException
        @test occursin("functional NVIDIA CUDA GPU", sprint(showerror, err))
    end
    @test info_windowed[:use_gpu] == cuda_ok
    @test info_windowed[:backend] == (cuda_ok ? :cuda : :cpu)
    @test info_windowed[:show_progress] == true
    @test occursin("PAM geometric ASA windowed reconstruction", windowed_progress)
    @test occursin("window 1/1", windowed_progress)
    @test occursin("complete", windowed_progress)
    @test info_windowed[:used_window_count] == 1
    @test info_windowed[:skipped_window_count] == 0
    @test info_cropped[:time_origin] ≈ cropped_origin
    @test info_cropped[:use_gpu] == cuda_ok
    @test info_cropped[:backend] == (cuda_ok ? :cuda : :cpu)
    @test stats_cropped[:success_rate] == 1.0

    cached_results = reconstruct_pam_case(
        rf,
        c,
        [source],
        cfg;
        simulation_info=Dict(:receiver_row => receiver_row(cfg), :receiver_cols => receiver_col_range(cfg)),
        frequencies=[source.frequency],
        use_gpu=cuda_ok,
    )
    @test cached_results[:simulation][:receiver_row] == receiver_row(cfg)
    @test cached_results[:use_gpu] == cuda_ok
    @test cached_results[:show_progress] == false
    @test cached_results[:geo_info][:use_gpu] == cuda_ok
    @test cached_results[:hasa_info][:use_gpu] == cuda_ok
    @test cached_results[:geo_info][:backend] == (cuda_ok ? :cuda : :cpu)
    @test cached_results[:hasa_info][:backend] == (cuda_ok ? :cuda : :cpu)
    @test cached_results[:stats_geo][:success_rate] == 1.0
    @test cached_results[:stats_hasa][:success_rate] == 1.0

    windowed_results = reconstruct_pam_case(
        rf,
        c,
        [source],
        cfg;
        simulation_info=Dict(:receiver_row => receiver_row(cfg), :receiver_cols => receiver_col_range(cfg)),
        frequencies=[source.frequency],
        reconstruction_mode=:windowed,
        window_config=one_window,
    )
    @test windowed_results[:reconstruction_mode] == :windowed
    @test windowed_results[:geo_info][:used_window_count] == 1

    duplicate_source_events = [source, PointSource2D(depth=source.depth, lateral=source.lateral, frequency=source.frequency)]
    truth_mask = pam_truth_mask([source], kgrid, cfg; radius=cfg.success_tolerance)
    event_results = reconstruct_pam_case(
        rf,
        c,
        duplicate_source_events,
        cfg;
        simulation_info=Dict(:receiver_row => receiver_row(cfg), :receiver_cols => receiver_col_range(cfg)),
        frequencies=[source.frequency],
        analysis_mode=:detection,
        detection_truth_mask=truth_mask,
        analysis_sources=[source],
    )
    @test event_results[:analysis_source_count] == 1
    @test event_results[:stats_geo][:num_truth_sources] == 1

    pam_script = read(joinpath(@__DIR__, "..", "scripts", "run_pam.jl"), String)
    @test occursin("\"recon-progress\" => \"false\"", pam_script)
    @test occursin("show_progress=parse_bool(opts[\"recon-progress\"])", pam_script)
end

@testset "PAM windowing helpers" begin
    cfg = PAMWindowConfig(enabled=true, window_duration=10e-6, hop=5e-6)
    exact_ranges, exact_win, exact_hop = TranscranialFUS._pam_window_ranges(100, 0.1e-6, cfg)
    @test exact_ranges == [1:100]
    @test exact_win == 100
    @test exact_hop == 50

    overlap_ranges, overlap_win, overlap_hop = TranscranialFUS._pam_window_ranges(250, 0.1e-6, cfg)
    @test overlap_win == 100
    @test overlap_hop == 50
    @test overlap_ranges == [1:100, 51:150, 101:200, 151:250]

    short_ranges, short_win, short_hop = TranscranialFUS._pam_window_ranges(40, 0.1e-6, cfg)
    @test short_ranges == [1:40]
    @test short_win == 40
    @test short_hop == 50

    p1 = fill(1.0 + 0.0im, 2, 2)
    p2 = fill(-1.0 + 0.0im, 2, 2)
    @test all(abs2.(p1 .+ p2) .== 0.0)
    @test all(abs2.(p1) .+ abs2.(p2) .== 2.0)
    @test TranscranialFUS.pam_reconstruction_mode(:auto, :squiggle) == :windowed
    @test TranscranialFUS.pam_reconstruction_mode(:auto, :point) == :full
    @test TranscranialFUS.pam_reconstruction_mode(:full, :squiggle) == :full
end

@testset "PAM analysis metrics" begin
    cfg = PAMConfig(
        dx=0.5e-3,
        dz=0.5e-3,
        axial_dim=0.05,
        transverse_dim=0.04,
        receiver_aperture=nothing,
        PML_GUARD=5,
        peak_suppression_radius=2e-3,
        success_tolerance=1.5e-3,
    )
    kgrid = pam_grid(cfg)
    depth = depth_coordinates(kgrid, cfg)
    lateral = kgrid.y_vec
    sources = [
        PointSource2D(depth=0.015, lateral=-0.004, frequency=0.4e6),
        PointSource2D(depth=0.028, lateral=0.006, frequency=0.4e6),
    ]

    intensity = zeros(Float64, kgrid.Nx, kgrid.Ny)
    σd = 0.8e-3
    σl = 0.8e-3
    for src in sources
        for i in 1:kgrid.Nx, j in 1:kgrid.Ny
            intensity[i, j] += exp(-((depth[i] - src.depth)^2 / (2σd^2) + (lateral[j] - src.lateral)^2 / (2σl^2)))
        end
    end

    stats = analyse_pam_2d(intensity, kgrid, cfg, sources)
    @test stats[:mean_radial_error_mm] < 0.6
    @test stats[:num_success] == 2
    @test length(stats[:axial_fwhm_mm]) == 2
    @test all(stats[:axial_fwhm_mm] .> 0)
    @test all(stats[:lateral_fwhm_mm] .> 0)

    metrics = pam_intensity_metrics(intensity, kgrid, cfg; threshold_ratio=0.5, reference_intensity=2.0)
    @test metrics[:peak_intensity] ≈ maximum(intensity)
    @test metrics[:relative_peak_intensity] ≈ maximum(intensity) / 2.0
    @test metrics[:integrated_intensity_m2] > 0
    @test metrics[:active_area_mm2] > 0
    @test isfinite(metrics[:centroid_depth_mm])
    @test isfinite(metrics[:centroid_lateral_mm])
end

@testset "Gaussian pulse cluster emissions" begin
    dt = 20e-9
    nt = 1000
    src = GaussianPulseCluster2D(
        depth=0.03,
        lateral=0.0,
        fundamental=0.5e6,
        amplitude=1.0,
        n_bubbles=1.0,
        harmonics=[2],
        harmonic_amplitudes=[1.0],
        harmonic_phases=[0.0],
        gate_duration=10e-6,
    )
    signal = TranscranialFUS._source_signal(nt, dt, src)
    active = findall(!iszero, signal)

    @test !isempty(active)
    @test maximum(abs, signal) > 0.5
    @test abs(signal[first(active)]) < 0.05 * maximum(abs, signal)
    @test abs(signal[last(active)]) < 0.05 * maximum(abs, signal)
    @test emission_frequencies(src) == [1.0e6]
    @test cavitation_model(src) == :gaussian_pulse

    spectrum = abs.(fft(signal))
    freq_axis = collect(0:(nt - 1)) ./ (nt * dt)
    pos_bins = 2:(fld(nt, 2) + 1)
    peak_bin = pos_bins[argmax(spectrum[pos_bins])]
    @test abs(freq_axis[peak_bin] - 1.0e6) <= 1 / (nt * dt)

    harmonic = BubbleCluster2D(
        depth=0.03,
        lateral=0.0,
        fundamental=0.5e6,
        harmonics=[2],
        harmonic_amplitudes=[1.0],
        harmonic_phases=[0.0],
        gate_duration=10e-6,
    )
    @test cavitation_model(harmonic) == :harmonic_cos
    @test TranscranialFUS._normalize_cavitation_model("harmonic-cos") == :harmonic_cos
    @test TranscranialFUS._normalize_cavitation_model("gaussian-pulse") == :gaussian_pulse
    @test_throws ErrorException TranscranialFUS._normalize_cavitation_model("haromnic-cos")
end

@testset "Source phase modes" begin
    @test TranscranialFUS._normalize_source_phase_mode(:coherent) == :coherent
    @test TranscranialFUS._normalize_source_phase_mode(:random_static_phase) == :random_static_phase
    @test TranscranialFUS._normalize_source_phase_mode(:random_phase_per_window) == :random_phase_per_window
    @test TranscranialFUS._normalize_source_phase_mode(:random_phase_per_realization) == :random_phase_per_realization
    @test TranscranialFUS._normalize_source_phase_mode("random-phase-per-realization") == :random_phase_per_realization
    @test_throws ErrorException TranscranialFUS._normalize_source_phase_mode(:unknown_mode)

    @test TranscranialFUS._normalize_cluster_phase_mode(:random_static_phase) == :random
    @test TranscranialFUS._normalize_cluster_phase_mode("random_static_phase") == :random
    @test TranscranialFUS._normalize_cluster_phase_mode(:coherent) == :coherent

    rng_r = Random.MersenneTwister(7)
    sources_orig = [
        BubbleCluster2D(depth=0.03, lateral=0.0, fundamental=0.5e6,
            harmonics=[2, 3], harmonic_amplitudes=[1.0, 0.6],
            harmonic_phases=[0.1, 0.2], gate_duration=10e-6),
        PointSource2D(depth=0.02, lateral=0.005, frequency=1.0e6, phase=0.5),
        GaussianPulseCluster2D(depth=0.04, lateral=-0.005, fundamental=0.5e6,
            harmonics=[2], harmonic_amplitudes=[1.0], harmonic_phases=[0.3],
            gate_duration=10e-6),
    ]
    resampled = TranscranialFUS._resample_source_phases(sources_orig, rng_r)

    @test resampled[1].depth == sources_orig[1].depth
    @test resampled[1].lateral == sources_orig[1].lateral
    @test resampled[1].harmonic_phases != sources_orig[1].harmonic_phases
    @test resampled[2].phase != sources_orig[2].phase
    @test resampled[3].harmonic_phases != sources_orig[3].harmonic_phases

end

@testset "SourceVariabilityConfig" begin
    rng = Random.MersenneTwister(42)
    src = BubbleCluster2D(depth=0.03, lateral=0.0, fundamental=0.5e6,
        harmonics=[2, 3], harmonic_amplitudes=[1.0, 0.6],
        harmonic_phases=[0.1, 0.2], gate_duration=50e-6)

    expanded, n = TranscranialFUS._expand_sources_per_window(
        [src], 10e-6, 5e-6, 80e-6, Random.MersenneTwister(1))
    @test n == 15
    @test length(expanded) == 15
    @test all(s.amplitude == src.amplitude for s in expanded)
    @test all(s.fundamental == src.fundamental for s in expanded)

    # frequency jitter: fundamentals vary across copies
    exp_fj, _ = TranscranialFUS._expand_sources_per_window(
        [src], 10e-6, 5e-6, 80e-6, Random.MersenneTwister(99);
        variability=SourceVariabilityConfig(frequency_jitter_fraction=0.05))
    @test length(unique(round.(Float64[s.fundamental for s in exp_fj]; digits=0))) > 1
end

@testset "Squiggle bubble sources" begin
    squiggle_clusters, squiggle_meta = make_squiggle_bubble_sources(
        [(0.03, 0.0)];
        root_length=12e-3,
        squiggle_amplitude=1.5e-3,
        squiggle_wavelength=6e-3,
        source_spacing=1e-3,
        position_jitter=0.0,
        min_separation=0.0,
        lateral_bounds=(-0.02, 0.02),
        rng=Random.MersenneTwister(41),
    )

    @test squiggle_meta[:source_model] == :squiggle
    @test squiggle_meta[:cavitation_model] == :harmonic_cos
    @test all(src -> src isa BubbleCluster2D, squiggle_clusters)
    @test length(squiggle_meta[:centerlines]) == 1
    @test maximum(src.lateral for src in squiggle_clusters) - minimum(src.lateral for src in squiggle_clusters) > 10e-3
    @test maximum(src.depth for src in squiggle_clusters) - minimum(src.depth for src in squiggle_clusters) > 2e-3

    pulse_clusters, pulse_meta = make_squiggle_bubble_sources(
        [(0.03, 0.0)];
        cavitation_model=:gaussian_pulse,
        root_length=12e-3,
        squiggle_amplitude=1.5e-3,
        squiggle_wavelength=6e-3,
        source_spacing=1e-3,
        position_jitter=0.0,
        min_separation=0.0,
        lateral_bounds=(-0.02, 0.02),
        rng=Random.MersenneTwister(41),
    )

    @test pulse_meta[:cavitation_model] == :gaussian_pulse
    @test all(src -> src isa GaussianPulseCluster2D, pulse_clusters)
    @test length(pulse_clusters) == length(squiggle_clusters)

    multi_clusters, multi_meta = make_squiggle_bubble_sources(
        [(0.03, -0.004), (0.035, 0.004)];
        root_length=8e-3,
        source_spacing=1e-3,
        position_jitter=0.0,
        min_separation=0.0,
        max_sources_per_anchor=20,
        lateral_bounds=(-0.02, 0.02),
        rng=Random.MersenneTwister(43),
    )
    @test length(multi_clusters) > length(squiggle_clusters)
    @test multi_meta[:source_model] == :squiggle
    @test length(multi_meta[:centerlines]) == 2
end

@testset "PAM detection metrics" begin
    cfg = PAMConfig(
        dx=0.5e-3,
        dz=0.5e-3,
        axial_dim=0.05,
        transverse_dim=0.04,
        receiver_aperture=nothing,
        PML_GUARD=5,
        success_tolerance=1.0e-3,
    )
    kgrid = pam_grid(cfg)
    sources = [
        PointSource2D(depth=0.015, lateral=-0.004, frequency=0.4e6),
        PointSource2D(depth=0.028, lateral=0.006, frequency=0.4e6),
    ]
    intensity = zeros(Float64, kgrid.Nx, kgrid.Ny)
    row1, col1 = source_grid_index(sources[1], cfg, kgrid)
    row_false, col_false = source_grid_index(PointSource2D(depth=0.038, lateral=-0.012), cfg, kgrid)
    intensity[row1, col1] = 1.0
    intensity[row_false, col_false] = 0.8

    stats = analyse_pam_detection_2d(
        intensity,
        kgrid,
        cfg,
        sources;
        truth_radius=1.0e-3,
        threshold_ratio=0.5,
        frequencies=[0.4e6],
        psf_axial_fwhm=2.0e-3,
        psf_lateral_fwhm=2.0e-3,
    )

    @test 0 < stats[:precision] < 1
    @test 0 < stats[:recall] < 1
    @test stats[:false_positive_pixels] > 0
    @test stats[:false_negative_pixels] > 0
    @test stats[:spurious_prediction_components] == 1
    @test 0 < stats[:energy_fraction_inside_mask] < 1
    @test stats[:energy_fraction_inside_mask] + stats[:energy_fraction_outside_mask] ≈ 1.0
    @test stats[:energy_fraction_inside_predicted_mask] > stats[:energy_fraction_inside_mask]
    @test stats[:energy_fraction_inside_predicted_mask] + stats[:energy_fraction_outside_predicted_mask] ≈ 1.0
    @test isfinite(stats[:centroid_error_mm])
    @test stats[:axial_spread_mm] > 0
    @test stats[:lateral_spread_mm] > 0
    @test haskey(stats, :psf_target_correlation)
    @test isfinite(stats[:psf_target_normalized_l2_error])

    truth_override = falses(kgrid.Nx, kgrid.Ny)
    truth_override[row_false, col_false] = true
    override_stats = analyse_pam_detection_2d(
        intensity,
        kgrid,
        cfg,
        sources;
        truth_radius=1.0e-3,
        threshold_ratio=0.5,
        truth_mask=truth_override,
        psf_axial_fwhm=2.0e-3,
        psf_lateral_fwhm=2.0e-3,
    )
    @test override_stats[:truth_mask_mode] == :provided
    @test override_stats[:psf_target_mode] == :provided_mask
    @test override_stats[:true_positive_pixels] == 1
    @test override_stats[:false_positive_pixels] == 1
    @test override_stats[:false_negative_pixels] == 0

    source_map = pam_source_map(sources, kgrid, cfg; weights=:uniform)
    @test sum(source_map) == length(sources)
    blurred_truth = pam_psf_blurred_truth_map(
        sources,
        kgrid,
        cfg;
        psf_axial_fwhm=2.0e-3,
        psf_lateral_fwhm=2.0e-3,
        weights=:uniform,
    )
    @test size(blurred_truth) == size(intensity)
    @test sum(blurred_truth) ≈ sum(source_map) atol=1e-8
end

@testset "CLEAN peak detection" begin
    # Two sources whose focal shoulders overlap: argmax with a suppression
    # radius smaller than the focus picks sidelobes of one source instead of
    # the second source. CLEAN should find both.
    cfg = PAMConfig(
        dx=0.5e-3,
        dz=0.5e-3,
        axial_dim=0.05,
        transverse_dim=0.04,
        receiver_aperture=40e-3,
        PML_GUARD=5,
        peak_suppression_radius=1e-3,   # smaller than focus FWHM
        success_tolerance=2e-3,
    )
    kgrid = pam_grid(cfg)
    depth = depth_coordinates(kgrid, cfg)
    lateral = kgrid.y_vec
    truths = [
        PointSource2D(depth=0.02, lateral=-0.003, frequency=0.4e6),
        PointSource2D(depth=0.02, lateral=0.003, frequency=0.4e6),
    ]
    intensity = zeros(Float64, kgrid.Nx, kgrid.Ny)
    σd = 3e-3  # broad axial focus to break argmax
    σl = 2e-3
    # unequal amplitudes to mimic a real coherent-interference scene
    amps = (1.0, 0.7)
    for (src, amp) in zip(truths, amps)
        for i in 1:kgrid.Nx, j in 1:kgrid.Ny
            intensity[i, j] += amp * exp(-((depth[i] - src.depth)^2 / (2σd^2) + (lateral[j] - src.lateral)^2 / (2σl^2)))
        end
    end

    stats_argmax = analyse_pam_2d(intensity, kgrid, cfg, truths; peak_method=:argmax)
    stats_clean = analyse_pam_2d(
        intensity, kgrid, cfg, truths;
        peak_method=:clean,
        frequencies=[0.4e6],
        clean_psf_axial_fwhm=2.355 * σd,
        clean_psf_lateral_fwhm=2.355 * σl,
    )

    @test stats_clean[:mean_radial_error_mm] < stats_argmax[:mean_radial_error_mm]
    @test stats_clean[:num_success] == 2
end

@testset "PAM sweep target presets" begin
    preset, axial, lateral = TranscranialFUS._resolve_pam_sweep_targets(:paper)
    @test preset == :paper
    @test axial == [30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    @test lateral == [-20.0, -10.0, 0.0, 10.0, 20.0]

    preset, axial, lateral = TranscranialFUS._resolve_pam_sweep_targets(:quick)
    @test preset == :quick
    @test axial == [40.0, 60.0, 80.0]
    @test lateral == [-10.0, 0.0, 10.0]

    preset, axial, lateral = TranscranialFUS._resolve_pam_sweep_targets(
        :paper;
        axial_targets_mm=[55.0, 35.0],
        lateral_targets_mm=[10.0, -5.0, 0.0],
    )
    @test preset == :custom
    @test axial == [35.0, 55.0]
    @test lateral == [-5.0, 0.0, 10.0]

    @test_throws ErrorException TranscranialFUS._resolve_pam_sweep_targets(:custom)
    @test_throws ErrorException TranscranialFUS._resolve_pam_sweep_targets(:paper; axial_targets_mm=[40.0])
end

@testset "PAM sweep example selection" begin
    targets = vec([
        PointSource2D(depth=axial_mm * 1e-3, lateral=lateral_mm * 1e-3, frequency=1e6)
        for axial_mm in (40.0, 60.0, 80.0), lateral_mm in (-10.0, 0.0, 10.0)
    ])
    examples = TranscranialFUS._default_pam_sweep_examples(targets)
    @test examples == [(40.0, 0.0), (60.0, 0.0), (80.0, 0.0)]
end

@testset "PAM sweep aggregation" begin
    cfg = PAMConfig(
        dx=1e-3,
        dz=1e-3,
        axial_dim=0.09,
        transverse_dim=0.05,
        receiver_aperture=0.03,
        t_max=20e-6,
        dt=100e-9,
    )
    targets = vec([
        PointSource2D(depth=axial_mm * 1e-3, lateral=lateral_mm * 1e-3, frequency=1e6)
        for axial_mm in (40.0, 60.0), lateral_mm in (-10.0, 0.0, 10.0)
    ])
    c, rho, _ = make_pam_medium(cfg; aberrator=:none)
    sweep = run_pam_sweep(
        c,
        rho,
        targets,
        cfg;
        frequencies=[1e6],
        example_targets_mm=[(40.0, 0.0), (60.0, 0.0)],
        runner=fake_pam_sweep_runner,
    )

    @test sweep[:axial_targets_mm] == [40.0, 60.0]
    @test sweep[:lateral_targets_mm] == [-10.0, 0.0, 10.0]
    @test size(sweep[:geo_error_mm]) == (2, 3)
    @test size(sweep[:hasa_error_mm]) == (2, 3)
    @test sweep[:geo_error_mm][1, 1] ≈ 1.4
    @test sweep[:geo_error_mm][2, 3] ≈ 1.6
    @test sweep[:hasa_error_mm][1, 2] ≈ 0.2
    @test length(sweep[:cases]) == 6
    @test length(sweep[:example_cases]) == 2
    @test sweep[:example_targets_mm] == [(40.0, 0.0), (60.0, 0.0)]
end

@testset "PAM external PML sizing" begin
    cfg_fine = PAMConfig(dx=0.2e-3, dz=0.2e-3, axial_dim=0.03, transverse_dim=0.03)
    cfg_coarse = PAMConfig(dx=0.5e-3, dz=0.5e-3, axial_dim=0.03, transverse_dim=0.03)

    @test TranscranialFUS._pam_pml_guard(cfg_fine) == 20
    @test TranscranialFUS._pam_pml_guard(cfg_coarse) == 8
    @test receiver_row(cfg_coarse) == 1

    kgrid = pam_grid(cfg_coarse)
    src = PointSource2D(depth=0.015, lateral=0.003, frequency=0.4e6)
    row, _ = source_grid_index(src, cfg_coarse, kgrid)
    @test row == 31
    @test row <= pam_Nx(cfg_coarse)
end

@testset "PAM skull sweep setup" begin
    base_cfg = PAMConfig(dx=1e-3, dz=1e-3, axial_dim=0.04, transverse_dim=0.06)
    targets = [
        PointSource2D(depth=axial_mm * 1e-3, lateral=0.0, frequency=1e6)
        for axial_mm in (40.0, 60.0, 80.0)
    ]
    fitted_cfg = fit_pam_config(
        base_cfg,
        targets;
        min_bottom_margin=5e-3,
        reference_depth=30e-3,
    )
    hu_vol = synthetic_hu_volume()
    c, _, info = make_pam_medium(
        fitted_cfg;
        aberrator=:skull,
        hu_vol=hu_vol,
        spacing_m=(1e-3, 1e-3, 1e-3),
        slice_index=2,
        skull_to_transducer=30e-3,
        hu_bone_thr=200,
    )

    @test info[:outer_row] == receiver_row(fitted_cfg) + 30
    @test info[:outer_row] < info[:inner_row]
    for src in targets
        row, col = source_grid_index(src, fitted_cfg, pam_grid(fitted_cfg))
        @test row > info[:inner_row]
        @test row <= pam_Nx(fitted_cfg)
        @test c[row, col] ≈ fitted_cfg.c0
    end
end

@testset "PAM skull target filtering" begin
    cfg = PAMConfig(dx=1e-3, dz=1e-3, axial_dim=0.09, transverse_dim=0.06)
    hu_vol = synthetic_hu_volume()
    c, _, info = make_pam_medium(
        cfg;
        aberrator=:skull,
        hu_vol=hu_vol,
        spacing_m=(1e-3, 1e-3, 1e-3),
        slice_index=2,
        skull_to_transducer=30e-3,
        hu_bone_thr=200,
    )

    targets = PointSource2D[
        PointSource2D(depth=(info[:inner_row] - 2) * 1e-3, lateral=0.0, frequency=1e6),
        PointSource2D(depth=(info[:inner_row] + 5) * 1e-3, lateral=0.0, frequency=1e6),
        PointSource2D(depth=(info[:inner_row] + 8) * 1e-3, lateral=25e-3, frequency=1e6),
    ]

    valid_targets, dropped_targets, cavity_start_rows = TranscranialFUS._filter_pam_targets_in_skull_cavity(
        c,
        cfg,
        targets;
        min_margin=1e-3,
    )

    @test length(valid_targets) == 1
    @test only(valid_targets).lateral ≈ 0.0
    @test length(dropped_targets) == 2
    @test Set(drop[:reason] for drop in dropped_targets) == Set((:too_shallow_for_cavity, :no_skull_above))
    center_col = source_grid_index(only(valid_targets), cfg, pam_grid(cfg))[2]
    @test cavity_start_rows[center_col] == info[:inner_row] + 2
end

@testset "PAM config fitting and skull placement" begin
    base_cfg = PAMConfig(dx=1e-3, dz=1e-3, axial_dim=0.03, transverse_dim=0.06)
    sources = [PointSource2D(depth=0.04, lateral=0.0)]
    fitted_cfg = fit_pam_config(
        base_cfg,
        sources;
        min_bottom_margin=5e-3,
        reference_depth=30e-3,
    )

    @test pam_Nx(fitted_cfg) == 46
    @test fitted_cfg.axial_dim ≈ 46e-3

    hu_vol = synthetic_hu_volume()
    c, rho, info = make_pam_medium(
        fitted_cfg;
        aberrator=:skull,
        hu_vol=hu_vol,
        spacing_m=(1e-3, 1e-3, 1e-3),
        slice_index=2,
        skull_to_transducer=30e-3,
        hu_bone_thr=200,
    )

    @test size(c) == (pam_Nx(fitted_cfg), pam_Ny(fitted_cfg))
    @test size(rho) == size(c)
    @test info[:outer_row] == receiver_row(fitted_cfg) + 30
    @test info[:outer_row] < info[:inner_row]
    @test maximum(c[info[:outer_row]:info[:inner_row], :]) > fitted_cfg.c0

    source_row, source_col = source_grid_index(first(sources), fitted_cfg, pam_grid(fitted_cfg))
    @test c[source_row, source_col] ≈ fitted_cfg.c0
    @test all(c[(source_row + 1):end, source_col] .≈ fitted_cfg.c0)
end

@testset "PAM deep-domain fitting is padding-stable" begin
    base_cfg = PAMConfig(
        dx=1e-3,
        dz=1e-3,
        axial_dim=0.08,
        transverse_dim=0.06,
        receiver_aperture=0.04,
        t_max=80e-6,
        dt=50e-9,
    )
    cluster = BubbleCluster2D(
        depth=0.08,
        lateral=0.0,
        fundamental=0.5e6,
        gate_duration=50e-6,
    )
    fitted_cfg = fit_pam_config(base_cfg, [cluster]; min_bottom_margin=5e-3)
    @test fitted_cfg.t_max >= TranscranialFUS._required_pam_t_max(base_cfg, [cluster])
    @test fitted_cfg.t_max > 110e-6

    cfg80 = PAMConfig(dx=1e-3, dz=1e-3, axial_dim=0.08, transverse_dim=0.03)
    cfg200 = PAMConfig(dx=1e-3, dz=1e-3, axial_dim=0.20, transverse_dim=0.03)
    c80 = fill(cfg80.c0, pam_Nx(cfg80), pam_Ny(cfg80))
    c200 = fill(cfg200.c0, pam_Nx(cfg200), pam_Ny(cfg200))
    c80[30:35, :] .= 2500.0
    c200[30:35, :] .= 2500.0
    source = PointSource2D(depth=0.055, lateral=0.0)

    ref80 = TranscranialFUS._pam_reference_sound_speed(c80, cfg80, [source])
    ref200 = TranscranialFUS._pam_reference_sound_speed(c200, cfg200, [source])
    @test ref80 ≈ ref200
    @test mean(c200) < mean(c80)

    rf = zeros(Float64, pam_Ny(cfg80), pam_Nt(cfg80))
    _, _, info = reconstruct_pam(
        rf,
        c80,
        cfg80;
        frequencies=[0.5e6],
        corrected=false,
        reference_sound_speed=1540.0,
        axial_step=0.25e-3,
    )
    @test info[:reference_sound_speed] == 1540.0
    @test info[:axial_step] ≈ 0.25e-3
    @test info[:axial_substeps_per_cell] == 4
    @test TranscranialFUS._pam_axial_substeps(0.2e-3, 50e-6) == 4
end

@testset "k-Wave smoke tests" begin
    if get(ENV, "TRANSCRANIALFUS_RUN_KWAVE_TESTS", "0") == "1" && kwave_available()
        cfg = SimulationConfig(
            z_focus=0.01,
            dx=1e-3,
            dz=1e-3,
            transverse_dim=0.02,
            trans_aperture=0.01,
            axial_padding=2.0,
            PML_GUARD=5,
            t_max=8e-6,
            dt=50e-9,
        )
        cfg.trans_index = Nx(cfg) - cfg.PML_GUARD
        c = fill(Float32(cfg.c0), Nx(cfg), Nz(cfg))
        rho = fill(Float32(cfg.rho0), Nx(cfg), Nz(cfg))

        p_rms, _, _ = focus(c, rho, GEOMETRIC, cfg, SweepSettings(record=:p_rms))
        @test size(p_rms) == (Nx(cfg), Nz(cfg))

        p_ts, _, _ = focus(c, rho, GEOMETRIC, cfg, SweepSettings(record=:p))
        @test size(p_ts, 1) == Nx(cfg)
        @test size(p_ts, 2) == Nz(cfg)
        @test size(p_ts, 3) == Nt(cfg)

        pam_cfg = PAMConfig(
            dx=0.5e-3,
            dz=0.5e-3,
            axial_dim=0.03,
            transverse_dim=0.03,
            receiver_aperture=0.03,
            PML_GUARD=20,
            t_max=30e-6,
            dt=50e-9,
            zero_pad_factor=2,
            peak_suppression_radius=1.0e-3,
            success_tolerance=1.5e-3,
        )
        c_pam, rho_pam, _ = make_pam_medium(pam_cfg; aberrator=:none)
        sources = [PointSource2D(depth=0.015, lateral=0.003, frequency=0.4e6, amplitude=5e4, num_cycles=4)]
        rf, kgrid_pam, sim_info = simulate_point_sources(c_pam, rho_pam, sources, pam_cfg)
        @test size(rf) == (pam_Ny(pam_cfg), pam_Nt(pam_cfg))
        @test sim_info[:receiver_row] == receiver_row(pam_cfg)
        @test sim_info[:receiver_row] == 1

        pam_map, _, pam_info = reconstruct_pam(rf, c_pam, pam_cfg; frequencies=[0.4e6], corrected=false)
        pam_stats = analyse_pam_2d(pam_map, kgrid_pam, pam_cfg, sources)
        @test pam_info[:corrected] == false
        @test pam_stats[:mean_radial_error_mm] <= 1.5

        sweep_sources = [
            PointSource2D(depth=0.012, lateral=-0.003, frequency=0.4e6, amplitude=5e4, num_cycles=4),
            PointSource2D(depth=0.012, lateral=0.003, frequency=0.4e6, amplitude=5e4, num_cycles=4),
            PointSource2D(depth=0.018, lateral=-0.003, frequency=0.4e6, amplitude=5e4, num_cycles=4),
            PointSource2D(depth=0.018, lateral=0.003, frequency=0.4e6, amplitude=5e4, num_cycles=4),
        ]
        sweep_results = run_pam_sweep(
            c_pam,
            rho_pam,
            sweep_sources,
            pam_cfg;
            frequencies=[0.4e6],
            example_targets_mm=[(12.0, -3.0), (12.0, 3.0), (18.0, 0.0 + 3.0)],
        )
        @test size(sweep_results[:geo_error_mm]) == (2, 2)
        @test size(sweep_results[:hasa_error_mm]) == (2, 2)
        @test all(isfinite.(vec(sweep_results[:geo_error_mm])))
        @test all(isfinite.(vec(sweep_results[:hasa_error_mm])))
        @test length(sweep_results[:cases]) == 4
    else
        @info "Skipping k-Wave smoke tests. Set TRANSCRANIALFUS_RUN_KWAVE_TESTS=1 to enable them."
    end
end

@testset "Optional CT integration" begin
    if isdir(DEFAULT_CT_PATH) && get(ENV, "TRANSCRANIALFUS_RUN_INTEGRATION", "0") == "1" && kwave_available()
        hu_vol, _ = load_default_ct()
        cfg = SimulationConfig(z_focus=0.05, trans_aperture=0.05)
        stats, pressure, _, _, _, _ = run_focus_case(hu_vol, cfg, SKULL_IN_WATER, GEOMETRIC, SweepSettings(); slice_index=250)
        @test pressure !== nothing
        @test stats !== nothing
        @test stats[:p_peak] > 0
    else
        @info "Skipping CT integration test. Requires the local CT dataset, k-Wave availability, and TRANSCRANIALFUS_RUN_INTEGRATION=1."
    end
end
