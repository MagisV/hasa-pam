@testset "workflow case assembly" begin
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

    one_window = PAMWindowConfig(
        enabled=true,
        window_duration=pam_Nt(cfg) * cfg.dt,
        hop=pam_Nt(cfg) * cfg.dt,
        taper=:none,
        min_energy_ratio=0.0,
    )
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
end

@testset "reference speed is source-depth scoped" begin
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
end
