@testset "config" begin
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

    base_cfg = PAMConfig(dx=1e-3, dz=1e-3, axial_dim=0.03, transverse_dim=0.06)
    fitted_cfg = fit_pam_config(
        base_cfg,
        [PointSource2D(depth=0.04, lateral=0.0)];
        min_bottom_margin=5e-3,
        reference_depth=30e-3,
    )
    @test pam_Nx(fitted_cfg) == 46
    @test fitted_cfg.axial_dim ≈ 46e-3

    deep_cfg = PAMConfig(
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
    fitted_deep = fit_pam_config(deep_cfg, [cluster]; min_bottom_margin=5e-3)
    @test fitted_deep.t_max >= TranscranialFUS._required_pam_t_max(deep_cfg, [cluster])
    @test fitted_deep.t_max > 110e-6
    @test TranscranialFUS._pam_axial_substeps(0.2e-3, 50e-6) == 4

    @test TranscranialFUS.pam_reconstruction_mode(:auto, :squiggle) == :windowed
    @test TranscranialFUS.pam_reconstruction_mode(:auto, :point) == :full
    @test TranscranialFUS.pam_reconstruction_mode(:full, :squiggle) == :full
    @test_throws ErrorException TranscranialFUS.pam_reconstruction_mode(:auto, :unknown)
end

@testset "windowing helpers" begin
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
end
