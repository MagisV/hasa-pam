@testset "simulation defaults" begin
    cfg2 = PAMConfig(dx=1e-3, dz=1e-3, axial_dim=0.02, transverse_dim=0.01, receiver_aperture=0.004)
    info2 = TranscranialFUS.default_simulation_info(cfg2)
    @test info2[:receiver_row] == receiver_row(cfg2)
    @test info2[:receiver_cols] == receiver_col_range(cfg2)
    @test isempty(info2[:source_indices])

    cfg3 = PAMConfig3D(
        dx=1e-3,
        dy=1e-3,
        dz=1e-3,
        axial_dim=0.02,
        transverse_dim_y=0.01,
        transverse_dim_z=0.012,
        receiver_aperture_y=0.004,
        receiver_aperture_z=0.006,
    )
    info3 = TranscranialFUS.default_simulation_info(cfg3)
    @test info3[:receiver_row] == receiver_row(cfg3)
    @test info3[:receiver_cols_y] == receiver_col_range_y(cfg3)
    @test info3[:receiver_cols_z] == receiver_col_range_z(cfg3)
    @test isempty(info3[:source_indices])
end

@testset "default frequencies and analytic sampling" begin
    sources = [
        PointSource2D(depth=0.01, lateral=0.0, frequency=0.4e6),
        BubbleCluster2D(depth=0.02, lateral=0.0, fundamental=0.5e6, harmonics=[2, 3]),
    ]
    @test TranscranialFUS.default_recon_frequencies(sources) == [0.4e6, 1.0e6, 1.5e6]
    @test TranscranialFUS._sample_source_signal([0.0, 10.0], 0.5, 1.0) ≈ 5.0
    @test TranscranialFUS._sample_source_signal([1.0], -1.0, 1.0) == 0.0

    cfg3 = PAMConfig3D(
        dx=1e-3,
        dy=1e-3,
        dz=1e-3,
        axial_dim=0.012,
        transverse_dim_y=0.006,
        transverse_dim_z=0.006,
        t_max=20e-6,
        dt=0.1e-6,
    )
    src3 = PointSource3D(depth=0.004, frequency=0.5e6, amplitude=1.0, num_cycles=3)
    rf, grid, info = TranscranialFUS.analytic_rf_for_point_sources_3d(cfg3, [src3])
    @test size(rf) == (pam_Ny(cfg3), pam_Nz(cfg3), pam_Nt(cfg3))
    @test length(grid.t) == pam_Nt(cfg3)
    @test info[:source_indices] == [source_grid_index_3d(src3, cfg3)]
    @test any(!iszero, rf)
end
