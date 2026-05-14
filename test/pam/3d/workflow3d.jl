@testset "workflow3d rejects CPU reconstruction" begin
    cfg = PAMConfig3D(
        dx=1e-3,
        dy=1e-3,
        dz=1e-3,
        axial_dim=0.012,
        transverse_dim_y=0.008,
        transverse_dim_z=0.008,
        t_max=20e-6,
    )
    c, rho, _ = make_pam_medium_3d(cfg; aberrator=:none)
    source = PointSource3D(depth=0.004, frequency=0.5e6)

    @test_throws ErrorException run_pam_case_3d(
        c,
        rho,
        [source],
        cfg;
        recon_use_gpu=false,
        simulation_backend=:analytic,
    )
end

@testset "analytic RF defaults" begin
    cfg = PAMConfig3D(
        dx=1e-3,
        dy=1e-3,
        dz=1e-3,
        axial_dim=0.012,
        transverse_dim_y=0.006,
        transverse_dim_z=0.006,
        t_max=20e-6,
        dt=0.1e-6,
    )
    source = PointSource3D(depth=0.004, frequency=0.5e6, amplitude=1.0, num_cycles=3)
    rf, grid, info = TranscranialFUS.analytic_rf_for_point_sources_3d(cfg, [source])

    @test size(rf) == (pam_Ny(cfg), pam_Nz(cfg), pam_Nt(cfg))
    @test grid === pam_grid_3d(cfg) || length(grid.t) == pam_Nt(cfg)
    @test info[:receiver_row] == receiver_row(cfg)
    @test info[:receiver_cols_y] == receiver_col_range_y(cfg)
    @test info[:receiver_cols_z] == receiver_col_range_z(cfg)
    @test info[:source_indices] == [source_grid_index_3d(source, cfg)]
    @test any(!iszero, rf)
end
