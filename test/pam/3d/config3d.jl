@testset "config3d" begin
    cfg = PAMConfig3D(
        dx=1e-3,
        dy=0.5e-3,
        dz=0.5e-3,
        axial_dim=0.04,
        transverse_dim_y=0.02,
        transverse_dim_z=0.03,
        receiver_aperture_y=0.01,
        receiver_aperture_z=0.015,
    )

    @test pam_Nx(cfg) == 40
    @test pam_Ny(cfg) == 40
    @test pam_Nz(cfg) == 60
    @test pam_Nt(cfg) == 2000
    @test receiver_row(cfg) == 1
    @test length(receiver_col_range_y(cfg)) == 20
    @test length(receiver_col_range_z(cfg)) == 30

    grid = pam_grid_3d(cfg)
    @test length(grid.x) == pam_Nx(cfg)
    @test length(grid.y) == pam_Ny(cfg)
    @test length(grid.z) == pam_Nz(cfg)

    src = PointSource3D(depth=0.025, lateral_y=0.001, lateral_z=-0.002)
    row, iy, iz = source_grid_index_3d(src, cfg)
    @test row == receiver_row(cfg) + 25
    @test grid.y[iy] ≈ src.lateral_y atol=cfg.dy
    @test grid.z[iz] ≈ src.lateral_z atol=cfg.dz

    fitted = fit_pam_config_3d(
        cfg,
        [PointSource3D(depth=0.055)];
        min_bottom_margin=5e-3,
        reference_depth=0.03,
    )
    @test pam_Nx(fitted) >= receiver_row(cfg) + 60
    @test fitted.axial_dim >= 60e-3
    @test fitted.t_max >= cfg.t_max
end
