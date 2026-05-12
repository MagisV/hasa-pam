@testset "medium" begin
    cfg = PAMConfig(dx=1e-3, dz=1e-3, axial_dim=0.04, transverse_dim=0.06)
    c_water, rho_water, info_water = make_pam_medium(cfg; aberrator=:none)
    @test size(c_water) == (pam_Nx(cfg), pam_Ny(cfg))
    @test size(rho_water) == size(c_water)
    @test all(c_water .≈ cfg.c0)
    @test all(rho_water .≈ cfg.rho0)
    @test info_water[:aberrator] == :none

    targets = [
        PointSource2D(depth=axial_mm * 1e-3, lateral=0.0, frequency=1e6)
        for axial_mm in (40.0, 60.0, 80.0)
    ]
    fitted_cfg = fit_pam_config(
        cfg,
        targets;
        min_bottom_margin=5e-3,
        reference_depth=30e-3,
    )
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

    for src in targets
        row, col = source_grid_index(src, fitted_cfg, pam_grid(fitted_cfg))
        @test row > info[:inner_row]
        @test row <= pam_Nx(fitted_cfg)
        @test c[row, col] ≈ fitted_cfg.c0
    end
end
