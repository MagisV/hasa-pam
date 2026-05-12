@testset "medium3d" begin
    cfg = PAMConfig3D(
        dx=1e-3,
        dy=1e-3,
        dz=1e-3,
        axial_dim=0.04,
        transverse_dim_y=0.03,
        transverse_dim_z=0.005,
    )

    c_water, rho_water, info_water = make_pam_medium_3d(cfg; aberrator=:water)
    @test size(c_water) == (pam_Nx(cfg), pam_Ny(cfg), pam_Nz(cfg))
    @test size(rho_water) == size(c_water)
    @test all(c_water .≈ cfg.c0)
    @test all(rho_water .≈ cfg.rho0)
    @test info_water[:aberrator] == :water

    hu_vol = synthetic_hu_volume()
    c, rho, info = make_pam_medium_3d(
        cfg;
        aberrator=:skull,
        hu_vol=hu_vol,
        spacing_m=(1e-3, 1e-3, 1e-3),
        slice_index_z=2,
        skull_to_transducer=20e-3,
        hu_bone_thr=200,
    )
    @test size(c) == (pam_Nx(cfg), pam_Ny(cfg), pam_Nz(cfg))
    @test size(rho) == size(c)
    @test info[:aberrator] == :skull
    @test info[:outer_row] == receiver_row(cfg) + 20
    @test info[:outer_row] < info[:inner_row]
    @test maximum(c[info[:outer_row]:info[:inner_row], :, :]) > cfg.c0
end
