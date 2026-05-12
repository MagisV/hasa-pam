@testset "plot projection helpers" begin
    cfg = PAMConfig3D(
        dx=1e-3,
        dy=1e-3,
        dz=1e-3,
        axial_dim=0.004,
        transverse_dim_y=0.003,
        transverse_dim_z=0.005,
    )
    grid = pam_grid_3d(cfg)
    values = reshape(Float64.(1:(pam_Nx(cfg) * pam_Ny(cfg) * pam_Nz(cfg))), pam_Nx(cfg), pam_Ny(cfg), pam_Nz(cfg))

    depth_y = TranscranialFUS._project3d_values(values, :depth_y)
    depth_z = TranscranialFUS._project3d_values(values, :depth_z)
    y_z = TranscranialFUS._project3d_values(values, :y_z)
    @test size(depth_y) == (pam_Nx(cfg), pam_Ny(cfg))
    @test size(depth_z) == (pam_Nx(cfg), pam_Nz(cfg))
    @test size(y_z) == (pam_Ny(cfg), pam_Nz(cfg))
    @test y_z == dropdims(maximum(values; dims=1); dims=1)

    xvals, yvals, xlabel, ylabel = TranscranialFUS._projection_axes_3d(grid, cfg, :depth_z)
    @test length(xvals) == pam_Nz(cfg)
    @test length(yvals) == pam_Nx(cfg)
    @test occursin("Z", xlabel)
    @test occursin("Depth", ylabel)

    sources = [PointSource3D(depth=0.01, lateral_y=0.002, lateral_z=-0.003)]
    @test TranscranialFUS.source_triples_mm(sources) == [(10.0, 2.0, -3.0)]
    @test_throws ErrorException TranscranialFUS._project3d_values(values, :unknown)
end
