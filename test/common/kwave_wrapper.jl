@testset "k-Wave wrapper helpers" begin
    @test TranscranialFUS._normalize_record(:p_rms) == :p_rms
    @test TranscranialFUS._normalize_record("p") == :p
    @test_throws ErrorException TranscranialFUS._normalize_record(:pressure)

    mat = reshape(1:6, 2, 3)
    @test TranscranialFUS._as_sensor_matrix(mat, 2, 3) == Float64.(mat)
    @test TranscranialFUS._as_sensor_matrix(permutedims(mat), 2, 3) == Float64.(mat)
    @test TranscranialFUS._as_sensor_matrix([1, 2, 3], 3, 1) == reshape(Float64[1, 2, 3], 3, 1)
    @test_throws ErrorException TranscranialFUS._as_sensor_matrix(zeros(2, 2, 2), 2, 4)
end

@testset "PAM k-Wave validation paths" begin
    cfg = PAMConfig(
        dx=1e-3,
        dz=1e-3,
        axial_dim=0.01,
        transverse_dim=0.008,
        t_max=4e-6,
        dt=0.1e-6,
        PML_GUARD=2,
    )
    source = PointSource2D(depth=0.003, lateral=0.0)
    c = fill(Float32(cfg.c0), pam_Nx(cfg), pam_Ny(cfg))
    rho = fill(Float32(cfg.rho0), size(c))

    @test_throws ErrorException simulate_point_sources(c, rho, PointSource2D[], cfg)
    @test_throws ErrorException simulate_point_sources(c, rho[:, 1:(end - 1)], [source], cfg)
    @test_throws ErrorException simulate_point_sources(c[1:(end - 1), :], rho[1:(end - 1), :], [source], cfg)

    cfg3 = PAMConfig3D(
        dx=1e-3,
        dy=1e-3,
        dz=1e-3,
        axial_dim=0.01,
        transverse_dim_y=0.006,
        transverse_dim_z=0.006,
        t_max=4e-6,
        dt=0.1e-6,
        PML_GUARD=2,
    )
    source3 = PointSource3D(depth=0.003)
    c3 = fill(Float32(cfg3.c0), pam_Nx(cfg3), pam_Ny(cfg3), pam_Nz(cfg3))
    rho3 = fill(Float32(cfg3.rho0), size(c3))

    @test_throws ErrorException simulate_point_sources_3d(c3, rho3, PointSource3D[], cfg3)
    @test_throws ErrorException simulate_point_sources_3d(c3, rho3[:, 1:(end - 1), :], [source3], cfg3)
    @test_throws ErrorException simulate_point_sources_3d(c3[1:(end - 1), :, :], rho3[1:(end - 1), :, :], [source3], cfg3)
end
