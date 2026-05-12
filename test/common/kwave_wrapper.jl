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

@testset "k-Wave deterministic planning helpers" begin
    cfg = PAMConfig(
        dx=1e-3,
        dz=1e-3,
        axial_dim=0.012,
        transverse_dim=0.008,
        t_max=4e-6,
        dt=0.1e-6,
        PML_GUARD=2,
    )
    kgrid = pam_grid(cfg)
    c = fill(Float32(cfg.c0), pam_Nx(cfg), pam_Ny(cfg))
    rho = fill(Float32(cfg.rho0), size(c))
    source_a = PointSource2D(depth=0.003, lateral=0.0, frequency=0.4e6)
    source_b = PointSource2D(depth=0.003, lateral=0.0, frequency=0.6e6)

    @test TranscranialFUS._validate_point_source_inputs(c, rho, [source_a], cfg) == size(c)
    indexed = TranscranialFUS._indexed_sources_2d([source_b, source_a], cfg, kgrid, pam_Nx(cfg))
    grouped = TranscranialFUS._group_sources_2d(indexed)
    @test length(grouped) == 1
    @test length(last(only(grouped))) == 2
    @test sort(TranscranialFUS._unique_emission_frequencies(indexed)) == [0.4e6, 0.6e6]
    info = TranscranialFUS._kwave_info_2d(receiver_row(cfg), receiver_col_range(cfg), [first(only(grouped))], [source_a, source_b], grouped)
    @test info[:num_input_sources] == 2
    @test info[:num_source_points] == 1

    cfg3 = PAMConfig3D(
        dx=1e-3,
        dy=1e-3,
        dz=1e-3,
        axial_dim=0.012,
        transverse_dim_y=0.006,
        transverse_dim_z=0.006,
        t_max=4e-6,
        dt=0.1e-6,
        PML_GUARD=2,
    )
    c3 = fill(Float32(cfg3.c0), pam_Nx(cfg3), pam_Ny(cfg3), pam_Nz(cfg3))
    rho3 = fill(Float32(cfg3.rho0), size(c3))
    source3a = PointSource3D(depth=0.003, frequency=0.5e6)
    source3b = PointSource3D(depth=0.003, frequency=0.75e6)

    @test TranscranialFUS._validate_point_source_inputs_3d(c3, rho3, [source3a], cfg3) == size(c3)
    indexed3 = TranscranialFUS._indexed_sources_3d([source3b, source3a], cfg3, pam_Nx(cfg3), pam_Ny(cfg3))
    grouped3 = TranscranialFUS._group_sources_3d(indexed3)
    @test length(grouped3) == 1
    @test length(last(only(grouped3))) == 2
    @test sort(TranscranialFUS._unique_emission_frequencies(indexed3)) == [0.5e6, 0.75e6]
    info3 = TranscranialFUS._kwave_info_3d(receiver_row(cfg3), receiver_col_range_y(cfg3), receiver_col_range_z(cfg3), [first(only(grouped3))], [source3a, source3b], grouped3)
    @test info3[:receiver_cols_y] == receiver_col_range_y(cfg3)
    @test info3[:num_input_sources] == 2
    @test info3[:num_source_points] == 1
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
