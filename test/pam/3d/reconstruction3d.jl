@testset "frequency bin selection" begin
    dt = 1e-6
    nt = 64
    target_bin = 9
    target_freq = (target_bin - 1) / (nt * dt)
    rf = zeros(Float64, 3, 4, nt)
    for iy in axes(rf, 1), iz in axes(rf, 2), it in axes(rf, 3)
        rf[iy, iz, it] = sin(2 * pi * target_freq * (it - 1) * dt)
    end

    freqs, bins = TranscranialFUS._select_frequency_bins_3d(rf, dt, nothing)
    @test only(bins) == target_bin
    @test only(freqs) ≈ target_freq

    wide_freqs, wide_bins = TranscranialFUS._select_frequency_bins_3d(rf, dt, [target_freq]; bandwidth=2 / (nt * dt))
    @test target_bin in wide_bins
    @test length(wide_freqs) >= 2
end

@testset "axial gain" begin
    cfg = PAMConfig3D(dx=0.5e-3, axial_gain_power=1.5)
    intensity = ones(Float64, 4, 2, 2)

    TranscranialFUS._apply_axial_gain_3d!(intensity, cfg)

    @test intensity[1, 1, 1] ≈ 1.0
    @test intensity[2, 1, 1] ≈ 1.0
    @test intensity[3, 1, 1] ≈ 2.0^1.5
    @test intensity[4, 1, 1] ≈ 3.0^1.5
end

@testset "reference speed and spectral tapers" begin
    cfg = PAMConfig3D(dx=1e-3, axial_dim=0.006, transverse_dim_y=0.004, transverse_dim_z=0.004)
    source = PointSource3D(depth=2e-3)
    c = fill(1500.0, pam_Nx(cfg), pam_Ny(cfg), pam_Nz(cfg))
    c[3:end, :, :] .= 1800.0

    @test TranscranialFUS._pam_reference_sound_speed(c, cfg, EmissionSource3D[]; margin=0.0) ≈ mean(c)
    @test TranscranialFUS._pam_reference_sound_speed(c, cfg, [source]; margin=0.0) ≈ mean(c[1:3, :, :])

    shifted = TranscranialFUS._ifftshift_2d([1 2 3; 4 5 6])
    @test shifted == circshift([1 2 3; 4 5 6], (-1, -1))

    k_radii = [0.0 1.0 1.5 2.0 3.0]
    taper = TranscranialFUS._tukey_radial(k_radii, 2.0, 0.5)
    @test taper[1] == 1.0
    @test taper[2] == 1.0
    @test 0.0 < taper[3] < 1.0
    @test taper[4] ≈ 0.0 atol=eps(Float64)
    @test taper[5] == 0.0
    @test all(TranscranialFUS._tukey_radial(k_radii, 0.0, 0.5) .== 1.0)
end

@testset "padding helpers" begin
    rf = reshape(1:12, 2, 2, 3)
    padded, range_y, range_z = TranscranialFUS._zero_pad_receiver_rf_3d(rf, 4, 6)
    @test size(padded) == (4, 6, 3)
    @test range_y == 2:3
    @test range_z == 3:4
    @test padded[range_y, range_z, :] == rf
    @test count(!iszero, padded) == length(rf)

    c = reshape(Float64.(1:8), 2, 2, 2)
    edge, cy, cz = TranscranialFUS._edge_pad_lateral_3d(c, 4, 4)
    @test size(edge) == (2, 4, 4)
    @test cy == 2:3
    @test cz == 2:3
    @test edge[:, cy, cz] == c
    @test edge[:, 1:1, cz] == c[:, 1:1, :]
    @test edge[:, end:end, cz] == c[:, end:end, :]

    @test_throws ErrorException TranscranialFUS._zero_pad_receiver_rf_3d(rf, 1, 2)
    @test_throws ErrorException TranscranialFUS._zero_pad_receiver_rf_3d(rf, 2, 1)
    @test_throws ErrorException TranscranialFUS._edge_pad_lateral_3d(c, 1, 2)
    @test_throws ErrorException TranscranialFUS._edge_pad_lateral_3d(c, 2, 1)
end

@testset "small CPU reconstruction metadata" begin
    cfg = PAMConfig3D(
        dx=1e-3,
        dy=1e-3,
        dz=1e-3,
        axial_dim=0.006,
        transverse_dim_y=0.004,
        transverse_dim_z=0.004,
        dt=1e-6,
        t_max=64e-6,
        zero_pad_factor=1,
        PML_GUARD=1,
        axial_gain_power=0.0,
    )
    c = fill(Float64(cfg.c0), pam_Nx(cfg), pam_Ny(cfg), pam_Nz(cfg))
    rf = zeros(Float64, pam_Ny(cfg), pam_Nz(cfg), pam_Nt(cfg))
    frequency = 4 / (pam_Nt(cfg) * cfg.dt)
    rf[2, 2, :] .= sin.(2π .* frequency .* collect(0:(pam_Nt(cfg) - 1)) .* cfg.dt)

    intensity, grid, info = reconstruct_pam_3d(
        rf,
        c,
        cfg;
        frequencies=[frequency],
        corrected=false,
        reference_sound_speed=cfg.c0,
        axial_step=0.5e-3,
        time_origin=2e-6,
        use_gpu=false,
    )

    @test size(intensity) == size(c)
    @test length(grid.t) == pam_Nt(cfg)
    @test info[:corrected] == false
    @test info[:backend] == :cpu
    @test info[:use_gpu] == false
    @test info[:reference_sound_speed] == cfg.c0
    @test info[:axial_step] ≈ 0.5e-3
    @test info[:axial_substeps_per_cell] == 2
    @test info[:time_origin] ≈ 2e-6
    @test info[:crop_range_y] == 1:pam_Ny(cfg)
    @test info[:crop_range_z] == 1:pam_Nz(cfg)
    @test only(info[:frequencies]) ≈ frequency

    @test_throws ErrorException reconstruct_pam_3d(rf[1:(end - 1), :, :], c, cfg; frequencies=[frequency])
    @test_throws ErrorException reconstruct_pam_3d(rf, c, cfg; frequencies=[frequency], reference_sound_speed=0.0)
end
