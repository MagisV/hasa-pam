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
end
