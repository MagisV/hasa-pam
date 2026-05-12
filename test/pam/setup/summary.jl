@testset "summary helpers" begin
    p2 = PointSource2D(depth=0.01, lateral=0.002, frequency=0.4e6, amplitude=2.0, phase=0.3, delay=1e-6, num_cycles=4)
    b2 = BubbleCluster2D(depth=0.02, lateral=-0.001, fundamental=0.5e6, harmonics=[2], harmonic_amplitudes=[0.7])
    p3 = PointSource3D(depth=0.03, lateral_y=0.001, lateral_z=-0.002, frequency=0.6e6)
    b3 = BubbleCluster3D(depth=0.04, lateral_y=0.002, lateral_z=0.003, fundamental=0.5e6)

    @test source_summary(p2)["kind"] == "point"
    @test source_summary(b2)["kind"] == "bubble_cluster"
    @test source_summary(p3)["kind"] == "point3d"
    @test source_summary(b3)["kind"] == "bubble3d"

    @test TranscranialFUS.source_model_from_meta(Dict("source_model" => "vascular"), [b2]) == :squiggle
    @test TranscranialFUS.source_model_from_meta(Dict("source_model" => "network3d"), [b3]) == :network
    @test TranscranialFUS.source_model_from_meta(Dict{String, Any}(), [p2]) == :point
    @test TranscranialFUS.source_model_from_meta(Dict{String, Any}(), [b2]) == :squiggle

    meta = Dict(
        "squiggle" => Dict(
            "centerlines_m" => [
                [[0.01, -0.001], [0.02, 0.001]],
            ],
        ),
    )
    centerlines = TranscranialFUS.centerlines_from_emission_meta(meta)
    @test length(centerlines) == 1
    @test first(centerlines)[1] == (0.01, -0.001)

    cfg = PAMConfig(dx=1e-3, dz=1e-3, axial_dim=0.04, transverse_dim=0.02)
    mask = TranscranialFUS.detection_truth_mask_from_meta(meta, pam_grid(cfg), cfg, 1e-3)
    @test count(mask) > 0

    info = Dict(
        :total_window_count => 2,
        :used_window_count => 1,
        :skipped_window_count => 1,
        :window_samples => 100,
        :hop_samples => 50,
        :energy_threshold => 0.1,
        :used_window_ranges => [1:100],
        :skipped_window_ranges => [51:150],
        :accumulation => :intensity,
    )
    compact = TranscranialFUS.compact_window_info(info)
    @test compact["used_window_ranges"] == [[1, 100]]
    @test compact["skipped_window_ranges"] == [[51, 150]]
    @test compact["accumulation"] == "intensity"
end
