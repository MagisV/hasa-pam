@testset "sources3d" begin
    point = PointSource3D(depth=0.02, lateral_y=0.001, lateral_z=-0.001, frequency=0.6e6, num_cycles=4)
    cluster = BubbleCluster3D(depth=0.03, fundamental=0.5e6, harmonics=[2, 4], harmonic_amplitudes=[1.0, 0.25])

    @test emission_frequencies(point) == [0.6e6]
    @test emission_frequencies(cluster) == [1.0e6, 2.0e6]
    @test TranscranialFUS._source_duration(point) ≈ 4 / 0.6e6
    @test TranscranialFUS._source_duration(cluster) ≈ cluster.gate_duration

    signal = TranscranialFUS._source_signal(512, 20e-9, point)
    @test any(!iszero, signal)
    @test maximum(abs, signal) > 0

    original = [cluster, point]
    resampled = TranscranialFUS._resample_source_phases_3d(original, Random.MersenneTwister(11))
    @test resampled[1].harmonic_phases != cluster.harmonic_phases
    @test resampled[2].phase != point.phase
end

@testset "squiggle and network sources3d" begin
    squiggle, squiggle_meta = make_squiggle_bubble_sources_3d(
        [(0.03, 0.0, 0.0)];
        root_length=6e-3,
        source_spacing=1e-3,
        position_jitter=0.0,
        min_separation=0.0,
        max_sources_per_anchor=12,
        rng=Random.MersenneTwister(12),
    )
    @test squiggle_meta[:source_model] == :squiggle
    @test all(src -> src isa BubbleCluster3D, squiggle)
    @test length(squiggle) <= 12
    @test length(squiggle_meta[:centerlines]) == 1

    network, network_meta = make_network_bubble_sources_3d(
        [(0.035, 0.0, 0.0)];
        root_count=3,
        generations=1,
        branch_length=2e-3,
        branch_step=1e-3,
        source_spacing=1e-3,
        min_separation=0.0,
        max_sources_per_center=20,
        rng=Random.MersenneTwister(13),
    )
    @test network_meta[:source_model] == :network
    @test all(src -> src isa BubbleCluster3D, network)
    @test !isempty(network_meta[:centerlines])
    @test length(network) <= 20
end
