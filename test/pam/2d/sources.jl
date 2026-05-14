@testset "bubble cluster emissions" begin
    dt = 20e-9
    nt = 1000
    src = BubbleCluster2D(
        depth=0.03,
        lateral=0.0,
        fundamental=0.5e6,
        amplitude=1.0,
        harmonics=[2],
        harmonic_amplitudes=[1.0],
        harmonic_phases=[0.0],
        gate_duration=10e-6,
    )
    signal = TranscranialFUS._source_signal(nt, dt, src)
    active = findall(!iszero, signal)

    @test !isempty(active)
    @test maximum(abs, signal) > 0.5
    @test abs(signal[first(active)]) < 0.05 * maximum(abs, signal)
    @test abs(signal[last(active)]) < 0.05 * maximum(abs, signal)
    @test emission_frequencies(src) == [1.0e6]

    spectrum = abs.(fft(signal))
    freq_axis = collect(0:(nt - 1)) ./ (nt * dt)
    pos_bins = 2:(fld(nt, 2) + 1)
    peak_bin = pos_bins[argmax(spectrum[pos_bins])]
    @test abs(freq_axis[peak_bin] - 1.0e6) <= 1 / (nt * dt)
end

@testset "source phase modes" begin
    @test TranscranialFUS._normalize_source_phase_mode(:coherent) == :coherent
    @test TranscranialFUS._normalize_source_phase_mode(:random_static_phase) == :random_static_phase
    @test TranscranialFUS._normalize_source_phase_mode(:random_phase_per_window) == :random_phase_per_window
    @test_throws ErrorException TranscranialFUS._normalize_source_phase_mode(:unknown_mode)

    @test TranscranialFUS._normalize_cluster_phase_mode(:random_static_phase) == :random
    @test TranscranialFUS._normalize_cluster_phase_mode("random_static_phase") == :random
    @test TranscranialFUS._normalize_cluster_phase_mode(:coherent) == :coherent

    sources_orig = [
        BubbleCluster2D(depth=0.03, lateral=0.0, fundamental=0.5e6,
            harmonics=[2, 3], harmonic_amplitudes=[1.0, 0.6],
            harmonic_phases=[0.1, 0.2], gate_duration=10e-6),
        PointSource2D(depth=0.02, lateral=0.005, frequency=1.0e6, phase=0.5),
    ]
    resampled = TranscranialFUS._resample_source_phases(sources_orig, Random.MersenneTwister(7))

    @test resampled[1].depth == sources_orig[1].depth
    @test resampled[1].lateral == sources_orig[1].lateral
    @test resampled[1].harmonic_phases != sources_orig[1].harmonic_phases
    @test resampled[2].phase != sources_orig[2].phase
end

@testset "source variability" begin
    src = BubbleCluster2D(depth=0.03, lateral=0.0, fundamental=0.5e6,
        harmonics=[2, 3], harmonic_amplitudes=[1.0, 0.6],
        harmonic_phases=[0.1, 0.2], gate_duration=50e-6)

    expanded, n = TranscranialFUS._expand_sources_per_window(
        [src], 10e-6, 5e-6, 80e-6, Random.MersenneTwister(1))
    @test n == 15
    @test length(expanded) == 15
    @test all(s.amplitude == src.amplitude for s in expanded)
    @test all(s.fundamental == src.fundamental for s in expanded)

    exp_fj, _ = TranscranialFUS._expand_sources_per_window(
        [src], 10e-6, 5e-6, 80e-6, Random.MersenneTwister(99);
        variability=SourceVariabilityConfig(frequency_jitter_fraction=0.05))
    @test length(unique(round.(Float64[s.fundamental for s in exp_fj]; digits=0))) > 1
end

@testset "squiggle bubble sources" begin
    squiggle_clusters, squiggle_meta = make_squiggle_bubble_sources(
        [(0.03, 0.0)];
        root_length=12e-3,
        squiggle_amplitude=1.5e-3,
        squiggle_wavelength=6e-3,
        source_spacing=1e-3,
        position_jitter=0.0,
        min_separation=0.0,
        lateral_bounds=(-0.02, 0.02),
        rng=Random.MersenneTwister(41),
    )

    @test squiggle_meta[:source_model] == :squiggle
    @test all(src -> src isa BubbleCluster2D, squiggle_clusters)
    @test length(squiggle_meta[:centerlines]) == 1
    @test maximum(src.lateral for src in squiggle_clusters) - minimum(src.lateral for src in squiggle_clusters) > 10e-3
    @test maximum(src.depth for src in squiggle_clusters) - minimum(src.depth for src in squiggle_clusters) > 2e-3

    multi_clusters, multi_meta = make_squiggle_bubble_sources(
        [(0.03, -0.004), (0.035, 0.004)];
        root_length=8e-3,
        source_spacing=1e-3,
        position_jitter=0.0,
        min_separation=0.0,
        max_sources_per_anchor=20,
        lateral_bounds=(-0.02, 0.02),
        rng=Random.MersenneTwister(43),
    )
    @test length(multi_clusters) > length(squiggle_clusters)
    @test multi_meta[:source_model] == :squiggle
    @test length(multi_meta[:centerlines]) == 2
end
