@testset "plot data helpers" begin
    map = [0.0 1.0; 10.0 100.0]

    @test TranscranialFUS.map_norm(map, 100.0) ≈ [0.0 0.01; 0.1 1.0]
    db = TranscranialFUS.map_db(map, 100.0)
    @test db[2, 2] ≈ 0.0
    @test db[1, 1] < -100

    sources = [
        PointSource2D(depth=0.01, lateral=-0.002),
        PointSource2D(depth=0.02, lateral=0.003),
    ]
    pairs = TranscranialFUS.source_pairs_mm(sources)
    @test pairs == [(10.0, -2.0), (20.0, 3.0)]

    stats = Dict(:mean_radial_error_mm => 0.25, :num_success => 1, :num_truth_sources => 2, :mean_norm_peak_intensity => 0.75)
    line = TranscranialFUS.summary_line(stats)
    @test occursin("err=0.25 mm", line)
    @test occursin("success=1/2", line)
end
