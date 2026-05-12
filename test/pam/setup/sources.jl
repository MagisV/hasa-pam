@testset "source parsing" begin
    opts, _ = TranscranialFUS.parse_cli([
        "--source-model=point",
        "--sources-mm=30:-2,40:3",
        "--source-frequencies-mhz=0.4,0.6",
        "--source-amplitudes-pa=1.5",
        "--phases-deg=0,90",
        "--delays-us=0,2",
    ])
    sources, meta = TranscranialFUS.parse_point_sources(opts)
    @test length(sources) == 2
    @test sources[1].depth ≈ 30e-3
    @test sources[2].lateral ≈ 3e-3
    @test sources[2].frequency ≈ 0.6e6
    @test all(src.amplitude ≈ 1.5 for src in sources)
    @test sources[2].phase ≈ pi / 2
    @test sources[2].delay ≈ 2e-6
    @test meta["source_model"] == "point"
    @test meta["physical_source_count"] == 2

    opts3, _ = TranscranialFUS.parse_cli([
        "--dimension=3",
        "--source-model=point",
        "--sources-mm=30:1:-1,35:2:3",
        "--source-frequencies-mhz=0.5",
    ])
    sources3, meta3 = TranscranialFUS.parse_point_sources_3d(opts3)
    @test length(sources3) == 2
    @test sources3[1].lateral_y ≈ 1e-3
    @test sources3[1].lateral_z ≈ -1e-3
    @test all(src.frequency ≈ 0.5e6 for src in sources3)
    @test meta3["source_model"] == "point3d"

    @test TranscranialFUS.expand_source_values(Float64[], 3, 2.0) == [2.0, 2.0, 2.0]
    @test TranscranialFUS.expand_source_values([4.0], 2, 0.0) == [4.0, 4.0]
    @test_throws ErrorException TranscranialFUS.expand_source_values([1.0, 2.0], 3, 0.0)
    @test TranscranialFUS.parse_coordinate_pairs_mm("10:1,20:-2", "sources-mm") == [(10e-3, 1e-3), (20e-3, -2e-3)]
    @test TranscranialFUS.parse_coordinate_triples_mm("10:1:2", "sources-mm") == [(10e-3, 1e-3, 2e-3)]
end

@testset "squiggle source parsing" begin
    opts, _ = TranscranialFUS.parse_cli([
        "--source-model=squiggle",
        "--anchors-mm=35:0",
        "--vascular-length-mm=4",
        "--vascular-source-spacing-mm=1",
        "--vascular-position-jitter-mm=0",
        "--vascular-min-separation-mm=0",
        "--vascular-max-sources-per-anchor=8",
    ])
    cfg = PAMConfig(transverse_dim=0.02)
    sources, meta = TranscranialFUS.parse_squiggle_sources(opts, cfg)
    @test !isempty(sources)
    @test length(sources) <= 8
    @test meta["source_model"] == "squiggle"
    @test haskey(meta, "squiggle")

    cfg3 = PAMConfig3D(transverse_dim_y=0.02, transverse_dim_z=0.02)
    opts3, _ = TranscranialFUS.parse_cli([
        "--dimension=3",
        "--source-model=squiggle",
        "--anchors-mm=35:0:0",
        "--vascular-length-mm=4",
        "--vascular-source-spacing-mm=1",
        "--vascular-position-jitter-mm=0",
        "--vascular-min-separation-mm=0",
        "--vascular-max-sources-per-anchor=8",
    ])
    sources3, meta3 = TranscranialFUS.parse_squiggle_sources_3d(opts3, cfg3)
    @test !isempty(sources3)
    @test length(sources3) <= 8
    @test meta3["source_model"] == "squiggle3d"
    @test haskey(meta3, "squiggle")
end
