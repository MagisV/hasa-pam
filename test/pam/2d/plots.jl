using CairoMakie

@testset "plot data helpers" begin
    map = [0.0 1.0; 10.0 100.0]

    @test TranscranialFUS.map_norm(map, 100.0) ≈ [0.0 0.01; 0.1 1.0]
    @test TranscranialFUS.map_norm(map, 0.0)[2, 2] ≈ 100.0 / eps(Float64)
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

    detection_line = TranscranialFUS.summary_line(Dict(:f1 => 0.5, :precision => 0.25, :recall => 1.0))
    @test occursin("F1=0.5", detection_line)
    @test occursin("recall=1.0", detection_line)
    @test TranscranialFUS.summary_line(Dict(:other => 1)) == "Dict(:other => 1)"
end

@testset "plot rendering helpers" begin
    fig = CairoMakie.Figure(size=(300, 240))
    ax = CairoMakie.Axis(fig[1, 1])
    c_water = fill(1500.0, 6, 6)
    @test TranscranialFUS.overlay_skull_2d!(ax, c_water, 1:6, 1:6) === nothing
    @test TranscranialFUS.lines_centerlines!(ax, nothing) === nothing
    @test TranscranialFUS.lines_centerlines!(ax, [[(0.001, 0.0)]]) === nothing
    @test TranscranialFUS.lines_centerlines!(ax, [[(0.001, -0.001), (0.002, 0.001)]]) === nothing

    cfg = PAMConfig(
        dx=1e-3,
        dz=1e-3,
        axial_dim=0.008,
        transverse_dim=0.008,
        peak_suppression_radius=1e-3,
        success_tolerance=1e-3,
        PML_GUARD=1,
    )
    kgrid = pam_grid(cfg)
    source = PointSource2D(depth=0.003, lateral=0.0)
    intensity = zeros(Float64, pam_Nx(cfg), pam_Ny(cfg))
    intensity[source_grid_index(source, cfg, kgrid)...] = 1.0
    truth_mask = pam_truth_mask([source], kgrid, cfg; radius=cfg.success_tolerance)

    mktempdir() do dir
        path = joinpath(dir, "boundary.png")
        metrics = TranscranialFUS.save_threshold_boundary_detection(
            path,
            intensity,
            0.8 .* intensity,
            kgrid,
            cfg,
            [source];
            threshold_ratios=[0.5, 0.9],
            truth_radius=cfg.success_tolerance,
            truth_mask=truth_mask,
            truth_centerlines=[[(0.002, -0.001), (0.004, 0.001)]],
            frequencies=[source.frequency],
            c=c_water,
        )
        @test isfile(path)
        @test metrics["threshold_ratios"] == [0.5, 0.9]
        @test length(metrics["geometric"]) == 2
        @test haskey(first(metrics["hasa"]), "f1")
    end
end
