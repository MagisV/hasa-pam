@testset "CLI config parsing" begin
    opts, provided = TranscranialFUS.parse_cli(String[])
    @test isempty(provided)
    @test opts["dimension"] == "2"
    @test opts["source-model"] == "squiggle"
    @test opts["recon-progress"] == "false"
    @test TranscranialFUS.parse_bool(opts["recon-progress"]) == false

    opts3, provided3 = TranscranialFUS.parse_cli(["--dimension=3"])
    @test "dimension" in provided3
    @test opts3["source-model"] == "point"
    @test opts3["sources-mm"] == "30:0:0"
    @test opts3["frequency-mhz"] == "0.5"

    @test TranscranialFUS.parse_dimension("2d") == 2
    @test TranscranialFUS.parse_dimension("3D") == 3
    @test_throws ErrorException TranscranialFUS.parse_dimension("4")
    @test TranscranialFUS.parse_float_list("1, 2.5,,3") == [1.0, 2.5, 3.0]
    @test TranscranialFUS.parse_int_list("1, 3,5") == [1, 3, 5]
    @test TranscranialFUS.parse_threshold_ratios("0.5,0.2,0.5") == [0.2, 0.5]
    @test_throws ErrorException TranscranialFUS.parse_threshold_ratios("")

    search_opts = copy(opts)
    search_opts["auto-threshold-min"] = "0.2"
    search_opts["auto-threshold-max"] = "0.5"
    search_opts["auto-threshold-step"] = "0.2"
    @test TranscranialFUS.parse_threshold_search_ratios(search_opts) == [0.2, 0.4, 0.5]

    @test TranscranialFUS.parse_source_model("point") == :point
    @test TranscranialFUS.parse_aberrator("water") == :water
    @test TranscranialFUS.parse_simulation_backend("analytic") == :analytic
    @test TranscranialFUS.parse_source_phase_mode("random-phase-per-window") == :random_phase_per_window
    @test TranscranialFUS.parse_analysis_mode("auto", :squiggle) == :detection
    @test TranscranialFUS.parse_analysis_mode("auto", :point) == :localization
    @test TranscranialFUS.parse_window_taper("rectangular") == :rectangular
    @test TranscranialFUS.parse_receiver_aperture_mm("full") === nothing
    @test TranscranialFUS.parse_receiver_aperture_mm("25") ≈ 25e-3
    @test TranscranialFUS.parse_transducer_mm("-30:5") == (-30e-3, 5e-3)
end

@testset "window and output config" begin
    opts, _ = TranscranialFUS.parse_cli(["--recon-window-us=12", "--recon-hop-us=6", "--recon-window-taper=tukey"])
    win = TranscranialFUS.make_window_config(opts, :windowed)
    @test win.enabled
    @test win.window_duration ≈ 12e-6
    @test win.hop ≈ 6e-6
    @test win.taper == :tukey

    sources = [PointSource2D(depth=0.03, lateral=0.0, frequency=0.4e6)]
    cfg = PAMConfig(axial_dim=0.04, transverse_dim=0.03)
    out = TranscranialFUS.default_output_dir(opts, sources, cfg, Dict("source_model" => :point))
    @test occursin("run_pam_2d", out)
    @test occursin("point", out)

    @test occursin("reconstruct_example_run", basename(TranscranialFUS.default_reconstruction_output_dir("/tmp/example_run")))
    @test_throws ErrorException TranscranialFUS.reject_cached_simulation_options!(Set(["sources-mm"]), ["sources-mm", "dx-mm"])
end
