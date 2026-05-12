@testset "run_pam script entrypoint" begin
    include("../../scripts/run_pam.jl")

    mktempdir() do dir
        out2d = joinpath(dir, "out2d")
        dry2d = main([
            "--source-model=point",
            "--sources-mm=30:0",
            "--out-dir=$out2d",
            "--boundary-threshold-ratios=0.4,0.8",
        ]; dry_run=true)
        @test dry2d[:branch] == :pam2d_simulation
        @test dry2d[:out_dir] == out2d
        @test dry2d[:source_model] == :point
        @test dry2d[:source_count] == 1
        @test dry2d[:threshold_ratios] == [0.4, 0.8]

        out3d = joinpath(dir, "out3d")
        dry3d = main([
            "--dimension=3",
            "--source-model=point",
            "--sources-mm=20:0:0",
            "--out-dir=$out3d",
            "--auto-threshold-search=true",
            "--auto-threshold-min=0.2",
            "--auto-threshold-max=0.5",
            "--auto-threshold-step=0.2",
        ]; dry_run=true)
        @test dry3d[:branch] == :pam3d
        @test dry3d[:out_dir] == out3d
        @test dry3d[:source_model] == :point
        @test dry3d[:threshold_score_ratios] == [0.2, 0.4, 0.5]

        cached_dir = joinpath(dir, "cached")
        mkpath(cached_dir)
        cached_path = joinpath(cached_dir, "result.jld2")
        write(cached_path, UInt8[])
        cached = main([
            "--from-run-dir=$cached_dir",
            "--out-dir=$(joinpath(dir, "cached_out"))",
        ]; dry_run=true)
        @test cached[:branch] == :pam2d_cached
        @test cached[:cached_path] == cached_path
        @test cached[:reconstruction_source]["mode"] == "cached_rf"

        @test_throws ErrorException main(["--dimension=3", "--from-run-dir=$cached_dir"]; dry_run=true)
        @test_throws ErrorException main(["--dimension=3", "--aberrator=water"]; dry_run=true)
        @test_throws ErrorException main(["--from-run-dir=$(joinpath(dir, "missing"))"]; dry_run=true)
        @test_throws ErrorException main(["--from-run-dir=$cached_dir", "--sources-mm=30:0"]; dry_run=true)
    end
end
