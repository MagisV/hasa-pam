#!/usr/bin/env julia

include(joinpath(@__DIR__, "run_pam_overnight_sweep.jl"))

function aperture_sweep_opts(args)
    opts = parse_cli(args)
    get!(opts, "max-hours", "8.75")
    get!(opts, "per-run-timeout-min", "90")
    get!(opts, "output-root", "")
    get!(opts, "anchors-mm", "45:0")
    get!(opts, "simulation-random-seeds", "42,43,44")
    get!(opts, "apertures-mm", "50,60,100,full,40,30,20")
    get!(opts, "transverse-mm", "102.4")
    get!(opts, "slice-index", "250")
    get!(opts, "skull-transducer-distance-mm", "30")
    get!(opts, "boundary-threshold-ratios", "0.6,0.65,0.7")
    get!(opts, "random-seed", first(parse_string_list(opts["simulation-random-seeds"])))
    return opts
end

function aperture_label(aperture::AbstractString)
    lower = lowercase(strip(aperture))
    lower in ("full", "all", "none") && return "full"
    return replace(lower, "." => "p")
end

function aperture_sweep_jobs(opts)
    apertures = parse_string_list(opts["apertures-mm"])
    seeds = parse_string_list(opts["simulation-random-seeds"])
    isempty(apertures) && error("--apertures-mm must contain at least one aperture.")
    isempty(seeds) && error("--simulation-random-seeds must contain at least one seed.")

    base = Dict(
        "source-model" => "squiggle",
        "anchors-mm" => opts["anchors-mm"],
        "vascular-length-mm" => "12",
        "aberrator" => "skull",
        "slice-index" => opts["slice-index"],
        "skull-transducer-distance-mm" => opts["skull-transducer-distance-mm"],
        "transverse-mm" => opts["transverse-mm"],
        "receiver-aperture-mm" => "",
        "boundary-threshold-ratios" => opts["boundary-threshold-ratios"],
        "recon-min-window-energy-ratio" => "0.001",
        "cavitation-model" => "harmonic-cos",
        "harmonics" => "2,3,4",
        "harmonic-amplitudes" => "1.0,0.6,0.3",
        "source-phase-mode" => "random_phase_per_window",
        "recon-window-us" => "20",
        "recon-hop-us" => "10",
        "recon-bandwidth-khz" => "500",
        "t-max-us" => "500",
        "frequency-jitter-percent" => "1",
    )

    jobs = Dict{String, Any}[]
    for seed in seeds
        for aperture in apertures
            id = "aperture_ap$(aperture_label(aperture))_seed$(seed)"
            args = merge_args(
                base,
                Dict(
                    "receiver-aperture-mm" => aperture,
                    "random-seed" => seed,
                ),
            )
            note = "Focused aperture sweep for h234 random-phase-per-window w20/bw500/t500 on a $(opts["transverse-mm"]) mm transverse grid."
            push!(jobs, sim_job(id, args; note=note))
        end
    end
    return jobs
end

function aperture_sweep_main()
    opts = aperture_sweep_opts(ARGS)
    dry_run = parse_bool(opts["dry-run"])
    force = parse_bool(opts["force"])
    max_seconds = parse(Float64, opts["max-hours"]) * 3600
    timeout_seconds = parse(Float64, opts["per-run-timeout-min"]) * 60
    out_root = isempty(strip(opts["output-root"])) ?
        joinpath(PROJECT_ROOT, "outputs", "$(timestamp())_pam_aperture_sweep") :
        abspath(opts["output-root"])
    mkpath(out_root)

    jobs = aperture_sweep_jobs(opts)
    manifest = Dict{String, Any}(
        "created_at" => string(Dates.now()),
        "output_root" => out_root,
        "max_hours" => parse(Float64, opts["max-hours"]),
        "per_run_timeout_min" => parse(Float64, opts["per-run-timeout-min"]),
        "apertures_mm" => parse_string_list(opts["apertures-mm"]),
        "simulation_random_seeds" => parse_string_list(opts["simulation-random-seeds"]),
        "transverse_mm" => opts["transverse-mm"],
        "job_count" => length(jobs),
        "jobs" => jobs,
    )
    write_json(joinpath(out_root, "manifest.json"), manifest)

    println("Output root: ", out_root)
    println("Jobs queued: ", length(jobs))
    println("Apertures mm: ", opts["apertures-mm"])
    println("Seeds: ", opts["simulation-random-seeds"])
    println("Transverse grid mm: ", opts["transverse-mm"])
    println("Max hours: ", opts["max-hours"], " | per-run timeout min: ", opts["per-run-timeout-min"])
    dry_run && println("Dry run only; no jobs will be executed.")

    rows = Dict{String, Any}[]
    csv_path = joinpath(out_root, "results.csv")
    start_all = time()
    for (idx, job) in pairs(jobs)
        elapsed_all = time() - start_all
        remaining_seconds = max_seconds - elapsed_all
        if !dry_run && remaining_seconds <= 0
            println("Time budget exhausted before job ", idx, "; stopping.")
            break
        end
        job_timeout_seconds = dry_run ? timeout_seconds : min(timeout_seconds, remaining_seconds)
        row = run_job(job, out_root, idx, length(jobs), job_timeout_seconds, dry_run, force)
        push!(rows, row)
        append_csv(csv_path, row)
        write_json(joinpath(out_root, "results.json"), rows)
        write_json(joinpath(out_root, "leaderboard.json"), leaderboard_rows(rows))
    end

    leaders = leaderboard_rows(rows)
    println()
    println("Aperture sweep complete. Results: ", out_root)
    if isempty(leaders)
        println("No successful jobs with activity-boundary metrics yet.")
    else
        println("Top HASA F1:")
        for row in first(leaders, min(10, length(leaders)))
            @printf("  %.3f  thr=%.2f  %s  %s\n",
                Float64(row["best_hasa_f1"]),
                Float64(row["best_hasa_threshold"]),
                row["id"],
                row["out_dir"],
            )
        end
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    aperture_sweep_main()
end
