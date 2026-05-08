#!/usr/bin/env julia

include(joinpath(@__DIR__, "run_pam_3d_vessel_sweep.jl"))

function rescue_sweep_jobs(opts)
    base = vessel_base_args(opts)
    weak_locations = [
        ("depth52", "52:0:0", "Deeper centered vessel."),
        ("depth62", "62:0:0", "Deep centered vessel."),
        ("y_p9", "42:9:0", "Moderate positive lateral-y offset."),
        ("y_p18", "42:18:0", "Large positive lateral-y offset."),
        ("diag_p12_m12", "42:12:-12", "Mixed-sign positive-y diagonal offset."),
        ("diag_p12_p12", "42:12:12", "Positive-y/positive-z diagonal offset."),
    ]
    param_sets = [
        ("bw60_jitter2", Dict("recon-bandwidth-khz" => "60", "frequency-jitter-percent" => "2"), "60 kHz / 2% tight-band high-F1 setting."),
        ("bw80_jitter4", Dict("recon-bandwidth-khz" => "80", "frequency-jitter-percent" => "4"), "80 kHz / 4% formula-matched setting."),
    ]
    jobs = Dict{String, Any}[]
    for (loc_id, anchor, loc_note) in weak_locations
        for (param_id, params, param_note) in param_sets
            id = "rescue_$(loc_id)_$(param_id)"
            args = merge_args(base, Dict("anchors-mm" => anchor), params)
            push!(jobs, sim_job(id, args; note="$loc_note $param_note"))
        end
    end
    return jobs
end

function rescue_sweep_main()
    opts = vessel_sweep_opts(ARGS)
    dry_run = parse_bool(opts["dry-run"])
    force = parse_bool(opts["force"])
    max_seconds = parse(Float64, opts["max-hours"]) * 3600
    timeout_seconds = parse(Float64, opts["per-run-timeout-min"]) * 60
    out_root = isempty(strip(opts["output-root"])) ?
        joinpath(PROJECT_ROOT, "outputs", "$(timestamp())_pam_3d_location_rescue_sweep") :
        abspath(opts["output-root"])
    mkpath(out_root)

    jobs = rescue_sweep_jobs(opts)
    write_json(joinpath(out_root, "manifest.json"), Dict{String, Any}(
        "created_at" => string(Dates.now()),
        "output_root" => out_root,
        "max_hours" => parse(Float64, opts["max-hours"]),
        "per_run_timeout_min" => parse(Float64, opts["per-run-timeout-min"]),
        "job_count" => length(jobs),
        "selection_metric" => "source_f1",
        "jobs" => jobs,
    ))

    println("Output root: ", out_root)
    println("Jobs queued: ", length(jobs))
    println("Max hours: ", opts["max-hours"], " | per-run timeout min: ", opts["per-run-timeout-min"])
    dry_run && println("Dry run only; no jobs will be executed.")

    rows = Dict{String, Any}[]
    csv_path = joinpath(out_root, "results.csv")
    start_all = time()
    for (idx, job) in pairs(jobs)
        remaining_seconds = max_seconds - (time() - start_all)
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
    println("3D location rescue sweep complete. Results: ", out_root)
    if isempty(leaders)
        println("No successful jobs with activity-boundary metrics yet.")
    else
        println("Top HASA source-F1:")
        for row in first(leaders, min(10, length(leaders)))
            @printf("  %.3f  thr=%.2f  prec=%.3f  rec=%.3f  %s\n",
                Float64(row["best_hasa_f1"]),
                Float64(row["best_hasa_threshold"]),
                Float64(get(row, "best_hasa_precision", NaN)),
                Float64(get(row, "best_hasa_recall", NaN)),
                row["id"],
            )
        end
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    rescue_sweep_main()
end
