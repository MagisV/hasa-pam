#!/usr/bin/env julia

include(joinpath(@__DIR__, "run_pam_overnight_sweep.jl"))

function focused_sweep_opts(args)
    opts = parse_cli(args)
    provided = Set(first(split(arg[3:end], "="; limit=2)) for arg in args if startswith(arg, "--"))
    !("max-hours" in provided) && (opts["max-hours"] = "0.75")
    !("per-run-timeout-min" in provided) && (opts["per-run-timeout-min"] = "8")
    !("output-root" in provided) && (opts["output-root"] = "")
    !("random-seed" in provided) && (opts["random-seed"] = "42")
    !("boundary-threshold-ratios" in provided) &&
        (opts["boundary-threshold-ratios"] = "0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9")
    !("auto-threshold-search" in provided) && (opts["auto-threshold-search"] = "true")
    !("auto-threshold-min" in provided) && (opts["auto-threshold-min"] = "0.10")
    !("auto-threshold-max" in provided) && (opts["auto-threshold-max"] = "0.95")
    !("auto-threshold-step" in provided) && (opts["auto-threshold-step"] = "0.01")
    return opts
end

function focused_base_args(opts)
    return Dict(
        "dimension" => "3",
        "source-model" => "squiggle",
        "gate-us" => "45",
        "anchors-mm" => "42:0:0",
        "vascular-length-mm" => "12",
        "vascular-squiggle-amplitude-mm" => "0.3",
        "vascular-squiggle-amplitude-x-mm" => "0.2",
        "vascular-squiggle-wavelength-mm" => "8",
        "squiggle-phase-x-deg" => "90",
        "vascular-source-spacing-mm" => "0.5",
        "vascular-min-separation-mm" => "0.25",
        "harmonics" => "2,3,4",
        "harmonic-amplitudes" => "1.0,0.6,0.3",
        "aberrator" => "skull",
        "skull-transducer-distance-mm" => "20",
        "slice-index" => opts["slice-index"],
        "axial-mm" => "70",
        "transverse-mm" => "64",
        "dx-mm" => "0.2",
        "dy-mm" => "0.5",
        "dz-mm" => "0.5",
        "t-max-us" => "250",
        "frequency-mhz" => "0.5",
        "receiver-aperture-mm" => "full",
        "source-phase-mode" => "random_phase_per_window",
        "recon-window-us" => "40",
        "recon-hop-us" => "20",
        "recon-bandwidth-khz" => "500",
        "boundary-threshold-ratios" => opts["boundary-threshold-ratios"],
        "auto-threshold-search" => opts["auto-threshold-search"],
        "auto-threshold-min" => opts["auto-threshold-min"],
        "auto-threshold-max" => opts["auto-threshold-max"],
        "auto-threshold-step" => opts["auto-threshold-step"],
        "frequency-jitter-percent" => "1",
        "axial-gain-power" => "1.5",
        "sim-mode" => "kwave",
        "use-gpu" => "true",
        "window-batch" => "2",
        "recon-progress" => "false",
        "random-seed" => opts["random-seed"],
    )
end

function focused_sweep_jobs(opts)
    base = focused_base_args(opts)
    specs = [
        ("bw400", Dict("recon-bandwidth-khz" => "400"), "Narrower bandwidth: fewer FFT bins, possible artifact reduction."),
        ("bw300", Dict("recon-bandwidth-khz" => "300"), "Aggressive bandwidth trim for speed and elongated-artifact check."),
        ("axgain075_bw400", Dict("recon-bandwidth-khz" => "400", "axial-gain-power" => "0.75"), "Lower axial gain may reduce stretched deep artifacts."),
        ("axgain000_bw400", Dict("recon-bandwidth-khz" => "400", "axial-gain-power" => "0.0"), "No axial gain: artifact/precision diagnostic."),
        ("fjitter3_bw400", Dict("recon-bandwidth-khz" => "400", "frequency-jitter-percent" => "3"), "More source variability to decorrelate off-source interference."),
        ("fjitter5_bw400", Dict("recon-bandwidth-khz" => "400", "frequency-jitter-percent" => "5"), "High source variability stress test."),
        ("w30_bw400", Dict("recon-window-us" => "30", "recon-hop-us" => "15", "recon-bandwidth-khz" => "400"), "Shorter windows: more phase realizations, tighter time support."),
        ("w50_bw400", Dict("gate-us" => "50", "recon-window-us" => "50", "recon-hop-us" => "25", "recon-bandwidth-khz" => "400"), "Longer windows: fewer FFT batches and more per-window energy."),
        ("dense03_bw400", Dict("vascular-source-spacing-mm" => "0.3", "vascular-min-separation-mm" => "0.15", "recon-bandwidth-khz" => "400"), "More realistic denser bubble sampling, moderate source count."),
        ("batch4_bw400", Dict("recon-bandwidth-khz" => "400", "window-batch" => "4"), "Runtime-only check: larger window batch after accumulator memory fix."),
    ]
    return [sim_job(id, merge_args(base, args); note=note) for (id, args, note) in specs]
end

function focused_sweep_main()
    opts = focused_sweep_opts(ARGS)
    dry_run = parse_bool(opts["dry-run"])
    force = parse_bool(opts["force"])
    max_seconds = parse(Float64, opts["max-hours"]) * 3600
    timeout_seconds = parse(Float64, opts["per-run-timeout-min"]) * 60
    out_root = isempty(strip(opts["output-root"])) ?
        joinpath(PROJECT_ROOT, "outputs", "$(timestamp())_pam_3d_focused_sweep") :
        abspath(opts["output-root"])
    mkpath(out_root)

    jobs = focused_sweep_jobs(opts)
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
    println("Focused 3D PAM sweep complete. Results: ", out_root)
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
    focused_sweep_main()
end
