#!/usr/bin/env julia

include(joinpath(@__DIR__, "run_pam_overnight_sweep.jl"))

function jitter_bandwidth_opts(args)
    opts = parse_cli(args)
    provided = Set(first(split(arg[3:end], "="; limit=2)) for arg in args if startswith(arg, "--"))
    !("max-hours" in provided) && (opts["max-hours"] = "0.75")
    !("per-run-timeout-min" in provided) && (opts["per-run-timeout-min"] = "12")
    !("output-root" in provided) && (opts["output-root"] = "")
    !("random-seed" in provided) && (opts["random-seed"] = "42")
    !("auto-threshold-search" in provided) && (opts["auto-threshold-search"] = "true")
    !("auto-threshold-min" in provided) && (opts["auto-threshold-min"] = "0.10")
    !("auto-threshold-max" in provided) && (opts["auto-threshold-max"] = "0.95")
    !("auto-threshold-step" in provided) && (opts["auto-threshold-step"] = "0.01")
    return opts
end

function jitter_bandwidth_base_args(opts)
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
        "frequency-jitter-percent" => "10",
        "recon-window-us" => "40",
        "recon-hop-us" => "20",
        "auto-threshold-search" => opts["auto-threshold-search"],
        "auto-threshold-min" => opts["auto-threshold-min"],
        "auto-threshold-max" => opts["auto-threshold-max"],
        "auto-threshold-step" => opts["auto-threshold-step"],
        "sim-mode" => "kwave",
        "use-gpu" => "true",
        "window-batch" => "2",
        "recon-progress" => "false",
        "random-seed" => opts["random-seed"],
    )
end

function jitter_bandwidth_jobs(opts)
    base = jitter_bandwidth_base_args(opts)
    specs = [
        ("fjitter10_bw80",   Dict("recon-bandwidth-khz" => "80",  "frequency-jitter-percent" => "10"), "10% frequency jitter with very tight harmonic search bands."),
        ("fjitter75_bw150",  Dict("recon-bandwidth-khz" => "150", "frequency-jitter-percent" => "7.5"), "7.5% frequency jitter with 150 kHz harmonic search bands."),
        ("fjitter9_bw180",   Dict("recon-bandwidth-khz" => "180", "frequency-jitter-percent" => "9"),   "9% frequency jitter with 180 kHz harmonic search bands."),
        ("fjitter10_bw200",  Dict("recon-bandwidth-khz" => "200", "frequency-jitter-percent" => "10"),  "10% frequency jitter with moderate harmonic search bands."),
        ("fjitter11_bw220",  Dict("recon-bandwidth-khz" => "220", "frequency-jitter-percent" => "11"),  "11% frequency jitter with 220 kHz harmonic search bands."),
        ("fjitter10_bw400",  Dict("recon-bandwidth-khz" => "400", "frequency-jitter-percent" => "10"),  "10% frequency jitter with broad harmonic search bands."),
        ("fjitter4_bw80",    Dict("recon-bandwidth-khz" => "80",  "frequency-jitter-percent" => "4"),   "4% frequency jitter with 80 kHz bands (formula-matched)."),
        ("fjitter20_bw80",   Dict("recon-bandwidth-khz" => "80",  "frequency-jitter-percent" => "20"),  "20% frequency jitter with tight 80 kHz bands (under-matched)."),
        ("fjitter30_bw80",   Dict("recon-bandwidth-khz" => "80",  "frequency-jitter-percent" => "30"),  "30% frequency jitter with tight 80 kHz bands (heavily under-matched)."),
        ("fjitter4_bw80",    Dict("recon-bandwidth-khz" => "80",  "frequency-jitter-percent" => "4"),   "4% frequency jitter with 80 kHz bands (formula-matched, clean timing run)."),
        ("fjitter2_bw60",    Dict("recon-bandwidth-khz" => "60",  "frequency-jitter-percent" => "2"),   "2% frequency jitter with 60 kHz bands (under-matched, tight regime)."),
        ("fjitter1_bw40",    Dict("recon-bandwidth-khz" => "40",  "frequency-jitter-percent" => "1"),   "1% frequency jitter with 40 kHz bands (tight regime)."),
        ("fjitter05_bw20",   Dict("recon-bandwidth-khz" => "20",  "frequency-jitter-percent" => "0.5"), "0.5% frequency jitter with 20 kHz bands (very tight regime)."),
        ("fjitter05_bw40",   Dict("recon-bandwidth-khz" => "40",  "frequency-jitter-percent" => "0.5"), "0.5% frequency jitter with 40 kHz bands (under-matched)."),
        ("fjitter15_bw40",   Dict("recon-bandwidth-khz" => "40",  "frequency-jitter-percent" => "1.5"), "1.5% frequency jitter with 40 kHz bands (over-matched)."),
        ("fjitter1_bw20",    Dict("recon-bandwidth-khz" => "20",  "frequency-jitter-percent" => "1"),   "1% frequency jitter with 20 kHz bands."),
    ]
    return [sim_job(id, merge_args(base, args); note=note) for (id, args, note) in specs]
end

function jitter_bandwidth_main()
    opts = jitter_bandwidth_opts(ARGS)
    dry_run = parse_bool(opts["dry-run"])
    force = parse_bool(opts["force"])
    max_seconds = parse(Float64, opts["max-hours"]) * 3600
    timeout_seconds = parse(Float64, opts["per-run-timeout-min"]) * 60
    out_root = isempty(strip(opts["output-root"])) ?
        joinpath(PROJECT_ROOT, "outputs", "$(timestamp())_pam_3d_jitter_bandwidth_sweep") :
        abspath(opts["output-root"])
    mkpath(out_root)

    jobs = jitter_bandwidth_jobs(opts)
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
    end

    completed = [row for row in rows if get(row, "status", "") in ("success", "skipped_existing") && haskey(row, "best_hasa_f1")]
    if !isempty(completed)
        best = first(completed)
        for row in completed[2:end]
            if get(row, "best_hasa_f1", -Inf) > get(best, "best_hasa_f1", -Inf)
                best = row
            end
        end
        println("\nBest HASA source F1: ", get(best, "best_hasa_f1", "n/a"), " (", best["id"], ")")
    end
    println("Results written to ", out_root)
end

jitter_bandwidth_main()
