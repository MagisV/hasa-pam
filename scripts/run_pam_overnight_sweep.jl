#!/usr/bin/env julia

using Dates
using JSON3
using Printf

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const PAM_SCRIPT = joinpath(PROJECT_ROOT, "scripts", "run_pam.jl")

function parse_cli(args)
    opts = Dict{String, String}(
        "max-hours" => "8.75",
        "per-run-timeout-min" => "90",
        "output-root" => "",
        "from-run-dir" => "",
        "anchors-mm" => "45:0",
        "random-seed" => "42",
        "simulation-random-seeds" => "42,43,44,45,46",
        "slice-index" => "250",
        "skull-transducer-distance-mm" => "30",
        "boundary-threshold-ratios" => "0.6,0.65,0.7",
        "auto-threshold-search" => "true",
        "auto-threshold-min" => "0.10",
        "auto-threshold-max" => "0.95",
        "auto-threshold-step" => "0.01",
        "use-gpu" => "false",
        "dry-run" => "false",
        "force" => "false",
    )
    for arg in args
        startswith(arg, "--") || error("Unsupported argument format: $arg")
        key_value = split(arg[3:end], "="; limit=2)
        length(key_value) == 2 || error("Arguments must use --name=value, got: $arg")
        opts[key_value[1]] = key_value[2]
    end
    return opts
end

parse_bool(s::AbstractString) = lowercase(strip(s)) in ("1", "true", "yes", "on")
parse_string_list(s::AbstractString) = [strip(item) for item in split(s, ",") if !isempty(strip(item))]

function timestamp()
    return Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
end

function slug(s::AbstractString)
    out = lowercase(strip(s))
    out = replace(out, r"[^a-z0-9]+" => "_")
    out = replace(out, r"^_+|_+$" => "")
    return isempty(out) ? "job" : out
end

function merge_args(pairs...)
    args = Dict{String, String}()
    for group in pairs
        for (key, value) in group
            args[String(key)] = String(value)
        end
    end
    return args
end

function common_sim_args(opts; seed=opts["random-seed"])
    return Dict(
        "source-model" => "squiggle",
        "anchors-mm" => opts["anchors-mm"],
        "vascular-length-mm" => "12",
        "aberrator" => "skull",
        "slice-index" => opts["slice-index"],
        "skull-transducer-distance-mm" => opts["skull-transducer-distance-mm"],
        "random-seed" => seed,
        "use-gpu" => opts["use-gpu"],
        "boundary-threshold-ratios" => opts["boundary-threshold-ratios"],
        "auto-threshold-search" => opts["auto-threshold-search"],
        "auto-threshold-min" => opts["auto-threshold-min"],
        "auto-threshold-max" => opts["auto-threshold-max"],
        "auto-threshold-step" => opts["auto-threshold-step"],
        "recon-min-window-energy-ratio" => "0.001",
    )
end

function recon_job(id, args; note="")
    return Dict{String, Any}(
        "id" => id,
        "kind" => "cached_reconstruction",
        "args" => args,
        "note" => note,
    )
end

function sim_job(id, args; note="")
    return Dict{String, Any}(
        "id" => id,
        "kind" => "simulation",
        "args" => args,
        "note" => note,
    )
end

function cached_reconstruction_jobs(opts)
    from_run_dir = opts["from-run-dir"]
    isempty(strip(from_run_dir)) && return Dict{String, Any}[]
    base = Dict(
        "from-run-dir" => from_run_dir,
        "boundary-threshold-ratios" => opts["boundary-threshold-ratios"],
        "auto-threshold-search" => opts["auto-threshold-search"],
        "auto-threshold-min" => opts["auto-threshold-min"],
        "auto-threshold-max" => opts["auto-threshold-max"],
        "auto-threshold-step" => opts["auto-threshold-step"],
    )
    specs = [
        ("cached_w20_bw300_step50", Dict("recon-window-us" => "20", "recon-hop-us" => "10", "recon-bandwidth-khz" => "300", "recon-step-um" => "50")),
        ("cached_w20_bw500_step50", Dict("recon-window-us" => "20", "recon-hop-us" => "10", "recon-bandwidth-khz" => "500", "recon-step-um" => "50")),
        ("cached_w20_bw700_step50", Dict("recon-window-us" => "20", "recon-hop-us" => "10", "recon-bandwidth-khz" => "700", "recon-step-um" => "50")),
        ("cached_w30_bw300_step50", Dict("recon-window-us" => "30", "recon-hop-us" => "15", "recon-bandwidth-khz" => "300", "recon-step-um" => "50")),
        ("cached_w30_bw500_step50", Dict("recon-window-us" => "30", "recon-hop-us" => "15", "recon-bandwidth-khz" => "500", "recon-step-um" => "50")),
        ("cached_w40_bw500_step50", Dict("recon-window-us" => "40", "recon-hop-us" => "20", "recon-bandwidth-khz" => "500", "recon-step-um" => "50")),
        ("cached_w20_bw500_step25", Dict("recon-window-us" => "20", "recon-hop-us" => "10", "recon-bandwidth-khz" => "500", "recon-step-um" => "25")),
        ("cached_w30_bw500_step25", Dict("recon-window-us" => "30", "recon-hop-us" => "15", "recon-bandwidth-khz" => "500", "recon-step-um" => "25")),
    ]
    return [recon_job(id, merge_args(base, args); note="Reuse cached RF and sweep only reconstruction/window/bandwidth settings.") for (id, args) in specs]
end

function simulation_jobs(opts)
    seeds = parse_string_list(opts["simulation-random-seeds"])
    isempty(seeds) && (seeds = [opts["random-seed"]])
    primary_seed = first(seeds)
    secondary_seeds = seeds[2:end]

    priority_specs = [
        (
            "sim_squiggle_h234_rpw_w20_bw500_t500",
            Dict(
                "cavitation-model" => "harmonic-cos",
                "harmonics" => "2,3,4",
                "harmonic-amplitudes" => "1.0,0.6,0.3",
                "source-phase-mode" => "random_phase_per_window",
                "recon-window-us" => "20",
                "recon-hop-us" => "10",
                "recon-bandwidth-khz" => "500",
                "t-max-us" => "500",
                "frequency-jitter-percent" => "1",
            ),
            "Current best harmonic squiggle random-phase-per-window setting.",
        ),
        (
            "sim_squiggle_h234_rpw_w30_bw500_t500",
            Dict(
                "cavitation-model" => "harmonic-cos",
                "harmonics" => "2,3,4",
                "harmonic-amplitudes" => "1.0,0.6,0.3",
                "source-phase-mode" => "random_phase_per_window",
                "recon-window-us" => "30",
                "recon-hop-us" => "15",
                "recon-bandwidth-khz" => "500",
                "t-max-us" => "500",
                "frequency-jitter-percent" => "1",
            ),
            "Longer window comparison.",
        ),
        (
            "sim_squiggle_gaussian_h234_rpw_w20_bw700_t500",
            Dict(
                "cavitation-model" => "gaussian-pulse",
                "harmonics" => "2,3,4",
                "harmonic-amplitudes" => "1.0,0.6,0.3",
                "source-phase-mode" => "random_phase_per_window",
                "recon-window-us" => "20",
                "recon-hop-us" => "10",
                "recon-bandwidth-khz" => "700",
                "t-max-us" => "500",
                "frequency-jitter-percent" => "1",
            ),
            "Gaussian pulse comparison using the same squiggle geometry.",
        ),
    ]

    jobs = Dict{String, Any}[]

    for (id, args, note) in priority_specs
        push!(jobs, sim_job(id, merge_args(common_sim_args(opts; seed=primary_seed), args); note=note))
    end

    for seed in secondary_seeds
        for (id, args, note) in priority_specs[1:min(7, length(priority_specs))]
            seeded_id = "$(id)_seed$(seed)"
            seeded_note = "$note Seed repeat to check robustness."
            push!(jobs, sim_job(seeded_id, merge_args(common_sim_args(opts; seed=seed), args); note=seeded_note))
        end
    end

    return jobs
end

function make_command(job, out_dir)
    args = copy(job["args"])
    args["out-dir"] = out_dir
    cmd = `$(Base.julia_cmd()) --project=$(PROJECT_ROOT) $(PAM_SCRIPT)`
    for key in sort(collect(keys(args)))
        cmd = `$cmd --$key=$(args[key])`
    end
    return cmd
end

function maybe_float_property(obj, field::Symbol)
    hasproperty(obj, field) || return nothing
    value = getproperty(obj, field)
    isnothing(value) && return nothing
    try
        return Float64(value)
    catch
        return nothing
    end
end

function copy_float_property!(metrics, key::AbstractString, obj, field::Symbol)
    value = maybe_float_property(obj, field)
    isnothing(value) || (metrics[key] = value)
    return metrics
end

function read_summary_metrics(out_dir)
    summary_path = joinpath(out_dir, "summary.json")
    isfile(summary_path) || return Dict{String, Any}("summary_found" => false)
    summary = try
        JSON3.read(read(summary_path, String))
    catch err
        return Dict{String, Any}(
            "summary_found" => false,
            "summary_error" => sprint(showerror, err),
        )
    end
    metrics = Dict{String, Any}("summary_found" => true)
    if hasproperty(summary, :activity_boundary_metrics)
        boundary_metrics = summary.activity_boundary_metrics
        best_hasa = nothing
        best_hasa_f1 = -Inf
        best_geo = nothing
        best_geo_f1 = -Inf
        if hasproperty(boundary_metrics, :hasa)
            for entry in boundary_metrics.hasa
                f1 = maybe_float_property(entry, :f1)
                if !isnothing(f1) && f1 > best_hasa_f1
                    best_hasa = entry
                    best_hasa_f1 = f1
                end
            end
        end
        if hasproperty(boundary_metrics, :geometric)
            for entry in boundary_metrics.geometric
                f1 = maybe_float_property(entry, :f1)
                if !isnothing(f1) && f1 > best_geo_f1
                    best_geo = entry
                    best_geo_f1 = f1
                end
            end
        end
        if !isnothing(best_hasa)
            metrics["best_hasa_f1"] = best_hasa_f1
            copy_float_property!(metrics, "best_hasa_threshold", best_hasa, :threshold_ratio)
            copy_float_property!(metrics, "best_hasa_precision", best_hasa, :precision)
            copy_float_property!(metrics, "best_hasa_recall", best_hasa, :recall)
        end
        if !isnothing(best_geo)
            metrics["best_geo_f1"] = best_geo_f1
            copy_float_property!(metrics, "best_geo_threshold", best_geo, :threshold_ratio)
        end
    end
    if hasproperty(summary, :hasa)
        copy_float_property!(metrics, "hasa_psf_corr", summary.hasa, :psf_target_correlation)
        copy_float_property!(metrics, "hasa_psf_l2", summary.hasa, :psf_target_normalized_l2_error)
        copy_float_property!(metrics, "hasa_centroid_error_mm", summary.hasa, :centroid_error_mm)
    end
    return metrics
end

function write_json(path, value)
    open(path, "w") do io
        JSON3.pretty(io, value)
        println(io)
    end
end

function append_csv(path, row)
    is_new = !isfile(path)
    headers = [
        "id",
        "kind",
        "status",
        "elapsed_min",
        "best_hasa_f1",
        "best_hasa_threshold",
        "best_hasa_precision",
        "best_hasa_recall",
        "best_geo_f1",
        "hasa_psf_corr",
        "hasa_psf_l2",
        "out_dir",
    ]
    open(path, "a") do io
        if is_new
            println(io, join(headers, ","))
        end
        vals = [replace(string(get(row, h, "")), "," => ";") for h in headers]
        println(io, join(vals, ","))
    end
end

function terminate_process!(proc, io; grace_seconds::Real=15)
    try
        kill(proc)
    catch err
        println(io, "# SIGTERM error: ", err)
        flush(io)
    end

    deadline = time() + grace_seconds
    while process_running(proc) && time() < deadline
        sleep(1)
    end

    if process_running(proc)
        println(io, "# Process still running after SIGTERM; sending SIGKILL.")
        flush(io)
        try
            kill(proc, Base.SIGKILL)
        catch err
            println(io, "# SIGKILL error: ", err)
            flush(io)
        end
        deadline = time() + 5
        while process_running(proc) && time() < deadline
            sleep(0.5)
        end
    end
    return !process_running(proc)
end

function wait_if_finished(proc, io)
    process_running(proc) && return false
    try
        wait(proc)
        return true
    catch err
        println(io)
        println(io, "# wait(proc) error: ", err)
        flush(io)
        return false
    end
end

function process_success(proc)
    try
        return success(proc)
    catch
        return false
    end
end

function run_job(job, out_root, index, total, timeout_seconds, dry_run, force)
    id = job["id"]
    job_dir = joinpath(out_root, @sprintf("%02d_%s", index, slug(id)))
    mkpath(job_dir)
    out_dir = joinpath(job_dir, "run")
    log_path = joinpath(job_dir, "run.log")
    status_path = joinpath(job_dir, "status.json")
    cmd = make_command(job, out_dir)

    if !force && isfile(joinpath(out_dir, "summary.json"))
        metrics = read_summary_metrics(out_dir)
        row = merge(
            Dict{String, Any}(
                "id" => id,
                "kind" => job["kind"],
                "status" => "skipped_existing",
                "elapsed_min" => 0.0,
                "out_dir" => out_dir,
                "cmd" => string(cmd),
            ),
            metrics,
        )
        write_json(status_path, row)
        return row
    end

    row = Dict{String, Any}(
        "id" => id,
        "kind" => job["kind"],
        "status" => dry_run ? "dry_run" : "running",
        "elapsed_min" => 0.0,
        "out_dir" => out_dir,
        "cmd" => string(cmd),
        "note" => get(job, "note", ""),
    )
    write_json(status_path, row)

    println()
    println("[$index/$total] ", id)
    println("  kind: ", job["kind"])
    println("  out:  ", out_dir)
    println("  log:  ", log_path)
    println("  cmd:  ", cmd)

    if dry_run
        return row
    end

    start = time()
    timed_out = false
    exit_ok = false
    open(log_path, "w") do io
        println(io, "# ", Dates.now())
        println(io, "# ", cmd)
        flush(io)
        proc = run(pipeline(cmd; stdout=io, stderr=io); wait=false)
        while process_running(proc)
            sleep(5)
            if timeout_seconds > 0 && (time() - start) > timeout_seconds
                timed_out = true
                println(io)
                println(io, "# TIMEOUT after ", round((time() - start) / 60; digits=2), " min; killing process.")
                flush(io)
                terminate_process!(proc, io)
                break
            end
        end
        finished = wait_if_finished(proc, io)
        exit_ok = !timed_out && finished && process_success(proc)
    end

    elapsed_min = (time() - start) / 60
    metrics = read_summary_metrics(out_dir)
    row = merge(
        row,
        metrics,
        Dict{String, Any}(
            "status" => timed_out ? "timeout" : (exit_ok ? "success" : "failed"),
            "elapsed_min" => round(elapsed_min; digits=2),
        ),
    )
    write_json(status_path, row)
    return row
end

function leaderboard_rows(rows)
    successful = [row for row in rows if get(row, "status", "") in ("success", "skipped_existing") && haskey(row, "best_hasa_f1")]
    sort!(successful; by=row -> Float64(row["best_hasa_f1"]), rev=true)
    return successful
end

function main()
    opts = parse_cli(ARGS)
    dry_run = parse_bool(opts["dry-run"])
    force = parse_bool(opts["force"])
    max_seconds = parse(Float64, opts["max-hours"]) * 3600
    timeout_seconds = parse(Float64, opts["per-run-timeout-min"]) * 60
    out_root = isempty(strip(opts["output-root"])) ?
        joinpath(PROJECT_ROOT, "outputs", "$(timestamp())_overnight_pam_sweep") :
        abspath(opts["output-root"])
    mkpath(out_root)

    jobs = vcat(cached_reconstruction_jobs(opts), simulation_jobs(opts))
    manifest = Dict{String, Any}(
        "created_at" => string(Dates.now()),
        "output_root" => out_root,
        "max_hours" => parse(Float64, opts["max-hours"]),
        "per_run_timeout_min" => parse(Float64, opts["per-run-timeout-min"]),
        "from_run_dir" => opts["from-run-dir"],
        "job_count" => length(jobs),
        "jobs" => jobs,
    )
    write_json(joinpath(out_root, "manifest.json"), manifest)

    println("Output root: ", out_root)
    println("Jobs queued: ", length(jobs))
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
    println("Sweep complete. Results: ", out_root)
    if isempty(leaders)
        println("No successful jobs with activity-boundary metrics yet.")
    else
        println("Top HASA F1:")
        for row in first(leaders, min(5, length(leaders)))
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
    main()
end
