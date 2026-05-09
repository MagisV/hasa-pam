#!/usr/bin/env julia

include(joinpath(@__DIR__, "run_pam_overnight_sweep.jl"))

function vessel_sweep_opts(args)
    opts = parse_cli(args)
    provided = Set(first(split(arg[3:end], "="; limit=2)) for arg in args if startswith(arg, "--"))
    !("max-hours" in provided) && (opts["max-hours"] = "11")
    !("per-run-timeout-min" in provided) && (opts["per-run-timeout-min"] = "25")
    !("output-root" in provided) && (opts["output-root"] = "")
    !("random-seed" in provided) && (opts["random-seed"] = "42")
    !("auto-threshold-search" in provided) && (opts["auto-threshold-search"] = "true")
    !("auto-threshold-min" in provided) && (opts["auto-threshold-min"] = "0.10")
    !("auto-threshold-max" in provided) && (opts["auto-threshold-max"] = "0.95")
    !("auto-threshold-step" in provided) && (opts["auto-threshold-step"] = "0.01")
    return opts
end

function vessel_base_args(opts)
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
        "vascular-radius-mm" => "1.0",
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
        "frequency-jitter-percent" => "1",
        "recon-window-us" => "40",
        "recon-hop-us" => "20",
        "recon-bandwidth-khz" => "40",
        "auto-threshold-search" => opts["auto-threshold-search"],
        "auto-threshold-min" => opts["auto-threshold-min"],
        "auto-threshold-max" => opts["auto-threshold-max"],
        "auto-threshold-step" => opts["auto-threshold-step"],
        "sim-mode" => "kwave",
        "kwave-use-gpu" => "true",
        "recon-use-gpu" => "true",
        "window-batch" => "2",
        "recon-progress" => "false",
        "random-seed" => opts["random-seed"],
    )
end

function vessel_sweep_jobs(opts)
    base = vessel_base_args(opts)
    specs = [
        ("baseline_len12_mild_center", Dict(), "Current 40 kHz / 1% / hop 20 us baseline."),

        ("size_len6_center", Dict("vascular-length-mm" => "6"), "Short vessel segment at the baseline location."),
        ("size_len18_center", Dict("vascular-length-mm" => "18"), "Longer vessel segment at the baseline location."),
        ("size_dense_spacing03", Dict("vascular-source-spacing-mm" => "0.3", "vascular-min-separation-mm" => "0.15"), "Denser bubble sampling on the same centerline."),
        ("size_sparse_spacing075", Dict("vascular-source-spacing-mm" => "0.75", "vascular-min-separation-mm" => "0.35"), "Sparser bubble sampling on the same centerline."),
        ("metric_radius05", Dict("vascular-radius-mm" => "0.5"), "Tighter source/truth radius scoring sensitivity."),
        ("metric_radius15", Dict("vascular-radius-mm" => "1.5"), "Looser source/truth radius scoring sensitivity."),

        ("shape_straight", Dict("vascular-squiggle-amplitude-mm" => "0", "vascular-squiggle-amplitude-x-mm" => "0"), "Straight centerline control."),
        ("shape_wide_sine", Dict("vascular-squiggle-amplitude-mm" => "0.8", "vascular-squiggle-amplitude-x-mm" => "0.5"), "Higher-amplitude 3D squiggle."),
        ("shape_tight_wave", Dict("vascular-squiggle-amplitude-mm" => "0.8", "vascular-squiggle-amplitude-x-mm" => "0.5", "vascular-squiggle-wavelength-mm" => "4"), "High-curvature short-wavelength squiggle."),
        ("shape_long_wave", Dict("vascular-squiggle-amplitude-mm" => "0.8", "vascular-squiggle-amplitude-x-mm" => "0.5", "vascular-squiggle-wavelength-mm" => "12"), "Smoother long-wavelength squiggle."),
        ("shape_axial_tilt", Dict("vascular-squiggle-slope" => "0.25"), "Centerline tilted in depth along its length."),

        ("loc_depth32_center", Dict("anchors-mm" => "32:0:0"), "Shallower centered vessel."),
        ("loc_depth52_center", Dict("anchors-mm" => "52:0:0"), "Deeper centered vessel."),
        ("loc_depth62_center", Dict("anchors-mm" => "62:0:0"), "Deep centered vessel near the lower part of the domain."),
        ("loc_y_m18", Dict("anchors-mm" => "42:-18:0"), "Lateral-y offset toward one side of the aperture."),
        ("loc_y_m9", Dict("anchors-mm" => "42:-9:0"), "Moderate negative lateral-y offset."),
        ("loc_y_p9", Dict("anchors-mm" => "42:9:0"), "Moderate positive lateral-y offset."),
        ("loc_y_p18", Dict("anchors-mm" => "42:18:0"), "Lateral-y offset toward the opposite side of the aperture."),
        ("loc_z_m18", Dict("anchors-mm" => "42:0:-18"), "Lateral-z offset toward one side of the aperture."),
        ("loc_z_m9", Dict("anchors-mm" => "42:0:-9"), "Moderate negative lateral-z offset."),
        ("loc_z_p9", Dict("anchors-mm" => "42:0:9"), "Moderate positive lateral-z offset."),
        ("loc_z_p18", Dict("anchors-mm" => "42:0:18"), "Lateral-z offset toward the opposite side of the aperture."),
        ("loc_diag_m12_m12", Dict("anchors-mm" => "42:-12:-12"), "Diagonal negative y/z offset."),
        ("loc_diag_m12_p12", Dict("anchors-mm" => "42:-12:12"), "Mixed-sign diagonal offset."),
        ("loc_diag_p12_m12", Dict("anchors-mm" => "42:12:-12"), "Mixed-sign diagonal offset."),
        ("loc_diag_p12_p12", Dict("anchors-mm" => "42:12:12"), "Diagonal positive y/z offset."),
    ]
    return [sim_job(id, merge_args(base, args); note=note) for (id, args, note) in specs]
end

function vessel_sweep_main()
    opts = vessel_sweep_opts(ARGS)
    dry_run = parse_bool(opts["dry-run"])
    force = parse_bool(opts["force"])
    max_seconds = parse(Float64, opts["max-hours"]) * 3600
    timeout_seconds = parse(Float64, opts["per-run-timeout-min"]) * 60
    out_root = isempty(strip(opts["output-root"])) ?
        joinpath(PROJECT_ROOT, "outputs", "$(timestamp())_pam_3d_vessel_sweep") :
        abspath(opts["output-root"])
    mkpath(out_root)

    jobs = vessel_sweep_jobs(opts)
    write_json(joinpath(out_root, "manifest.json"), Dict{String, Any}(
        "created_at" => string(Dates.now()),
        "output_root" => out_root,
        "max_hours" => parse(Float64, opts["max-hours"]),
        "per_run_timeout_min" => parse(Float64, opts["per-run-timeout-min"]),
        "job_count" => length(jobs),
        "fixed_reconstruction" => Dict(
            "recon_bandwidth_khz" => 40,
            "frequency_jitter_percent" => 1,
            "recon_window_us" => 40,
            "recon_hop_us" => 20,
        ),
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
    println("3D vessel sweep complete. Results: ", out_root)
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
    vessel_sweep_main()
end
