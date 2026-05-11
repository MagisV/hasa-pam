#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Random
using JLD2
using JSON3
using TranscranialFUS
import TranscranialFUS: parse_cli, parse_bool, parse_dimension, parse_float_list,
    parse_threshold_ratios, parse_threshold_search_ratios, parse_source_model, parse_aberrator,
    parse_simulation_backend, parse_source_phase_mode, parse_source_variability,
    source_variability_from_summary, parse_analysis_mode, resolve_reconstruction_mode,
    make_window_config, parse_receiver_aperture_mm, parse_point_sources_3d,
    parse_squiggle_sources_3d, parse_network_sources_3d, parse_sources,
    default_simulation_info, default_recon_frequencies, default_output_dir,
    default_reconstruction_output_dir, reject_cached_simulation_options!, json3_to_any,
    source_model_from_meta, centerlines_from_emission_meta, detection_truth_mask_from_meta,
    save_overview, save_threshold_boundary_detection, save_threshold_boundary_detection_3d,
    save_best_threshold_volume_3d, save_napari_npz_3d, compact_window_info,
    source_summary, string_key_dict, run_pam_case_3d

opts, provided_keys = parse_cli(ARGS)
dimension = parse_dimension(opts["dimension"])
source_model = parse_source_model(opts["source-model"])
from_run_dir = strip(opts["from-run-dir"])
detection_truth_radius_m = parse(Float64, opts["vascular-radius-mm"]) * 1e-3
detection_threshold_ratio = parse(Float64, opts["detection-threshold-ratio"])
boundary_threshold_ratios = parse_threshold_ratios(opts["boundary-threshold-ratios"])
auto_threshold_search = parse_bool(opts["auto-threshold-search"])
threshold_score_ratios = auto_threshold_search ? parse_threshold_search_ratios(opts) : boundary_threshold_ratios

if dimension == 3
    isempty(from_run_dir) || error("--from-run-dir is not implemented for 3D PAM yet.")
    source_model in (:point, :squiggle, :network) ||
        error("3D PAM CLI supports --source-model=point, --source-model=squiggle, or --source-model=network.")
    aberrator = parse_aberrator(opts["aberrator"])
    aberrator in (:none, :skull) || error("3D PAM CLI currently supports only --aberrator=none or --aberrator=skull.")

    dy_mm = isempty(strip(opts["dy-mm"])) ? parse(Float64, opts["dz-mm"]) : parse(Float64, opts["dy-mm"])
    transverse_y_mm = isempty(strip(opts["transverse-y-mm"])) ? parse(Float64, opts["transverse-mm"]) : parse(Float64, opts["transverse-y-mm"])
    transverse_z_mm = isempty(strip(opts["transverse-z-mm"])) ? parse(Float64, opts["transverse-mm"]) : parse(Float64, opts["transverse-z-mm"])
    receiver_aperture_y_spec = isempty(strip(opts["receiver-aperture-y-mm"])) ? opts["receiver-aperture-mm"] : opts["receiver-aperture-y-mm"]
    receiver_aperture_z_spec = isempty(strip(opts["receiver-aperture-z-mm"])) ? opts["receiver-aperture-mm"] : opts["receiver-aperture-z-mm"]

    cfg_base = PAMConfig3D(
        dx=parse(Float64, opts["dx-mm"]) * 1e-3,
        dy=dy_mm * 1e-3,
        dz=parse(Float64, opts["dz-mm"]) * 1e-3,
        axial_dim=parse(Float64, opts["axial-mm"]) * 1e-3,
        transverse_dim_y=transverse_y_mm * 1e-3,
        transverse_dim_z=transverse_z_mm * 1e-3,
        t_max=parse(Float64, opts["t-max-us"]) * 1e-6,
        dt=parse(Float64, opts["dt-ns"]) * 1e-9,
        zero_pad_factor=parse(Int, opts["zero-pad-factor"]),
        receiver_aperture_y=parse_receiver_aperture_mm(receiver_aperture_y_spec),
        receiver_aperture_z=parse_receiver_aperture_mm(receiver_aperture_z_spec),
        peak_suppression_radius=parse(Float64, opts["peak-suppression-radius-mm"]) * 1e-3,
        success_tolerance=parse(Float64, opts["success-tolerance-mm"]) * 1e-3,
        axial_gain_power=parse(Float64, opts["axial-gain-power"]),
    )

    sources, emission_meta = if source_model == :point
        parse_point_sources_3d(opts)
    elseif source_model == :network
        parse_network_sources_3d(opts, cfg_base)
    else
        parse_squiggle_sources_3d(opts, cfg_base)
    end
    bottom_margin_m = parse(Float64, opts["bottom-margin-mm"]) * 1e-3
    cfg = fit_pam_config_3d(cfg_base, sources; min_bottom_margin=bottom_margin_m)

    out_dir = if haskey(opts, "out-dir") && !isempty(strip(opts["out-dir"]))
        opts["out-dir"]
    else
        default_output_dir(opts, sources, cfg, emission_meta)
    end
    mkpath(out_dir)

    c, rho, medium_info = make_pam_medium_3d(cfg;
        aberrator             = aberrator,
        ct_path               = opts["ct-path"],
        slice_index_z         = parse(Int, opts["slice-index"]),
        skull_to_transducer   = parse(Float64, opts["skull-transducer-distance-mm"]) * 1e-3,
        hu_bone_thr           = parse(Int, opts["hu-bone-thr"]),
    )
    recon_frequencies = if haskey(opts, "recon-frequencies-mhz") && !isempty(strip(opts["recon-frequencies-mhz"]))
        parse_float_list(opts["recon-frequencies-mhz"]) .* 1e6
    else
        default_recon_frequencies(sources)
    end
    reconstruction_mode = resolve_reconstruction_mode(opts["recon-mode"], source_model)
    recon_bandwidth_hz = parse(Float64, opts["recon-bandwidth-khz"]) * 1e3
    window_config = make_window_config(opts, reconstruction_mode)
    source_phase_mode = parse_source_phase_mode(opts["source-phase-mode"])
    rng_sim = Random.MersenneTwister(parse(Int, opts["random-seed"]) + 1)
    source_variability = parse_source_variability(opts)
    if source_model in (:squiggle, :network)
        emission_meta["activity_model"] = Dict(
            "activity_mode" => String(source_phase_mode),
            "frequency_jitter_percent" => source_variability.frequency_jitter_fraction * 100.0,
        )
    end

    simulation_backend = parse_simulation_backend(opts["simulation-backend"])
    simulation_backend == :analytic && aberrator == :skull &&
        error("--simulation-backend=analytic is not compatible with --aberrator=skull; use --simulation-backend=kwave.")
    results = run_pam_case_3d(
        c,
        rho,
        sources,
        cfg;
        frequencies=recon_frequencies,
        bandwidth=recon_bandwidth_hz,
        kwave_use_gpu=parse_bool(opts["kwave-use-gpu"]),
        recon_use_gpu=parse_bool(opts["recon-use-gpu"]),
        reconstruction_axial_step=parse(Float64, opts["recon-step-um"]) * 1e-6,
        reconstruction_mode=reconstruction_mode,
        window_config=window_config,
        show_progress=parse_bool(opts["recon-progress"]),
        benchmark=parse_bool(opts["benchmark"]),
        window_batch=parse(Int, opts["window-batch"]),
        simulation_backend=simulation_backend,
        source_phase_mode=source_phase_mode,
        rng=rng_sim,
        source_variability=source_variability,
    )

    medium_summary = Dict{String, Any}()
    for (key, value) in medium_info
        key == :mask && continue
        medium_summary[String(key)] = value
    end

    activity_boundary_path = joinpath(out_dir, "activity_boundaries.png")
    activity_boundary_metrics = save_threshold_boundary_detection_3d(
        activity_boundary_path,
        results[:pam_geo],
        results[:pam_hasa],
        results[:kgrid],
        cfg,
        sources;
        threshold_ratios=threshold_score_ratios,
        truth_radius=detection_truth_radius_m,
        c=c,
    )
    activity_boundary_metrics["auto_threshold_search"] = auto_threshold_search
    activity_boundary_metrics["display_threshold_mode"] = "selected_best_recall_precision"
    best_volume_path = joinpath(out_dir, "best_threshold_3d.png")
    best_volume_metrics = save_best_threshold_volume_3d(
        best_volume_path,
        results[:pam_hasa],
        results[:kgrid],
        cfg,
        sources;
        threshold=activity_boundary_metrics["best_hasa_threshold"],
        truth_radius=detection_truth_radius_m,
    )

    summary = Dict(
        "out_dir" => out_dir,
        "dimension" => 3,
        "reconstruction_source" => Dict("mode" => String(simulation_backend)),
        "simulation_backend" => String(simulation_backend),
        "activity_boundary_figure" => activity_boundary_path,
        "activity_boundary_metrics" => activity_boundary_metrics,
        "best_threshold_3d_figure" => best_volume_path,
        "best_threshold_3d_metrics" => best_volume_metrics,
        "sources" => [source_summary(src) for src in sources],
        "emission_meta" => emission_meta,
        "config" => Dict(
            "dx" => cfg.dx,
            "dy" => cfg.dy,
            "dz" => cfg.dz,
            "axial_dim" => cfg.axial_dim,
            "transverse_dim_y" => cfg.transverse_dim_y,
            "transverse_dim_z" => cfg.transverse_dim_z,
            "receiver_aperture_y" => cfg.receiver_aperture_y,
            "receiver_aperture_z" => cfg.receiver_aperture_z,
            "t_max" => cfg.t_max,
            "dt" => cfg.dt,
            "c0" => cfg.c0,
            "rho0" => cfg.rho0,
            "zero_pad_factor" => cfg.zero_pad_factor,
            "peak_suppression_radius" => cfg.peak_suppression_radius,
            "success_tolerance" => cfg.success_tolerance,
            "axial_gain_power" => cfg.axial_gain_power,
            "bottom_margin" => bottom_margin_m,
        ),
        "medium" => medium_summary,
        "reconstruction_frequencies_hz" => recon_frequencies,
        "reconstruction_bandwidth_hz" => recon_bandwidth_hz,
        "reconstruction_mode" => String(results[:reconstruction_mode]),
        "reconstruction_progress" => parse_bool(opts["recon-progress"]),
        "source_phase_mode" => String(results[:source_phase_mode]),
        "n_frames" => Int(get(results, :n_frames, 1)),
        "source_variability" => Dict(
            "frequency_jitter_percent" => source_variability.frequency_jitter_fraction * 100.0,
        ),
        "threshold_search" => Dict(
            "auto" => auto_threshold_search,
            "min_ratio" => minimum(threshold_score_ratios),
            "max_ratio" => maximum(threshold_score_ratios),
            "step" => auto_threshold_search ? parse(Float64, opts["auto-threshold-step"]) : nothing,
            "count" => length(threshold_score_ratios),
            "selection_metric" => "source_f1",
            "display_threshold_mode" => "selected_best_recall_precision",
        ),
        "physical_source_count" => length(sources),
        "emission_event_count" => Int(get(results, :emission_event_count, length(sources))),
        "window_config" => string_key_dict(results[:window_config]),
        "window_info" => Dict(
            "geometric" => compact_window_info(results[:geo_info]),
            "hasa" => compact_window_info(results[:hasa_info]),
        ),
        "benchmark" => parse_bool(opts["benchmark"]),
        "gpu_timing" => Dict(
            "geometric" => get(results[:geo_info], :gpu_timing, nothing),
            "hasa" => get(results[:hasa_info], :gpu_timing, nothing),
        ),
        "reconstruction_axial_step_m" => results[:geo_info][:axial_step],
        "reference_sound_speed_m_per_s" => results[:geo_info][:reference_sound_speed],
        "analysis_mode" => String(results[:analysis_mode]),
        "simulation" => Dict(
            "receiver_row" => results[:simulation][:receiver_row],
            "receiver_cols_y" => [first(results[:simulation][:receiver_cols_y]), last(results[:simulation][:receiver_cols_y])],
            "receiver_cols_z" => [first(results[:simulation][:receiver_cols_z]), last(results[:simulation][:receiver_cols_z])],
            "source_indices" => [[row, col_y, col_z] for (row, col_y, col_z) in get(results[:simulation], :source_indices, NTuple{3, Int}[])],
        ),
        "geometric" => results[:stats_geo],
        "hasa" => results[:stats_hasa],
    )

    open(joinpath(out_dir, "summary.json"), "w") do io
        JSON3.pretty(io, summary)
    end

    @save joinpath(out_dir, "result.jld2") c rho cfg sources results medium_info

    save_napari_npz_3d(
        out_dir,
        results[:pam_geo],
        results[:pam_hasa],
        c, rho,
        results[:kgrid],
        cfg,
        sources;
        truth_radius=detection_truth_radius_m,
    )

    println("Saved 3D PAM outputs to $out_dir")
    exit()
end

if isempty(from_run_dir)
    cfg_base = PAMConfig(
        dx=parse(Float64, opts["dx-mm"]) * 1e-3,
        dz=parse(Float64, opts["dz-mm"]) * 1e-3,
        axial_dim=parse(Float64, opts["axial-mm"]) * 1e-3,
        transverse_dim=parse(Float64, opts["transverse-mm"]) * 1e-3,
        receiver_aperture=parse_receiver_aperture_mm(opts["receiver-aperture-mm"]),
        t_max=parse(Float64, opts["t-max-us"]) * 1e-6,
        dt=parse(Float64, opts["dt-ns"]) * 1e-9,
        zero_pad_factor=parse(Int, opts["zero-pad-factor"]),
        peak_suppression_radius=parse(Float64, opts["peak-suppression-radius-mm"]) * 1e-3,
        success_tolerance=parse(Float64, opts["success-tolerance-mm"]) * 1e-3,
    )

    sources, emission_meta = parse_sources(opts, cfg_base)
    source_model = source_model_from_meta(emission_meta, sources)

    aberrator = parse_aberrator(opts["aberrator"])
    bottom_margin_m = parse(Float64, opts["bottom-margin-mm"]) * 1e-3
    cfg = fit_pam_config(
        cfg_base,
        sources;
        min_bottom_margin=bottom_margin_m,
        reference_depth=aberrator == :skull ? parse(Float64, opts["skull-transducer-distance-mm"]) * 1e-3 : nothing,
    )

    out_dir = if haskey(opts, "out-dir") && !isempty(strip(opts["out-dir"]))
        opts["out-dir"]
    else
        default_output_dir(opts, sources, cfg, emission_meta)
    end
    mkpath(out_dir)

    c, rho, medium_info = make_pam_medium(
        cfg;
        aberrator=aberrator,
        ct_path=opts["ct-path"],
        slice_index=parse(Int, opts["slice-index"]),
        skull_to_transducer=parse(Float64, opts["skull-transducer-distance-mm"]) * 1e-3,
        hu_bone_thr=parse(Int, opts["hu-bone-thr"]),
    )

    recon_frequencies = if haskey(opts, "recon-frequencies-mhz") && !isempty(strip(opts["recon-frequencies-mhz"]))
        parse_float_list(opts["recon-frequencies-mhz"]) .* 1e6
    else
        default_recon_frequencies(sources)
    end
    reconstruction_mode = resolve_reconstruction_mode(opts["recon-mode"], source_model)
    recon_bandwidth_hz = parse(Float64, opts["recon-bandwidth-khz"]) * 1e3
    window_config = make_window_config(opts, reconstruction_mode)
    analysis_mode = parse_analysis_mode(opts["analysis-mode"], source_model)
    truth_centerlines = centerlines_from_emission_meta(emission_meta)
    detection_truth_mask = detection_truth_mask_from_meta(emission_meta, pam_grid(cfg), cfg, detection_truth_radius_m)

    source_phase_mode = parse_source_phase_mode(opts["source-phase-mode"])
    rng_sim = Random.MersenneTwister(parse(Int, opts["random-seed"]) + 1)
    source_variability = parse_source_variability(opts)
    if source_model == :squiggle
        emission_meta["activity_model"] = Dict(
            "activity_mode" => String(source_phase_mode),
            "frequency_jitter_percent" => source_variability.frequency_jitter_fraction * 100.0,
        )
    end

    results = run_pam_case(
        c,
        rho,
        sources,
        cfg;
        frequencies=recon_frequencies,
        bandwidth=recon_bandwidth_hz,
        use_gpu=parse_bool(opts["recon-use-gpu"]),
        kwave_use_gpu=parse_bool(opts["kwave-use-gpu"]),
        reconstruction_axial_step=parse(Float64, opts["recon-step-um"]) * 1e-6,
        analysis_mode=analysis_mode,
        detection_truth_radius=detection_truth_radius_m,
        detection_threshold_ratio=detection_threshold_ratio,
        detection_truth_mask=detection_truth_mask,
        reconstruction_mode=reconstruction_mode,
        window_config=window_config,
        source_phase_mode=source_phase_mode,
        rng=rng_sim,
        source_variability=source_variability,
        show_progress=parse_bool(opts["recon-progress"]),
        benchmark=parse_bool(opts["benchmark"]),
        window_batch=parse(Int, opts["window-batch"]),
    )
    reconstruction_source = Dict("mode" => "simulation")
else
    reject_cached_simulation_options!(
        provided_keys,
        (
            "source-model", "sources-mm", "anchors-mm", "frequency-mhz", "fundamental-mhz",
            "amplitude-pa", "source-amplitudes-pa", "source-frequencies-mhz", "phases-deg",
            "num-cycles", "harmonics", "harmonic-amplitudes",
            "gate-us", "taper-ratio", "axial-mm", "transverse-mm", "dx-mm", "dz-mm",
            "receiver-aperture-mm", "t-max-us", "dt-ns", "zero-pad-factor",
            "peak-suppression-radius-mm", "success-tolerance-mm", "aberrator", "ct-path",
            "slice-index", "skull-transducer-distance-mm", "bottom-margin-mm", "hu-bone-thr",
            "simulation-backend", "phase-mode", "phase-jitter-rad", "random-seed",
            "transducer-mm", "delays-us", "vascular-length-mm", "vascular-squiggle-amplitude-mm",
            "vascular-squiggle-amplitude-x-mm", "vascular-squiggle-wavelength-mm",
            "vascular-squiggle-slope", "squiggle-phase-x-deg",
            "vascular-source-spacing-mm", "vascular-position-jitter-mm",
            "vascular-min-separation-mm", "vascular-max-sources-per-anchor",
            "network-axial-radius-mm", "network-lateral-y-radius-mm",
            "network-lateral-z-radius-mm", "network-root-count", "network-generations",
            "network-branch-length-mm", "network-branch-step-mm", "network-branch-angle-deg",
            "network-tortuosity", "network-orientation", "network-density-sigma-mm", "network-density-axial-sigma-mm",
            "network-density-lateral-y-sigma-mm", "network-density-lateral-z-sigma-mm",
            "network-max-sources-per-center",
            "source-phase-mode", "frequency-jitter-percent",
        ),
    )
    cached_path = joinpath(from_run_dir, "result.jld2")
    isfile(cached_path) || error("--from-run-dir must contain result.jld2, missing: $cached_path")
    cached = load(cached_path)
    c = cached["c"]
    rho = haskey(cached, "rho") ? cached["rho"] : fill(Float32(cached["cfg"].rho0), size(c))
    cfg = cached["cfg"]
    sources = haskey(cached, "sources") ? cached["sources"] : cached["clusters"]
    cached_results = cached["results"]
    rf = cached_results[:rf]
    medium_info = haskey(cached, "medium_info") ? cached["medium_info"] : Dict{Symbol, Any}(:aberrator => :cached)
    bottom_margin_m = nothing
    cached_summary_path = joinpath(from_run_dir, "summary.json")
    cached_summary = isfile(cached_summary_path) ? JSON3.read(read(cached_summary_path, String)) : nothing
    source_variability = source_variability_from_summary(cached_summary)
    emission_meta = if !isnothing(cached_summary) && hasproperty(cached_summary, :emission_meta)
        Dict{String, Any}(json3_to_any(cached_summary.emission_meta))
    else
        Dict{String, Any}(
            "source_model" => source_model_from_meta(Dict{String, Any}(), sources) |> String,
            "n_emission_sources" => length(sources),
        )
    end
    emission_meta["from_run_dir"] = abspath(from_run_dir)
    source_model = source_model_from_meta(emission_meta, sources)

    out_dir = if haskey(opts, "out-dir") && !isempty(strip(opts["out-dir"]))
        opts["out-dir"]
    else
        default_reconstruction_output_dir(from_run_dir)
    end
    mkpath(out_dir)

    recon_frequencies = if haskey(opts, "recon-frequencies-mhz") && !isempty(strip(opts["recon-frequencies-mhz"]))
        parse_float_list(opts["recon-frequencies-mhz"]) .* 1e6
    else
        default_recon_frequencies(sources)
    end
    reconstruction_mode = resolve_reconstruction_mode(opts["recon-mode"], source_model)
    recon_bandwidth_hz = parse(Float64, opts["recon-bandwidth-khz"]) * 1e3
    window_config = make_window_config(opts, reconstruction_mode)
    analysis_mode = parse_analysis_mode(opts["analysis-mode"], source_model)
    simulation_info = haskey(cached_results, :simulation) ? cached_results[:simulation] : default_simulation_info(cfg)
    truth_centerlines = centerlines_from_emission_meta(emission_meta)
    detection_truth_mask = detection_truth_mask_from_meta(emission_meta, pam_grid(cfg), cfg, detection_truth_radius_m)
    results = reconstruct_pam_case(
        rf,
        c,
        sources,
        cfg;
        simulation_info=simulation_info,
        frequencies=recon_frequencies,
        bandwidth=recon_bandwidth_hz,
        use_gpu=parse_bool(opts["recon-use-gpu"]),
        reconstruction_axial_step=parse(Float64, opts["recon-step-um"]) * 1e-6,
        analysis_mode=analysis_mode,
        detection_truth_radius=detection_truth_radius_m,
        detection_threshold_ratio=detection_threshold_ratio,
        detection_truth_mask=detection_truth_mask,
        reconstruction_mode=reconstruction_mode,
        window_config=window_config,
        show_progress=parse_bool(opts["recon-progress"]),
        benchmark=parse_bool(opts["benchmark"]),
        window_batch=parse(Int, opts["window-batch"]),
    )
    reconstruction_source = Dict(
        "mode" => "cached_rf",
        "from_run_dir" => abspath(from_run_dir),
        "from_result_jld2" => abspath(cached_path),
    )
end

medium_summary = Dict{String, Any}()
for (key, value) in medium_info
    key == :mask && continue
    medium_summary[String(key)] = value
end

save_overview(
    joinpath(out_dir, "overview.png"),
    c, results[:rf], results[:pam_geo], results[:pam_hasa],
    results[:kgrid], cfg, sources, results[:stats_geo], results[:stats_hasa],
)

activity_boundary_path = joinpath(out_dir, "activity_boundaries.png")
activity_boundary_metrics = save_threshold_boundary_detection(
    activity_boundary_path,
    results[:pam_geo],
    results[:pam_hasa],
    results[:kgrid],
    cfg,
    sources;
    threshold_ratios=boundary_threshold_ratios,
    truth_radius=detection_truth_radius_m,
    truth_mask=detection_truth_mask,
    truth_centerlines=truth_centerlines,
    frequencies=recon_frequencies,
    c=c,
)

summary = Dict(
    "out_dir" => out_dir,
    "reconstruction_source" => reconstruction_source,
    "activity_boundary_figure" => activity_boundary_path,
    "activity_boundary_metrics" => activity_boundary_metrics,
    "sources" => [source_summary(src) for src in sources],
    "emission_meta" => emission_meta,
    "config" => Dict(
        "dx" => cfg.dx,
        "dz" => cfg.dz,
        "axial_dim" => cfg.axial_dim,
        "transverse_dim" => cfg.transverse_dim,
        "receiver_aperture" => cfg.receiver_aperture,
        "t_max" => cfg.t_max,
        "dt" => cfg.dt,
        "c0" => cfg.c0,
        "rho0" => cfg.rho0,
        "zero_pad_factor" => cfg.zero_pad_factor,
        "peak_suppression_radius" => cfg.peak_suppression_radius,
        "success_tolerance" => cfg.success_tolerance,
        "bottom_margin" => bottom_margin_m,
    ),
    "medium" => medium_summary,
    "reconstruction_frequencies_hz" => recon_frequencies,
    "reconstruction_bandwidth_hz" => recon_bandwidth_hz,
    "reconstruction_mode" => String(results[:reconstruction_mode]),
    "reconstruction_progress" => parse_bool(opts["recon-progress"]),
    "source_phase_mode" => String(get(results, :source_phase_mode, :coherent)),
    "source_variability" => Dict(
        "frequency_jitter_percent" => source_variability.frequency_jitter_fraction * 100.0,
    ),
    "window_config" => string_key_dict(results[:window_config]),
    "window_info" => Dict(
        "geometric" => compact_window_info(results[:geo_info]),
        "hasa" => compact_window_info(results[:hasa_info]),
    ),
    "benchmark" => parse_bool(opts["benchmark"]),
    "gpu_timing" => Dict(
        "geometric" => get(results[:geo_info], :gpu_timing, nothing),
        "hasa" => get(results[:hasa_info], :gpu_timing, nothing),
    ),
    "reconstruction_axial_step_m" => results[:geo_info][:axial_step],
    "reference_sound_speed_m_per_s" => results[:geo_info][:reference_sound_speed],
    "activity_model" => get(emission_meta, "activity_model", Dict("activity_mode" => "static")),
    "physical_source_count" => get(emission_meta, "physical_source_count", length(sources)),
    "emission_event_count" => get(emission_meta, "emission_event_count", length(sources)),
    "analysis_mode" => String(analysis_mode),
    "detection_truth_radius_m" => detection_truth_radius_m,
    "detection_truth_mode" => isnothing(detection_truth_mask) ? "source_disks" : "centerline_tube",
    "detection_threshold_ratio" => detection_threshold_ratio,
    "boundary_threshold_ratios" => boundary_threshold_ratios,
    "simulation" => Dict(
        "receiver_row" => results[:simulation][:receiver_row],
        "receiver_cols" => [first(results[:simulation][:receiver_cols]), last(results[:simulation][:receiver_cols])],
        "source_indices" => [[row, col] for (row, col) in get(results[:simulation], :source_indices, Tuple{Int, Int}[])],
    ),
    "geometric" => results[:stats_geo],
    "hasa" => results[:stats_hasa],
)

open(joinpath(out_dir, "summary.json"), "w") do io
    JSON3.pretty(io, summary)
end

@save joinpath(out_dir, "result.jld2") c rho cfg sources results medium_info

println("Saved PAM outputs to $out_dir")
