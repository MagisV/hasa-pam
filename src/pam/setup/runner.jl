# Testable runner planning helpers for scripts/run_pam.jl.

"""
    run_pam_medium_summary(medium_info)

Convert PAM medium metadata into a JSON-friendly summary, omitting mask arrays.
"""
function run_pam_medium_summary(medium_info)
    medium_summary = Dict{String, Any}()
    for (key, value) in medium_info
        key == :mask && continue
        medium_summary[String(key)] = value
    end
    return medium_summary
end

"""
    run_pam_dry_plan(args)

Parse PAM CLI arguments and return the branch/output plan without running a simulation.
"""
function run_pam_dry_plan(args::AbstractVector{<:AbstractString})
    opts, provided_keys = parse_cli(String.(args))
    dimension = parse_dimension(opts["dimension"])
    source_model = parse_source_model(opts["source-model"])
    from_run_dir = strip(opts["from-run-dir"])
    detection_truth_radius_m = parse(Float64, opts["vascular-radius-mm"]) * 1e-3
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
        cfg = fit_pam_config_3d(cfg_base, sources; min_bottom_margin=parse(Float64, opts["bottom-margin-mm"]) * 1e-3)
        out_dir = if haskey(opts, "out-dir") && !isempty(strip(opts["out-dir"]))
            opts["out-dir"]
        else
            default_output_dir(opts, sources, cfg, emission_meta)
        end
        simulation_backend = parse_simulation_backend(opts["simulation-backend"])
        simulation_backend == :analytic && aberrator == :skull &&
            error("--simulation-backend=analytic is not compatible with --aberrator=skull; use --simulation-backend=kwave.")
        return Dict(
            :branch => :pam3d,
            :out_dir => out_dir,
            :source_model => source_model,
            :source_count => length(sources),
            :threshold_score_ratios => threshold_score_ratios,
            :detection_truth_radius_m => detection_truth_radius_m,
            :simulation_backend => simulation_backend,
        )
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
        cfg = fit_pam_config(
            cfg_base,
            sources;
            min_bottom_margin=parse(Float64, opts["bottom-margin-mm"]) * 1e-3,
            reference_depth=aberrator == :skull ? parse(Float64, opts["skull-transducer-distance-mm"]) * 1e-3 : nothing,
        )
        out_dir = if haskey(opts, "out-dir") && !isempty(strip(opts["out-dir"]))
            opts["out-dir"]
        else
            default_output_dir(opts, sources, cfg, emission_meta)
        end
        return Dict(
            :branch => :pam2d_simulation,
            :out_dir => out_dir,
            :source_model => source_model,
            :source_count => length(sources),
            :threshold_ratios => boundary_threshold_ratios,
            :detection_truth_radius_m => detection_truth_radius_m,
        )
    end

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
    out_dir = if haskey(opts, "out-dir") && !isempty(strip(opts["out-dir"]))
        opts["out-dir"]
    else
        default_reconstruction_output_dir(from_run_dir)
    end
    return Dict(
        :branch => :pam2d_cached,
        :out_dir => out_dir,
        :cached_path => cached_path,
        :reconstruction_source => Dict(
            "mode" => "cached_rf",
            "from_run_dir" => abspath(from_run_dir),
            "from_result_jld2" => abspath(cached_path),
        ),
    )
end
