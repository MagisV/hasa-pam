# CLI/config parsing helpers for PAM runner scripts.

Base.@kwdef struct CLIOption
    name::String
    default::String
    value::String = "string"
    category::String = "General"
    applies_to::String = "PAM"
    choices::Vector{String} = String[]
    description::String = ""
end

function _cli_option(name, default, value, category, applies_to, description; choices=String[])
    return CLIOption(
        name=name,
        default=default,
        value=value,
        category=category,
        applies_to=applies_to,
        choices=choices,
        description=description,
    )
end

function pam_cli_options()
    return CLIOption[
        _cli_option("dimension", "2", "2|3", "General", "PAM", "Selects the 2D or 3D PAM workflow."; choices=["2", "3"]),
        _cli_option("source-model", "squiggle", "point|squiggle|network", "General", "PAM", "Selects explicit point sources, a squiggly vascular source, or a synthetic 3D network."; choices=["point", "squiggle", "network"]),
        _cli_option("from-run-dir", "", "path", "General", "2D reconstruction only", "Loads RF data, medium, grid, and sources from a previous output directory and reruns reconstruction/analysis only."),
        _cli_option("random-seed", "42", "integer", "General", "PAM", "Seed used for stochastic phases, source placement jitter, and generated vascular/network geometry."),
        _cli_option("benchmark", "false", "bool", "General", "PAM", "Prints additional timing information from simulation and reconstruction."),

        _cli_option("sources-mm", "30:0", "depth:lateral[,depth:lateral] or depth:y:z", "Source geometry", "point", "Point source coordinates in millimeters. 2D uses depth:lateral; 3D uses depth:y:z."),
        _cli_option("anchors-mm", "45:0", "depth:lateral[,depth:lateral] or depth:y:z", "Source geometry", "squiggle, network", "Anchor coordinates for generated vascular or network activity. 2D uses depth:lateral; 3D uses depth:y:z."),
        _cli_option("frequency-mhz", "0.4", "MHz", "Source signal", "point", "Tone-burst frequency for point sources unless per-source frequencies are supplied."),
        _cli_option("fundamental-mhz", "0.5", "MHz", "Source signal", "squiggle, network", "Fundamental activity frequency. Harmonic frequencies are integer multiples of this value."),
        _cli_option("amplitude-pa", "1.0", "pressure", "Source signal", "PAM", "Default pressure amplitude for generated sources."),
        _cli_option("source-amplitudes-pa", "", "comma list", "Source signal", "point", "Optional per-point-source amplitudes. Use one value for all sources or one value per source."),
        _cli_option("source-frequencies-mhz", "", "comma list", "Source signal", "point", "Optional per-point-source frequencies in MHz. Use one value for all sources or one value per source."),
        _cli_option("phases-deg", "", "comma list", "Source signal", "point", "Optional per-point-source phases in degrees before phase-mode randomization."),
        _cli_option("delays-us", "0", "comma list", "Source signal", "PAM", "Emission delays in microseconds. Use one value for all sources or one value per coordinate/anchor."),
        _cli_option("num-cycles", "4", "integer", "Source signal", "point", "Number of cycles in each point-source tone burst."),
        _cli_option("harmonics", "2,3,4", "comma list", "Source signal", "squiggle, network", "Harmonic orders emitted by generated bubble activity."),
        _cli_option("harmonic-amplitudes", "1.0,0.6,0.3", "comma list", "Source signal", "squiggle, network", "Relative amplitude for each harmonic listed in --harmonics."),
        _cli_option("gate-us", "50", "microseconds", "Source signal", "squiggle, network", "Duration of each activity emission gate."),
        _cli_option("taper-ratio", "0.25", "fraction", "Source signal", "squiggle, network", "Tukey taper fraction applied to generated activity gates."),
        _cli_option("phase-mode", "geometric", "coherent|random|jittered|geometric", "Source signal", "PAM", "Controls initial source phases. Point sources accept coherent, random, and jittered; generated activity also uses geometric travel-time phases."; choices=["coherent", "random", "jittered", "geometric"]),
        _cli_option("phase-jitter-rad", "0.2", "radians", "Source signal", "PAM", "Standard deviation for jittered source phases."),
        _cli_option("source-phase-mode", "random_phase_per_window", "coherent|random_static_phase|random_phase_per_window", "Source signal", "PAM", "Controls whether source phases are fixed or redrawn across reconstruction windows."; choices=["coherent", "random_static_phase", "random_phase_per_window"]),
        _cli_option("frequency-jitter-percent", "1", "percent", "Source signal", "squiggle, network", "Multiplicative jitter applied to generated source fundamentals before harmonics are formed."),
        _cli_option("transducer-mm", "-30:0", "depth:lateral", "Source geometry", "2D squiggle", "Reference transducer position used when computing geometric source phases in 2D."),

        _cli_option("vascular-length-mm", "12", "mm", "Vascular source", "squiggle", "Length of the generated squiggle centerline for each anchor."),
        _cli_option("vascular-squiggle-amplitude-mm", "1.5", "mm", "Vascular source", "squiggle", "Lateral squiggle amplitude in 2D, or y-amplitude in 3D."),
        _cli_option("vascular-squiggle-amplitude-x-mm", "1.0", "mm", "Vascular source", "3D squiggle", "Depth-direction squiggle amplitude for 3D vascular sources."),
        _cli_option("vascular-squiggle-wavelength-mm", "8", "mm", "Vascular source", "squiggle", "Spatial wavelength of the generated squiggle path."),
        _cli_option("vascular-squiggle-slope", "0.0", "slope", "Vascular source", "squiggle", "Linear slope added to the generated squiggle path."),
        _cli_option("squiggle-phase-x-deg", "90", "degrees", "Vascular source", "3D squiggle", "Phase offset for the 3D depth-direction squiggle component."),
        _cli_option("vascular-source-spacing-mm", "0.5", "mm", "Vascular source", "squiggle, network", "Approximate spacing between sampled bubble emitters along generated centerlines."),
        _cli_option("vascular-position-jitter-mm", "0.05", "mm", "Vascular source", "squiggle", "Random position jitter applied when sampling vascular sources."),
        _cli_option("vascular-min-separation-mm", "0.25", "mm", "Vascular source", "squiggle, network", "Minimum allowed distance between generated bubble emitters."),
        _cli_option("vascular-max-sources-per-anchor", "0", "integer", "Vascular source", "squiggle", "Caps generated sources per anchor. A value of 0 disables the cap."),
        _cli_option("vascular-radius-mm", "1.0", "mm", "Analysis", "squiggle, network", "Truth radius used when scoring activity detection around generated sources."),

        _cli_option("network-axial-radius-mm", "10.0", "mm", "Network source", "3D network", "Axial radius of the ellipsoid used to clip generated network activity."),
        _cli_option("network-lateral-y-radius-mm", "1.5", "mm", "Network source", "3D network", "Y radius of the generated network ellipsoid."),
        _cli_option("network-lateral-z-radius-mm", "1.5", "mm", "Network source", "3D network", "Z radius of the generated network ellipsoid."),
        _cli_option("network-root-count", "12", "integer", "Network source", "3D network", "Number of root branches grown for each network center."),
        _cli_option("network-generations", "3", "integer", "Network source", "3D network", "Number of branching generations in the synthetic network."),
        _cli_option("network-branch-length-mm", "5.0", "mm", "Network source", "3D network", "Nominal length of each generated branch segment."),
        _cli_option("network-branch-step-mm", "0.4", "mm", "Network source", "3D network", "Sampling step along generated network branches."),
        _cli_option("network-branch-angle-deg", "36", "degrees", "Network source", "3D network", "Nominal branching angle for synthetic network growth."),
        _cli_option("network-tortuosity", "0.18", "fraction", "Network source", "3D network", "Strength of random branch curvature in the synthetic network."),
        _cli_option("network-orientation", "isotropic", "isotropic|horizontal|axial", "Network source", "3D network", "Orientation prior for generated network branches."; choices=["isotropic", "horizontal", "axial"]),
        _cli_option("network-density-sigma-mm", "0", "mm", "Network source", "3D network", "Optional isotropic Gaussian density sigma. A value of 0 uses the anisotropic sigma options."),
        _cli_option("network-density-axial-sigma-mm", "10.0", "mm", "Network source", "3D network", "Axial Gaussian density sigma for network source sampling."),
        _cli_option("network-density-lateral-y-sigma-mm", "1.5", "mm", "Network source", "3D network", "Y Gaussian density sigma for network source sampling."),
        _cli_option("network-density-lateral-z-sigma-mm", "1.5", "mm", "Network source", "3D network", "Z Gaussian density sigma for network source sampling."),
        _cli_option("network-max-sources-per-center", "80", "integer", "Network source", "3D network", "Caps generated sources per network center. Values <= 0 disable the cap."),

        _cli_option("axial-mm", "80", "mm", "Grid", "PAM", "Requested axial domain depth. The runner may extend this to fit sources and time of flight."),
        _cli_option("transverse-mm", "102.4", "mm", "Grid", "PAM", "Default lateral domain width. In 3D this seeds y and z widths unless overridden."),
        _cli_option("transverse-y-mm", "", "mm", "Grid", "3D", "Overrides the 3D y-width when set."),
        _cli_option("transverse-z-mm", "", "mm", "Grid", "3D", "Overrides the 3D z-width when set."),
        _cli_option("dx-mm", "0.2", "mm", "Grid", "PAM", "Axial grid spacing."),
        _cli_option("dy-mm", "", "mm", "Grid", "3D", "3D y grid spacing. Defaults to --dz-mm when omitted."),
        _cli_option("dz-mm", "0.2", "mm", "Grid", "PAM", "2D lateral spacing or 3D z spacing."),
        _cli_option("t-max-us", "500", "microseconds", "Grid", "PAM", "Requested simulation duration. The runner may extend this when needed to capture source arrivals."),
        _cli_option("dt-ns", "20", "nanoseconds", "Grid", "PAM", "Simulation time step."),
        _cli_option("zero-pad-factor", "4", "integer", "Grid", "PAM", "Lateral FFT zero-padding factor used by ASA/HASA reconstruction."),
        _cli_option("bottom-margin-mm", "10", "mm", "Grid", "PAM", "Minimum margin below the deepest source when auto-fitting the PAM domain."),
        _cli_option("receiver-aperture-mm", "full", "mm|full", "Receiver", "PAM", "Receiver aperture width. Use full, all, or none to use the whole receiver plane."),
        _cli_option("receiver-aperture-y-mm", "", "mm|full", "Receiver", "3D", "Overrides the 3D receiver aperture in y."),
        _cli_option("receiver-aperture-z-mm", "", "mm|full", "Receiver", "3D", "Overrides the 3D receiver aperture in z."),
        _cli_option("peak-suppression-radius-mm", "8.0", "mm", "Analysis", "PAM", "Radius used to suppress neighboring peaks during localization analysis."),
        _cli_option("success-tolerance-mm", "1.5", "mm", "Analysis", "PAM", "Localization error threshold used when reporting success."),
        _cli_option("axial-gain-power", "1.5", "power", "Analysis", "3D", "Depth-gain exponent applied in 3D analysis/visualization."),

        _cli_option("aberrator", "none", "none|water|skull", "Medium", "PAM", "Selects homogeneous water/no aberrator or a CT-derived skull medium."; choices=["none", "water", "skull"]),
        _cli_option("ct-path", DEFAULT_CT_PATH, "path", "Medium", "skull", "Path to the private DICOM folder used for CT-backed skull media."),
        _cli_option("slice-index", "250", "integer", "Medium", "skull", "CT slice index used when building the skull medium."),
        _cli_option("skull-transducer-distance-mm", "30", "mm", "Medium", "skull", "Distance from the receiver/transducer plane to the outer skull surface."),
        _cli_option("hu-bone-thr", "200", "HU", "Medium", "skull", "Hounsfield-unit threshold used to identify bone in CT data."),

        _cli_option("simulation-backend", "kwave", "kwave|analytic", "Simulation", "PAM", "Forward model backend. CT skull runs require k-Wave."; choices=["kwave", "analytic"]),
        _cli_option("kwave-use-gpu", "true", "bool", "Simulation", "k-Wave", "Passes GPU execution to k-Wave where supported."),
        _cli_option("recon-use-gpu", "true", "bool", "Reconstruction", "PAM", "Uses the CUDA.jl reconstruction backend. 3D reconstruction currently requires this to be true."),
        _cli_option("recon-bandwidth-khz", "500", "kHz", "Reconstruction", "PAM", "Half-width bandwidth used to select frequency bins around reconstruction frequencies. Use 0 to keep only the target bins."),
        _cli_option("recon-step-um", "50", "micrometers", "Reconstruction", "PAM", "Axial integration step used by ASA/HASA reconstruction."),
        _cli_option("recon-mode", "auto", "auto|full|windowed", "Reconstruction", "PAM", "Reconstruction mode. Auto uses full for point sources and windowed for squiggle/network activity."; choices=["auto", "full", "windowed"]),
        _cli_option("recon-window-us", "20", "microseconds", "Reconstruction", "windowed", "Window duration for windowed incoherent reconstruction."),
        _cli_option("recon-hop-us", "10", "microseconds", "Reconstruction", "windowed", "Hop between consecutive reconstruction windows."),
        _cli_option("recon-window-taper", "hann", "hann|none|rectangular|tukey", "Reconstruction", "windowed", "Taper applied to each reconstruction window."; choices=["hann", "none", "rectangular", "tukey"]),
        _cli_option("recon-min-window-energy-ratio", "0.001", "ratio", "Reconstruction", "windowed", "Skips windows whose energy is below this fraction of the maximum window energy."),
        _cli_option("recon-progress", "false", "bool", "Reconstruction", "PAM", "Prints reconstruction progress updates."),
        _cli_option("window-batch", "1", "integer", "Reconstruction", "windowed GPU", "Number of reconstruction windows batched together on the GPU."),

        _cli_option("analysis-mode", "auto", "auto|localization|detection", "Analysis", "PAM", "Selects localization or activity-detection metrics. Auto uses detection for squiggle/network sources."; choices=["auto", "localization", "detection"]),
        _cli_option("detection-threshold-ratio", "0.2", "ratio", "Analysis", "detection", "Single threshold ratio used by basic detection analysis."),
        _cli_option("boundary-threshold-ratios", "0.5,0.55,0.6,0.65,0.7,0.75", "comma list", "Analysis", "detection", "Threshold ratios used for boundary overlays and threshold sweeps."),
        _cli_option("auto-threshold-search", "true", "bool", "Analysis", "detection", "Searches a dense threshold range and selects representative detection thresholds."),
        _cli_option("auto-threshold-min", "0.10", "ratio", "Analysis", "detection", "Minimum threshold ratio for automatic threshold search."),
        _cli_option("auto-threshold-max", "0.95", "ratio", "Analysis", "detection", "Maximum threshold ratio for automatic threshold search."),
        _cli_option("auto-threshold-step", "0.01", "ratio", "Analysis", "detection", "Threshold ratio spacing for automatic threshold search."),
    ]
end

function pam_cli_defaults()
    return Dict(option.name => option.default for option in pam_cli_options())
end

function parse_cli(args)
    opts = pam_cli_defaults()

    provided_keys = Set{String}()
    for arg in args
        startswith(arg, "--") || error("Unsupported argument format: $arg")
        parts = split(arg[3:end], "="; limit=2)
        length(parts) == 2 || error("Arguments must use --name=value, got: $arg")
        push!(provided_keys, parts[1])
        opts[parts[1]] = parts[2]
    end
    apply_model_defaults!(opts, provided_keys)
    return opts, provided_keys
end

slug_value(x; digits::Int=1) = replace(string(round(Float64(x); digits=digits)), "-" => "m", "." => "p")
parse_bool(s::AbstractString) = lowercase(strip(s)) in ("1", "true", "yes", "on")

function parse_dimension(s::AbstractString)
    value = strip(s)
    value in ("2", "2d", "2D") && return 2
    value in ("3", "3d", "3D") && return 3
    error("--dimension must be 2 or 3, got: $s")
end

function parse_float_list(spec::AbstractString)
    isempty(strip(spec)) && return Float64[]
    return [parse(Float64, strip(item)) for item in split(spec, ",") if !isempty(strip(item))]
end

function parse_int_list(spec::AbstractString)
    isempty(strip(spec)) && return Int[]
    return [parse(Int, strip(item)) for item in split(spec, ",") if !isempty(strip(item))]
end

function parse_threshold_ratios(spec::AbstractString)
    ratios = parse_float_list(spec)
    isempty(ratios) && error("At least one threshold ratio is required.")
    all(r -> r > 0, ratios) || error("Threshold ratios must be positive.")
    return sort(unique(ratios))
end

function parse_threshold_search_ratios(opts)
    min_ratio = parse(Float64, opts["auto-threshold-min"])
    max_ratio = parse(Float64, opts["auto-threshold-max"])
    step = parse(Float64, opts["auto-threshold-step"])
    min_ratio > 0 || error("--auto-threshold-min must be positive.")
    max_ratio >= min_ratio || error("--auto-threshold-max must be >= --auto-threshold-min.")
    step > 0 || error("--auto-threshold-step must be positive.")
    n = floor(Int, (max_ratio - min_ratio) / step + 1e-9)
    ratios = [round(min_ratio + i * step; digits=6) for i in 0:n]
    if isempty(ratios) || ratios[end] < max_ratio - 1e-9
        push!(ratios, round(max_ratio; digits=6))
    end
    return sort(unique(ratios))
end

function parse_source_model(s::AbstractString)
    value = Symbol(lowercase(strip(s)))
    value in (:point, :squiggle, :network) || error("--source-model must be point, squiggle, or network, got: $s")
    return value
end

function apply_model_defaults!(opts, provided_keys::Set{String})
    dimension = parse_dimension(opts["dimension"])
    if dimension == 3
        !("source-model" in provided_keys) && (opts["source-model"] = "point")
        !("sources-mm" in provided_keys) && (opts["sources-mm"] = "30:0:0")
        !("anchors-mm" in provided_keys) && (opts["anchors-mm"] = "45:0:0")
        !("vascular-squiggle-amplitude-x-mm" in provided_keys) && (opts["vascular-squiggle-amplitude-x-mm"] = "1.0")
        !("squiggle-phase-x-deg" in provided_keys) && (opts["squiggle-phase-x-deg"] = "90")
        !("frequency-mhz" in provided_keys) && (opts["frequency-mhz"] = "0.5")
        !("recon-bandwidth-khz" in provided_keys) && (opts["recon-bandwidth-khz"] = "0")
        !("receiver-aperture-mm" in provided_keys) && (opts["receiver-aperture-mm"] = "full")
        !("dx-mm" in provided_keys) && (opts["dx-mm"] = "0.2")
        !("dy-mm" in provided_keys) && (opts["dy-mm"] = "0.5")
        !("dz-mm" in provided_keys) && (opts["dz-mm"] = "0.5")
        !("axial-mm" in provided_keys) && (opts["axial-mm"] = "60")
        !("transverse-mm" in provided_keys) && (opts["transverse-mm"] = "32")
        !("dt-ns" in provided_keys) && (opts["dt-ns"] = "80")
        !("t-max-us" in provided_keys) && (opts["t-max-us"] = "60")
        !("zero-pad-factor" in provided_keys) && (opts["zero-pad-factor"] = "4")
        !("num-cycles" in provided_keys) && (opts["num-cycles"] = "5")
        !("phase-mode" in provided_keys) && (opts["phase-mode"] = "coherent")
        !("recon-step-um" in provided_keys) && (opts["recon-step-um"] = "50")
    end
    source_model = parse_source_model(opts["source-model"])
    if dimension == 3 && source_model in (:squiggle, :network)
        !("vascular-source-spacing-mm" in provided_keys) && (opts["vascular-source-spacing-mm"] = "0.5")
        !("vascular-min-separation-mm" in provided_keys) && (opts["vascular-min-separation-mm"] = "0.25")
        !("recon-bandwidth-khz" in provided_keys) && (opts["recon-bandwidth-khz"] = "40")
        !("recon-window-us" in provided_keys) && (opts["recon-window-us"] = "40")
        !("recon-hop-us" in provided_keys) && (opts["recon-hop-us"] = "20")
        !("boundary-threshold-ratios" in provided_keys) && (opts["boundary-threshold-ratios"] = "0.5,0.55,0.6,0.65,0.7,0.75")
    end
    if source_model == :point
        !("source-phase-mode" in provided_keys) && (opts["source-phase-mode"] = "coherent")
        !("recon-bandwidth-khz" in provided_keys) && (opts["recon-bandwidth-khz"] = "0")
        !("receiver-aperture-mm" in provided_keys) && (opts["receiver-aperture-mm"] = "50")
        !("transverse-mm" in provided_keys) && (opts["transverse-mm"] = "60")
        !("dt-ns" in provided_keys) && (opts["dt-ns"] = "40")
        !("t-max-us" in provided_keys) && (opts["t-max-us"] = "60")
        !("axial-mm" in provided_keys) && (opts["axial-mm"] = "60")
        !("phase-mode" in provided_keys) && (opts["phase-mode"] = "coherent")
    end
    return opts
end

function parse_aberrator(s::AbstractString)
    value = Symbol(lowercase(strip(s)))
    value in (:none, :water, :skull) || error("Unknown aberrator: $s")
    return value
end

function parse_simulation_backend(s::AbstractString)
    value = Symbol(lowercase(strip(s)))
    value in (:analytic, :kwave) || error("Unknown --simulation-backend: $s (must be analytic or kwave)")
    return value
end

function parse_source_phase_mode(s::AbstractString)
    value = Symbol(replace(lowercase(strip(s)), "-" => "_"))
    value in (:coherent, :random_static_phase, :random_phase_per_window) ||
        error("--source-phase-mode must be coherent, random_static_phase, or random_phase_per_window, got: $s")
    return value
end

parse_source_variability(opts) = SourceVariabilityConfig(
    frequency_jitter_fraction=parse(Float64, opts["frequency-jitter-percent"]) / 100.0,
)

function source_variability_from_summary(summary)
    if isnothing(summary) || !hasproperty(summary, :source_variability)
        return SourceVariabilityConfig()
    end
    sv = summary.source_variability
    if hasproperty(sv, :frequency_jitter_percent)
        return SourceVariabilityConfig(frequency_jitter_fraction=Float64(sv.frequency_jitter_percent) / 100.0)
    end
    return SourceVariabilityConfig()
end

function parse_analysis_mode(s::AbstractString, source_model::Symbol)
    value = Symbol(lowercase(strip(s)))
    value == :auto && return source_model in (:squiggle, :network) ? :detection : :localization
    value in (:localization, :detection) || error("--analysis-mode must be auto, localization, or detection, got: $s")
    return value
end

resolve_reconstruction_mode(s::AbstractString, source_model::Symbol) =
    TranscranialFUS.pam_reconstruction_mode(s, source_model)

function parse_window_taper(s::AbstractString)
    value = Symbol(replace(lowercase(strip(s)), "-" => "_"))
    value in (:hann, :none, :rect, :rectangular, :tukey) ||
        error("--recon-window-taper must be hann, none, rectangular, or tukey, got: $s")
    return value
end

function make_window_config(opts, reconstruction_mode::Symbol)
    return PAMWindowConfig(
        enabled=reconstruction_mode == :windowed,
        window_duration=parse(Float64, opts["recon-window-us"]) * 1e-6,
        hop=parse(Float64, opts["recon-hop-us"]) * 1e-6,
        taper=parse_window_taper(opts["recon-window-taper"]),
        min_energy_ratio=parse(Float64, opts["recon-min-window-energy-ratio"]),
        accumulation=:intensity,
    )
end

function parse_receiver_aperture_mm(s::AbstractString)
    value = lowercase(strip(s))
    value in ("none", "full", "all") && return nothing
    return parse(Float64, value) * 1e-3
end

function parse_transducer_mm(s::AbstractString)
    parts = split(strip(s), ":"; limit=2)
    length(parts) == 2 || error("--transducer-mm must be depth_mm:lateral_mm, got: $s")
    return parse(Float64, strip(parts[1])) * 1e-3, parse(Float64, strip(parts[2])) * 1e-3
end

function default_output_dir(opts, sources, cfg, emission_meta)
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    source_model = lowercase(String(emission_meta["source_model"]))
    lateral_slug = if cfg isa PAMConfig3D
        "laty$(slug_value(cfg.transverse_dim_y * 1e3; digits=0))mm_latz$(slug_value(cfg.transverse_dim_z * 1e3; digits=0))mm"
    else
        "lat$(slug_value(cfg.transverse_dim * 1e3; digits=0))mm"
    end
    parts = String[
        timestamp,
        "run_pam",
        cfg isa PAMConfig3D ? "3d" : "2d",
        lowercase(opts["aberrator"]),
        source_model,
        "$(length(sources))src",
        "ax$(slug_value(cfg.axial_dim * 1e3; digits=0))mm",
        lateral_slug,
    ]
    if occursin("squiggle", source_model) || occursin("network", source_model)
        count_key = haskey(emission_meta, "n_anchor_clusters") ? "n_anchor_clusters" : "n_network_centers"
        label = occursin("network", source_model) ? "centers" : "anchors"
        insert!(parts, 5, "$(emission_meta[count_key])$(label)")
        push!(parts, "f$(slug_value(parse(Float64, opts["fundamental-mhz"]); digits=2))mhz")
        push!(parts, "h$(replace(opts["harmonics"], "," => ""))")
        push!(parts, replace(lowercase(opts["source-phase-mode"]), "_" => ""))
    else
        push!(parts, "f$(slug_value(parse(Float64, opts["frequency-mhz"]); digits=2))mhz")
    end
    if lowercase(opts["aberrator"]) == "skull"
        insert!(parts, length(parts), "slice" * opts["slice-index"])
        insert!(parts, length(parts), "st$(slug_value(parse(Float64, opts["skull-transducer-distance-mm"]); digits=1))mm")
    end
    return joinpath(pwd(), "outputs", join(parts, "_"))
end

function default_reconstruction_output_dir(source_dir::AbstractString)
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    source_name = basename(normpath(source_dir))
    return joinpath(pwd(), "outputs", "$(timestamp)_reconstruct_$(source_name)")
end

function reject_cached_simulation_options!(provided_keys::Set{String}, blocked_keys)
    illegal = sort(collect(intersect(provided_keys, Set(blocked_keys))))
    isempty(illegal) && return nothing
    formatted = join(["--$key" for key in illegal], ", ")
    error("--from-run-dir reuses the previous RF simulation, medium, sources, and grid. Remove simulation-specific option(s): $formatted")
end
