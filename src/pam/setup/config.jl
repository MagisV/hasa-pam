# CLI/config parsing helpers for PAM runner scripts.

function parse_cli(args)
    opts = Dict{String, String}(
        "dimension" => "2",
        "source-model" => "squiggle",
        "sources-mm" => "30:0",
        "anchors-mm" => "45:0",
        "frequency-mhz" => "0.4",
        "fundamental-mhz" => "0.5",
        "amplitude-pa" => "1.0",
        "source-amplitudes-pa" => "",
        "source-frequencies-mhz" => "",
        "phases-deg" => "",
        "num-cycles" => "4",
        "harmonics" => "2,3,4",
        "harmonic-amplitudes" => "1.0,0.6,0.3",
        "gate-us" => "50",
        "taper-ratio" => "0.25",
        "axial-mm" => "80",
        "transverse-mm" => "102.4",
        "dx-mm" => "0.2",
        "dz-mm" => "0.2",
        "dy-mm" => "",
        "transverse-y-mm" => "",
        "transverse-z-mm" => "",
        "axial-gain-power" => "1.5",
        "receiver-aperture-mm" => "full",
        "receiver-aperture-y-mm" => "",
        "receiver-aperture-z-mm" => "",
        "t-max-us" => "500",
        "dt-ns" => "20",
        "zero-pad-factor" => "4",
        "peak-suppression-radius-mm" => "8.0",
        "success-tolerance-mm" => "1.5",
        "aberrator" => "none",
        "simulation-backend" => "kwave",
        "ct-path" => DEFAULT_CT_PATH,
        "slice-index" => "250",
        "skull-transducer-distance-mm" => "30",
        "bottom-margin-mm" => "10",
        "hu-bone-thr" => "200",
        "kwave-use-gpu" => "true",
        "recon-use-gpu" => "true",
        "recon-bandwidth-khz" => "500",
        "recon-step-um" => "50",
        "recon-mode" => "auto",
        "recon-window-us" => "20",
        "recon-hop-us" => "10",
        "recon-window-taper" => "hann",
        "recon-min-window-energy-ratio" => "0.001",
        "recon-progress" => "false",
        "benchmark" => "false",
        "window-batch" => "1",
        "phase-mode" => "geometric",
        "phase-jitter-rad" => "0.2",
        "random-seed" => "42",
        "source-phase-mode" => "random_phase_per_window",
        "frequency-jitter-percent" => "1",
        "transducer-mm" => "-30:0",
        "delays-us" => "0",
        "vascular-length-mm" => "12",
        "vascular-squiggle-amplitude-mm" => "1.5",
        "vascular-squiggle-amplitude-x-mm" => "1.0",
        "vascular-squiggle-wavelength-mm" => "8",
        "vascular-squiggle-slope" => "0.0",
        "squiggle-phase-x-deg" => "90",
        "vascular-source-spacing-mm" => "0.5",
        "vascular-position-jitter-mm" => "0.05",
        "vascular-min-separation-mm" => "0.25",
        "vascular-max-sources-per-anchor" => "0",
        "vascular-radius-mm" => "1.0",
        "network-axial-radius-mm" => "10.0",
        "network-lateral-y-radius-mm" => "1.5",
        "network-lateral-z-radius-mm" => "1.5",
        "network-root-count" => "12",
        "network-generations" => "3",
        "network-branch-length-mm" => "5.0",
        "network-branch-step-mm" => "0.4",
        "network-branch-angle-deg" => "36",
        "network-tortuosity" => "0.18",
        "network-orientation" => "isotropic",
        "network-density-sigma-mm" => "0",
        "network-density-axial-sigma-mm" => "10.0",
        "network-density-lateral-y-sigma-mm" => "1.5",
        "network-density-lateral-z-sigma-mm" => "1.5",
        "network-max-sources-per-center" => "80",
        "analysis-mode" => "auto",
        "detection-threshold-ratio" => "0.2",
        "boundary-threshold-ratios" => "0.5,0.55,0.6,0.65,0.7,0.75",
        "auto-threshold-search" => "true",
        "auto-threshold-min" => "0.10",
        "auto-threshold-max" => "0.95",
        "auto-threshold-step" => "0.01",
        "from-run-dir" => "",
    )

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
