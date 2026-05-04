#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Dates
using Printf
using Random
using CairoMakie
using JLD2
using JSON3
using TranscranialFUS

function parse_cli(args)
    opts = Dict{String, String}(
        "sources-mm" => "30:0",
        "frequency-mhz" => "0.4",
        "amplitude-pa" => "5e4",
        "num-cycles" => "4",
        "axial-mm" => "60",
        "transverse-mm" => "60",
        "dx-mm" => "0.2",
        "dz-mm" => "0.2",
        "receiver-aperture-mm" => "50",
        "t-max-us" => "60",
        "dt-ns" => "40",
        "zero-pad-factor" => "4",
        "peak-suppression-radius-mm" => "2.0",
        "success-tolerance-mm" => "1.0",
        "aberrator" => "lens",
        "ct-path" => DEFAULT_CT_PATH,
        "slice-index" => "250",
        "skull-transducer-distance-mm" => "30",
        "bottom-margin-mm" => "10",
        "hu-bone-thr" => "200",
        "lens-depth-mm" => "12",
        "lens-lateral-mm" => "0",
        "lens-axial-radius-mm" => "3",
        "lens-lateral-radius-mm" => "12",
        "aberrator-c" => "1700",
        "aberrator-rho" => "1150",
        "use-gpu" => "false",
        "recon-bandwidth-khz" => "0",
        "recon-step-um" => "50",
        "phase-mode" => "coherent",
        "phase-jitter-rad" => "0.2",
        "random-seed" => "0",
        "peak-method" => "argmax",
        "clean-loop-gain" => "0.1",
        "clean-max-iter" => "500",
        "clean-threshold-ratio" => "0.01",
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
    return opts, provided_keys
end

slug_value(x; digits::Int=1) = replace(string(round(Float64(x); digits=digits)), "-" => "m", "." => "p")

parse_bool(s::AbstractString) = lowercase(strip(s)) in ("1", "true", "yes", "on")

function parse_float_list(spec::AbstractString)
    isempty(strip(spec)) && return Float64[]
    return [parse(Float64, strip(item)) for item in split(spec, ",") if !isempty(strip(item))]
end

function parse_aberrator(s::AbstractString)
    value = Symbol(lowercase(strip(s)))
    value in (:none, :lens, :skull) || error("Unknown aberrator: $s")
    return value
end

function parse_receiver_aperture_mm(s::AbstractString)
    value = lowercase(strip(s))
    value in ("none", "full", "all") && return nothing
    return parse(Float64, value) * 1e-3
end

function expand_source_values(values::Vector{Float64}, n::Int, default::Float64)
    if isempty(values)
        return fill(default, n)
    elseif length(values) == 1
        return fill(values[1], n)
    elseif length(values) == n
        return values
    end
    error("Source parameter list must have length 1 or match the number of sources ($n).")
end

function parse_sources(opts)
    coord_tokens = [strip(token) for token in split(opts["sources-mm"], ",") if !isempty(strip(token))]
    1 <= length(coord_tokens) <= 5 || error("Provide between 1 and 5 sources via --sources-mm=depth:lateral,...")

    phases_deg = expand_source_values(
        haskey(opts, "phases-deg") ? parse_float_list(opts["phases-deg"]) : Float64[],
        length(coord_tokens),
        0.0,
    )
    delays_us = expand_source_values(
        haskey(opts, "delays-us") ? parse_float_list(opts["delays-us"]) : Float64[],
        length(coord_tokens),
        0.0,
    )
    amplitudes_pa = expand_source_values(
        haskey(opts, "source-amplitudes-pa") ? parse_float_list(opts["source-amplitudes-pa"]) : Float64[],
        length(coord_tokens),
        parse(Float64, opts["amplitude-pa"]),
    )
    frequencies_mhz = expand_source_values(
        haskey(opts, "source-frequencies-mhz") ? parse_float_list(opts["source-frequencies-mhz"]) : Float64[],
        length(coord_tokens),
        parse(Float64, opts["frequency-mhz"]),
    )
    num_cycles = parse(Int, opts["num-cycles"])

    phase_mode = lowercase(strip(opts["phase-mode"]))
    if phase_mode == "coherent"
        # use --phases-deg as provided
    elseif phase_mode == "random"
        Random.seed!(parse(Int, opts["random-seed"]))
        phases_deg = rand(length(coord_tokens)) .* 360.0
    elseif phase_mode == "jittered"
        Random.seed!(parse(Int, opts["random-seed"]))
        jitter_rad = parse(Float64, opts["phase-jitter-rad"])
        phases_deg = phases_deg .+ randn(length(coord_tokens)) .* (jitter_rad * 180 / π)
    else
        error("Unknown phase-mode: $phase_mode (expected: coherent|random|jittered)")
    end

    sources = PointSource2D[]
    for (idx, token) in pairs(coord_tokens)
        parts = split(token, ":"; limit=2)
        length(parts) == 2 || error("Each source must be specified as depth_mm:lateral_mm, got: $token")
        depth_mm = parse(Float64, strip(parts[1]))
        lateral_mm = parse(Float64, strip(parts[2]))
        push!(sources, PointSource2D(
            depth=depth_mm * 1e-3,
            lateral=lateral_mm * 1e-3,
            frequency=frequencies_mhz[idx] * 1e6,
            amplitude=amplitudes_pa[idx],
            phase=phases_deg[idx] * π / 180,
            delay=delays_us[idx] * 1e-6,
            num_cycles=num_cycles,
        ))
    end
    return sources
end

function default_output_dir(opts, sources, cfg)
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    parts = String[
        timestamp,
        "run_pam",
        lowercase(opts["aberrator"]),
        "$(length(sources))src",
        "f$(slug_value(parse(Float64, opts["frequency-mhz"]); digits=2))mhz",
        "ax$(slug_value(cfg.axial_dim * 1e3; digits=0))mm",
        "lat$(slug_value(cfg.transverse_dim * 1e3; digits=0))mm",
        "dx$(slug_value(cfg.dx * 1e3; digits=2))mm"
    ]
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

function default_simulation_info(cfg::PAMConfig)
    return Dict{Symbol, Any}(
        :receiver_row => receiver_row(cfg),
        :receiver_cols => receiver_col_range(cfg),
        :source_indices => Tuple{Int, Int}[],
    )
end

function map_db(map::AbstractMatrix{<:Real}, ref::Real)
    safe_ref = max(Float64(ref), eps(Float64))
    return 10 .* log10.(max.(Float64.(map), eps(Float64)) ./ safe_ref)
end

function source_pairs_mm(sources)
    return [(src.depth * 1e3, src.lateral * 1e3) for src in sources]
end

function scatter_source_points!(ax, sources; color=:red, marker=:x, markersize=14, strokewidth=2)
    truth = source_pairs_mm(sources)
    scatter!(
        ax,
        last.(truth),
        first.(truth);
        color=color,
        marker=marker,
        markersize=markersize,
        strokewidth=strokewidth,
    )
end

function scatter_predicted_points!(ax, stats; color=:white, marker=:circle, markersize=12, strokewidth=2)
    pred = stats[:predicted_mm]
    scatter!(
        ax,
        last.(pred),
        first.(pred);
        color=color,
        marker=marker,
        markersize=markersize,
        strokecolor=color,
        strokewidth=strokewidth,
    )
end

function save_overview(path, c, rf, pam_geo, pam_hasa, kgrid, cfg, sources, stats_geo, stats_hasa)
    depth_mm = depth_coordinates(kgrid, cfg) .* 1e3
    lateral_mm = kgrid.y_vec .* 1e3
    time_us = collect(0:(size(rf, 2) - 1)) .* cfg.dt .* 1e6
    map_ref = max(maximum(Float64.(pam_geo)), maximum(Float64.(pam_hasa)), eps(Float64))
    pam_geo_db = map_db(pam_geo, map_ref)
    pam_hasa_db = map_db(pam_hasa, map_ref)

    fig = Figure(size=(1500, 1000))

    ax_medium = Axis(
        fig[1, 1];
        title="Simulation Medium",
        xlabel="Lateral position [mm]",
        ylabel="Depth below receiver [mm]",
        aspect=DataAspect(),
    )
    hm_medium = heatmap!(ax_medium, lateral_mm, depth_mm, Float64.(c)'; colormap=:thermal)
    hlines!(ax_medium, [0.0]; color=:white, linestyle=:dash)
    scatter_source_points!(ax_medium, sources)
    Colorbar(fig[1, 2], hm_medium; label="Sound speed [m/s]")

    ax_rf = Axis(
        fig[1, 3];
        title="Recorded RF Data",
        xlabel="Time [μs]",
        ylabel="Lateral position [mm]",
    )
    hm_rf = heatmap!(ax_rf, time_us, lateral_mm, rf'; colormap=:balance)
    Colorbar(fig[1, 4], hm_rf; label="Pressure [Pa]")

    geo_title = "Geometric ASA | mean radial error = $(round(stats_geo[:mean_radial_error_mm]; digits=2)) mm"
    ax_geo = Axis(
        fig[2, 1];
        title=geo_title,
        xlabel="Lateral position [mm]",
        ylabel="Depth below receiver [mm]",
        aspect=DataAspect(),
    )
    hm_geo = heatmap!(ax_geo, lateral_mm, depth_mm, pam_geo_db'; colormap=:viridis, colorrange=(-30, 0))
    hlines!(ax_geo, [0.0]; color=:white, linestyle=:dash)
    scatter_source_points!(ax_geo, sources)
    scatter_predicted_points!(ax_geo, stats_geo)

    hasa_title = "HASA | mean radial error = $(round(stats_hasa[:mean_radial_error_mm]; digits=2)) mm"
    ax_hasa = Axis(
        fig[2, 3];
        title=hasa_title,
        xlabel="Lateral position [mm]",
        ylabel="Depth below receiver [mm]",
        aspect=DataAspect(),
    )
    hm_hasa = heatmap!(ax_hasa, lateral_mm, depth_mm, pam_hasa_db'; colormap=:viridis, colorrange=(-30, 0))
    hlines!(ax_hasa, [0.0]; color=:white, linestyle=:dash)
    scatter_source_points!(ax_hasa, sources)
    scatter_predicted_points!(ax_hasa, stats_hasa)
    Colorbar(fig[2, 4], hm_hasa; label="PAM intensity [dB]")

    save(path, fig)
end

format_sci(value::Real) = @sprintf("%.2e", Float64(value))

function string_key_dict(dict::AbstractDict)
    return Dict(String(key) => value for (key, value) in dict)
end

function overlay_medium_contour!(ax, c::AbstractMatrix{<:Real}, lateral_mm, depth_mm, cfg; tol::Real=5.0)
    medium = Float64.(abs.(Float64.(c) .- cfg.c0) .> Float64(tol))
    any(medium .> 0.0) || return nothing
    contour!(ax, lateral_mm, depth_mm, medium'; levels=[0.5], color=(:white, 0.45), linewidth=1.2)
    return nothing
end

function pam_heatmap_title(label::AbstractString, metrics)
    rel_peak = round(metrics[:relative_peak_intensity]; digits=2)
    peak = format_sci(metrics[:peak_intensity])
    total = format_sci(metrics[:integrated_intensity_m2])
    return "$label | rel peak=$rel_peak, peak=$peak, sum=$total"
end

function add_pam_heatmap_panel!(
    fig,
    row::Int,
    label::AbstractString,
    intensity::AbstractMatrix{<:Real},
    c,
    kgrid,
    cfg,
    sources,
    global_ref::Real,
    metrics,
)
    depth_mm = depth_coordinates(kgrid, cfg) .* 1e3
    lateral_mm = kgrid.y_vec .* 1e3
    norm_intensity = clamp.(Float64.(intensity) ./ max(Float64(global_ref), eps(Float64)), 0.0, 1.0)

    ax = Axis(
        fig[row, 1];
        title=pam_heatmap_title(label, metrics),
        xlabel=row == 2 ? "Lateral distance [mm]" : "",
        ylabel="Axial distance [mm]",
        aspect=DataAspect(),
    )
    hm = heatmap!(ax, lateral_mm, depth_mm, norm_intensity'; colormap=:turbo, colorrange=(0, 1))
    overlay_medium_contour!(ax, c, lateral_mm, depth_mm, cfg)
    scatter_source_points!(ax, sources; color=(:white, 0.85), marker=:circle, markersize=7, strokewidth=0)
    xlims!(ax, minimum(lateral_mm), maximum(lateral_mm))
    ylims!(ax, minimum(depth_mm), maximum(depth_mm))
    return hm
end

function save_pam_heatmap(path, c, pam_geo, pam_hasa, kgrid, cfg, sources; threshold_ratio::Real=0.2)
    global_ref = max(maximum(Float64.(pam_geo)), maximum(Float64.(pam_hasa)), eps(Float64))
    geo_metrics = pam_intensity_metrics(
        pam_geo,
        kgrid,
        cfg;
        threshold_ratio=threshold_ratio,
        reference_intensity=global_ref,
    )
    hasa_metrics = pam_intensity_metrics(
        pam_hasa,
        kgrid,
        cfg;
        threshold_ratio=threshold_ratio,
        reference_intensity=global_ref,
    )

    fig = Figure(size=(900, 1100), fontsize=22)
    hm = add_pam_heatmap_panel!(fig, 1, "Uncorrected", pam_geo, c, kgrid, cfg, sources, global_ref, geo_metrics)
    add_pam_heatmap_panel!(fig, 2, "Corrected", pam_hasa, c, kgrid, cfg, sources, global_ref, hasa_metrics)
    Colorbar(fig[1:2, 2], hm; label="Norm. PAM intensity (shared max)")

    save(path, fig)
    return Dict(
        "global_reference_intensity" => global_ref,
        "geometric" => string_key_dict(geo_metrics),
        "hasa" => string_key_dict(hasa_metrics),
    )
end

opts, provided_keys = parse_cli(ARGS)
from_run_dir = strip(opts["from-run-dir"])
recon_bandwidth_hz = parse(Float64, opts["recon-bandwidth-khz"]) * 1e3
peak_method = Symbol(lowercase(strip(opts["peak-method"])))
peak_method in (:argmax, :clean) || error("--peak-method must be argmax or clean, got: $(opts["peak-method"])")

if isempty(from_run_dir)
    sources = parse_sources(opts)
    cfg = PAMConfig(
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

    aberrator = parse_aberrator(opts["aberrator"])
    bottom_margin_m = parse(Float64, opts["bottom-margin-mm"]) * 1e-3
    cfg = fit_pam_config(
        cfg,
        sources;
        min_bottom_margin=bottom_margin_m,
        reference_depth=aberrator == :skull ? parse(Float64, opts["skull-transducer-distance-mm"]) * 1e-3 : nothing,
    )
    out_dir = if haskey(opts, "out-dir") && !isempty(strip(opts["out-dir"]))
        opts["out-dir"]
    else
        default_output_dir(opts, sources, cfg)
    end
    mkpath(out_dir)

    c, rho, medium_info = make_pam_medium(
        cfg;
        aberrator=aberrator,
        lens_center_depth=parse(Float64, opts["lens-depth-mm"]) * 1e-3,
        lens_center_lateral=parse(Float64, opts["lens-lateral-mm"]) * 1e-3,
        lens_axial_radius=parse(Float64, opts["lens-axial-radius-mm"]) * 1e-3,
        lens_lateral_radius=parse(Float64, opts["lens-lateral-radius-mm"]) * 1e-3,
        c_aberrator=parse(Float64, opts["aberrator-c"]),
        rho_aberrator=parse(Float64, opts["aberrator-rho"]),
        ct_path=opts["ct-path"],
        slice_index=parse(Int, opts["slice-index"]),
        skull_to_transducer=parse(Float64, opts["skull-transducer-distance-mm"]) * 1e-3,
        hu_bone_thr=parse(Int, opts["hu-bone-thr"]),
    )

    recon_frequencies = if haskey(opts, "recon-frequencies-mhz")
        parse_float_list(opts["recon-frequencies-mhz"]) .* 1e6
    else
        sort(unique(Float64[src.frequency for src in sources]))
    end

    results = run_pam_case(
        c,
        rho,
        sources,
        cfg;
        frequencies=recon_frequencies,
        bandwidth=recon_bandwidth_hz,
        use_gpu=parse_bool(opts["use-gpu"]),
        reconstruction_axial_step=parse(Float64, opts["recon-step-um"]) * 1e-6,
        peak_method=peak_method,
        clean_loop_gain=parse(Float64, opts["clean-loop-gain"]),
        clean_max_iter=parse(Int, opts["clean-max-iter"]),
        clean_threshold_ratio=parse(Float64, opts["clean-threshold-ratio"]),
    )
    reconstruction_source = Dict("mode" => "simulation")
    phase_mode_summary = lowercase(strip(opts["phase-mode"]))
    random_seed_summary = parse(Int, opts["random-seed"])
else
    reject_cached_simulation_options!(
        provided_keys,
        (
            "sources-mm", "frequency-mhz", "amplitude-pa", "num-cycles",
            "phases-deg", "delays-us", "source-amplitudes-pa", "source-frequencies-mhz",
            "phase-mode", "phase-jitter-rad", "random-seed",
            "axial-mm", "transverse-mm", "dx-mm", "dz-mm", "receiver-aperture-mm",
            "t-max-us", "dt-ns", "zero-pad-factor", "peak-suppression-radius-mm",
            "success-tolerance-mm", "aberrator", "ct-path", "slice-index",
            "skull-transducer-distance-mm", "bottom-margin-mm", "hu-bone-thr",
            "lens-depth-mm", "lens-lateral-mm", "lens-axial-radius-mm", "lens-lateral-radius-mm",
            "aberrator-c", "aberrator-rho", "use-gpu",
        ),
    )
    cached_path = joinpath(from_run_dir, "result.jld2")
    isfile(cached_path) || error("--from-run-dir must contain result.jld2, missing: $cached_path")
    cached = load(cached_path)
    c = cached["c"]
    rho = haskey(cached, "rho") ? cached["rho"] : fill(Float32(cached["cfg"].rho0), size(c))
    cfg = cached["cfg"]
    sources = cached["sources"]
    cached_results = cached["results"]
    rf = cached_results[:rf]
    medium_info = haskey(cached, "medium_info") ? cached["medium_info"] : Dict{Symbol, Any}(:aberrator => :cached)
    bottom_margin_m = nothing

    out_dir = if haskey(opts, "out-dir") && !isempty(strip(opts["out-dir"]))
        opts["out-dir"]
    else
        default_reconstruction_output_dir(from_run_dir)
    end
    mkpath(out_dir)

    recon_frequencies = if haskey(opts, "recon-frequencies-mhz")
        parse_float_list(opts["recon-frequencies-mhz"]) .* 1e6
    else
        sort(unique(Float64[src.frequency for src in sources]))
    end
    simulation_info = haskey(cached_results, :simulation) ? cached_results[:simulation] : default_simulation_info(cfg)
    results = reconstruct_pam_case(
        rf,
        c,
        sources,
        cfg;
        simulation_info=simulation_info,
        frequencies=recon_frequencies,
        bandwidth=recon_bandwidth_hz,
        reconstruction_axial_step=parse(Float64, opts["recon-step-um"]) * 1e-6,
        peak_method=peak_method,
        clean_loop_gain=parse(Float64, opts["clean-loop-gain"]),
        clean_max_iter=parse(Int, opts["clean-max-iter"]),
        clean_threshold_ratio=parse(Float64, opts["clean-threshold-ratio"]),
    )
    reconstruction_source = Dict(
        "mode" => "cached_rf",
        "from_run_dir" => abspath(from_run_dir),
        "from_result_jld2" => abspath(cached_path),
    )
    phase_mode_summary = nothing
    random_seed_summary = nothing
end

medium_summary = Dict{String, Any}()
for (key, value) in medium_info
    key == :mask && continue
    medium_summary[String(key)] = value
end

save_overview(
    joinpath(out_dir, "overview.png"),
    c,
    results[:rf],
    results[:pam_geo],
    results[:pam_hasa],
    results[:kgrid],
    cfg,
    sources,
    results[:stats_geo],
    results[:stats_hasa],
)

heatmap_path = joinpath(out_dir, "pam_heatmap.png")
heatmap_metrics = save_pam_heatmap(
    heatmap_path,
    c,
    results[:pam_geo],
    results[:pam_hasa],
    results[:kgrid],
    cfg,
    sources,
)

summary = Dict(
    "out_dir" => out_dir,
    "reconstruction_source" => reconstruction_source,
    "heatmap_figure" => heatmap_path,
    "pam_heatmap_metrics" => heatmap_metrics,
    "sources" => [Dict(
        "depth_m" => src.depth,
        "lateral_m" => src.lateral,
        "frequency_hz" => src.frequency,
        "amplitude_pa" => src.amplitude,
        "phase_rad" => src.phase,
        "delay_s" => src.delay,
        "num_cycles" => src.num_cycles,
    ) for src in sources],
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
        "PML_GUARD" => cfg.PML_GUARD,
        "effective_pml_guard" => TranscranialFUS._pam_pml_guard(cfg),
        "zero_pad_factor" => cfg.zero_pad_factor,
        "peak_suppression_radius" => cfg.peak_suppression_radius,
        "success_tolerance" => cfg.success_tolerance,
        "bottom_margin" => bottom_margin_m,
    ),
    "medium" => medium_summary,
    "reconstruction_frequencies_hz" => recon_frequencies,
    "reconstruction_bandwidth_hz" => recon_bandwidth_hz,
    "reconstruction_axial_step_m" => results[:geo_info][:axial_step],
    "reference_sound_speed_m_per_s" => results[:geo_info][:reference_sound_speed],
    "phase_mode" => phase_mode_summary,
    "random_seed" => random_seed_summary,
    "peak_method" => String(peak_method),
    "clean_loop_gain" => parse(Float64, opts["clean-loop-gain"]),
    "clean_max_iter" => parse(Int, opts["clean-max-iter"]),
    "clean_threshold_ratio" => parse(Float64, opts["clean-threshold-ratio"]),
    "simulation" => Dict(
        "receiver_row" => results[:simulation][:receiver_row],
        "receiver_cols" => [first(results[:simulation][:receiver_cols]), last(results[:simulation][:receiver_cols])],
        "source_indices" => [[row, col] for (row, col) in results[:simulation][:source_indices]],
    ),
    "geometric" => results[:stats_geo],
    "hasa" => results[:stats_hasa],
)

open(joinpath(out_dir, "summary.json"), "w") do io
    JSON3.pretty(io, summary)
end

@save joinpath(out_dir, "result.jld2") c rho cfg sources results medium_info

println("Saved PAM outputs to $out_dir")
