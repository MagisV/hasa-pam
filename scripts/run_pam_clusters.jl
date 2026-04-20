#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Dates
using Random
using CairoMakie
using JLD2
using JSON3
using TranscranialFUS

function parse_cli(args)
    opts = Dict{String, String}(
        "clusters-mm" => "30:0",
        "fundamental-mhz" => "0.4",
        "amplitude-pa" => "1.0",
        "n-bubbles" => "10",
        "harmonics" => "2,3",
        "harmonic-amplitudes" => "1.0,0.6",
        "gate-us" => "50",
        "taper-ratio" => "0.25",
        "axial-mm" => "60",
        "transverse-mm" => "60",
        "dx-mm" => "0.2",
        "dz-mm" => "0.2",
        "receiver-aperture-mm" => "50",
        "t-max-us" => "80",
        "dt-ns" => "20",
        "zero-pad-factor" => "4",
        "peak-suppression-radius-mm" => "8.0",
        "success-tolerance-mm" => "1.5",
        "aberrator" => "none",
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
        "recon-bandwidth-khz" => "20",
        "phase-mode" => "geometric",
        "phase-jitter-rad" => "0.2",
        "random-seed" => "0",
        "transducer-mm" => "-30:0",
        "delays-us" => "0",
    )

    for arg in args
        startswith(arg, "--") || error("Unsupported argument format: $arg")
        parts = split(arg[3:end], "="; limit=2)
        length(parts) == 2 || error("Arguments must use --name=value, got: $arg")
        opts[parts[1]] = parts[2]
    end
    return opts
end

slug_value(x; digits::Int=1) = replace(string(round(Float64(x); digits=digits)), "-" => "m", "." => "p")
parse_bool(s::AbstractString) = lowercase(strip(s)) in ("1", "true", "yes", "on")

function parse_float_list(spec::AbstractString)
    isempty(strip(spec)) && return Float64[]
    return [parse(Float64, strip(item)) for item in split(spec, ",") if !isempty(strip(item))]
end

function parse_int_list(spec::AbstractString)
    isempty(strip(spec)) && return Int[]
    return [parse(Int, strip(item)) for item in split(spec, ",") if !isempty(strip(item))]
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

function parse_transducer_mm(s::AbstractString)
    parts = split(strip(s), ":"; limit=2)
    length(parts) == 2 || error("--transducer-mm must be depth_mm:lateral_mm, got: $s")
    return parse(Float64, strip(parts[1])) * 1e-3, parse(Float64, strip(parts[2])) * 1e-3
end

function expand_cluster_values(values::Vector{Float64}, n::Int, default::Float64)
    isempty(values) && return fill(default, n)
    length(values) == 1 && return fill(values[1], n)
    length(values) == n && return values
    error("Per-cluster parameter list must have length 1 or match the number of clusters ($n).")
end

function geometric_drive_phase(depth::Real, lateral::Real, tx_depth::Real, tx_lateral::Real, n::Int, f0::Real, c0::Real)
    d = hypot(depth - tx_depth, lateral - tx_lateral)
    return -2π * n * f0 * d / c0
end

function parse_clusters(opts, c0::Real)
    coord_tokens = [strip(token) for token in split(opts["clusters-mm"], ",") if !isempty(strip(token))]
    1 <= length(coord_tokens) <= 10 || error("Provide between 1 and 10 clusters via --clusters-mm=depth:lateral,...")

    f0 = parse(Float64, opts["fundamental-mhz"]) * 1e6
    harmonics = parse_int_list(opts["harmonics"])
    isempty(harmonics) && error("--harmonics must be a non-empty integer list.")
    harmonic_amplitudes = parse_float_list(opts["harmonic-amplitudes"])
    length(harmonic_amplitudes) == length(harmonics) ||
        error("--harmonic-amplitudes must have the same length as --harmonics ($(length(harmonics))).")

    gate = parse(Float64, opts["gate-us"]) * 1e-6
    taper = parse(Float64, opts["taper-ratio"])
    per_bubble_amp = parse(Float64, opts["amplitude-pa"])

    n_clusters = length(coord_tokens)
    n_bubbles_per = expand_cluster_values(parse_float_list(opts["n-bubbles"]), n_clusters, 10.0)
    delays_us = expand_cluster_values(parse_float_list(opts["delays-us"]), n_clusters, 0.0)

    tx_depth, tx_lateral = parse_transducer_mm(opts["transducer-mm"])
    phase_mode = lowercase(strip(opts["phase-mode"]))
    phase_mode in ("coherent", "geometric", "random", "jittered") ||
        error("Unknown phase-mode: $phase_mode (expected coherent|geometric|random|jittered).")
    phase_rng_used = phase_mode in ("random", "jittered")
    if phase_rng_used
        Random.seed!(parse(Int, opts["random-seed"]))
    end
    jitter_rad = parse(Float64, opts["phase-jitter-rad"])

    clusters = BubbleCluster2D[]
    for (idx, token) in pairs(coord_tokens)
        parts = split(token, ":"; limit=2)
        length(parts) == 2 || error("Each cluster must be specified as depth_mm:lateral_mm, got: $token")
        depth_m = parse(Float64, strip(parts[1])) * 1e-3
        lateral_m = parse(Float64, strip(parts[2])) * 1e-3

        phases = Vector{Float64}(undef, length(harmonics))
        for (h_idx, n) in pairs(harmonics)
            base = if phase_mode in ("geometric", "jittered")
                geometric_drive_phase(depth_m, lateral_m, tx_depth, tx_lateral, n, f0, c0)
            elseif phase_mode == "random"
                2π * rand()
            else
                0.0
            end
            if phase_mode == "jittered"
                base += randn() * jitter_rad
            end
            phases[h_idx] = base
        end

        push!(clusters, BubbleCluster2D(
            depth=depth_m,
            lateral=lateral_m,
            fundamental=f0,
            amplitude=per_bubble_amp,
            n_bubbles=n_bubbles_per[idx],
            harmonics=copy(harmonics),
            harmonic_amplitudes=copy(harmonic_amplitudes),
            harmonic_phases=phases,
            gate_duration=gate,
            taper_ratio=taper,
            delay=delays_us[idx] * 1e-6,
        ))
    end

    meta = Dict{String, Any}(
        "phase_mode" => phase_mode,
        "fundamental_hz" => f0,
        "harmonics" => harmonics,
        "harmonic_amplitudes" => harmonic_amplitudes,
        "gate_duration_s" => gate,
        "transducer_m" => (tx_depth, tx_lateral),
        "phase_jitter_rad" => jitter_rad,
        "random_seed" => parse(Int, opts["random-seed"]),
        "n_bubbles_per_cluster" => n_bubbles_per,
        "delays_s" => delays_us .* 1e-6,
    )
    return clusters, meta
end

function default_output_dir(opts, clusters, cfg)
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    parts = String[
        "run_pam_clusters",
        lowercase(opts["aberrator"]),
        "$(length(clusters))cl",
        "f$(slug_value(parse(Float64, opts["fundamental-mhz"]); digits=2))mhz",
        "h$(replace(opts["harmonics"], "," => ""))",
        lowercase(opts["phase-mode"]),
        "ax$(slug_value(cfg.axial_dim * 1e3; digits=0))mm",
        "lat$(slug_value(cfg.transverse_dim * 1e3; digits=0))mm",
        timestamp,
    ]
    if lowercase(opts["aberrator"]) == "skull"
        insert!(parts, length(parts), "slice" * opts["slice-index"])
        insert!(parts, length(parts), "st$(slug_value(parse(Float64, opts["skull-transducer-distance-mm"]); digits=1))mm")
    end
    return joinpath(pwd(), "outputs", join(parts, "_"))
end

function map_db(map::AbstractMatrix{<:Real}, ref::Real)
    safe_ref = max(Float64(ref), eps(Float64))
    return 10 .* log10.(max.(Float64.(map), eps(Float64)) ./ safe_ref)
end

function scatter_cluster_points!(ax, clusters; color=:red, marker=:x, markersize=14, strokewidth=2)
    depths_mm = [cl.depth * 1e3 for cl in clusters]
    laterals_mm = [cl.lateral * 1e3 for cl in clusters]
    scatter!(ax, laterals_mm, depths_mm; color=color, marker=marker, markersize=markersize, strokewidth=strokewidth)
end

function scatter_predicted_points!(ax, stats; color=:white, marker=:circle, markersize=12, strokewidth=2)
    pred = stats[:predicted_mm]
    scatter!(ax, last.(pred), first.(pred); color=color, marker=marker, markersize=markersize, strokecolor=color, strokewidth=strokewidth)
end

function save_overview(path, c, rf, pam_geo, pam_hasa, kgrid, cfg, clusters, stats_geo, stats_hasa)
    depth_mm = depth_coordinates(kgrid, cfg) .* 1e3
    lateral_mm = kgrid.y_vec .* 1e3
    time_us = collect(0:(size(rf, 2) - 1)) .* cfg.dt .* 1e6
    map_ref = max(maximum(Float64.(pam_geo)), maximum(Float64.(pam_hasa)), eps(Float64))
    pam_geo_db = map_db(pam_geo, map_ref)
    pam_hasa_db = map_db(pam_hasa, map_ref)

    fig = Figure(size=(1500, 1000))
    ax_medium = Axis(fig[1, 1]; title="Simulation Medium", xlabel="Lateral [mm]", ylabel="Depth [mm]", aspect=DataAspect())
    hm_medium = heatmap!(ax_medium, lateral_mm, depth_mm, Float64.(c)'; colormap=:thermal)
    hlines!(ax_medium, [0.0]; color=:white, linestyle=:dash)
    scatter_cluster_points!(ax_medium, clusters)
    Colorbar(fig[1, 2], hm_medium; label="c [m/s]")

    ax_rf = Axis(fig[1, 3]; title="Recorded RF", xlabel="Time [μs]", ylabel="Lateral [mm]")
    hm_rf = heatmap!(ax_rf, time_us, lateral_mm, rf'; colormap=:balance)
    Colorbar(fig[1, 4], hm_rf; label="Pressure [Pa]")

    ax_geo = Axis(fig[2, 1]; title="Geometric ASA | err = $(round(stats_geo[:mean_radial_error_mm]; digits=2)) mm",
                  xlabel="Lateral [mm]", ylabel="Depth [mm]", aspect=DataAspect())
    hm_geo = heatmap!(ax_geo, lateral_mm, depth_mm, pam_geo_db'; colormap=:viridis, colorrange=(-30, 0))
    hlines!(ax_geo, [0.0]; color=:white, linestyle=:dash)
    scatter_cluster_points!(ax_geo, clusters)
    scatter_predicted_points!(ax_geo, stats_geo)

    ax_hasa = Axis(fig[2, 3]; title="HASA | err = $(round(stats_hasa[:mean_radial_error_mm]; digits=2)) mm",
                   xlabel="Lateral [mm]", ylabel="Depth [mm]", aspect=DataAspect())
    hm_hasa = heatmap!(ax_hasa, lateral_mm, depth_mm, pam_hasa_db'; colormap=:viridis, colorrange=(-30, 0))
    hlines!(ax_hasa, [0.0]; color=:white, linestyle=:dash)
    scatter_cluster_points!(ax_hasa, clusters)
    scatter_predicted_points!(ax_hasa, stats_hasa)
    Colorbar(fig[2, 4], hm_hasa; label="Intensity [dB]")

    save(path, fig)
end

opts = parse_cli(ARGS)

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

clusters, cluster_meta = parse_clusters(opts, cfg_base.c0)

aberrator = parse_aberrator(opts["aberrator"])
cfg = fit_pam_config(
    cfg_base,
    clusters;
    min_bottom_margin=parse(Float64, opts["bottom-margin-mm"]) * 1e-3,
    reference_depth=aberrator == :skull ? parse(Float64, opts["skull-transducer-distance-mm"]) * 1e-3 : nothing,
)

out_dir = if haskey(opts, "out-dir") && !isempty(strip(opts["out-dir"]))
    opts["out-dir"]
else
    default_output_dir(opts, clusters, cfg)
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

medium_summary = Dict{String, Any}()
for (key, value) in medium_info
    key == :mask && continue
    medium_summary[String(key)] = value
end

recon_frequencies = if haskey(opts, "recon-frequencies-mhz") && !isempty(strip(opts["recon-frequencies-mhz"]))
    parse_float_list(opts["recon-frequencies-mhz"]) .* 1e6
else
    sort(unique(Float64[n * cluster_meta["fundamental_hz"] for n in cluster_meta["harmonics"]]))
end
recon_bandwidth_hz = parse(Float64, opts["recon-bandwidth-khz"]) * 1e3

results = run_pam_case(
    c,
    rho,
    clusters,
    cfg;
    frequencies=recon_frequencies,
    bandwidth=recon_bandwidth_hz,
    use_gpu=parse_bool(opts["use-gpu"]),
)

save_overview(
    joinpath(out_dir, "overview.png"),
    c, results[:rf], results[:pam_geo], results[:pam_hasa],
    results[:kgrid], cfg, clusters, results[:stats_geo], results[:stats_hasa],
)

summary = Dict(
    "out_dir" => out_dir,
    "clusters" => [Dict(
        "depth_m" => cl.depth,
        "lateral_m" => cl.lateral,
        "fundamental_hz" => cl.fundamental,
        "amplitude_pa" => cl.amplitude,
        "n_bubbles" => cl.n_bubbles,
        "harmonics" => cl.harmonics,
        "harmonic_amplitudes" => cl.harmonic_amplitudes,
        "harmonic_phases_rad" => cl.harmonic_phases,
        "gate_duration_s" => cl.gate_duration,
        "delay_s" => cl.delay,
    ) for cl in clusters],
    "emission_meta" => cluster_meta,
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
        "bottom_margin" => parse(Float64, opts["bottom-margin-mm"]) * 1e-3,
    ),
    "medium" => medium_summary,
    "reconstruction_frequencies_hz" => recon_frequencies,
    "reconstruction_bandwidth_hz" => recon_bandwidth_hz,
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

@save joinpath(out_dir, "result.jld2") c rho cfg clusters results medium_info

println("Saved PAM cluster outputs to $out_dir")
