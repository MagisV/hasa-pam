#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using JSON3
using TranscranialFUS

function parse_cli(args)
    opts = Dict{String, String}(
        "ct-path" => DEFAULT_CT_PATH,
        "slice-index" => "250",
        "frequency-mhz" => "0.5",
        "focal-cm" => "6.0",
        "lateral-cm" => "0.0",
        "aperture-cm" => "10.0",
        "medium" => "skull_in_water",
        "out-dir" => joinpath(pwd(), "outputs", "compare_estimators"),
    )

    for arg in args
        startswith(arg, "--") || error("Unsupported argument format: $arg")
        parts = split(arg[3:end], "="; limit=2)
        length(parts) == 2 || error("Arguments must use --name=value, got: $arg")
        opts[parts[1]] = parts[2]
    end
    return opts
end

parse_medium(s) = lowercase(s) == "water" ? WATER : lowercase(s) == "skull_in_water" ? SKULL_IN_WATER : error("Unknown medium: $s")

function pressure_figure(path, p_geo, p_hasa, kgrid, geo_stats, hasa_stats)
    vmin = min(minimum(p_geo), minimum(p_hasa))
    vmax = max(maximum(p_geo), maximum(p_hasa))

    fig = Figure(size=(1400, 600))
    ax1 = Axis(fig[1, 1]; title="Geometric", xlabel="Lateral position [mm]", ylabel="Axial position [mm]")
    ax2 = Axis(fig[1, 2]; title="HASA", xlabel="Lateral position [mm]", ylabel="Axial position [mm]")

    hm1 = heatmap!(ax1, kgrid.y_vec .* 1e3, kgrid.x_vec .* 1e3, p_geo'; colormap=:viridis, colorrange=(vmin, vmax))
    hm2 = heatmap!(ax2, kgrid.y_vec .* 1e3, kgrid.x_vec .* 1e3, p_hasa'; colormap=:viridis, colorrange=(vmin, vmax))
    Colorbar(fig[1, 3], hm2; label="Pressure")

    if geo_stats !== nothing
        px, py = geo_stats[:peak_mm]
        scatter!(ax1, [py], [px]; color=:red, markersize=12)
    end
    if hasa_stats !== nothing
        px, py = hasa_stats[:peak_mm]
        scatter!(ax2, [py], [px]; color=:red, markersize=12)
    end

    save(path, fig)
end

opts = parse_cli(ARGS)
out_dir = opts["out-dir"]
mkpath(out_dir)

hu_vol, spacing_m = load_default_ct(ct_path=opts["ct-path"])
cfg_kwargs = (
    fc=parse(Float64, opts["frequency-mhz"]) * 1e6,
    z_focus=parse(Float64, opts["focal-cm"]) * 1e-2,
    x_focus=parse(Float64, opts["lateral-cm"]) * 1e-2,
    trans_aperture=parse(Float64, opts["aperture-cm"]) * 1e-2,
)

stats_geo, p_geo, _, kgrid, _, _ = run_focus_case(
    hu_vol,
    SimulationConfig(; cfg_kwargs...),
    parse_medium(opts["medium"]),
    GEOMETRIC,
    SweepSettings();
    slice_index=parse(Int, opts["slice-index"]),
)
stats_hasa, p_hasa, _, _, _, _ = run_focus_case(
    hu_vol,
    SimulationConfig(; cfg_kwargs...),
    parse_medium(opts["medium"]),
    HASA,
    SweepSettings();
    slice_index=parse(Int, opts["slice-index"]),
)

pressure_figure(joinpath(out_dir, "comparison.png"), p_geo, p_hasa, kgrid, stats_geo, stats_hasa)

summary = Dict(
    "ct_path" => opts["ct-path"],
    "slice_index" => parse(Int, opts["slice-index"]),
    "spacing_m" => spacing_m,
    "geometric" => stats_geo,
    "hasa" => stats_hasa,
)
open(joinpath(out_dir, "summary.json"), "w") do io
    JSON3.pretty(io, summary)
end

println("Saved comparison outputs to $out_dir")
