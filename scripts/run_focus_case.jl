#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using JLD2
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
        "estimator" => "hasa",
        "medium" => "skull_in_water",
        "out-dir" => joinpath(pwd(), "outputs", "run_focus_case"),
    )

    for arg in args
        startswith(arg, "--") || error("Unsupported argument format: $arg")
        parts = split(arg[3:end], "="; limit=2)
        length(parts) == 2 || error("Arguments must use --name=value, got: $arg")
        opts[parts[1]] = parts[2]
    end
    return opts
end

parse_est(s) = lowercase(s) == "geometric" ? GEOMETRIC : lowercase(s) == "hasa" ? HASA : error("Unknown estimator: $s")
parse_medium(s) = lowercase(s) == "water" ? WATER : lowercase(s) == "skull_in_water" ? SKULL_IN_WATER : error("Unknown medium: $s")

function save_pressure_plot(path::AbstractString, pressure, kgrid, stats, title::AbstractString)
    fig = Figure(size=(900, 600))
    ax = Axis(fig[1, 1]; title=title, xlabel="Lateral position [mm]", ylabel="Axial position [mm]")
    hm = heatmap!(ax, kgrid.y_vec .* 1e3, kgrid.x_vec .* 1e3, pressure'; colormap=:viridis)
    Colorbar(fig[1, 2], hm; label="Pressure")
    if stats !== nothing
        peak_x, peak_y = stats[:peak_mm]
        scatter!(ax, [peak_y], [peak_x]; color=:red, markersize=12)
    end
    save(path, fig)
end

opts = parse_cli(ARGS)
out_dir = opts["out-dir"]
mkpath(out_dir)

hu_vol, spacing_m = load_default_ct(ct_path=opts["ct-path"])
ct_info = CTInfo(hu_vol, spacing_m)
@info "Loaded CT volume" size=size(hu_vol) spacing_m spacing_info=ct_info

focus_depth = haskey(opts, "focus-depth-from-inner-skull-mm") ? parse(Float64, opts["focus-depth-from-inner-skull-mm"]) * 1e-3 : nothing
cfg = SimulationConfig(
    fc=parse(Float64, opts["frequency-mhz"]) * 1e6,
    z_focus=parse(Float64, opts["focal-cm"]) * 1e-2,
    x_focus=parse(Float64, opts["lateral-cm"]) * 1e-2,
    trans_aperture=parse(Float64, opts["aperture-cm"]) * 1e-2,
    focus_depth_from_inner_skull=focus_depth,
)

stats, pressure, c_map, kgrid, cfg, hasa_info = run_focus_case(
    hu_vol,
    cfg,
    parse_medium(opts["medium"]),
    parse_est(opts["estimator"]),
    SweepSettings();
    slice_index=parse(Int, opts["slice-index"]),
    return_c=true,
)

summary = Dict(
    "ct_path" => opts["ct-path"],
    "slice_index" => parse(Int, opts["slice-index"]),
    "spacing_m" => spacing_m,
    "stats" => stats,
    "config" => Dict(
        "fc" => cfg.fc,
        "x_focus" => cfg.x_focus,
        "z_focus" => cfg.z_focus,
        "trans_aperture" => cfg.trans_aperture,
        "focus_depth_from_inner_skull" => cfg.focus_depth_from_inner_skull,
    ),
)

open(joinpath(out_dir, "summary.json"), "w") do io
    JSON3.pretty(io, summary)
end

hasa_save = copy(hasa_info)
hasa_save[:fig] = nothing
@save joinpath(out_dir, "result.jld2") pressure c_map kgrid cfg stats hasa_save

if pressure !== nothing && ndims(pressure) == 2
    save_pressure_plot(
        joinpath(out_dir, "pressure.png"),
        pressure,
        kgrid,
        stats,
        "Focus Run: $(opts["estimator"]) in $(opts["medium"])",
    )
end

println("Saved outputs to $out_dir")
