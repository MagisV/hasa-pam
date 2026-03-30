#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Dates
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

function default_focus_depth_m(opts, medium_type)
    if haskey(opts, "focus-depth-from-inner-skull-mm")
        return parse(Float64, opts["focus-depth-from-inner-skull-mm"]) * 1e-3
    end
    return medium_type == SKULL_IN_WATER ? 30e-3 : nothing
end

slug_value(x; digits::Int=1) = replace(string(round(Float64(x); digits=digits)), "-" => "m", "." => "p")

function default_output_dir(opts, focus_depth)
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    parts = String[
        "run_focus",
        lowercase(opts["estimator"]),
        lowercase(opts["medium"]),
        "f$(slug_value(parse(Float64, opts["frequency-mhz"]); digits=2))mhz",
        "z$(slug_value(parse(Float64, opts["focal-cm"]); digits=1))cm",
        "x$(slug_value(parse(Float64, opts["lateral-cm"]); digits=1))cm",
        "ap$(slug_value(parse(Float64, opts["aperture-cm"]); digits=1))cm",
        "slice$(parse(Int, opts["slice-index"]))",
    ]
    if !isnothing(focus_depth)
        push!(parts, "fd$(slug_value(focus_depth * 1e3; digits=1))mm")
    end
    push!(parts, timestamp)
    return joinpath(pwd(), "outputs", join(parts, "_"))
end

function target_mm(kgrid, cfg)
    x_tgt_mm = (kgrid.x_vec[cfg.trans_index] - cfg.z_focus) * 1e3
    y_tgt_mm = (kgrid.y_vec[fld(kgrid.Ny, 2) + 1] + cfg.x_focus) * 1e3
    return x_tgt_mm, y_tgt_mm
end

function overlay_skull!(ax, c_map, kgrid)
    c_map === nothing && return nothing
    skull_mask = skull_mask_from_c_columnwise(c_map; mask_outside=false)
    any(skull_mask) || return nothing

    overlay = fill(Float64(NaN), size(c_map))
    overlay[skull_mask] .= Float64.(c_map[skull_mask]) ./ maximum(Float64.(c_map[skull_mask]))
    return heatmap!(
        ax,
        kgrid.y_vec .* 1e3,
        kgrid.x_vec .* 1e3,
        overlay';
        colormap=:grays,
        alpha=0.35,
        colorrange=(0, 1),
        nan_color=CairoMakie.RGBAf(0, 0, 0, 0),
    )
end

function save_pressure_plot(path::AbstractString, pressure, c_map, kgrid, cfg, stats, title::AbstractString)
    fig = Figure(size=(900, 600))
    ax = Axis(
        fig[1, 1];
        title=title,
        xlabel="Lateral position [mm]",
        ylabel="Axial position [mm]",
        aspect=DataAspect(),
    )
    hm = heatmap!(ax, kgrid.y_vec .* 1e3, kgrid.x_vec .* 1e3, pressure'; colormap=:viridis)
    overlay_skull!(ax, c_map, kgrid)
    Colorbar(fig[1, 2], hm; label="Pressure")

    x_tgt_mm, y_tgt_mm = target_mm(kgrid, cfg)
    scatter!(ax, [y_tgt_mm], [x_tgt_mm]; color=:red, marker=:x, markersize=16, strokewidth=2)
    if stats !== nothing
        peak_x, peak_y = stats[:peak_mm]
        scatter!(ax, [peak_y], [peak_x]; color=:white, marker=:circle, markersize=12, strokecolor=:white, strokewidth=2)
    end
    save(path, fig)
end

opts = parse_cli(ARGS)
medium_type = parse_medium(opts["medium"])
focus_depth = default_focus_depth_m(opts, medium_type)
out_dir = if haskey(opts, "out-dir") && !isempty(strip(opts["out-dir"]))
    opts["out-dir"]
else
    default_output_dir(opts, focus_depth)
end
mkpath(out_dir)

hu_vol, spacing_m = load_default_ct(ct_path=opts["ct-path"])
ct_info = CTInfo(hu_vol, spacing_m)
@info "Loaded CT volume" size=size(hu_vol) spacing_m spacing_info=ct_info

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
    medium_type,
    parse_est(opts["estimator"]),
    SweepSettings();
    slice_index=parse(Int, opts["slice-index"]),
    return_c=true,
)

summary = Dict(
    "out_dir" => out_dir,
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
        c_map,
        kgrid,
        cfg,
        stats,
        "Focus Run: $(opts["estimator"]) in $(opts["medium"])",
    )
end

println("Saved outputs to $out_dir")
