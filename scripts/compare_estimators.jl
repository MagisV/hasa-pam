#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Dates
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
        "placement" => "auto",
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
parse_placement(s) = begin
    norm = replace(lowercase(s), "-" => "_")
    norm in ("auto", "fixed_transducer", "fixed_focus_depth") || error("Unknown placement mode: $s")
    Symbol(norm)
end

function resolve_placement(opts, medium_type)
    placement = parse_placement(opts["placement"])
    requested_focus_depth = haskey(opts, "focus-depth-from-inner-skull-mm") ?
        parse(Float64, opts["focus-depth-from-inner-skull-mm"]) * 1e-3 :
        nothing

    if placement == :fixed_transducer
        isnothing(requested_focus_depth) || error("`--focus-depth-from-inner-skull-mm` is incompatible with `--placement=fixed_transducer`.")
        return placement, nothing
    end

    if placement == :fixed_focus_depth
        if !isnothing(requested_focus_depth)
            return placement, requested_focus_depth
        elseif medium_type == SKULL_IN_WATER
            return placement, 30e-3
        else
            error("`--placement=fixed_focus_depth` requires `--focus-depth-from-inner-skull-mm` for this medium.")
        end
    end

    if !isnothing(requested_focus_depth)
        return :fixed_focus_depth, requested_focus_depth
    elseif medium_type == SKULL_IN_WATER
        return :fixed_focus_depth, 30e-3
    else
        return :fixed_transducer, nothing
    end
end

slug_value(x; digits::Int=1) = replace(string(round(Float64(x); digits=digits)), "-" => "m", "." => "p")

function default_output_dir(opts, placement_mode, focus_depth)
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    parts = String[
        "compare_estimators",
        lowercase(opts["medium"]),
        String(placement_mode),
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

function pressure_figure(path, p_geo, p_hasa, c_geo, c_hasa, kgrid, cfg_geo, cfg_hasa, geo_stats, hasa_stats)
    vmin = min(minimum(p_geo), minimum(p_hasa))
    vmax = max(maximum(p_geo), maximum(p_hasa))

    fig = Figure(size=(1400, 600))
    ax1 = Axis(fig[1, 1]; title="Geometric", xlabel="Lateral position [mm]", ylabel="Axial position [mm]", aspect=DataAspect())
    ax2 = Axis(fig[1, 2]; title="HASA", xlabel="Lateral position [mm]", ylabel="Axial position [mm]", aspect=DataAspect())

    hm1 = heatmap!(ax1, kgrid.y_vec .* 1e3, kgrid.x_vec .* 1e3, p_geo'; colormap=:viridis, colorrange=(vmin, vmax))
    hm2 = heatmap!(ax2, kgrid.y_vec .* 1e3, kgrid.x_vec .* 1e3, p_hasa'; colormap=:viridis, colorrange=(vmin, vmax))
    overlay_skull!(ax1, c_geo, kgrid)
    overlay_skull!(ax2, c_hasa, kgrid)
    Colorbar(fig[1, 3], hm2; label="Pressure")

    geo_x_tgt_mm, geo_y_tgt_mm = target_mm(kgrid, cfg_geo)
    hasa_x_tgt_mm, hasa_y_tgt_mm = target_mm(kgrid, cfg_hasa)
    scatter!(ax1, [geo_y_tgt_mm], [geo_x_tgt_mm]; color=:red, marker=:x, markersize=16, strokewidth=2)
    scatter!(ax2, [hasa_y_tgt_mm], [hasa_x_tgt_mm]; color=:red, marker=:x, markersize=16, strokewidth=2)

    if geo_stats !== nothing
        px, py = geo_stats[:peak_mm]
        scatter!(ax1, [py], [px]; color=:white, marker=:circle, markersize=12, strokecolor=:white, strokewidth=2)
    end
    if hasa_stats !== nothing
        px, py = hasa_stats[:peak_mm]
        scatter!(ax2, [py], [px]; color=:white, marker=:circle, markersize=12, strokecolor=:white, strokewidth=2)
    end

    save(path, fig)
end

opts = parse_cli(ARGS)
medium_type = parse_medium(opts["medium"])
placement_mode, focus_depth = resolve_placement(opts, medium_type)
out_dir = if haskey(opts, "out-dir") && !isempty(strip(opts["out-dir"]))
    opts["out-dir"]
else
    default_output_dir(opts, placement_mode, focus_depth)
end
mkpath(out_dir)

hu_vol, spacing_m = load_default_ct(ct_path=opts["ct-path"])
cfg_kwargs = (
    fc=parse(Float64, opts["frequency-mhz"]) * 1e6,
    z_focus=parse(Float64, opts["focal-cm"]) * 1e-2,
    x_focus=parse(Float64, opts["lateral-cm"]) * 1e-2,
    trans_aperture=parse(Float64, opts["aperture-cm"]) * 1e-2,
    focus_depth_from_inner_skull=focus_depth,
)

stats_geo, p_geo, c_geo, kgrid, cfg_geo, _ = run_focus_case(
    hu_vol,
    SimulationConfig(; cfg_kwargs...),
    medium_type,
    GEOMETRIC,
    SweepSettings();
    slice_index=parse(Int, opts["slice-index"]),
    return_c=true,
)
stats_hasa, p_hasa, c_hasa, _, cfg_hasa, _ = run_focus_case(
    hu_vol,
    SimulationConfig(; cfg_kwargs...),
    medium_type,
    HASA,
    SweepSettings();
    slice_index=parse(Int, opts["slice-index"]),
    return_c=true,
)

pressure_figure(joinpath(out_dir, "comparison.png"), p_geo, p_hasa, c_geo, c_hasa, kgrid, cfg_geo, cfg_hasa, stats_geo, stats_hasa)

summary = Dict(
    "out_dir" => out_dir,
    "ct_path" => opts["ct-path"],
    "slice_index" => parse(Int, opts["slice-index"]),
    "spacing_m" => spacing_m,
    "placement_mode" => String(placement_mode),
    "geometric" => stats_geo,
    "hasa" => stats_hasa,
)
open(joinpath(out_dir, "summary.json"), "w") do io
    JSON3.pretty(io, summary)
end

println("Saved comparison outputs to $out_dir")
