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
        "frequency-mhz" => "1.0",
        "amplitude-pa" => "5e4",
        "num-cycles" => "4",
        "axial-mm" => "90",
        "transverse-mm" => "60",
        "dx-mm" => "0.2",
        "dz-mm" => "0.2",
        "receiver-aperture-mm" => "50",
        "t-max-us" => "80",
        "dt-ns" => "40",
        "zero-pad-factor" => "4",
        "peak-suppression-radius-mm" => "2.0",
        "success-tolerance-mm" => "1.0",
        "sweep-preset" => "paper",
        "aberrator" => "skull",
        "ct-path" => DEFAULT_CT_PATH,
        "slice-index" => "250",
        "skull-transducer-distance-mm" => "30",
        "skull-cavity-margin-mm" => "1.0",
        "bottom-margin-mm" => "10",
        "hu-bone-thr" => "200",
        "lens-depth-mm" => "12",
        "lens-lateral-mm" => "0",
        "lens-axial-radius-mm" => "3",
        "lens-lateral-radius-mm" => "12",
        "aberrator-c" => "1700",
        "aberrator-rho" => "1150",
        "kwave-use-gpu" => "true",
        "recon-use-gpu" => "true",
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

function parse_target_pairs_mm(spec::AbstractString)
    isempty(strip(spec)) && return Tuple{Float64, Float64}[]
    pairs_mm = Tuple{Float64, Float64}[]
    for token in split(spec, ",")
        stripped = strip(token)
        isempty(stripped) && continue
        parts = split(stripped, ":"; limit=2)
        length(parts) == 2 || error("Expected target specification axial_mm:lateral_mm, got: $token")
        push!(pairs_mm, (parse(Float64, strip(parts[1])), parse(Float64, strip(parts[2]))))
    end
    return pairs_mm
end

function build_sweep_targets(axial_targets_mm, lateral_targets_mm, opts)
    frequency_hz = parse(Float64, opts["frequency-mhz"]) * 1e6
    amplitude_pa = parse(Float64, opts["amplitude-pa"])
    num_cycles = parse(Int, opts["num-cycles"])

    targets = PointSource2D[]
    for axial_mm in axial_targets_mm, lateral_mm in lateral_targets_mm
        push!(targets, PointSource2D(
            depth=axial_mm * 1e-3,
            lateral=lateral_mm * 1e-3,
            frequency=frequency_hz,
            amplitude=amplitude_pa,
            num_cycles=num_cycles,
        ))
    end
    return targets
end

function default_output_dir(opts, sweep_preset, cfg)
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    parts = String[
        timestamp,
        "run_pam_sweep",
        lowercase(opts["aberrator"]),
        String(sweep_preset),
        "f$(slug_value(parse(Float64, opts["frequency-mhz"]); digits=2))mhz",
        "ap$(isnothing(cfg.receiver_aperture) ? "full" : slug_value(cfg.receiver_aperture * 1e3; digits=1) * "mm")",
        "dx$(slug_value(cfg.dx * 1e3; digits=2))mm",
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

function bubble_sizes(values::AbstractVector{<:Real}; min_size::Real=10, max_size::Real=34)
    finite_values = Float64[v for v in values if isfinite(v)]
    isempty(finite_values) && return fill(Float64(min_size), length(values))

    lo = minimum(finite_values)
    hi = maximum(finite_values)
    if isapprox(lo, hi; atol=1e-9)
        return fill(Float64((min_size + max_size) / 2), length(values))
    end

    scale = (Float64(max_size) - Float64(min_size)) / (hi - lo)
    return [isfinite(v) ? Float64(min_size) + (Float64(v) - lo) * scale : Float64(min_size) for v in values]
end

function matrix_points(axial_targets_mm, lateral_targets_mm, error_mm::AbstractMatrix)
    xs = Float64[]
    ys = Float64[]
    vals = Float64[]
    for (row, axial_mm) in pairs(axial_targets_mm), (col, lateral_mm) in pairs(lateral_targets_mm)
        value = Float64(error_mm[row, col])
        isfinite(value) || continue
        push!(xs, lateral_mm)
        push!(ys, axial_mm)
        push!(vals, value)
    end
    return xs, ys, vals
end

function maybe_overlay_medium!(ax, c, depth_mm, lateral_mm, cfg)
    if maximum(Float64.(c)) > cfg.c0 + 10
        level = cfg.c0 + 0.5 * (maximum(Float64.(c)) - cfg.c0)
        contour!(
            ax,
            lateral_mm,
            depth_mm,
            Float64.(c)';
            levels=[level],
            color=:white,
            linewidth=1.5,
        )
    end
end

function scatter_truth!(ax, truth_mm)
    scatter!(ax, [truth_mm[2]], [truth_mm[1]]; color=:red, marker=:x, markersize=14, strokewidth=2)
end

function scatter_prediction!(ax, predicted_mm)
    scatter!(ax, [predicted_mm[2]], [predicted_mm[1]]; color=:white, marker=:circle, markersize=12, strokecolor=:black, strokewidth=1.0)
end

function case_filename(truth_mm)
    return "case_z$(slug_value(truth_mm[1]; digits=1))mm_x$(slug_value(truth_mm[2]; digits=1))mm.png"
end

function save_case_overview(path, c, cfg, case)
    truth = case[:truth_mm]
    geo_pred = case[:geo_predicted_mm]
    hasa_pred = case[:hasa_predicted_mm]
    kgrid = case[:kgrid]
    depth_mm = depth_coordinates(kgrid, cfg) .* 1e3
    lateral_mm = kgrid.y_vec .* 1e3
    ref = max(maximum(Float64.(case[:pam_geo])), maximum(Float64.(case[:pam_hasa])), eps(Float64))

    fig = Figure(size=(1000, 500))
    geo_title = "Uncorrected | err=$(round(case[:stats_geo][:mean_radial_error_mm]; digits=2)) mm"
    hasa_title = "Corrected | err=$(round(case[:stats_hasa][:mean_radial_error_mm]; digits=2)) mm"

    ax_geo = Axis(
        fig[1, 1];
        title=geo_title,
        xlabel="Lateral [mm]",
        ylabel="Depth [mm]",
        aspect=DataAspect(),
        yreversed=true,
    )
    heatmap!(ax_geo, lateral_mm, depth_mm, map_db(case[:pam_geo], ref)'; colormap=:viridis, colorrange=(-30, 0))
    maybe_overlay_medium!(ax_geo, c, depth_mm, lateral_mm, cfg)
    scatter_truth!(ax_geo, truth)
    scatter_prediction!(ax_geo, geo_pred)

    ax_hasa = Axis(
        fig[1, 2];
        title=hasa_title,
        xlabel="Lateral [mm]",
        ylabel="Depth [mm]",
        aspect=DataAspect(),
        yreversed=true,
    )
    heatmap!(ax_hasa, lateral_mm, depth_mm, map_db(case[:pam_hasa], ref)'; colormap=:viridis, colorrange=(-30, 0))
    maybe_overlay_medium!(ax_hasa, c, depth_mm, lateral_mm, cfg)
    scatter_truth!(ax_hasa, truth)
    scatter_prediction!(ax_hasa, hasa_pred)

    Label(
        fig[0, 1:2],
        "z=$(round(truth[1]; digits=1)) mm, x=$(round(truth[2]; digits=1)) mm";
        fontsize=18,
        tellwidth=false,
    )

    save(path, fig)
end

function save_sweep_overview(path, c, cfg, sweep_results)
    example_cases = sweep_results[:example_cases]
    n_examples = length(example_cases)
    n_examples > 0 || error("No example cases available for PAM sweep overview plotting.")

    ref = eps(Float64)
    for case in example_cases
        ref = max(ref, maximum(Float64.(case[:pam_geo])), maximum(Float64.(case[:pam_hasa])))
    end

    fig = Figure(size=(1800, 1100))

    for (row, case) in pairs(example_cases)
        truth = case[:truth_mm]
        geo_pred = case[:geo_predicted_mm]
        hasa_pred = case[:hasa_predicted_mm]
        kgrid = case[:kgrid]
        depth_mm = depth_coordinates(kgrid, cfg) .* 1e3
        lateral_mm = kgrid.y_vec .* 1e3

        geo_title = "Uncorrected | z=$(round(truth[1]; digits=0)) mm, x=$(round(truth[2]; digits=0)) mm"
        hasa_title = "Corrected | z=$(round(truth[1]; digits=0)) mm, x=$(round(truth[2]; digits=0)) mm"

        ax_geo = Axis(
            fig[row, 1];
            title=geo_title,
            xlabel=row == n_examples ? "Lateral [mm]" : "",
            ylabel="Depth [mm]",
            aspect=DataAspect(),
            yreversed=true,
        )
        heatmap!(ax_geo, lateral_mm, depth_mm, map_db(case[:pam_geo], ref)'; colormap=:viridis, colorrange=(-30, 0))
        maybe_overlay_medium!(ax_geo, c, depth_mm, lateral_mm, cfg)
        scatter_truth!(ax_geo, truth)
        scatter_prediction!(ax_geo, geo_pred)

        ax_hasa = Axis(
            fig[row, 2];
            title=hasa_title,
            xlabel=row == n_examples ? "Lateral [mm]" : "",
            ylabel=row == 1 ? "" : "Depth [mm]",
            aspect=DataAspect(),
            yreversed=true,
        )
        heatmap!(ax_hasa, lateral_mm, depth_mm, map_db(case[:pam_hasa], ref)'; colormap=:viridis, colorrange=(-30, 0))
        maybe_overlay_medium!(ax_hasa, c, depth_mm, lateral_mm, cfg)
        scatter_truth!(ax_hasa, truth)
        scatter_prediction!(ax_hasa, hasa_pred)
    end

    xs_geo, ys_geo, vals_geo = matrix_points(
        sweep_results[:axial_targets_mm],
        sweep_results[:lateral_targets_mm],
        sweep_results[:geo_error_mm],
    )
    xs_hasa, ys_hasa, vals_hasa = matrix_points(
        sweep_results[:axial_targets_mm],
        sweep_results[:lateral_targets_mm],
        sweep_results[:hasa_error_mm],
    )
    finite_vals = vcat(vals_geo, vals_hasa)
    color_max = isempty(finite_vals) ? 1.0 : max(maximum(finite_vals), eps(Float64))

    ax_geo_err = Axis(
        fig[1:n_examples, 3];
        title="Uncorrected",
        xlabel="Lateral target [mm]",
        ylabel="Axial target [mm]",
        yreversed=true,
        xticks=sweep_results[:lateral_targets_mm],
        yticks=sweep_results[:axial_targets_mm],
    )
    sc_geo = scatter!(
        ax_geo_err,
        xs_geo,
        ys_geo;
        color=vals_geo,
        colormap=:thermal,
        colorrange=(0, color_max),
        markersize=bubble_sizes(vals_geo),
        strokecolor=:black,
        strokewidth=0.75,
    )

    ax_hasa_err = Axis(
        fig[1:n_examples, 4];
        title="Corrected",
        xlabel="Lateral target [mm]",
        ylabel="Axial target [mm]",
        yreversed=true,
        xticks=sweep_results[:lateral_targets_mm],
        yticks=sweep_results[:axial_targets_mm],
    )
    sc_hasa = scatter!(
        ax_hasa_err,
        xs_hasa,
        ys_hasa;
        color=vals_hasa,
        colormap=:thermal,
        colorrange=(0, color_max),
        markersize=bubble_sizes(vals_hasa),
        strokecolor=:black,
        strokewidth=0.75,
    )
    Colorbar(fig[1:n_examples, 5], sc_hasa; label="Error [mm]")

    save(path, fig)
end

function case_summary(case)
    geo_pred = case[:geo_predicted_mm]
    hasa_pred = case[:hasa_predicted_mm]
    return Dict(
        "truth_mm" => [case[:truth_mm][1], case[:truth_mm][2]],
        "geometric" => Dict(
            "predicted_mm" => [geo_pred[1], geo_pred[2]],
            "stats" => case[:stats_geo],
        ),
        "hasa" => Dict(
            "predicted_mm" => [hasa_pred[1], hasa_pred[2]],
            "stats" => case[:stats_hasa],
        ),
    )
end

opts = parse_cli(ARGS)
aberrator = parse_aberrator(opts["aberrator"])

custom_axial_targets = haskey(opts, "axial-targets-mm") ? parse_float_list(opts["axial-targets-mm"]) : nothing
custom_lateral_targets = haskey(opts, "lateral-targets-mm") ? parse_float_list(opts["lateral-targets-mm"]) : nothing
sweep_preset, axial_targets_mm, lateral_targets_mm = TranscranialFUS._resolve_pam_sweep_targets(
    opts["sweep-preset"];
    axial_targets_mm=custom_axial_targets,
    lateral_targets_mm=custom_lateral_targets,
)

targets = build_sweep_targets(axial_targets_mm, lateral_targets_mm, opts)
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
cfg = fit_pam_config(
    cfg,
    targets;
    min_bottom_margin=parse(Float64, opts["bottom-margin-mm"]) * 1e-3,
    reference_depth=aberrator == :skull ? parse(Float64, opts["skull-transducer-distance-mm"]) * 1e-3 : nothing,
)

out_dir = if haskey(opts, "out-dir") && !isempty(strip(opts["out-dir"]))
    opts["out-dir"]
else
    default_output_dir(opts, sweep_preset, cfg)
end
mkpath(out_dir)

example_targets_mm = haskey(opts, "example-targets-mm") ? parse_target_pairs_mm(opts["example-targets-mm"]) : nothing

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

requested_targets = targets
dropped_targets = Dict{Symbol, Any}[]
cavity_start_rows = nothing
if aberrator == :skull
    targets, dropped_targets, cavity_start_rows = TranscranialFUS._filter_pam_targets_in_skull_cavity(
        c,
        cfg,
        targets;
        min_margin=parse(Float64, opts["skull-cavity-margin-mm"]) * 1e-3,
    )
    isempty(targets) && error("No PAM sweep targets remain after enforcing skull-cavity placement.")
end

case_dir = joinpath(out_dir, "cases")
mkpath(case_dir)
case_file_records = Dict{String, Any}[]

function case_callback(case, results)
    plot_case = copy(case)
    plot_case[:pam_geo] = results[:pam_geo]
    plot_case[:pam_hasa] = results[:pam_hasa]
    plot_case[:kgrid] = results[:kgrid]
    filename = case_filename(case[:truth_mm])
    relpath = joinpath("cases", filename)
    save_case_overview(joinpath(out_dir, relpath), c, cfg, plot_case)
    push!(case_file_records, Dict(
        "truth_mm" => [case[:truth_mm][1], case[:truth_mm][2]],
        "file" => relpath,
    ))
end

sweep_results = run_pam_sweep(
    c,
    rho,
    targets,
    cfg;
    frequencies=[parse(Float64, opts["frequency-mhz"]) * 1e6],
    example_targets_mm=example_targets_mm,
    use_gpu=parse_bool(opts["recon-use-gpu"]),
    kwave_use_gpu=parse_bool(opts["kwave-use-gpu"]),
    case_callback=case_callback,
)

save_sweep_overview(joinpath(out_dir, "overview.png"), c, cfg, sweep_results)

medium_summary = Dict{String, Any}()
for (key, value) in medium_info
    key == :mask && continue
    medium_summary[String(key)] = value
end

summary = Dict(
    "out_dir" => out_dir,
    "sweep_preset" => String(sweep_preset),
    "requested_axial_targets_mm" => axial_targets_mm,
    "requested_lateral_targets_mm" => lateral_targets_mm,
    "axial_targets_mm" => sweep_results[:axial_targets_mm],
    "lateral_targets_mm" => sweep_results[:lateral_targets_mm],
    "example_targets_mm" => [[target[1], target[2]] for target in sweep_results[:example_targets_mm]],
    "requested_target_count" => length(requested_targets),
    "retained_target_count" => length(targets),
    "dropped_targets" => [
        Dict(
            "truth_mm" => [drop[:truth_mm][1], drop[:truth_mm][2]],
            "row" => drop[:row],
            "col" => drop[:col],
            "reason" => String(drop[:reason]),
            "required_row" => get(drop, :required_row, nothing),
        ) for drop in dropped_targets
    ],
    "per_case_files" => case_file_records,
    "source_frequency_hz" => parse(Float64, opts["frequency-mhz"]) * 1e6,
    "source_amplitude_pa" => parse(Float64, opts["amplitude-pa"]),
    "source_num_cycles" => parse(Int, opts["num-cycles"]),
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
        "skull_cavity_margin" => parse(Float64, opts["skull-cavity-margin-mm"]) * 1e-3,
        "bottom_margin" => parse(Float64, opts["bottom-margin-mm"]) * 1e-3,
    ),
    "medium" => medium_summary,
    "cases" => [case_summary(case) for case in sweep_results[:cases]],
    "geometric_error_mm" => sweep_results[:geo_error_mm],
    "hasa_error_mm" => sweep_results[:hasa_error_mm],
    "geometric_peak_intensity" => sweep_results[:geo_peak_intensity],
    "hasa_peak_intensity" => sweep_results[:hasa_peak_intensity],
)

if !isnothing(cavity_start_rows)
    summary["medium"]["cavity_start_rows"] = cavity_start_rows
end

open(joinpath(out_dir, "summary.json"), "w") do io
    JSON3.pretty(io, summary)
end

@save joinpath(out_dir, "result.jld2") c rho cfg targets sweep_results medium_info sweep_preset axial_targets_mm lateral_targets_mm dropped_targets

println("Saved PAM sweep outputs to $out_dir")
