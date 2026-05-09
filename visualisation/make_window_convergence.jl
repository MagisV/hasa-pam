#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using CUDA
using Dates
using JLD2
using JSON3
using Printf
using Random
using Statistics
using TranscranialFUS

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

parse_bool(s) = lowercase(strip(String(s))) in ("1", "true", "yes", "on")

function parse_cli(args)
    opts = Dict{String, String}(
        "out-dir" => "",
        "random-seed" => "42",
        "aberrator" => "skull",
        "t-max-us" => "300",
        "dy-mm" => "0.2",
        "dz-mm" => "0.2",
        "max-windows" => "0",
        "fps" => "12",
        "frames-only" => "false",
        "dry-run" => "false",
        "recon-progress" => "false",
        "kwave-use-gpu" => "true",
        "recon-use-gpu" => "true",
        "zero-pad-factor" => "2",
        "from-data" => "",
    )
    for arg in args
        startswith(arg, "--") || error("Unsupported argument format: $arg")
        parts = split(arg[3:end], "="; limit=2)
        length(parts) == 2 || error("Arguments must use --name=value, got: $arg")
        opts[parts[1]] = parts[2]
    end
    return opts
end

function timestamp()
    return Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
end

function parse_aberrator(value::AbstractString)
    sym = Symbol(lowercase(strip(value)))
    sym in (:none, :water, :skull) || error("--aberrator must be none, water, or skull.")
    return sym == :water ? :none : sym
end

function source_summary_3d(src)
    if src isa BubbleCluster3D
        return Dict(
            "type" => "BubbleCluster3D",
            "depth_m" => src.depth,
            "lateral_y_m" => src.lateral_y,
            "lateral_z_m" => src.lateral_z,
            "fundamental_hz" => src.fundamental,
            "harmonics" => src.harmonics,
            "amplitude" => src.amplitude,
        )
    elseif src isa PointSource3D
        return Dict(
            "type" => "PointSource3D",
            "depth_m" => src.depth,
            "lateral_y_m" => src.lateral_y,
            "lateral_z_m" => src.lateral_z,
            "frequency_hz" => src.frequency,
            "amplitude" => src.amplitude,
        )
    end
    return Dict("type" => string(typeof(src)))
end

function make_demo_config(opts)
    dy = parse(Float64, opts["dy-mm"]) * 1e-3
    dz = parse(Float64, opts["dz-mm"]) * 1e-3
    return PAMConfig3D(
        dx = 0.2e-3,
        dy = dy,
        dz = dz,
        axial_dim = 70e-3,
        transverse_dim_y = 64e-3,
        transverse_dim_z = 64e-3,
        dt = 40e-9,
        t_max = parse(Float64, opts["t-max-us"]) * 1e-6,
        receiver_aperture_y = nothing,
        receiver_aperture_z = nothing,
        zero_pad_factor = parse(Int, opts["zero-pad-factor"]),
        peak_suppression_radius = 2e-3,
        success_tolerance = 1e-3,
        axial_gain_power = 1.5,
    )
end

function make_demo_sources(cfg::PAMConfig3D, seed::Integer)
    rng = Random.MersenneTwister(seed)
    centers = [(45e-3, 0.0, 0.0)]
    sources, meta = make_network_bubble_sources_3d(
        centers;
        axial_radius = 10e-3,
        lateral_y_radius = 1.5e-3,
        lateral_z_radius = 1.5e-3,
        root_count = 12,
        generations = 3,
        branch_length = 5.0e-3,
        branch_step = 0.4e-3,
        branch_angle = 36 * pi / 180,
        tortuosity = 0.18,
        network_orientation = :isotropic,
        source_spacing = 0.5e-3,
        density_sigma_depth = 10.0e-3,
        density_sigma_y = 1.5e-3,
        density_sigma_z = 1.5e-3,
        min_separation = 0.25e-3,
        max_sources_per_center = 80,
        depth_bounds = (0.0, cfg.axial_dim),
        lateral_y_bounds = (-cfg.transverse_dim_y / 2, cfg.transverse_dim_y / 2),
        lateral_z_bounds = (-cfg.transverse_dim_z / 2, cfg.transverse_dim_z / 2),
        fundamental = 0.5e6,
        harmonics = [2, 3, 4],
        harmonic_amplitudes = [1.0, 0.6, 0.3],
        gate_duration = 50e-6,
        taper_ratio = 0.25,
        # The actual simulated events are expanded below with fresh random
        # phases per reconstruction window. Keep the template sources neutral
        # so the saved metadata does not imply geometric phase driving.
        phase_mode = :random,
        transducer_depth = -30e-3,
        rng = rng,
    )
    meta[:activity_model] = Dict(
        "activity_mode" => "random_phase_per_window",
        "template_phase_mode" => "random",
    )
    return sources, meta
end

function default_recon_frequencies(sources)
    freqs = Float64[]
    for src in sources
        append!(freqs, emission_frequencies(src))
    end
    return sort(unique(freqs))
end

function source_detected_flags_3d(pred, grid, cfg::PAMConfig3D, sources; radius::Real)
    radius_m = Float64(radius)
    x = collect(grid.x)
    y = collect(grid.y)
    z = collect(grid.z)
    r0 = receiver_row(cfg)
    row_r = ceil(Int, radius_m / cfg.dx)
    col_r_y = ceil(Int, radius_m / cfg.dy)
    col_r_z = ceil(Int, radius_m / cfg.dz)
    radius2 = radius_m^2

    flags = falses(length(sources))
    distances_mm = Float64[]
    for (idx, src) in pairs(sources)
        src_x = x[r0] + src.depth
        row0 = r0 + round(Int, src.depth / cfg.dx)
        col0_y = argmin(abs.(y .- src.lateral_y))
        col0_z = argmin(abs.(z .- src.lateral_z))
        best_d2 = Inf
        for row in max(r0 + 1, row0 - row_r):min(size(pred, 1), row0 + row_r)
            dx2 = (x[row] - src_x)^2
            for iy in max(1, col0_y - col_r_y):min(size(pred, 2), col0_y + col_r_y)
                dy2 = (y[iy] - src.lateral_y)^2
                for iz in max(1, col0_z - col_r_z):min(size(pred, 3), col0_z + col_r_z)
                    pred[row, iy, iz] || continue
                    d2 = dx2 + dy2 + (z[iz] - src.lateral_z)^2
                    if d2 <= radius2 && d2 < best_d2
                        best_d2 = d2
                    end
                end
            end
        end
        if isfinite(best_d2)
            flags[idx] = true
            push!(distances_mm, sqrt(best_d2) * 1e3)
        end
    end
    return flags, distances_mm
end

function fixed_threshold_stats_3d(intensity, cutoff::Real, truth_mask, grid, cfg, sources; truth_radius::Real)
    pred = intensity .>= Float64(cutoff)
    tp = count(pred .& truth_mask)
    fp = count(pred .& .!truth_mask)
    fn = count(.!pred .& truth_mask)
    precision = tp + fp == 0 ? 0.0 : tp / (tp + fp)
    voxel_recall = tp + fn == 0 ? 0.0 : tp / (tp + fn)
    voxel_f1 = precision + voxel_recall == 0 ? 0.0 : 2 * precision * voxel_recall / (precision + voxel_recall)
    flags, distances_mm = source_detected_flags_3d(pred, grid, cfg, sources; radius=truth_radius)
    detected = count(identity, flags)
    source_recall = isempty(sources) ? 0.0 : detected / length(sources)
    source_f1 = precision + source_recall == 0 ? 0.0 : 2 * precision * source_recall / (precision + source_recall)
    return Dict{Symbol, Any}(
        :source_f1 => source_f1,
        :voxel_f1 => voxel_f1,
        :precision => precision,
        :source_recall => source_recall,
        :voxel_recall => voxel_recall,
        :detected_source_count => detected,
        :num_truth_sources => length(sources),
        :predicted_voxels => count(pred),
        :true_positive_voxels => tp,
        :false_positive_voxels => fp,
        :false_negative_voxels => fn,
        :truth_voxels => count(truth_mask),
        :mean_detected_source_distance_mm => isempty(distances_mm) ? nothing : mean(distances_mm),
        :max_detected_source_distance_mm => isempty(distances_mm) ? nothing : maximum(distances_mm),
        :detected_flags => flags,
    )
end

function best_final_threshold(final_intensity, truth_mask, grid, cfg, sources; truth_radius::Real)
    local_ref = max(maximum(Float64.(final_intensity)), eps(Float64))
    thresholds = collect(0.10:0.01:0.95)
    entries = threshold_detection_stats_3d(
        final_intensity,
        grid,
        cfg,
        sources;
        threshold_ratios=thresholds,
        truth_radius=truth_radius,
        truth_mask=truth_mask,
    )
    best = best_threshold_entry_3d(entries)
    best[:absolute_cutoff] = Float64(best[:threshold_ratio]) * local_ref
    return best, entries
end

function qualify_windows(rf, cfg, window_config, max_windows::Integer)
    nt = size(rf, 3)
    ranges, window_samples, hop_samples = TranscranialFUS._pam_window_ranges(nt, cfg.dt, window_config)
    energies = [sum(abs2, @view rf[:, :, range]) for range in ranges]
    max_energy = isempty(energies) ? 0.0 : maximum(energies)
    threshold = max_energy * window_config.min_energy_ratio
    selected = UnitRange{Int}[]
    selected_energy = Float64[]
    skipped = UnitRange{Int}[]
    for (range, energy) in zip(ranges, energies)
        if energy < threshold || energy <= 0
            push!(skipped, range)
        else
            push!(selected, range)
            push!(selected_energy, Float64(energy))
        end
    end
    if max_windows > 0 && length(selected) > max_windows
        selected = selected[1:max_windows]
        selected_energy = selected_energy[1:max_windows]
    end
    return selected, selected_energy, skipped, window_samples, hop_samples, threshold
end

function reconstruct_one_window(rf, c, cfg, range, window_config, recon_frequencies, bandwidth_hz;
    use_gpu::Bool, show_progress::Bool)
    taper = TranscranialFUS._pam_temporal_taper(length(range), window_config.taper)
    rf_window = Float64.(@view rf[:, :, range]) .* reshape(taper, 1, 1, :)
    t0 = Float64(first(range) - 1) * cfg.dt
    intensity, grid, _ = reconstruct_pam_3d(
        rf_window,
        c,
        cfg;
        frequencies = recon_frequencies,
        bandwidth = bandwidth_hz,
        corrected = true,
        axial_step = 50e-6,
        time_origin = t0,
        use_gpu = use_gpu,
        show_progress = show_progress,
        window_batch = 1,
    )
    return intensity, grid
end

depth_mm(grid, cfg) = (collect(grid.x) .- grid.x[receiver_row(cfg)]) .* 1e3
y_mm(grid) = collect(grid.y) .* 1e3
z_mm(grid) = collect(grid.z) .* 1e3

function mask_projection(mask, projection::Symbol)
    if projection == :depth_y
        return dropdims(any(mask; dims=3); dims=3)
    elseif projection == :depth_z
        return dropdims(any(mask; dims=2); dims=2)
    elseif projection == :y_z
        return dropdims(any(mask; dims=1); dims=1)
    end
    error("Unknown projection: $projection")
end

function volume_projection(volume, projection::Symbol)
    if projection == :depth_y
        return dropdims(maximum(volume; dims=3); dims=3)
    elseif projection == :depth_z
        return dropdims(maximum(volume; dims=2); dims=2)
    elseif projection == :y_z
        return dropdims(maximum(volume; dims=1); dims=1)
    end
    error("Unknown projection: $projection")
end

function projection_axes(grid, cfg, projection::Symbol)
    if projection == :depth_y
        return depth_mm(grid, cfg), y_mm(grid), "Depth [mm]", "Y [mm]"
    elseif projection == :depth_z
        return depth_mm(grid, cfg), z_mm(grid), "Depth [mm]", "Z [mm]"
    elseif projection == :y_z
        return y_mm(grid), z_mm(grid), "Y [mm]", "Z [mm]"
    end
    error("Unknown projection: $projection")
end

function normalized_projection(volume, projection::Symbol)
    proj = Float64.(volume_projection(volume, projection))
    ref = max(maximum(proj), eps(Float64))
    return log10.(1 .+ 99 .* (proj ./ ref)) ./ 2
end

function add_contour_if_valid!(ax, xs, ys, mask; color, linewidth=2.0, linestyle=:solid)
    any(mask) || return
    any(.!mask) || return
    contour!(ax, xs, ys, Float64.(mask); levels=[0.5], color=color, linewidth=linewidth, linestyle=linestyle)
end

function threshold_voxel_points(mask, intensity, grid, cfg; max_points::Int=25000)
    idxs = findall(mask)
    isempty(idxs) && return Float64[], Float64[], Float64[], Float64[]
    if length(idxs) > max_points
        keep = unique(round.(Int, range(1, length(idxs); length=max_points)))
        idxs = idxs[keep]
    end
    rr = receiver_row(cfg)
    xs = Float64[]
    ys = Float64[]
    zs = Float64[]
    cs = Float64[]
    max_i = max(maximum(Float64.(intensity)), eps(Float64))
    sizehint!(xs, length(idxs))
    sizehint!(ys, length(idxs))
    sizehint!(zs, length(idxs))
    sizehint!(cs, length(idxs))
    for idx in idxs
        push!(xs, grid.y[idx[2]] * 1e3)
        push!(ys, grid.z[idx[3]] * 1e3)
        push!(zs, (grid.x[idx[1]] - grid.x[rr]) * 1e3)
        push!(cs, Float64(intensity[idx]) / max_i)
    end
    return xs, ys, zs, cs
end

function render_frame(path, cumulative, current, truth_mask, fixed_cutoff, best_ratio, grid, cfg, sources, centerlines, stats, frame_idx, total_frames)
    pred_mask = cumulative .>= fixed_cutoff
    current_mask = current .>= fixed_cutoff
    flags = stats[:detected_flags]

    update_theme!(fontsize=15)
    fig = Figure(size=(1600, 950), backgroundcolor=:black)

    ax3 = Axis3(
        fig[1:3, 1:2];
        xlabel="Y [mm]",
        ylabel="Z [mm]",
        zlabel="Depth [mm]",
        title="3D cumulative HASA detection",
        aspect=:data,
        perspectiveness=0.35,
        azimuth=5pi / 8,
        elevation=pi / 7,
        backgroundcolor=:black,
        titlecolor=:white,
        xlabelcolor=:white,
        ylabelcolor=:white,
        zlabelcolor=:white,
        xticklabelcolor=:white,
        yticklabelcolor=:white,
        zticklabelcolor=:white,
    )

    for line in centerlines
        ys = [p[2] * 1e3 for p in line]
        zs = [p[3] * 1e3 for p in line]
        ds = [p[1] * 1e3 for p in line]
        lines!(ax3, ys, zs, ds; color=(:white, 0.55), linewidth=1.1)
    end

    vx, vy, vz, vc = threshold_voxel_points(pred_mask, cumulative, grid, cfg; max_points=22000)
    if !isempty(vx)
        scatter!(ax3, vx, vy, vz; color=vc, colormap=:Oranges, colorrange=(0.0, 1.0), markersize=2.2, alpha=0.28)
    end

    src_y = [src.lateral_y * 1e3 for src in sources]
    src_z = [src.lateral_z * 1e3 for src in sources]
    src_d = [src.depth * 1e3 for src in sources]
    src_colors = [flags[i] ? :lime : (:gray80, 0.45) for i in eachindex(sources)]
    scatter!(ax3, src_y, src_z, src_d; color=src_colors, markersize=8, strokecolor=:black, strokewidth=0.7)

    projections = [
        (:depth_y, "Depth-Y MIP"),
        (:depth_z, "Depth-Z MIP"),
        (:y_z, "Y-Z MIP"),
    ]
    for (col, (projection, title)) in enumerate(projections)
        xs, ys, xlabel, ylabel = projection_axes(grid, cfg, projection)
        ax = Axis(
            fig[1, col + 2];
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            aspect=DataAspect(),
            backgroundcolor=:black,
            titlecolor=:white,
            xlabelcolor=:white,
            ylabelcolor=:white,
            xticklabelcolor=:white,
            yticklabelcolor=:white,
        )
        heatmap!(ax, xs, ys, normalized_projection(cumulative, projection); colormap=:inferno, colorrange=(0, 1))
        add_contour_if_valid!(ax, xs, ys, mask_projection(truth_mask, projection); color=(:white, 0.9), linewidth=2.0, linestyle=:dash)
        add_contour_if_valid!(ax, xs, ys, mask_projection(pred_mask, projection); color=(:orange, 0.95), linewidth=2.2)
        add_contour_if_valid!(ax, xs, ys, mask_projection(current_mask, projection); color=(:deepskyblue, 0.75), linewidth=1.5)
    end

    stats_text = @sprintf(
        "window %d / %d\nfixed threshold: %.2f of final max\nsource F1: %.3f\nprecision: %.3f\nsource recall: %.3f\ndetected: %d / %d\npredicted voxels: %d",
        frame_idx,
        total_frames,
        best_ratio,
        Float64(stats[:source_f1]),
        Float64(stats[:precision]),
        Float64(stats[:source_recall]),
        Int(stats[:detected_source_count]),
        Int(stats[:num_truth_sources]),
        Int(stats[:predicted_voxels]),
    )
    Label(
        fig[2:3, 3:5],
        stats_text;
        color=:white,
        fontsize=24,
        halign=:left,
        valign=:top,
        tellheight=false,
        tellwidth=false,
    )

    Legend(
        fig[4, 1:5],
        [
            LineElement(color=(:white, 0.9), linestyle=:dash, linewidth=2),
            LineElement(color=(:orange, 0.95), linewidth=3),
            LineElement(color=(:deepskyblue, 0.75), linewidth=2),
            MarkerElement(color=:lime, marker=:circle, markersize=12),
            MarkerElement(color=(:gray80, 0.45), marker=:circle, markersize=12),
        ],
        ["truth mask", "cumulative prediction", "current window", "detected source", "not yet detected"];
        orientation=:horizontal,
        framevisible=false,
        labelcolor=:white,
        tellheight=true,
    )

    save(path, fig)
    return path
end

function find_ffmpeg()
    ffmpeg = Sys.which("ffmpeg")
    !isnothing(ffmpeg) && return ffmpeg
    if Sys.iswindows() && haskey(ENV, "LOCALAPPDATA")
        winget_root = joinpath(ENV["LOCALAPPDATA"], "Microsoft", "WinGet", "Packages")
        if isdir(winget_root)
            for (root, _, files) in walkdir(winget_root)
                if "ffmpeg.exe" in files
                    return joinpath(root, "ffmpeg.exe")
                end
            end
        end
    end
    return nothing
end

function encode_mp4(frames_dir, mp4_path, fps::Integer)
    ffmpeg = find_ffmpeg()
    pattern = joinpath(frames_dir, "frame_%04d.png")
    if isnothing(ffmpeg)
        println("ffmpeg not found on PATH. Frames are ready in: $frames_dir")
        println("Encode with:")
        println("ffmpeg -y -framerate $fps -i \"$pattern\" -vf format=yuv420p \"$mp4_path\"")
        return false
    end
    cmd = Cmd([ffmpeg, "-y", "-framerate", string(fps), "-i", pattern, "-vf", "format=yuv420p", mp4_path])
    run(cmd)
    return true
end

function print_dry_run(opts, cfg, fitted_cfg, sources, out_dir, window_config)
    nt = pam_Nt(fitted_cfg)
    ranges, window_samples, hop_samples = TranscranialFUS._pam_window_ranges(nt, fitted_cfg.dt, window_config)
    println("Dry run: PAM window convergence visualisation")
    println("  output dir       : $out_dir")
    println("  aberrator        : ", opts["aberrator"])
    println("  sources          : ", length(sources))
    println("  grid             : $(pam_Nx(fitted_cfg)) x $(pam_Ny(fitted_cfg)) x $(pam_Nz(fitted_cfg))")
    println("  spacing mm       : dx=$(fitted_cfg.dx * 1e3), dy=$(fitted_cfg.dy * 1e3), dz=$(fitted_cfg.dz * 1e3)")
    println("  t_max us         : ", fitted_cfg.t_max * 1e6)
    println("  nt               : $nt")
    println("  windows          : $(length(ranges))")
    println("  window samples   : $window_samples")
    println("  hop samples      : $hop_samples")
    println("  recon freqs MHz  : ", join(round.(default_recon_frequencies(sources) ./ 1e6; digits=3), ", "))
    println("  CUDA functional  : ", CUDA.functional())
    println("  k-Wave available : ", kwave_available())
    println("  ffmpeg           : ", something(find_ffmpeg(), "not found"))
    return nothing
end

function load_cached_inputs(path::AbstractString)
    d = load(path)
    required = ["cfg", "sources", "network_meta", "c", "rho", "medium_info", "rf", "grid", "simulation_info"]
    missing = filter(key -> !haskey(d, key), required)
    isempty(missing) || error("--from-data is missing required keys: $(join(missing, ", "))")
    return (
        cfg = d["cfg"],
        sources = d["sources"],
        network_meta = d["network_meta"],
        c = d["c"],
        rho = d["rho"],
        medium_info = d["medium_info"],
        rf = d["rf"],
        grid = d["grid"],
        simulation_info = d["simulation_info"],
        sim_sources = haskey(d, "sim_sources") ? d["sim_sources"] : EmissionSource3D[],
        n_frames = haskey(d, "n_frames") ? d["n_frames"] : missing,
    )
end

function main()
    opts = parse_cli(ARGS)
    from_data = strip(opts["from-data"])
    seed = parse(Int, opts["random-seed"])
    aberrator = parse_aberrator(opts["aberrator"])
    dry_run = parse_bool(opts["dry-run"])
    frames_only = parse_bool(opts["frames-only"])
    fps = parse(Int, opts["fps"])
    max_windows = parse(Int, opts["max-windows"])
    use_gpu_sim = parse_bool(opts["kwave-use-gpu"])
    use_gpu_recon = parse_bool(opts["recon-use-gpu"])
    show_progress = parse_bool(opts["recon-progress"])

    out_dir = isempty(strip(opts["out-dir"])) ?
        joinpath(@__DIR__, "outputs", "$(timestamp())_window_convergence") :
        abspath(opts["out-dir"])
    frames_dir = joinpath(out_dir, "frames")
    data_path = joinpath(out_dir, "data.jld2")
    summary_path = joinpath(out_dir, "summary.json")
    mp4_path = joinpath(out_dir, "pam_window_convergence.mp4")

    window_config = PAMWindowConfig(
        enabled=true,
        window_duration=40e-6,
        hop=20e-6,
        taper=:hann,
        min_energy_ratio=0.001,
        accumulation=:intensity,
    )

    cfg_base = make_demo_config(opts)
    if isempty(from_data)
        sources, network_meta = make_demo_sources(cfg_base, seed)
        cfg = fit_pam_config_3d(
            cfg_base,
            sources;
            min_bottom_margin=10e-3,
            reference_depth=aberrator == :skull ? 20e-3 : nothing,
        )
    else
        cached = load_cached_inputs(abspath(from_data))
        cfg = cached.cfg
        sources = cached.sources
        network_meta = cached.network_meta
    end

    if dry_run
        print_dry_run(opts, cfg_base, cfg, sources, out_dir, window_config)
        if !isempty(from_data)
            println("  from data        : ", abspath(from_data))
        end
        return nothing
    end

    mkpath(frames_dir)

    if isempty(from_data)
        println("Building medium...")
        c, rho, medium_info = make_pam_medium_3d(
            cfg;
            aberrator=aberrator,
            skull_to_transducer=20e-3,
            slice_index_z=250,
        )

        println("Expanding random-phase-per-window source events...")
        rng_sim = Random.MersenneTwister(seed + 1)
        sim_sources, n_frames = TranscranialFUS._expand_sources_per_window(
            sources,
            window_config.window_duration,
            window_config.hop,
            cfg.t_max,
            rng_sim;
            variability=SourceVariabilityConfig(frequency_jitter_fraction=0.01),
        )

        println("Simulating RF data with $(length(sim_sources)) emission events...")
        rf, grid, simulation_info = simulate_point_sources_3d(c, rho, sim_sources, cfg; use_gpu=use_gpu_sim)
        println("Saving RF/medium cache to $data_path")
        @save data_path cfg sources network_meta c rho medium_info rf grid simulation_info sim_sources n_frames window_config
    else
        println("Loading cached RF/medium data from $(abspath(from_data))")
        cached = load_cached_inputs(abspath(from_data))
        c = cached.c
        rho = cached.rho
        medium_info = cached.medium_info
        rf = cached.rf
        grid = cached.grid
        simulation_info = cached.simulation_info
        sim_sources = cached.sim_sources
        n_frames = cached.n_frames
    end

    selected_ranges, selected_energy, skipped_ranges, window_samples, hop_samples, energy_threshold =
        qualify_windows(rf, cfg, window_config, max_windows)
    isempty(selected_ranges) && error("No qualifying windows were found.")

    recon_frequencies = default_recon_frequencies(sources)
    bandwidth_hz = 40e3
    truth_radius = 1.0e-3
    truth_mask = pam_truth_mask_3d(sources, grid, cfg; radius=truth_radius)
    centerlines = network_meta[:centerlines]

    println("Accumulating $(length(selected_ranges)) reconstructed windows to choose final threshold...")
    cumulative_sum = zeros(Float64, pam_Nx(cfg), pam_Ny(cfg), pam_Nz(cfg))
    final_grid = grid
    for (idx, range) in pairs(selected_ranges)
        @printf("  accumulation window %d/%d samples %d:%d\n", idx, length(selected_ranges), first(range), last(range))
        current, final_grid = reconstruct_one_window(
            rf, c, cfg, range, window_config, recon_frequencies, bandwidth_hz;
            use_gpu=use_gpu_recon,
            show_progress=show_progress,
        )
        cumulative_sum .+= current
        GC.gc(false)
        CUDA.reclaim()
    end
    final_cumulative = cumulative_sum ./ length(selected_ranges)
    best_threshold, threshold_entries = best_final_threshold(
        final_cumulative,
        truth_mask,
        final_grid,
        cfg,
        sources;
        truth_radius=truth_radius,
    )
    fixed_cutoff = Float64(best_threshold[:absolute_cutoff])
    best_ratio = Float64(best_threshold[:threshold_ratio])
    @printf("Selected final threshold %.2f of final max: source F1 %.3f, precision %.3f, source recall %.3f\n",
        best_ratio,
        Float64(best_threshold[:source_f1]),
        Float64(best_threshold[:precision]),
        Float64(best_threshold[:source_recall]),
    )

    metrics_by_frame = Dict{Symbol, Any}[]
    println("Rendering frames with fixed final threshold...")
    fill!(cumulative_sum, 0.0)
    for (idx, range) in pairs(selected_ranges)
        @printf("  render window %d/%d\n", idx, length(selected_ranges))
        current, final_grid = reconstruct_one_window(
            rf, c, cfg, range, window_config, recon_frequencies, bandwidth_hz;
            use_gpu=use_gpu_recon,
            show_progress=show_progress,
        )
        cumulative_sum .+= current
        cumulative = cumulative_sum ./ idx
        stats = fixed_threshold_stats_3d(cumulative, fixed_cutoff, truth_mask, final_grid, cfg, sources; truth_radius=truth_radius)
        stats[:frame_index] = idx
        stats[:sample_range] = [first(range), last(range)]
        push!(metrics_by_frame, copy(stats))
        frame_path = joinpath(frames_dir, @sprintf("frame_%04d.png", idx))
        render_frame(
            frame_path,
            cumulative,
            current,
            truth_mask,
            fixed_cutoff,
            best_ratio,
            final_grid,
            cfg,
            sources,
            centerlines,
            stats,
            idx,
            length(selected_ranges),
        )
        GC.gc(false)
        CUDA.reclaim()
    end

    summary = Dict(
        "out_dir" => out_dir,
        "data_path" => data_path,
        "frames_dir" => frames_dir,
        "mp4_path" => mp4_path,
        "random_seed" => seed,
        "aberrator" => String(aberrator),
        "from_data" => isempty(from_data) ? nothing : abspath(from_data),
        "grid" => Dict(
            "nx" => pam_Nx(cfg),
            "ny" => pam_Ny(cfg),
            "nz" => pam_Nz(cfg),
            "nt" => pam_Nt(cfg),
            "dx_m" => cfg.dx,
            "dy_m" => cfg.dy,
            "dz_m" => cfg.dz,
            "t_max_s" => cfg.t_max,
            "zero_pad_factor" => cfg.zero_pad_factor,
        ),
        "source_model" => "network3d",
        "source_phase_mode" => "random_phase_per_window",
        "source_count" => length(sources),
        "emission_event_count" => length(sim_sources),
        "n_frames_from_source_expansion" => n_frames,
        "windowing" => Dict(
            "window_duration_s" => window_config.window_duration,
            "hop_s" => window_config.hop,
            "window_samples" => window_samples,
            "hop_samples" => hop_samples,
            "selected_window_count" => length(selected_ranges),
            "skipped_window_count" => length(skipped_ranges),
            "energy_threshold" => energy_threshold,
        ),
        "reconstruction" => Dict(
            "frequencies_hz" => recon_frequencies,
            "bandwidth_hz" => bandwidth_hz,
            "axial_step_m" => 50e-6,
            "use_gpu" => use_gpu_recon,
        ),
        "threshold" => Dict(
            "mode" => "final_best_source_f1_fixed_absolute_cutoff",
            "ratio_of_final_max" => best_ratio,
            "absolute_cutoff" => fixed_cutoff,
            "best_final_metrics" => best_threshold,
        ),
        "medium" => Dict(String(k) => v for (k, v) in medium_info),
        "simulation" => Dict(String(k) => v for (k, v) in simulation_info),
        "sources" => [source_summary_3d(src) for src in sources],
        "metrics_by_frame" => metrics_by_frame,
    )

    println("Saving data bundle...")
    selected_ranges_pairs = [[first(r), last(r)] for r in selected_ranges]
    skipped_ranges_pairs = [[first(r), last(r)] for r in skipped_ranges]
    @save data_path cfg sources network_meta c rho medium_info rf grid simulation_info sim_sources n_frames window_config recon_frequencies bandwidth_hz truth_radius truth_mask final_cumulative best_threshold threshold_entries selected_ranges_pairs skipped_ranges_pairs metrics_by_frame

    open(summary_path, "w") do io
        JSON3.pretty(io, summary)
    end

    if frames_only
        println("Frames written to $frames_dir")
    else
        encoded = encode_mp4(frames_dir, mp4_path, fps)
        encoded && println("Saved MP4 to $mp4_path")
    end
    println("Done: $out_dir")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
