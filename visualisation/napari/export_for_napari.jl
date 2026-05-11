#!/usr/bin/env julia
# Exports a window-convergence run to a single HDF5 file readable by show_convergence.py.
#
# Usage:
#   julia --project=. visualisation/napari/export_for_napari.jl --run-dir=visualisation/outputs/<timestamp>_window_convergence
#
# Reads:  <run-dir>/data.jld2  and  <run-dir>/recons/window_XXXX.jld2
# Writes: <run-dir>/napari_data.h5

using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using HDF5
using JLD2
using Printf
using TranscranialFUS

function parse_run_dir(args)
    for arg in args
        m = match(r"^--run-dir=(.+)$", arg)
        isnothing(m) || return strip(m.captures[1])
    end
    error("Required argument --run-dir=<path> not found.")
end

function main()
    run_dir   = abspath(parse_run_dir(ARGS))
    data_path = joinpath(run_dir, "data.jld2")
    recons_dir = joinpath(run_dir, "recons")
    out_path  = joinpath(run_dir, "napari_data.h5")

    isfile(data_path) || error("data.jld2 not found in $run_dir")

    println("Loading data from $data_path ...")
    d = load(data_path)

    cfg          = d["cfg"]
    sources      = d["sources"]
    network_meta = d["network_meta"]
    grid         = d["grid"]
    c            = d["c"]       # Float32 (nx, ny, nz) — SOS field

    rr = receiver_row(cfg)
    x_mm = Float32.(collect(grid.x) .* 1e3)
    y_mm = Float32.(collect(grid.y) .* 1e3)
    z_mm = Float32.(collect(grid.z) .* 1e3)

    println("Computing truth mask ...")
    truth_mask = pam_truth_mask_3d(sources, grid, cfg; radius=1.0e-3)

    src_depth_mm = Float32[src.depth     * 1e3 for src in sources]
    src_y_mm     = Float32[src.lateral_y * 1e3 for src in sources]
    src_z_mm     = Float32[src.lateral_z * 1e3 for src in sources]

    # Flatten centerlines to (N,3) and a parallel track-id vector
    centerlines = network_meta[:centerlines]
    cl_depth = Float32[]
    cl_y     = Float32[]
    cl_z     = Float32[]
    cl_ids   = Int32[]
    for (tid, line) in enumerate(centerlines)
        for pt in line
            push!(cl_depth, Float32(pt[1] * 1e3))
            push!(cl_y,     Float32(pt[2] * 1e3))
            push!(cl_z,     Float32(pt[3] * 1e3))
            push!(cl_ids,   Int32(tid))
        end
    end

    # Per-window intensities
    window_files = sort(filter(f -> startswith(f, "window_") && endswith(f, ".jld2"),
                               readdir(recons_dir)))
    isempty(window_files) && println("Warning: no window_XXXX.jld2 found in $recons_dir")

    println("Writing $out_path ...")
    h5open(out_path, "w") do f
        f["sos"]              = Float32.(c)
        f["x_mm"]             = x_mm
        f["y_mm"]             = y_mm
        f["z_mm"]             = z_mm
        f["receiver_row"]     = Int32(rr)
        f["source_depth_mm"]  = src_depth_mm
        f["source_y_mm"]      = src_y_mm
        f["source_z_mm"]      = src_z_mm
        f["truth_mask"]       = UInt8.(truth_mask)
        if !isempty(cl_ids)
            f["centerline_depth_mm"] = cl_depth
            f["centerline_y_mm"]     = cl_y
            f["centerline_z_mm"]     = cl_z
            f["centerline_ids"]      = cl_ids
        end
        for wf in window_files
            @printf("  loading %s ...\n", wf)
            wd  = load(joinpath(recons_dir, wf))
            key = splitext(wf)[1]
            f[key] = Float32.(wd["intensity"])
        end
    end

    println("Done: $out_path")
end

main()
