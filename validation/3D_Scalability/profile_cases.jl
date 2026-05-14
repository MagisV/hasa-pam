"""
Phase-breakdown profiling for the two problematic aperture cases.

Runs GPU HASA with benchmark=true to split each reconstruction into:
  setup_s       — operator + batch setup
  fft_s         — 2-D FFT time (accumulated over all march rows)
  elementwise_s — element-wise kernel time
  download_s    — GPU→CPU intensity transfer
  bandwidth_GBps — estimated memory bandwidth during march

25 mm is the reference (clean baseline after warm-up fix).
75 mm is the suspect case (erratic times, near-VRAM-limit).

GPU memory (free / total) is logged before and after every run to check
for leaks or fragmentation that would explain the 75 mm variance.

k-Wave RF data is cached to rf_cache.jld2 so repeated profiling runs do
not require re-simulation.

Run from the project root:
    julia --project=. validation/3D_Scalability/profile_cases.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using TranscranialFUS
using JLD2, Printf, Dates, CUDA, Statistics

# ── Parameters (consistent with scalability study) ────────────────────────────

const FREQ       = 1.0e6
const NUM_CYCLES = 10.0
const DX         = 0.2e-3
const DT         = 40e-9
const T_MAX      = 120e-6
const AXIAL_STEP = 50e-6
const BANDWIDTH  = 0.0

const SKULL_DIST  = 20e-3
const SLICE_INDEX = 250

const KW_APERTURE  = 64e-3
const KW_AXIAL_DIM = 70e-3

const PROFILE_APERTURES_MM = [25.0, 75.0]
const DEPTHS_MM             = [70.0]
const N_RUNS                = 5   # runs per (aperture, depth) point

# ── RF cache ──────────────────────────────────────────────────────────────────

const RF_CACHE = joinpath(@__DIR__, "rf_cache.jld2")

function load_or_simulate_rf()
    if isfile(RF_CACHE)
        println("Loading cached RF data from rf_cache.jld2…")
        d = load(RF_CACHE)
        return d["rf_base"]
    end

    println("No RF cache found — running k-Wave (GPU)…")
    kw_cfg = PAMConfig3D(
        dx               = DX, dy = DX, dz = DX,
        axial_dim        = KW_AXIAL_DIM,
        transverse_dim_y = KW_APERTURE,
        transverse_dim_z = KW_APERTURE,
        dt               = DT, t_max = T_MAX,
        receiver_aperture_y = KW_APERTURE,
        receiver_aperture_z = KW_APERTURE,
    )
    c_kw, rho_kw, _ = make_pam_medium_3d(kw_cfg;
        aberrator=:skull, skull_to_transducer=SKULL_DIST, slice_index_z=SLICE_INDEX)
    src = PointSource3D(
        depth=50e-3, lateral_y=0.0, lateral_z=0.0,
        frequency=FREQ, num_cycles=NUM_CYCLES,
    )
    kwave_tmp = mktempdir()
    t_sim = @elapsed begin
        rf_base, _, _ = simulate_point_sources_3d(c_kw, rho_kw, [src], kw_cfg;
            use_gpu=true, kwave_data_path=kwave_tmp)
    end
    rm(kwave_tmp; recursive=true, force=true)
    @printf("  k-Wave wall-clock: %.1f s\n", t_sim)

    jldsave(RF_CACHE; rf_base)
    println("  RF data cached → rf_cache.jld2\n")
    return rf_base
end

# ── RF resize (centre crop / zero-pad) ────────────────────────────────────────

function resize_rf(rf::AbstractArray, Ny_target::Int, Nz_target::Int)
    Ny_src, Nz_src, Nt = size(rf)
    rf_out = zeros(eltype(rf), Ny_target, Nz_target, Nt)
    y_range_src, y_range_dst = _centre_ranges(Ny_src, Ny_target)
    z_range_src, z_range_dst = _centre_ranges(Nz_src, Nz_target)
    rf_out[y_range_dst, z_range_dst, :] .= rf[y_range_src, z_range_src, :]
    return rf_out
end

function _centre_ranges(n_src, n_tgt)
    if n_tgt <= n_src
        lo = (n_src - n_tgt) ÷ 2 + 1
        n_src_r = lo:(lo + n_tgt - 1);  n_tgt_r = 1:n_tgt
    else
        lo = (n_tgt - n_src) ÷ 2 + 1
        n_src_r = 1:n_src;  n_tgt_r = lo:(lo + n_src - 1)
    end
    return n_src_r, n_tgt_r
end

# ── GPU memory helper ─────────────────────────────────────────────────────────

gpu_free_mb()  = CUDA.available_memory() / 1024^2
gpu_total_mb() = CUDA.total_memory()     / 1024^2

function print_gpu_mem(label)
    free  = gpu_free_mb()
    total = gpu_total_mb()
    used  = total - free
    @printf("  [GPU mem %-6s]  used = %6.0f / %6.0f MB  (free = %6.0f MB)\n",
            label, used, total, free)
end

# ── Warm-up ───────────────────────────────────────────────────────────────────

function warmup(apt_mm)
    print("  warm-up $(round(Int, apt_mm)) mm … ")
    flush(stdout)
    cfg_w = PAMConfig3D(
        dx = DX, dy = DX, dz = DX,
        axial_dim           = 45e-3,
        transverse_dim_y    = apt_mm * 1e-3,
        transverse_dim_z    = apt_mm * 1e-3,
        dt = DT, t_max = T_MAX,
        receiver_aperture_y = apt_mm * 1e-3,
        receiver_aperture_z = apt_mm * 1e-3,
    )
    c_w, _, _ = make_pam_medium_3d(cfg_w;
        aberrator=:skull, skull_to_transducer=SKULL_DIST, slice_index_z=SLICE_INDEX)
    rf_w = zeros(Float32, pam_Ny(cfg_w), pam_Nz(cfg_w), pam_Nt(cfg_w))
    reconstruct_pam_3d(rf_w, c_w, cfg_w;
        frequencies=[FREQ], bandwidth=BANDWIDTH, corrected=true,
        axial_step=AXIAL_STEP, use_gpu=true, show_progress=false, benchmark=true)
    GC.gc(true); CUDA.reclaim()
    println("done.")
end

# ── Main ──────────────────────────────────────────────────────────────────────

rf_base = load_or_simulate_rf()

println("GPU warm-up for profiled apertures…")
for apt_mm in PROFILE_APERTURES_MM
    warmup(apt_mm)
end
println()

for apt_mm in PROFILE_APERTURES_MM

    Ny_tgt    = round(Int, apt_mm * 1e-3 / DX)
    rf_apt    = resize_rf(rf_base, Ny_tgt, Ny_tgt)

    println("=" ^ 70)
    @printf("APERTURE = %g mm  (Ny = Nz = %d,  %d k voxels/plane)\n",
            apt_mm, Ny_tgt, round(Int, Ny_tgt^2 / 1000))
    println("=" ^ 70)

    for depth_mm in DEPTHS_MM

        recon_axial_dim = (depth_mm + SKULL_DIST * 1e3) * 1e-3
        n_march = round(Int, depth_mm * 1e-3 / AXIAL_STEP)

        println()
        @printf("  depth = %g mm below skull  (%d march rows)\n", depth_mm, n_march)
        println("  " * "─" ^ 66)

        cfg_recon = PAMConfig3D(
            dx = DX, dy = DX, dz = DX,
            axial_dim           = recon_axial_dim,
            transverse_dim_y    = apt_mm * 1e-3,
            transverse_dim_z    = apt_mm * 1e-3,
            dt = DT, t_max = T_MAX,
            receiver_aperture_y = apt_mm * 1e-3,
            receiver_aperture_z = apt_mm * 1e-3,
        )

        Ny_max  = round(Int, maximum(PROFILE_APERTURES_MM) * 1e-3 / DX)
        cfg_max = PAMConfig3D(
            dx = DX, dy = DX, dz = DX,
            axial_dim           = recon_axial_dim,
            transverse_dim_y    = maximum(PROFILE_APERTURES_MM) * 1e-3,
            transverse_dim_z    = maximum(PROFILE_APERTURES_MM) * 1e-3,
            dt = DT, t_max = T_MAX,
            receiver_aperture_y = maximum(PROFILE_APERTURES_MM) * 1e-3,
            receiver_aperture_z = maximum(PROFILE_APERTURES_MM) * 1e-3,
        )
        c_max, _, _ = make_pam_medium_3d(cfg_max;
            aberrator=:skull, skull_to_transducer=SKULL_DIST, slice_index_z=SLICE_INDEX)
        y_lo = (pam_Ny(cfg_max) - Ny_tgt) ÷ 2 + 1
        c_recon = c_max[:, y_lo:(y_lo + Ny_tgt - 1), y_lo:(y_lo + Ny_tgt - 1)]

        wall_s_runs = Float64[]
        fft_s_runs  = Float64[]
        ew_s_runs   = Float64[]

        for r in 1:N_RUNS
            print_gpu_mem("before")

            t_wall = @elapsed begin
                _, _, info = reconstruct_pam_3d(
                    rf_apt, c_recon, cfg_recon;
                    frequencies=[FREQ], bandwidth=BANDWIDTH,
                    corrected=true, axial_step=AXIAL_STEP,
                    use_gpu=true, show_progress=false, benchmark=true,
                )
            end

            gt = info[:gpu_timing]
            fft_s = something(gt[:fft_s],         NaN)
            ew_s  = something(gt[:elementwise_s],  NaN)
            mg_s  = something(gt[:march_gpu_s],    NaN)
            mw_s  = something(gt[:march_wall_s],   NaN)
            dl_s  = something(gt[:download_s],     NaN)
            bw    = something(gt[:bandwidth_GBps], NaN)
            su_s  = something(gt[:setup_s],        NaN)

            fft_pct = isnan(mg_s) || mg_s == 0 ? NaN : 100 * fft_s / mg_s
            ew_pct  = isnan(mg_s) || mg_s == 0 ? NaN : 100 * ew_s  / mg_s

            @printf("  run %d/%d │ wall=%5.2fs  setup=%5.2fs  march_gpu=%5.2fs  wall=%5.2fs\n",
                    r, N_RUNS, t_wall, su_s, mg_s, mw_s)
            @printf("         │   fft=%5.2fs(%4.1f%%)  ew=%5.2fs(%4.1f%%)  download=%5.2fs  BW=%5.0f GB/s\n",
                    fft_s, fft_pct, ew_s, ew_pct, dl_s, bw)

            print_gpu_mem("after ")
            println()

            push!(wall_s_runs, t_wall)
            push!(fft_s_runs,  fft_s)
            push!(ew_s_runs,   ew_s)

            GC.gc(true); CUDA.reclaim()
        end

        @printf("  summary │ wall  median=%.2fs  min=%.2fs  max=%.2fs  σ=%.2fs\n",
                median(wall_s_runs), minimum(wall_s_runs), maximum(wall_s_runs), std(wall_s_runs))
        @printf("          │ fft   median=%.2fs  min=%.2fs  max=%.2fs\n",
                median(fft_s_runs), minimum(fft_s_runs), maximum(fft_s_runs))
        @printf("          │ ew    median=%.2fs  min=%.2fs  max=%.2fs\n",
                median(ew_s_runs),  minimum(ew_s_runs),  maximum(ew_s_runs))

        c_recon = nothing; c_max = nothing
        GC.gc(true); CUDA.reclaim()
    end

    rf_apt = nothing
    GC.gc(true); CUDA.reclaim()
    println()
end
