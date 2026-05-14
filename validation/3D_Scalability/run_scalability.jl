"""
3D HASA GPU Scalability Study — synthetic RF, no k-Wave or DICOM required.

RF data is synthetic (uninitialised Float32 noise of the correct shape) and the
medium is a uniform homogeneous array (c = 1500 m/s everywhere).  HASA still
executes its full GPU code path; the correction terms are just trivial with a
homogeneous medium.  This avoids slow k-Wave simulations and allows sweeping
large apertures to find the GPU memory limit.

benchmark=true captures per-phase GPU timing:
  march_gpu_s   — GPU-measured march time (CUDA.@elapsed)
  fft_s         — FFT share of the march
  elementwise_s — element-wise kernel share
  setup_s       — one-time operator setup (pre-computed operator stack)
  download_s    — GPU→CPU intensity transfer

Sweep:
  APERTURES_MM  — extended until OOM; out-of-memory points are marked NaN
  DEPTHS_MM     — depth below skull surface

All timing arrays (wall, setup, march_gpu, fft, ew, bandwidth) are saved to a
timestamped JLD2 file.  The progress file is updated after every point so a
crash loses at most one run.

Run from the project root:
    julia --project=. validation/3D_Scalability/run_scalability.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using TranscranialFUS
using JLD2, Printf, Dates, CUDA, Statistics

# ── Fixed parameters ──────────────────────────────────────────────────────────

const FREQ       = 1.0e6
const DX         = 0.2e-3
const DT         = 40e-9
const T_MAX      = 120e-6
const AXIAL_STEP = 50e-6
const BANDWIDTH  = 0.0
const C0         = 1500f0   # homogeneous medium sound speed [m/s]

const SKULL_DIST = 20e-3    # kept so depth labelling matches the rest of the study
const N_RUNS     = 3        # timed repetitions per point; median is stored

# ── Sweep parameters ──────────────────────────────────────────────────────────

const APERTURES_MM = [25.0, 50.0, 64.0, 75.0, 100.0, 125.0]
const DEPTHS_MM    = [20.0, 40.0, 60.0, 80.0, 100.0]   # depth below skull

let apts = round.(Int, APERTURES_MM .* 1e-3 ./ DX)
    println("Aperture sweep : $(round.(Int, APERTURES_MM)) mm  →  Ny=Nz = $apts")
    println("  voxels/plane : $(apts .^ 2)")
end
let steps = round.(Int, DEPTHS_MM .* 1e-3 ./ AXIAL_STEP)
    println("Depth sweep    : $(round.(Int, DEPTHS_MM)) mm below skull")
    println("  march steps  : $steps (×4 substeps for HASA)")
end
println()

# ── Storage ───────────────────────────────────────────────────────────────────

n_depths    = length(DEPTHS_MM)
n_apertures = length(APERTURES_MM)

timing_wall_s       = fill(NaN, n_depths, n_apertures)
timing_setup_s      = fill(NaN, n_depths, n_apertures)
timing_march_gpu_s  = fill(NaN, n_depths, n_apertures)
timing_march_wall_s = fill(NaN, n_depths, n_apertures)
timing_fft_s        = fill(NaN, n_depths, n_apertures)
timing_ew_s         = fill(NaN, n_depths, n_apertures)
timing_bw_GBps      = fill(NaN, n_depths, n_apertures)
oom_flags           = fill(false, n_depths, n_apertures)   # true = OOM at this point

# ── Progress file ─────────────────────────────────────────────────────────────

const PROGRESS_FILE = joinpath(@__DIR__, "progress.jld2")

function save_progress()
    jldsave(PROGRESS_FILE;
        timing_wall_s, timing_setup_s,
        timing_march_gpu_s, timing_march_wall_s,
        timing_fft_s, timing_ew_s, timing_bw_GBps,
        oom_flags,
        APERTURES_MM, DEPTHS_MM,
        FREQ, DX, DT, T_MAX, AXIAL_STEP, BANDWIDTH, C0, SKULL_DIST, N_RUNS,
    )
end

if isfile(PROGRESS_FILE)
    prev      = load(PROGRESS_FILE)
    prev_apts = get(prev, "APERTURES_MM", Float64[])
    prev_deps = get(prev, "DEPTHS_MM",    Float64[])
    ai_map = [findfirst(==(a), prev_apts) for a in APERTURES_MM]
    di_map = [findfirst(==(d), prev_deps) for d in DEPTHS_MM]
    for (di_new, di_old) in enumerate(di_map)
        di_old === nothing && continue
        for (ai_new, ai_old) in enumerate(ai_map)
            ai_old === nothing && continue
            for field in (:timing_wall_s, :timing_setup_s, :timing_march_gpu_s,
                          :timing_march_wall_s, :timing_fft_s, :timing_ew_s,
                          :timing_bw_GBps, :oom_flags)
                key = string(field)
                haskey(prev, key) && (eval(field)[di_new, ai_new] = prev[key][di_old, ai_old])
            end
        end
    end
    n_done = count(!isnan, timing_wall_s) + count(identity, oom_flags)
    println("Resuming from progress.jld2 — $n_done/$(n_depths*n_apertures) points already done.\n")
end

# ── Warm-up — one call per aperture size to trigger JIT compilation ───────────

if any(isnan, timing_wall_s) && !all(identity, oom_flags)
    println("GPU warm-up (one call per aperture size, axial_dim = 45 mm)…")
    for apt_mm in APERTURES_MM
        Ny = round(Int, apt_mm * 1e-3 / DX)
        print("  aperture = $(round(Int, apt_mm)) mm (Ny=Nz=$Ny) … ")
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
        Nx_w = pam_Nx(cfg_w)
        Nt_w = pam_Nt(cfg_w)
        c_w  = fill(C0, Nx_w, Ny, Ny)
        rf_w = Array{Float32}(undef, Ny, Ny, Nt_w)
        try
            reconstruct_pam_3d(rf_w, c_w, cfg_w;
                frequencies=[FREQ], bandwidth=BANDWIDTH, corrected=true,
                axial_step=AXIAL_STEP, use_gpu=true, show_progress=false, benchmark=true)
            println("done.")
        catch e
            isa(e, CUDA.OutOfGPUMemoryError) ? println("OOM — skipping this aperture.") : rethrow(e)
        end
        c_w = nothing; rf_w = nothing
        GC.gc(true); CUDA.reclaim()
    end
    println()
end

# ── Scalability sweep ─────────────────────────────────────────────────────────

t_sweep_start = time()
n_total = n_depths * n_apertures
s_count = 0
n_run   = 0

for (di, depth_mm) in enumerate(DEPTHS_MM)

    recon_axial_dim = (depth_mm + SKULL_DIST * 1e3) * 1e-3

    for (ai, apt_mm) in enumerate(APERTURES_MM)

        global s_count += 1

        if !isnan(timing_wall_s[di, ai]) || oom_flags[di, ai]
            status = oom_flags[di, ai] ? "OOM" : "done"
            @printf("[%2d/%d] depth=%3.0f mm, aperture=%3.0f mm  [skipped — %s]\n",
                    s_count, n_total, depth_mm, apt_mm, status)
            continue
        end

        n_remaining = count(isnan, timing_wall_s) - count(identity, oom_flags)
        elapsed     = time() - t_sweep_start
        eta_s       = n_run > 0 ? elapsed / n_run * max(n_remaining - 1, 0) : NaN
        eta_str     = isnan(eta_s) ? "?" : @sprintf("%.0f", eta_s / 60)

        Ny_tgt     = round(Int, apt_mm * 1e-3 / DX)
        n_march    = round(Int, depth_mm * 1e-3 / AXIAL_STEP)
        @printf("[%2d/%d] depth=%3.0f mm, aperture=%3.0f mm (Ny=Nz=%d, %d march rows, ETA ~%s min)\n",
                s_count, n_total, depth_mm, apt_mm, Ny_tgt, n_march, eta_str)
        flush(stdout)

        cfg_recon = PAMConfig3D(
            dx = DX, dy = DX, dz = DX,
            axial_dim           = recon_axial_dim,
            transverse_dim_y    = apt_mm * 1e-3,
            transverse_dim_z    = apt_mm * 1e-3,
            dt = DT, t_max = T_MAX,
            receiver_aperture_y = apt_mm * 1e-3,
            receiver_aperture_z = apt_mm * 1e-3,
        )
        Nx_tgt = pam_Nx(cfg_recon)
        Nt_tgt = pam_Nt(cfg_recon)

        c_recon = fill(C0, Nx_tgt, Ny_tgt, Ny_tgt)
        rf_apt  = Array{Float32}(undef, Ny_tgt, Ny_tgt, Nt_tgt)

        run_wall_s    = Float64[]
        run_setup_s   = Float64[]
        run_mgpu_s    = Float64[]
        run_mwall_s   = Float64[]
        run_fft_s     = Float64[]
        run_ew_s      = Float64[]
        run_bw_GBps   = Float64[]

        point_oom = false

        for r in 1:N_RUNS
            @printf("  run %d/%d … ", r, N_RUNS); flush(stdout)
            try
                local info
                t = @elapsed begin
                    _, _, info = reconstruct_pam_3d(
                        rf_apt, c_recon, cfg_recon;
                        frequencies=[FREQ], bandwidth=BANDWIDTH,
                        corrected=true, axial_step=AXIAL_STEP,
                        use_gpu=true, show_progress=false, benchmark=true,
                    )
                end
                gt = info[:gpu_timing]
                push!(run_wall_s,   t)
                push!(run_setup_s,  something(gt[:setup_s],         NaN))
                push!(run_mgpu_s,   something(gt[:march_gpu_s],     NaN))
                push!(run_mwall_s,  something(gt[:march_wall_s],    NaN))
                push!(run_fft_s,    something(gt[:fft_s],           NaN))
                push!(run_ew_s,     something(gt[:elementwise_s],   NaN))
                push!(run_bw_GBps,  something(gt[:bandwidth_GBps],  NaN))
                @printf("%.2fs  (setup=%.2fs  march_gpu=%.2fs  fft=%.2fs  ew=%.2fs  BW=%.0f GB/s)\n",
                        t, run_setup_s[end], run_mgpu_s[end],
                        run_fft_s[end], run_ew_s[end], run_bw_GBps[end])
            catch e
                if isa(e, CUDA.OutOfGPUMemoryError)
                    println("OOM")
                    point_oom = true
                    break
                else
                    rethrow(e)
                end
            end
            GC.gc(true); CUDA.reclaim()
        end

        if point_oom
            oom_flags[di, ai] = true
            @printf("  → OOM at aperture=%g mm, depth=%g mm\n", apt_mm, depth_mm)
        else
            timing_wall_s[di, ai]       = median(run_wall_s)
            timing_setup_s[di, ai]      = median(run_setup_s)
            timing_march_gpu_s[di, ai]  = median(run_mgpu_s)
            timing_march_wall_s[di, ai] = median(run_mwall_s)
            timing_fft_s[di, ai]        = median(run_fft_s)
            timing_ew_s[di, ai]         = median(run_ew_s)
            timing_bw_GBps[di, ai]      = median(run_bw_GBps)
            @printf("  → median  wall=%.2fs  setup=%.2fs  march_gpu=%.2fs  fft=%.2fs  ew=%.2fs  BW=%.0f GB/s\n",
                    timing_wall_s[di, ai], timing_setup_s[di, ai],
                    timing_march_gpu_s[di, ai], timing_fft_s[di, ai],
                    timing_ew_s[di, ai], timing_bw_GBps[di, ai])
        end

        c_recon = nothing; rf_apt = nothing
        GC.gc(true); CUDA.reclaim()

        global n_run += 1
        n_done = count(!isnan, timing_wall_s) + count(identity, oom_flags)
        save_progress()
        @printf("    [progress saved — %d/%d done]\n\n", n_done, n_total)
    end
end

total_min = (time() - t_sweep_start) / 60
@printf("\nSweep complete in %.1f minutes.\n\n", total_min)

# ── Summary tables ────────────────────────────────────────────────────────────

for (label, arr) in [
        ("Wall-clock [s]",      timing_wall_s),
        ("GPU march [s]",       timing_march_gpu_s),
        ("Setup [s]",           timing_setup_s),
        ("Bandwidth [GB/s]",    timing_bw_GBps),
    ]
    println(label, "  (rows = depth from skull, cols = aperture)")
    @printf("  %10s │", "Depth\\Apt")
    for apt_mm in APERTURES_MM; @printf("  %6.0f mm", apt_mm); end
    println()
    for (di, dep) in enumerate(DEPTHS_MM)
        @printf("  %6.0f mm │", dep)
        for ai in 1:n_apertures
            oom_flags[di, ai] ? @printf("     OOM  ") : @printf("  %7.2f ", arr[di, ai])
        end
        println()
    end
    println()
end

# ── Save final results ────────────────────────────────────────────────────────

ts      = Dates.format(now(), "yyyymmdd_HHMMSS")
outfile = joinpath(@__DIR__, "results_$(ts).jld2")
save_progress()
mv(PROGRESS_FILE, outfile; force=true)
println("Results saved → $outfile")
