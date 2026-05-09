"""
3D HASA GPU Scalability Study — reconstruction time vs aperture and depth.

One k-Wave simulation (64 mm aperture, source at 50 mm — same parameters as
the speed benchmark) provides the base RF data.  The reconstruction config is
then swept over all (depth, aperture) combinations to isolate the two
independent GPU compute bottlenecks:

  Transverse grid size Ny×Nz  → FFT / memory-bandwidth bottleneck
  Axial reconstruction depth  → marching-step count, scales linearly

Aperture sweep uses crop / zero-pad of the base RF data so that no additional
k-Wave simulations are required; only GPU HASA reconstruction is timed.

Inner-loop structure for a single call (nfreq=1, nwindows=1):
  n_march_rows = depth_mm / (AXIAL_STEP*1e3)   (linear in depth)
  ops per row  = 4 substeps (HASA) × 2D FFT on Ny×Nz

Run from the project root:
    julia --project=. validation/3D_Scalability/run_scalability.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using TranscranialFUS
using JLD2, Printf, Dates

# ── Fixed parameters (consistent with 3D_PAM_Accuracy / 3D_Speed_Benchmark) ──

const FREQ       = 1.0e6
const NUM_CYCLES = 10.0
const DX         = 0.2e-3
const DT         = 40e-9
const T_MAX      = 120e-6
const AXIAL_STEP = 50e-6
const BANDWIDTH  = 0.0

const SKULL_DIST  = 20e-3
const SLICE_INDEX = 250

# ── Sweep parameters ──────────────────────────────────────────────────────────

const APERTURES_MM = [50.0, 75.0, 100.0]   # receiver aperture (y and z)
const DEPTHS_MM    = [20.0, 40.0, 60.0, 80.0, 100.0]   # depth below skull surface

let apts = round.(Int, APERTURES_MM .* 1e-3 ./ DX)
    println("Aperture sweep: $(round.(Int, APERTURES_MM)) mm → Ny=Nz = $apts")
    println("  voxels/plane  : $(apts .^ 2)")
end
let steps = round.(Int, DEPTHS_MM .* 1e-3 ./ AXIAL_STEP)
    println("Depth sweep    : $(round.(Int, DEPTHS_MM)) mm below skull")
    println("  march steps  : $steps (×4 substeps for HASA)")
end
println()

# ── k-Wave simulation (64 mm aperture — same as speed benchmark) ──────────────

const KW_APERTURE  = 64e-3
const KW_AXIAL_DIM = 70e-3

kw_cfg = PAMConfig3D(
    dx               = DX,
    dy               = DX,
    dz               = DX,
    axial_dim        = KW_AXIAL_DIM,
    transverse_dim_y = KW_APERTURE,
    transverse_dim_z = KW_APERTURE,
    dt               = DT,
    t_max            = T_MAX,
    receiver_aperture_y = KW_APERTURE,
    receiver_aperture_z = KW_APERTURE,
)

println("k-Wave grid : $(pam_Nx(kw_cfg))×$(pam_Ny(kw_cfg))×$(pam_Nz(kw_cfg)), Nt=$(pam_Nt(kw_cfg))")
println()

println("Building skull medium for k-Wave…")
c_kw, rho_kw, _ = make_pam_medium_3d(
    kw_cfg;
    aberrator           = :skull,
    skull_to_transducer = SKULL_DIST,
    slice_index_z       = SLICE_INDEX,
)

src = PointSource3D(
    depth      = 50e-3,
    lateral_y  = 0.0,
    lateral_z  = 0.0,
    frequency  = FREQ,
    num_cycles = NUM_CYCLES,
)

println("Running k-Wave (GPU)…")
t_sim = @elapsed begin
    rf_base, _, _ = simulate_point_sources_3d(c_kw, rho_kw, [src], kw_cfg; use_gpu=true)
end
@printf("  k-Wave wall-clock: %.1f s\n\n", t_sim)

const Ny_base = pam_Ny(kw_cfg)   # 320 (64 mm / 0.2 mm)
const Nz_base = pam_Nz(kw_cfg)   # 320

# ── RF resize helper (centre crop or zero-pad) ────────────────────────────────

function resize_rf(rf::AbstractArray, Ny_target::Int, Nz_target::Int)
    Ny_src, Nz_src, Nt = size(rf)
    rf_out = zeros(eltype(rf), Ny_target, Nz_target, Nt)

    if Ny_target <= Ny_src
        y_src_lo = (Ny_src - Ny_target) ÷ 2 + 1
        y_range_src = y_src_lo:(y_src_lo + Ny_target - 1)
        y_range_dst = 1:Ny_target
    else
        y_dst_lo = (Ny_target - Ny_src) ÷ 2 + 1
        y_range_src = 1:Ny_src
        y_range_dst = y_dst_lo:(y_dst_lo + Ny_src - 1)
    end

    if Nz_target <= Nz_src
        z_src_lo = (Nz_src - Nz_target) ÷ 2 + 1
        z_range_src = z_src_lo:(z_src_lo + Nz_target - 1)
        z_range_dst = 1:Nz_target
    else
        z_dst_lo = (Nz_target - Nz_src) ÷ 2 + 1
        z_range_src = 1:Nz_src
        z_range_dst = z_dst_lo:(z_dst_lo + Nz_src - 1)
    end

    rf_out[y_range_dst, z_range_dst, :] .= rf[y_range_src, z_range_src, :]
    return rf_out
end

# ── GPU warm-up ───────────────────────────────────────────────────────────────

print("GPU warm-up (small domain) … ")
flush(stdout)
let
    cfg_w = PAMConfig3D(
        dx = DX, dy = DX, dz = DX,
        axial_dim        = 30e-3,
        transverse_dim_y = 50e-3,
        transverse_dim_z = 50e-3,
        dt = DT, t_max = T_MAX,
        receiver_aperture_y = 50e-3,
        receiver_aperture_z = 50e-3,
    )
    c_w, _, _ = make_pam_medium_3d(cfg_w;
        aberrator=:skull, skull_to_transducer=SKULL_DIST, slice_index_z=SLICE_INDEX)
    rf_w = zeros(Float32, pam_Ny(cfg_w), pam_Nz(cfg_w), pam_Nt(cfg_w))
    reconstruct_pam_3d(rf_w, c_w, cfg_w;
        frequencies=[FREQ], bandwidth=BANDWIDTH, corrected=true,
        axial_step=AXIAL_STEP, use_gpu=true, show_progress=false)
end
println("done.\n")

# ── Scalability sweep ─────────────────────────────────────────────────────────

n_depths    = length(DEPTHS_MM)
n_apertures = length(APERTURES_MM)

timing_wall_s  = fill(NaN, n_depths, n_apertures)
timing_march_s = fill(NaN, n_depths, n_apertures)

t_sweep_start = time()
n_total = n_depths * n_apertures
s_count = 0

for (di, depth_mm) in enumerate(DEPTHS_MM)

    recon_axial_dim = (depth_mm + SKULL_DIST * 1e3) * 1e-3   # skull + depth in metres

    # Build medium for max aperture at this depth; crop for smaller apertures.
    cfg_max = PAMConfig3D(
        dx               = DX,
        dy               = DX,
        dz               = DX,
        axial_dim        = recon_axial_dim,
        transverse_dim_y = maximum(APERTURES_MM) * 1e-3,
        transverse_dim_z = maximum(APERTURES_MM) * 1e-3,
        dt               = DT,
        t_max            = T_MAX,
        receiver_aperture_y = maximum(APERTURES_MM) * 1e-3,
        receiver_aperture_z = maximum(APERTURES_MM) * 1e-3,
    )
    c_max, _, _ = make_pam_medium_3d(cfg_max;
        aberrator=:skull, skull_to_transducer=SKULL_DIST, slice_index_z=SLICE_INDEX)
    Ny_max = pam_Ny(cfg_max)   # 500

    for (ai, apt_mm) in enumerate(APERTURES_MM)

        global s_count += 1
        Ny_tgt = round(Int, apt_mm * 1e-3 / DX)
        @printf("[%2d/%d] depth=%3.0f mm, aperture=%3.0f mm (Ny=Nz=%d, %d march rows) … ",
                s_count, n_total, depth_mm, apt_mm, Ny_tgt,
                round(Int, depth_mm * 1e-3 / AXIAL_STEP))
        flush(stdout)

        # Reconstruction config for this (depth, aperture)
        cfg_recon = PAMConfig3D(
            dx               = DX,
            dy               = DX,
            dz               = DX,
            axial_dim        = recon_axial_dim,
            transverse_dim_y = apt_mm * 1e-3,
            transverse_dim_z = apt_mm * 1e-3,
            dt               = DT,
            t_max            = T_MAX,
            receiver_aperture_y = apt_mm * 1e-3,
            receiver_aperture_z = apt_mm * 1e-3,
        )

        # Crop medium from the max-aperture build (centre slice)
        y_lo = (Ny_max - Ny_tgt) ÷ 2 + 1
        y_hi = y_lo + Ny_tgt - 1
        c_recon = c_max[:, y_lo:y_hi, y_lo:y_hi]

        # RF resized to match aperture (crop or zero-pad from 64 mm base)
        rf_apt = resize_rf(rf_base, Ny_tgt, Ny_tgt)

        # GPU HASA reconstruction
        t_wall = @elapsed begin
            _, _, info = reconstruct_pam_3d(
                rf_apt, c_recon, cfg_recon;
                frequencies   = [FREQ],
                bandwidth     = BANDWIDTH,
                corrected     = true,
                axial_step    = AXIAL_STEP,
                use_gpu       = true,
                show_progress = false,
            )
            timing_march_s[di, ai] = get(get(info, :gpu_timing, Dict()), :march_wall_s, NaN)
        end

        timing_wall_s[di, ai] = t_wall

        @printf("%.2f s  (march %.2f s)\n", t_wall, timing_march_s[di, ai])
    end
end

total_min = (time() - t_sweep_start) / 60
@printf("\nSweep complete in %.1f minutes.\n\n", total_min)

# ── Summary table ─────────────────────────────────────────────────────────────

println("Wall-clock GPU HASA [s]  (rows = depth from skull, cols = aperture)")
@printf("  %10s │", "Depth \\ Apt")
for apt_mm in APERTURES_MM
    @printf("  %5.0f mm", apt_mm)
end
println()
println("  " * "─"^11 * "┼" * "─"^(9 * n_apertures))
for (di, depth_mm) in enumerate(DEPTHS_MM)
    @printf("  %6.0f mm   │", depth_mm)
    for (ai, _) in enumerate(APERTURES_MM)
        @printf("  %7.2f s", timing_wall_s[di, ai])
    end
    println()
end
println()

# ── Save ──────────────────────────────────────────────────────────────────────

outdir  = @__DIR__
ts      = Dates.format(now(), "yyyymmdd_HHMMSS")
outfile = joinpath(outdir, "results_$(ts).jld2")

jldsave(outfile;
    timing_wall_s,
    timing_march_s,
    APERTURES_MM,
    DEPTHS_MM,
    FREQ, NUM_CYCLES, DX, DT, T_MAX, AXIAL_STEP, BANDWIDTH,
    SKULL_DIST, SLICE_INDEX,
    sim_time_s = t_sim,
)
println("Results saved → $outfile")
