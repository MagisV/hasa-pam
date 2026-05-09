"""
3D PAM localization accuracy — single fixed aperture.

27 point sources (3 axial × 3 lateral-y × 3 lateral-z) are simulated through
a skull CT extruded to 3D with k-Wave.  PAM is reconstructed with geometric
ASA and HASA for a fixed 64 mm square aperture.  Radial localization errors
and GPU reconstruction timings are recorded.

Run from the project root:
    julia --project=. validation/3D_PAM_Accuracy/run_validation.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using TranscranialFUS
using Statistics, JLD2, Printf, Dates

# ── Configuration ──────────────────────────────────────────────────────────────

const FREQ       = 1.0e6       # Hz
const NUM_CYCLES = 10.0        # source duration = 10 µs at 1 MHz
const DX         = 0.2e-3     # m — isotropic grid spacing
const DT         = 40e-9      # s
const T_MAX      = 120e-6     # s — covers deepest source + diagonal aperture path
const AXIAL_STEP = 50e-6      # m — HASA march step (matches 2D validation)
const BANDWIDTH  = 0.0        # Hz — exact bin

const SKULL_DIST  = 20e-3    # m — outer skull surface to receiver
const SLICE_INDEX = 250       # CT slice (same as 2D validation)

const APERTURE_MM = 64.0      # mm — fixed square aperture in y and z

# Source grid: 3 × 3 × 3 = 27 sources
const AXIAL_MM    = [40.0, 50.0, 60.0]
const LATERAL_Y_MM = [-10.0, 0.0, 10.0]
const LATERAL_Z_MM = [-10.0, 0.0, 10.0]

# Transverse domain equals the aperture; k-Wave adds PML outside (pml_inside=false)
const TRANS_DIM = APERTURE_MM * 1e-3   # 64 mm

# Axial domain: skull at 20 mm + deepest source at 60 mm + 10 mm margin
const AXIAL_DIM = 70e-3   # m

# ── Domain and medium ──────────────────────────────────────────────────────────

sim_cfg = PAMConfig3D(
    dx               = DX,
    dy               = DX,
    dz               = DX,
    axial_dim        = AXIAL_DIM,
    transverse_dim_y = TRANS_DIM,
    transverse_dim_z = TRANS_DIM,
    dt               = DT,
    t_max            = T_MAX,
    receiver_aperture_y = APERTURE_MM * 1e-3,
    receiver_aperture_z = APERTURE_MM * 1e-3,
)

println("Domain: $(round(Int, AXIAL_DIM*1e3)) mm axial × $(round(Int, TRANS_DIM*1e3)) mm lateral (y & z)")
println("Grid  : $(pam_Nx(sim_cfg))×$(pam_Ny(sim_cfg))×$(pam_Nz(sim_cfg)) cells, Nt=$(pam_Nt(sim_cfg))")
println("Aperture: $(round(Int, APERTURE_MM)) mm × $(round(Int, APERTURE_MM)) mm")
println()

println("Building 3D skull medium (CT slice $SLICE_INDEX, skull at $(round(Int, SKULL_DIST*1e3)) mm)…")
c, rho, med_info = make_pam_medium_3d(
    sim_cfg;
    aberrator           = :skull,
    skull_to_transducer = SKULL_DIST,
    slice_index_z       = SLICE_INDEX,
)
println("  outer skull row : $(med_info[:outer_row])")
println("  inner skull row : $(med_info[:inner_row])")
println("  c₀ (water)      : $(sim_cfg.c0) m/s")
println()

# ── Aperture mask ─────────────────────────────────────────────────────────────

# The simulation always captures the full ny×nz receiver plane.  Restrict to the
# requested aperture by zeroing RF outside the active element range before
# passing to reconstruction, exactly as the 2D validation does.
apt_range_y = receiver_col_range_y(sim_cfg)
apt_range_z = receiver_col_range_z(sim_cfg)
println("Active receiver columns: y=$(apt_range_y), z=$(apt_range_z)")
println()

# ── Storage ────────────────────────────────────────────────────────────────────

n_ax  = length(AXIAL_MM)
n_y   = length(LATERAL_Y_MM)
n_z   = length(LATERAL_Z_MM)

geo_errors_mm    = fill(NaN, n_ax, n_y, n_z)
hasa_errors_mm   = fill(NaN, n_ax, n_y, n_z)
geo_recon_ms     = fill(NaN, n_ax, n_y, n_z)
hasa_recon_ms    = fill(NaN, n_ax, n_y, n_z)
sim_time_s       = fill(NaN, n_ax, n_y, n_z)

# Progress file: written after every source so a crash loses at most one run.
const PROGRESS_FILE = joinpath(@__DIR__, "progress.jld2")

function save_progress(;total_elapsed_min=NaN)
    jldsave(PROGRESS_FILE;
        geo_errors_mm, hasa_errors_mm,
        geo_recon_ms, hasa_recon_ms,
        sim_time_s,
        AXIAL_MM, LATERAL_Y_MM, LATERAL_Z_MM,
        APERTURE_MM, FREQ, NUM_CYCLES, AXIAL_STEP,
        DX, DT, T_MAX, AXIAL_DIM, TRANS_DIM,
        SKULL_DIST, SLICE_INDEX,
        med_info,
        total_time_min=total_elapsed_min,
    )
end

# ── Main sweep ────────────────────────────────────────────────────────────────

t_sweep_start = time()
n_total = n_ax * n_y * n_z
s_count = 0

for (ai, ax_mm) in enumerate(AXIAL_MM),
    (yi, ly_mm) in enumerate(LATERAL_Y_MM),
    (zi, lz_mm) in enumerate(LATERAL_Z_MM)

    global s_count += 1
    elapsed = time() - t_sweep_start
    eta_s = s_count > 1 ? elapsed / (s_count - 1) * (n_total - s_count + 1) : NaN
    eta_str = isnan(eta_s) ? "?" : @sprintf("%.0f", eta_s / 60)
    @printf("[%2d/%d] ax=%4.0fmm ly=%+5.1fmm lz=%+5.1fmm  (ETA ~%s min)\n",
            s_count, n_total, ax_mm, ly_mm, lz_mm, eta_str)

    src = PointSource3D(
        depth      = ax_mm  * 1e-3,
        lateral_y  = ly_mm  * 1e-3,
        lateral_z  = lz_mm  * 1e-3,
        frequency  = FREQ,
        num_cycles = NUM_CYCLES,
    )

    # 1. k-Wave simulation (GPU)
    t_sim_start = time()
    rf_full, kgrid, _ = simulate_point_sources_3d(c, rho, [src], sim_cfg; use_gpu=true)
    sim_time_s[ai, yi, zi] = time() - t_sim_start

    # Apply aperture mask: zero RF outside the 64 mm active element window
    rf = zeros(eltype(rf_full), size(rf_full))
    rf[apt_range_y, apt_range_z, :] .= rf_full[apt_range_y, apt_range_z, :]
    rf_full = nothing   # free memory

    # 2. Geometric ASA (corrected=false, GPU)
    I_geo, _, geo_info = reconstruct_pam_3d(
        rf, c, sim_cfg;
        frequencies = [FREQ],
        bandwidth   = BANDWIDTH,
        corrected   = false,
        axial_step  = AXIAL_STEP,
        use_gpu     = true,
        show_progress = false,
    )
    geo_recon_ms[ai, yi, zi] = geo_info[:gpu_timing][:march_wall_s] * 1e3
    stats_geo = analyse_pam_3d(I_geo, kgrid, sim_cfg, [src])
    geo_errors_mm[ai, yi, zi] = only(stats_geo[:radial_errors_mm])

    # 3. HASA (corrected=true, GPU)
    I_hasa, _, hasa_info = reconstruct_pam_3d(
        rf, c, sim_cfg;
        frequencies = [FREQ],
        bandwidth   = BANDWIDTH,
        corrected   = true,
        axial_step  = AXIAL_STEP,
        use_gpu     = true,
        show_progress = false,
    )
    hasa_recon_ms[ai, yi, zi] = hasa_info[:gpu_timing][:march_wall_s] * 1e3
    stats_hasa = analyse_pam_3d(I_hasa, kgrid, sim_cfg, [src])
    hasa_errors_mm[ai, yi, zi] = only(stats_hasa[:radial_errors_mm])

    @printf("    geo=%.2f mm  hasa=%.2f mm  (geo %.0f ms / hasa %.0f ms GPU march)\n",
            geo_errors_mm[ai, yi, zi], hasa_errors_mm[ai, yi, zi],
            geo_recon_ms[ai, yi, zi], hasa_recon_ms[ai, yi, zi])

    save_progress()
    @printf("    [progress saved — %d/%d done]\n", s_count, n_total)
end

total_time_min = (time() - t_sweep_start) / 60
@printf("\nSweep complete in %.1f minutes.\n\n", total_time_min)

# ── Print summary ─────────────────────────────────────────────────────────────

println("="^68)
println("3D PAM Localization Error — Fixed $(round(Int, APERTURE_MM)) mm Aperture")
println("="^68)
println()
@printf("  Sources: %d ax × %d y × %d z = %d total\n", n_ax, n_y, n_z, n_total)
@printf("  Frequency: %.1f MHz, Axial step: %.0f µm\n", FREQ/1e6, AXIAL_STEP*1e6)
println()

geo_all  = filter(!isnan, geo_errors_mm)
hasa_all = filter(!isnan, hasa_errors_mm)
@printf("  Geometric ASA : %.2f ± %.2f mm (mean ± std, n=%d)\n",
        mean(geo_all), std(geo_all), length(geo_all))
@printf("  HASA          : %.2f ± %.2f mm (mean ± std, n=%d)\n",
        mean(hasa_all), std(hasa_all), length(hasa_all))
println()
println("  Per-depth breakdown:")
println("  Depth │ Geo error mm   │ HASA error mm  │ Geo ms │ HASA ms")
println("  ──────┼────────────────┼────────────────┼────────┼────────")
for (ai, ax_mm) in enumerate(AXIAL_MM)
    gv = filter(!isnan, geo_errors_mm[ai, :, :])
    hv = filter(!isnan, hasa_errors_mm[ai, :, :])
    gm = filter(!isnan, geo_recon_ms[ai, :, :])
    hm = filter(!isnan, hasa_recon_ms[ai, :, :])
    @printf("  %3.0fmm │ %4.2f ± %4.2f mm  │ %4.2f ± %4.2f mm  │ %6.1f │ %6.1f\n",
            ax_mm, mean(gv), std(gv), mean(hv), std(hv), mean(gm), mean(hm))
end
println()

# ── Save results ──────────────────────────────────────────────────────────────

outdir  = @__DIR__
ts      = Dates.format(now(), "yyyymmdd_HHMMSS")
outfile = joinpath(outdir, "results_$(ts).jld2")

save_progress(total_elapsed_min=total_time_min)
mv(PROGRESS_FILE, outfile; force=true)
println("Results saved → $outfile")
