"""
Reproduce Fig. 7 / Table II of Schoen & Arvanitis (2020) IEEE TMI.

Point sources on a rectangular grid (axial 30–80 mm, lateral ±20 mm) are
simulated through a human skull with k-Wave.  The PAM is reconstructed with
the homogeneous ASA (geometric) and the heterogeneous ASA (HASA, Eq. 6).
Localization errors are tabulated for three receiver apertures (50, 75, 100 mm).

Each source is simulated once with the largest aperture; smaller apertures are
obtained by zeroing RF outside the requested element range, so reconstruction
is free.

Run from the project root:
    julia --project=. validation/2D_PAM_Accuracy/run_validation.jl [--recon-use-gpu]
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using TranscranialFUS
using Statistics, JLD2, Printf, Dates

# ── Configuration ──────────────────────────────────────────────────────────────

# Grid (matching paper: 200 µm pitch, 40 ns time step)
const DX           = 0.2e-3        # m – axial and lateral spacing
const DT           = 40e-9         # s
const T_MAX        = 100e-6        # s – covers 80 mm source + full 100 mm aperture
const AXIAL_DIM    = 100e-3        # m – total axial extent (skull at 20 mm, sources ≤ 80 mm)
const SKULL_DIST   = 20e-3         # m – outer skull surface below receiver
const SLICE_INDEX  = 250           # CT slice (same default as run_pam.jl)

# Source grid (paper: 30–80 mm axial, ±20 mm lateral, 1 MHz)
const FREQ         = 1.0e6         # Hz
const NUM_CYCLES   = 10            # source duration = 10 µs at 1 MHz
const AXIAL_MM     = [30.0, 40.0, 50.0, 60.0, 70.0, 80.0]   # mm
const LATERAL_MM   = [-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]  # mm

# Apertures to sweep (paper Fig. 7: 50, 75, 100 mm)
const APERTURES_MM = [50.0, 75.0, 100.0]
const MAX_APT_MM   = maximum(APERTURES_MM)

# Reconstruction
const AXIAL_STEP   = 50e-6         # m – Δz in HASA marching (matches paper)
const BANDWIDTH    = 0.0           # Hz – exact bin (no smoothing)

# k-Wave simulation always uses GPU.
# PAM reconstruction uses GPU only when --recon-use-gpu is passed.
const USE_GPU_RECON = "--recon-use-gpu" in ARGS

# ── Domain and medium ──────────────────────────────────────────────────────────

# Transverse domain must fit the largest aperture plus PML guard.
# PML guard at DX = 0.2 mm: default is max(4, round(4 mm / DX)) = 20 cells = 4 mm.
# Use 12 mm margin on each side → total = MAX_APT_MM + 24 mm.
const TRANS_DIM = (MAX_APT_MM + 24.0) * 1e-3

sim_cfg = PAMConfig(
    dx             = DX,
    dz             = DX,
    axial_dim      = AXIAL_DIM,
    transverse_dim = TRANS_DIM,
    receiver_aperture = MAX_APT_MM * 1e-3,
    t_max          = T_MAX,
    dt             = DT,
)

println("Domain: $(round(Int, AXIAL_DIM*1e3)) mm axial × $(round(Int, TRANS_DIM*1e3)) mm lateral")
println("Grid  : $(pam_Nx(sim_cfg)) × $(pam_Ny(sim_cfg)) cells, Nt = $(pam_Nt(sim_cfg))")
println("Active receiver: $(round(Int, MAX_APT_MM)) mm ($(round(Int, MAX_APT_MM*1e-3/DX)) elements)")
println()

println("Building skull medium (CT slice $SLICE_INDEX, skull at $(round(Int,SKULL_DIST*1e3)) mm)…")
c, rho, med_info = make_pam_medium(
    sim_cfg;
    aberrator          = :skull,
    skull_to_transducer = SKULL_DIST,
    slice_index        = SLICE_INDEX,
)
println("  outer skull row : $(med_info[:outer_row])")
println("  inner skull row : $(med_info[:inner_row])")
println("  c₀ (water)      : $(sim_cfg.c0) m/s")
println()

# ── Sweep storage ─────────────────────────────────────────────────────────────

n_ax    = length(AXIAL_MM)
n_lat   = length(LATERAL_MM)
n_apt   = length(APERTURES_MM)
n_total = n_ax * n_lat

# Shape: [n_apt × n_ax × n_lat]
geo_errors_mm  = fill(NaN, n_apt, n_ax, n_lat)
hasa_errors_mm = fill(NaN, n_apt, n_ax, n_lat)
geo_ax_err_mm  = fill(NaN, n_apt, n_ax, n_lat)
hasa_ax_err_mm = fill(NaN, n_apt, n_ax, n_lat)
geo_lat_err_mm = fill(NaN, n_apt, n_ax, n_lat)
hasa_lat_err_mm = fill(NaN, n_apt, n_ax, n_lat)

ny = pam_Ny(sim_cfg)
mid_col = ny ÷ 2 + 1   # center column index

# Pre-compute active column ranges for each aperture.
# Use start = mid - floor(n/2), end = start + n - 1 so the range always has
# exactly n_elem columns regardless of odd/even.
apt_ranges = Dict{Float64, UnitRange{Int}}()
for apt_mm in APERTURES_MM
    n_elem = round(Int, apt_mm * 1e-3 / DX)
    half   = n_elem ÷ 2          # floor division
    astart = mid_col - half
    aend   = astart + n_elem - 1
    (1 <= astart && aend <= ny) ||
        error("Aperture $(apt_mm) mm exceeds transverse domain $(round(Int, TRANS_DIM*1e3)) mm.")
    apt_ranges[apt_mm] = astart:aend
end

# ── Main sweep ────────────────────────────────────────────────────────────────

t_sweep_start = time()
s_count = 0

for (ai, ax_mm) in enumerate(AXIAL_MM), (li, lat_mm) in enumerate(LATERAL_MM)
    global s_count += 1
    elapsed = time() - t_sweep_start
    eta_s = s_count > 1 ? elapsed / (s_count - 1) * (n_total - s_count + 1) : NaN
    eta_str = isnan(eta_s) ? "?" : @sprintf("%.0f", eta_s / 60)
    @printf("[%2d/%d] ax=%4.0fmm lat=%+5.0fmm  (ETA ~%s min)\n",
            s_count, n_total, ax_mm, lat_mm, eta_str)

    src = PointSource2D(
        depth      = ax_mm  * 1e-3,
        lateral    = lat_mm * 1e-3,
        frequency  = FREQ,
        num_cycles = NUM_CYCLES,
    )

    # Simulate once with full (MAX_APT_MM) aperture
    rf_full, kgrid, _ = simulate_point_sources(c, rho, [src], sim_cfg; use_gpu=true)

    for (pi, apt_mm) in enumerate(APERTURES_MM)
        # Mask RF to requested aperture by zeroing outside active range
        apt_range = apt_ranges[apt_mm]
        rf_apt = zeros(Float64, ny, size(rf_full, 2))
        rf_apt[apt_range, :] .= rf_full[apt_range, :]

        # Reconstruct: homogeneous ASA (geo) and heterogeneous ASA (HASA)
        pam_geo,  _, _ = reconstruct_pam(rf_apt, c, sim_cfg;
            corrected             = false,
            frequencies           = [FREQ],
            bandwidth             = BANDWIDTH,
            axial_step            = AXIAL_STEP,
            use_gpu               = USE_GPU_RECON,  # --recon-use-gpu flag
        )
        pam_hasa, _, _ = reconstruct_pam(rf_apt, c, sim_cfg;
            corrected             = true,
            frequencies           = [FREQ],
            bandwidth             = BANDWIDTH,
            axial_step            = AXIAL_STEP,
            use_gpu               = USE_GPU_RECON,  # --recon-use-gpu flag
        )

        # Localization error via existing analysis function (1 source → argmax)
        stats_geo  = analyse_pam_2d(pam_geo,  kgrid, sim_cfg, [src])
        stats_hasa = analyse_pam_2d(pam_hasa, kgrid, sim_cfg, [src])

        geo_errors_mm[pi, ai, li]   = only(stats_geo[:radial_errors_mm])
        hasa_errors_mm[pi, ai, li]  = only(stats_hasa[:radial_errors_mm])
        geo_ax_err_mm[pi, ai, li]   = only(stats_geo[:axial_errors_mm])
        hasa_ax_err_mm[pi, ai, li]  = only(stats_hasa[:axial_errors_mm])
        geo_lat_err_mm[pi, ai, li]  = only(stats_geo[:lateral_errors_mm])
        hasa_lat_err_mm[pi, ai, li] = only(stats_hasa[:lateral_errors_mm])
    end
end

total_time_min = (time() - t_sweep_start) / 60
@printf("\nSweep complete in %.1f minutes.\n\n", total_time_min)

# ── Print results (Table II format) ───────────────────────────────────────────

println("="^72)
println("PAM Localization Error — 2D Skull (Schoen & Arvanitis 2020, Table II)")
println("="^72)
println()
@printf("  Sources: %d axial × %d lateral = %d total\n", n_ax, n_lat, n_total)
@printf("  Frequency: %.1f MHz, Axial step: %.0f µm\n", FREQ/1e6, AXIAL_STEP*1e6)
println()

println("  Aperture │ Uncorrected (geo)   │ Corrected (HASA)")
println("  ─────────┼─────────────────────┼────────────────────")
for (pi, apt_mm) in enumerate(APERTURES_MM)
    geo_vals  = filter(!isnan, geo_errors_mm[pi, :, :])
    hasa_vals = filter(!isnan, hasa_errors_mm[pi, :, :])
    @printf("  %3.0f mm   │ %4.1f ± %4.1f mm       │ %4.1f ± %4.1f mm\n",
            apt_mm,
            mean(geo_vals),  std(geo_vals),
            mean(hasa_vals), std(hasa_vals))
end
println()

println("  Reference (paper, Table II):")
println("  Aperture │ Uncorrected          │ Corrected")
println("  ─────────┼──────────────────────┼──────────────────────")
println("   50 mm   │  3.7 ± 2.2 mm        │  1.2 ± 0.7 mm")
println("   75 mm   │  2.5 ± 1.7 mm        │  0.9 ± 0.5 mm")
println("  100 mm   │  3.5 ± 1.9 mm        │  0.8 ± 0.4 mm")
println()

println("  Per-aperture breakdown (mean |axial| / |lateral| error):")
println("  Aperture │ Method │ |Axial| mm   │ |Lateral| mm")
println("  ─────────┼────────┼──────────────┼─────────────")
for (pi, apt_mm) in enumerate(APERTURES_MM)
    for (method, ax_err, lat_err) in [
            ("geo ",  geo_ax_err_mm,  geo_lat_err_mm),
            ("hasa", hasa_ax_err_mm, hasa_lat_err_mm),
        ]
        ax_vals  = filter(!isnan, abs.(ax_err[pi, :, :]))
        lat_vals = filter(!isnan, abs.(lat_err[pi, :, :]))
        @printf("  %3.0f mm   │ %-6s │ %5.2f ± %4.2f │ %5.2f ± %4.2f\n",
                apt_mm, method, mean(ax_vals), std(ax_vals), mean(lat_vals), std(lat_vals))
    end
end
println()

# ── Save results ──────────────────────────────────────────────────────────────

outdir  = @__DIR__
ts      = Dates.format(now(), "yyyymmdd_HHMMSS")
outfile = joinpath(outdir, "results_$(ts).jld2")

jldsave(outfile;
    geo_errors_mm, hasa_errors_mm,
    geo_ax_err_mm, hasa_ax_err_mm,
    geo_lat_err_mm, hasa_lat_err_mm,
    AXIAL_MM, LATERAL_MM, APERTURES_MM,
    FREQ, AXIAL_STEP, DX, DT, T_MAX,
    AXIAL_DIM, TRANS_DIM, SKULL_DIST, SLICE_INDEX,
    med_info,
    total_time_min,
)
println("Results saved → $outfile")
