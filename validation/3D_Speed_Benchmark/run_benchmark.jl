"""
3D HASA Speed Benchmark — CPU vs GPU, ASA vs HASA.

A single centre source (50 mm depth, lateral = 0) is simulated once with
k-Wave (GPU).  The RF data is then reconstructed with all four mode
combinations and wall-clock times are recorded for direct comparison:
  CPU ASA  |  CPU HASA
  GPU ASA  |  GPU HASA

Parameters match validation/3D_PAM_Accuracy/run_validation.jl exactly.

Run from the project root:
    julia --project=. validation/3D_Speed_Benchmark/run_benchmark.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using TranscranialFUS
using JLD2, Printf, Dates

# ── Configuration (identical to 3D_PAM_Accuracy) ──────────────────────────────

const FREQ       = 1.0e6
const NUM_CYCLES = 10.0
const DX         = 0.2e-3
const DT         = 40e-9
const T_MAX      = 120e-6
const AXIAL_STEP = 50e-6
const BANDWIDTH  = 0.0

const SKULL_DIST  = 20e-3
const SLICE_INDEX = 250
const APERTURE    = 64e-3

const AXIAL_DIM = 70e-3
const TRANS_DIM = APERTURE

# ── Domain and medium ─────────────────────────────────────────────────────────

sim_cfg = PAMConfig3D(
    dx               = DX,
    dy               = DX,
    dz               = DX,
    axial_dim        = AXIAL_DIM,
    transverse_dim_y = TRANS_DIM,
    transverse_dim_z = TRANS_DIM,
    dt               = DT,
    t_max            = T_MAX,
    receiver_aperture_y = APERTURE,
    receiver_aperture_z = APERTURE,
)

println("Domain  : $(round(Int, AXIAL_DIM*1e3)) mm axial × $(round(Int, TRANS_DIM*1e3)) mm transverse")
println("Grid    : $(pam_Nx(sim_cfg))×$(pam_Ny(sim_cfg))×$(pam_Nz(sim_cfg)) cells, Nt=$(pam_Nt(sim_cfg))")
println("Aperture: $(round(Int, APERTURE*1e3)) mm × $(round(Int, APERTURE*1e3)) mm")
println()

println("Building skull medium (CT slice $SLICE_INDEX, skull at $(round(Int, SKULL_DIST*1e3)) mm)…")
c, rho, med_info = make_pam_medium_3d(
    sim_cfg;
    aberrator           = :skull,
    skull_to_transducer = SKULL_DIST,
    slice_index_z       = SLICE_INDEX,
)
println("  outer skull row : $(med_info[:outer_row])")
println("  inner skull row : $(med_info[:inner_row])")
println()

# ── Single centre source ───────────────────────────────────────────────────────

src = PointSource3D(
    depth      = 50e-3,
    lateral_y  = 0.0,
    lateral_z  = 0.0,
    frequency  = FREQ,
    num_cycles = NUM_CYCLES,
)

# ── k-Wave simulation ─────────────────────────────────────────────────────────

println("Running k-Wave (GPU)…")
t_sim = @elapsed begin
    rf_full, kgrid, _ = simulate_point_sources_3d(c, rho, [src], sim_cfg; use_gpu=true)
end
@printf("  k-Wave wall-clock : %.1f s\n\n", t_sim)

apt_range_y = receiver_col_range_y(sim_cfg)
apt_range_z = receiver_col_range_z(sim_cfg)
rf = zeros(eltype(rf_full), size(rf_full))
rf[apt_range_y, apt_range_z, :] .= rf_full[apt_range_y, apt_range_z, :]
rf_full = nothing

# ── Four reconstructions ──────────────────────────────────────────────────────

const MODES = [
    (use_gpu=false, corrected=false, label="CPU_ASA"),
    (use_gpu=false, corrected=true,  label="CPU_HASA"),
    (use_gpu=true,  corrected=false, label="GPU_ASA"),
    (use_gpu=true,  corrected=true,  label="GPU_HASA"),
]

intensities = Dict{String,Array{Float32,3}}()
info_dicts  = Dict{String,Any}()
wall_times  = Dict{String,Float64}()

println("Reconstructions:")
for m in MODES
    @printf("  %-10s … ", m.label)
    flush(stdout)
    t = @elapsed begin
        I, _, info = reconstruct_pam_3d(
            rf, c, sim_cfg;
            frequencies   = [FREQ],
            bandwidth     = BANDWIDTH,
            corrected     = m.corrected,
            axial_step    = AXIAL_STEP,
            use_gpu       = m.use_gpu,
            show_progress = false,
        )
        intensities[m.label] = I
        info_dicts[m.label]  = info
    end
    wall_times[m.label] = t
    @printf("%.2f s\n", t)
end

# ── Summary ───────────────────────────────────────────────────────────────────

println()
println("="^52)
println("3D PAM Speed Benchmark — $(round(Int, APERTURE*1e3)) mm aperture, source at 50 mm")
println("="^52)
println()
println("  Mode       │ Wall-clock")
println("  ───────────┼───────────")
for m in MODES
    @printf("  %-10s │ %7.2f s\n", m.label, wall_times[m.label])
end
println()
@printf("  GPU/CPU speedup (HASA): %.1f×\n",
        wall_times["CPU_HASA"] / wall_times["GPU_HASA"])
@printf("  GPU/CPU speedup (ASA) : %.1f×\n",
        wall_times["CPU_ASA"]  / wall_times["GPU_ASA"])
println()

# ── Save ──────────────────────────────────────────────────────────────────────

outdir  = @__DIR__
ts      = Dates.format(now(), "yyyymmdd_HHMMSS")
outfile = joinpath(outdir, "results_$(ts).jld2")

jldsave(outfile;
    intensities,
    wall_times,
    info_dicts,
    sim_time_s = t_sim,
    FREQ, NUM_CYCLES, DX, DT, T_MAX, AXIAL_STEP, BANDWIDTH,
    AXIAL_DIM, TRANS_DIM, APERTURE, SKULL_DIST, SLICE_INDEX,
    med_info,
)
println("Results saved → $outfile")
