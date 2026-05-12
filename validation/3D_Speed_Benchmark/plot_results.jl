"""
Plot 3D PAM speed benchmark results.

Produces a figure with:
  - 2×2 grid: Y-Z cross-section at source depth for each mode
              (CPU ASA | GPU ASA) / (CPU HASA | GPU HASA)
  - Bar chart: wall-clock reconstruction time per mode (log scale)

Usage:
    julia --project=. validation/3D_Speed_Benchmark/plot_results.jl [results_file.jld2]
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CairoMakie, JLD2, TranscranialFUS, Printf

# ── Load ──────────────────────────────────────────────────────────────────────

results_dir  = @__DIR__
results_file = if !isempty(ARGS)
    ARGS[1]
else
    files = filter(f -> startswith(f, "results_") && endswith(f, ".jld2"), readdir(results_dir))
    isempty(files) && error("No results_*.jld2 in $results_dir. Run run_benchmark.jl first.")
    joinpath(results_dir, last(sort(files)))
end
println("Loading: $results_file")

d           = load(results_file)
intensities = d["intensities"]
wall_times  = d["wall_times"]
DX          = d["DX"]
DT          = d["DT"]
T_MAX       = d["T_MAX"]
APERTURE    = d["APERTURE"]
AXIAL_DIM   = d["AXIAL_DIM"]
TRANS_DIM   = d["TRANS_DIM"]
SKULL_DIST  = d["SKULL_DIST"]
SLICE_INDEX = d["SLICE_INDEX"]
FREQ        = d["FREQ"]
NUM_CYCLES  = d["NUM_CYCLES"]

# ── Rebuild config for coordinate arrays ──────────────────────────────────────

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

rr     = receiver_row(sim_cfg)
Ny     = pam_Ny(sim_cfg)
Nz     = pam_Nz(sim_cfg)
dx_mm  = DX * 1e3

y_coords = [(j - (Ny ÷ 2 + 1)) * dx_mm for j in 1:Ny]
z_coords = [(k - (Nz ÷ 2 + 1)) * dx_mm for k in 1:Nz]

# Source is at 50 mm depth from receiver
src_row = rr + round(Int, 50e-3 / DX)
src_row = clamp(src_row, 1, pam_Nx(sim_cfg))

# ── Print summary ─────────────────────────────────────────────────────────────

LABELS = ["CPU_ASA", "GPU_ASA", "CPU_HASA", "GPU_HASA"]

println()
println("  Mode       │ Wall-clock")
println("  ───────────┼───────────")
for lbl in LABELS
    @printf("  %-10s │ %7.2f s\n", lbl, wall_times[lbl])
end
println()

# ── Figure ────────────────────────────────────────────────────────────────────

update_theme!(fontsize = 9)

fig = Figure(size = (900, 720))

PANEL_LABELS = ["CPU — Geometric ASA", "GPU — Geometric ASA",
                "CPU — HASA",          "GPU — HASA"]
PANEL_KEYS   = ["CPU_ASA", "GPU_ASA", "CPU_HASA", "GPU_HASA"]
POSITIONS    = [(1, 1), (1, 2), (2, 1), (2, 2)]

for (lbl, title, pos) in zip(PANEL_KEYS, PANEL_LABELS, POSITIONS)
    I     = intensities[lbl]
    slice = I[src_row, :, :]          # (Ny, Nz) — Y-Z plane at source depth
    I_max = max(maximum(slice), 1f-30)
    slice_db = 20f0 .* log10.(slice ./ I_max .+ 1f-10)

    ax = Axis(fig[pos[1], pos[2]];
        title   = @sprintf("%s\n%.2f s", title, wall_times[lbl]),
        xlabel  = "Lateral Z [mm]",
        ylabel  = "Lateral Y [mm]",
        aspect  = DataAspect(),
    )
    heatmap!(ax, z_coords, y_coords, slice_db';
        colorrange = (-40, 0),
        colormap   = :inferno,
    )
end

# Shared colorbar
Colorbar(fig[1:2, 3];
    colorrange    = (-40, 0),
    colormap      = :inferno,
    label         = "Normalised intensity [dB]",
    width         = 14,
    ticklabelsize = 8,
    labelsize     = 9,
    ticks         = -40:10:0,
)

# Timing bar chart
ax_bar = Axis(fig[3, 1:2];
    title   = "Reconstruction wall-clock time",
    ylabel  = "Time [s]",
    yscale  = log10,
    xticks  = (1:4, LABELS),
    xticklabelsize = 9,
)

bar_vals = [wall_times[l] for l in LABELS]
bar_colors = [:steelblue, :coral, :steelblue, :coral]
barplot!(ax_bar, 1:4, bar_vals; color = bar_colors, strokewidth = 0.5, strokecolor = :black)

# Annotate each bar with its value
for (i, v) in enumerate(bar_vals)
    text!(ax_bar, i, v * 1.3;
        text      = @sprintf("%.1f s", v),
        align     = (:center, :bottom),
        fontsize  = 8,
    )
end

Legend(fig[3, 3],
    [PolyElement(color=:steelblue), PolyElement(color=:coral)],
    ["ASA", "HASA"];
    framevisible = false,
    labelsize    = 9,
)

rowgap!(fig.layout, 10)
colgap!(fig.layout, 8)
rowsize!(fig.layout, 3, Relative(0.28))

# ── Save ──────────────────────────────────────────────────────────────────────

figfile = replace(results_file, r"\.jld2$" => "_speed_benchmark.pdf")
save(figfile, fig)
println("Saved → $figfile")
