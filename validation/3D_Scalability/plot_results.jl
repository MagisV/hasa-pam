"""
Plot 3D HASA GPU scalability results.

Two-panel figure:
  (a) Wall-clock time vs aperture — one curve per depth, x-axis annotated
      with Ny×Nz voxels per plane (the FFT/memory-bandwidth bottleneck).
  (b) Wall-clock time vs reconstruction depth — one curve per aperture,
      showing the linear growth from increased marching-step count.

Usage:
    julia --project=. validation/3D_Scalability/plot_results.jl [results_file.jld2]
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CairoMakie, JLD2, Printf

# ── Load ──────────────────────────────────────────────────────────────────────

results_dir  = @__DIR__
results_file = if !isempty(ARGS)
    ARGS[1]
else
    files = filter(f -> startswith(f, "results_") && endswith(f, ".jld2"), readdir(results_dir))
    isempty(files) && error("No results_*.jld2 in $results_dir. Run run_scalability.jl first.")
    joinpath(results_dir, last(sort(files)))
end
println("Loading: $results_file")

d              = load(results_file)
timing_wall_s  = d["timing_wall_s"]    # (n_depths × n_apertures)
APERTURES_MM   = d["APERTURES_MM"]
DEPTHS_MM      = d["DEPTHS_MM"]
DX             = d["DX"]
AXIAL_STEP     = d["AXIAL_STEP"]

n_depths    = length(DEPTHS_MM)
n_apertures = length(APERTURES_MM)

# ── Derived labels ─────────────────────────────────────────────────────────────

Ny_vals = round.(Int, APERTURES_MM .* 1e-3 ./ DX)
vox_k   = round.(Int, Ny_vals .^ 2 ./ 1000)
march_steps = round.(Int, DEPTHS_MM .* 1e-3 ./ AXIAL_STEP)

# Tick labels for aperture axis: "50 mm\n(250×250=62k)"
apt_tick_labels = [@sprintf("%d mm\n(%d×%d=%dk)", round(Int,a), n, n, v)
                   for (a, n, v) in zip(APERTURES_MM, Ny_vals, vox_k)]

# Print summary table
println()
println("Wall-clock GPU HASA [s]  (rows = depth from skull, cols = aperture)")
@printf("  %12s │", "Depth \\ Apt")
for apt_mm in APERTURES_MM
    @printf("  %5.0f mm", apt_mm)
end
println()
for (di, depth_mm) in enumerate(DEPTHS_MM)
    @printf("  %8.0f mm │", depth_mm)
    for ai in 1:n_apertures
        @printf("   %6.2f s", timing_wall_s[di, ai])
    end
    println()
end
println()

# ── Figure ────────────────────────────────────────────────────────────────────

update_theme!(fontsize = 9)

DEPTH_COLORS  = [:steelblue, :coral, :forestgreen, :darkorchid, :darkorange]
APT_MARKERS   = [:circle, :diamond, :rect, :utriangle]

fig = Figure(size = (820, 370))

# ── (a) Time vs aperture (one curve per depth) ─────────────────────────────────

ax_l = Axis(fig[1, 1];
    title   = "(a) Scaling with transverse grid size",
    xlabel  = "Aperture [mm]  (Ny × Nz voxels/plane)",
    ylabel  = "GPU HASA wall-clock [s]",
    xticks  = (APERTURES_MM, apt_tick_labels),
    xticklabelsize = 8,
)

for (di, depth_mm) in enumerate(DEPTHS_MM)
    label = @sprintf("%d mm depth", round(Int, depth_mm))
    lines!(ax_l, APERTURES_MM, timing_wall_s[di, :];
        label     = label,
        color     = DEPTH_COLORS[di],
        linewidth = 1.5,
    )
    scatter!(ax_l, APERTURES_MM, timing_wall_s[di, :];
        color      = DEPTH_COLORS[di],
        markersize = 7,
    )
end

axislegend(ax_l; title = "Recon depth\n(below skull)", position = :lt,
           labelsize = 8, titlesize = 8)

# ── (b) Time vs depth (one curve per aperture) ─────────────────────────────────

ax_r = Axis(fig[1, 2];
    title   = "(b) Scaling with reconstruction depth",
    xlabel  = "Reconstruction depth below skull [mm]",
    ylabel  = "GPU HASA wall-clock [s]",
    xticks  = (DEPTHS_MM, string.(round.(Int, DEPTHS_MM))),
)

for (ai, apt_mm) in enumerate(APERTURES_MM)
    Ny = Ny_vals[ai]
    label = @sprintf("%d mm  (%d×%d = %dk vox)", round(Int,apt_mm), Ny, Ny, vox_k[ai])
    lines!(ax_r, DEPTHS_MM, timing_wall_s[:, ai];
        label     = label,
        color     = DEPTH_COLORS[ai],
        linewidth = 1.5,
    )
    scatter!(ax_r, DEPTHS_MM, timing_wall_s[:, ai];
        color      = DEPTH_COLORS[ai],
        marker     = APT_MARKERS[ai],
        markersize = 7,
    )
end

axislegend(ax_r; title = "Aperture\n(voxels/plane)", position = :lt,
           labelsize = 8, titlesize = 8)

colgap!(fig.layout, 22)

# ── Save ──────────────────────────────────────────────────────────────────────

figfile = replace(results_file, r"\.jld2$" => "_scalability.pdf")
save(figfile, fig)
println("Saved → $figfile")
