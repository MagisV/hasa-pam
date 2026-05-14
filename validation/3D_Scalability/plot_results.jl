"""
Plot 3D HASA GPU scalability results.

Three-panel figure:
  (a) GPU march time vs aperture — one curve per depth
  (b) GPU march time vs depth    — one curve per aperture
  (c) Memory bandwidth vs aperture — shows GPU utilisation trend

OOM points are excluded from curves; a dashed vertical line marks the
first aperture that hit OOM (if any).

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

d = load(results_file)

APERTURES_MM        = d["APERTURES_MM"]
DEPTHS_MM           = d["DEPTHS_MM"]
DX                  = d["DX"]
AXIAL_STEP          = d["AXIAL_STEP"]
timing_march_gpu_s  = get(d, "timing_march_gpu_s",  get(d, "timing_march_s", fill(NaN, length(DEPTHS_MM), length(APERTURES_MM))))
timing_bw_GBps      = get(d, "timing_bw_GBps",      fill(NaN, length(DEPTHS_MM), length(APERTURES_MM)))
oom_flags           = get(d, "oom_flags",            fill(false, length(DEPTHS_MM), length(APERTURES_MM)))

n_depths    = length(DEPTHS_MM)
n_apertures = length(APERTURES_MM)

# ── Derived labels ─────────────────────────────────────────────────────────────

Ny_vals = round.(Int, APERTURES_MM .* 1e-3 ./ DX)
vox_k   = round.(Int, Ny_vals .^ 2 ./ 1000)

apt_tick_labels = [@sprintf("%g mm\n(%d×%d=%dk)", a, n, n, v)
                   for (a, n, v) in zip(APERTURES_MM, Ny_vals, vox_k)]

# ── Print summary table ────────────────────────────────────────────────────────

println()
println("GPU march time [s]  (rows = depth from skull, cols = aperture)")
@printf("  %12s │", "Depth \\ Apt")
for apt_mm in APERTURES_MM; @printf("  %6.0f mm", apt_mm); end
println()
for (di, dep) in enumerate(DEPTHS_MM)
    @printf("  %8.0f mm │", dep)
    for ai in 1:n_apertures
        oom_flags[di, ai] ? @printf("     OOM  ") : @printf("  %7.3f s", timing_march_gpu_s[di, ai])
    end
    println()
end
println()

# ── Colours / markers ─────────────────────────────────────────────────────────

DEPTH_COLORS  = [:steelblue, :coral, :forestgreen, :darkorchid, :darkorange,
                 :crimson, :teal, :goldenrod]
APT_MARKERS   = [:circle, :diamond, :rect, :utriangle, :dtriangle, :star5, :hexagon, :cross]

depth_colors  = DEPTH_COLORS[1:n_depths]
apt_markers   = APT_MARKERS[1:n_apertures]

# First aperture that hit OOM at any depth (for vertical marker)
oom_apt_idx = findfirst(ai -> any(oom_flags[:, ai]), 1:n_apertures)

# ── Figure ────────────────────────────────────────────────────────────────────

update_theme!(fontsize = 9)
fig = Figure(size = (1100, 340))

# ── (a) march_gpu vs aperture ─────────────────────────────────────────────────

ax_a = Axis(fig[1, 1];
    title   = "(a) GPU march time vs aperture",
    xlabel  = "Aperture (Ny × Nz voxels/plane)",
    ylabel  = "GPU march time [s]",
    xticks  = (APERTURES_MM, apt_tick_labels),
    xticklabelsize = 7,
)

for (di, depth_mm) in enumerate(DEPTHS_MM)
    valid = findall(ai -> !oom_flags[di, ai] && !isnan(timing_march_gpu_s[di, ai]), 1:n_apertures)
    isempty(valid) && continue
    label = @sprintf("%g mm depth", depth_mm)
    lines!(ax_a, APERTURES_MM[valid], timing_march_gpu_s[di, valid];
        color=depth_colors[di], linewidth=1.5, label=label)
    scatter!(ax_a, APERTURES_MM[valid], timing_march_gpu_s[di, valid];
        color=depth_colors[di], markersize=6)
end

if oom_apt_idx !== nothing
    vlines!(ax_a, [APERTURES_MM[oom_apt_idx]]; color=:red, linestyle=:dash, linewidth=1,
            label="OOM boundary")
end

axislegend(ax_a; title="Depth\n(below skull)", position=:lt, labelsize=7, titlesize=7)

# ── (b) march_gpu vs depth ────────────────────────────────────────────────────

ax_b = Axis(fig[1, 2];
    title   = "(b) GPU march time vs depth",
    xlabel  = "Reconstruction depth below skull [mm]",
    ylabel  = "GPU march time [s]",
    xticks  = (DEPTHS_MM, string.(round.(Int, DEPTHS_MM))),
)

for (ai, apt_mm) in enumerate(APERTURES_MM)
    valid = findall(di -> !oom_flags[di, ai] && !isnan(timing_march_gpu_s[di, ai]), 1:n_depths)
    isempty(valid) && continue
    Ny = Ny_vals[ai]
    label = @sprintf("%g mm  (%d×%d=%dk)", apt_mm, Ny, Ny, vox_k[ai])
    lines!(ax_b, DEPTHS_MM[valid], timing_march_gpu_s[valid, ai];
        color=depth_colors[min(ai, length(depth_colors))], linewidth=1.5, label=label)
    scatter!(ax_b, DEPTHS_MM[valid], timing_march_gpu_s[valid, ai];
        color=depth_colors[min(ai, length(depth_colors))],
        marker=apt_markers[ai], markersize=6)
end

axislegend(ax_b; title="Aperture\n(voxels/plane)", position=:lt, labelsize=7, titlesize=7)

# ── (c) bandwidth vs aperture ─────────────────────────────────────────────────

ax_c = Axis(fig[1, 3];
    title   = "(c) Memory bandwidth vs aperture",
    xlabel  = "Aperture (Ny × Nz voxels/plane)",
    ylabel  = "Estimated bandwidth [GB/s]",
    xticks  = (APERTURES_MM, apt_tick_labels),
    xticklabelsize = 7,
)

for (di, depth_mm) in enumerate(DEPTHS_MM)
    valid = findall(ai -> !oom_flags[di, ai] && !isnan(timing_bw_GBps[di, ai]), 1:n_apertures)
    isempty(valid) && continue
    lines!(ax_c, APERTURES_MM[valid], timing_bw_GBps[di, valid];
        color=depth_colors[di], linewidth=1.5)
    scatter!(ax_c, APERTURES_MM[valid], timing_bw_GBps[di, valid];
        color=depth_colors[di], markersize=6)
end

if oom_apt_idx !== nothing
    vlines!(ax_c, [APERTURES_MM[oom_apt_idx]]; color=:red, linestyle=:dash, linewidth=1)
end

colgap!(fig.layout, 20)

# ── Save ──────────────────────────────────────────────────────────────────────

figfile = replace(results_file, r"\.jld2$" => "_scalability.pdf")
save(figfile, fig)
println("Saved → $figfile")
