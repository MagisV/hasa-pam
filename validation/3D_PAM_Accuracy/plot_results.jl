"""
Plot 3D PAM localization accuracy from saved validation results.

Layout matches Fig. 2 in the paper:
  • Receiver aperture outline at the TOP  (depth = 0)
  • Source error bubbles below, sized by radial localisation error
  • Dashed stem lines from each source up to skull depth as reference

Coordinate mapping:
  Plot X = lateral Y (simulation),  Plot Y = lateral Z (simulation),
  Plot Z = –depth  →  receiver at Z = 0 (top), deep sources at Z ≈ –60 (bottom).
  Z tick labels are shown as positive depth values in mm.

Saved as PNG (not PDF) to avoid 800+ MB vector output.

Usage:
    julia --project=. validation/3D_PAM_Accuracy/plot_results.jl [results_file.jld2]
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CairoMakie, JLD2, Statistics, Printf

# ── Load results ──────────────────────────────────────────────────────────────

results_dir  = @__DIR__
results_file = if !isempty(ARGS)
    ARGS[1]
else
    files = filter(f -> startswith(f, "results_") && endswith(f, ".jld2"), readdir(results_dir))
    isempty(files) && error("No results_*.jld2 in $results_dir. Run run_validation.jl first.")
    joinpath(results_dir, last(sort(files)))
end
println("Loading: $results_file")

d = load(results_file)
geo_errors_mm  = d["geo_errors_mm"]
hasa_errors_mm = d["hasa_errors_mm"]
AXIAL_MM       = d["AXIAL_MM"]
LATERAL_Y_MM   = d["LATERAL_Y_MM"]
LATERAL_Z_MM   = d["LATERAL_Z_MM"]
APERTURE_MM    = d["APERTURE_MM"]
FREQ           = d["FREQ"]
DX             = d["DX"]
DT             = d["DT"]
T_MAX          = d["T_MAX"]
AXIAL_DIM      = d["AXIAL_DIM"]
TRANS_DIM      = d["TRANS_DIM"]
SKULL_DIST     = d["SKULL_DIST"]
SLICE_INDEX    = d["SLICE_INDEX"]

# ── Print summary ─────────────────────────────────────────────────────────────

geo_all  = filter(!isnan, geo_errors_mm)
hasa_all = filter(!isnan, hasa_errors_mm)
println()
println("  Method        │ Mean ± Std error")
println("  ──────────────┼──────────────────────")
@printf("  Geometric ASA │ %.2f ± %.2f mm (n=%d)\n", mean(geo_all),  std(geo_all),  length(geo_all))
@printf("  HASA          │ %.2f ± %.2f mm (n=%d)\n", mean(hasa_all), std(hasa_all), length(hasa_all))
println()

# ── Coordinate mapping ────────────────────────────────────────────────────────
#
#  Plot axis → simulation quantity
#  X  =  lateral Y   (horizontal left–right)
#  Y  =  lateral Z   (horizontal front–back)
#  Z  = –depth       (vertical: 0 = receiver plane at top, negative = deeper)
#
#  This puts the receiver plate on top, sources below — matching Fig. 2 geometry.

skull_plot_z = -SKULL_DIST * 1e3   # e.g. –20 mm (used as stem reference)

# ── Error → visual encoding ───────────────────────────────────────────────────

MAX_ERR  = 6.0
MIN_MS   = 8.0
MAX_MS   = 30.0
errcmap  = :plasma

err_to_ms(e) = MIN_MS + (MAX_MS - MIN_MS) * clamp(e, 0.0, MAX_ERR) / MAX_ERR

# Z axis ticks: actual values are negative, labels shown as positive depth
max_depth     = maximum(AXIAL_MM)
tick_depths   = 0.0:10.0:max_depth              # positive depth values
ztick_vals    = collect(-tick_depths)            # e.g. [0, -10, -20, ..., -60]
ztick_labels  = [string(Int(d)) for d in tick_depths]

# ── Figure ────────────────────────────────────────────────────────────────────

update_theme!(fontsize = 10)
fig = Figure(size = (1000, 520))

panels = [("Geometric ASA", geo_errors_mm), ("HASA", hasa_errors_mm)]

for (panel_col, (panel_label, errors)) in enumerate(panels)
    ax = Axis3(fig[1, panel_col];
        xlabel          = "Lateral Y [mm]",
        ylabel          = "Lateral Z [mm]",
        zlabel          = "Depth [mm]",
        title           = panel_label,
        aspect          = :data,
        perspectiveness = 0.35,
        azimuth         = 5π / 4,   # standard isometric-ish: both lateral axes visible
        elevation       = π / 5,    # ~36° top-down so depth separation is clear
        zticks          = (ztick_vals, ztick_labels),
        xlabeloffset    = 40,
        ylabeloffset    = 40,
        zlabeloffset    = 55,
    )

    # Receiver aperture outline at top (Z = 0)
    half = APERTURE_MM / 2.0
    apy  = [-half,  half, half, -half, -half]
    apz  = [-half, -half, half,  half, -half]
    lines!(ax, apy, apz, fill(0.0, 5); color = :black, linewidth = 1.4)

    # Collect sources in plot coordinates
    xs_plot = Float64[]   # lateral Y
    ys_plot = Float64[]   # lateral Z
    zs_plot = Float64[]   # -depth
    cs      = Float64[]
    ms      = Float64[]
    for (ai, ax_mm) in enumerate(AXIAL_MM),
        (yi, ly_mm) in enumerate(LATERAL_Y_MM),
        (zi, lz_mm) in enumerate(LATERAL_Z_MM)
        e = errors[ai, yi, zi]
        isnan(e) && continue
        push!(xs_plot, Float64(ly_mm))
        push!(ys_plot, Float64(lz_mm))
        push!(zs_plot, -Float64(ax_mm))
        push!(cs, clamp(e, 0.0, MAX_ERR))
        push!(ms, err_to_ms(e))
    end

    # Dashed stem lines from each source UP to the skull plane
    for (xp, yp, zp) in zip(xs_plot, ys_plot, zs_plot)
        lines!(ax, [xp, xp], [yp, yp], [skull_plot_z, zp];
               color = :gray55, linewidth = 0.9, linestyle = :dash)
    end
    # Cross markers at skull level to anchor the stems
    scatter!(ax, xs_plot, ys_plot, fill(skull_plot_z, length(xs_plot));
             marker = :cross, markersize = 7, color = :gray45, strokewidth = 0)

    # Error bubbles (on top of stems)
    scatter!(ax, xs_plot, ys_plot, zs_plot;
             color       = cs,
             colorrange  = (0.0, MAX_ERR),
             colormap    = errcmap,
             markersize  = ms,
             strokewidth = 0.8,
             strokecolor = :black)
end

# Shared error colorbar
Colorbar(fig[1, 3];
    colormap      = errcmap,
    colorrange    = (0.0, MAX_ERR),
    label         = "Radial Error [mm]",
    width         = 14,
    ticklabelsize = 9,
    labelsize     = 10,
    ticks         = 0:1:Int(MAX_ERR),
)

# Bubble size legend
leg_ax = Axis(fig[2, 1:2]; height = 44, aspect = nothing,
              xautolimitmargin = (0,0), yautolimitmargin = (0,0))
hidedecorations!(leg_ax); hidespines!(leg_ax)
for (i, e) in enumerate([0.0, 2.0, 4.0, 6.0])
    scatter!(leg_ax, [i * 70.0], [22.0];
             color = :gray80, markersize = err_to_ms(e),
             strokewidth = 0.8, strokecolor = :black)
    text!(leg_ax, i * 70.0, 3.0; text = "$(Int(e)) mm",
          align = (:center, :bottom), fontsize = 8)
end
text!(leg_ax, 8.0, 22.0; text = "Bubble size = error:",
      align = (:left, :center), fontsize = 8)
xlims!(leg_ax, 0, 330); ylims!(leg_ax, 0, 44)

rowsize!(fig.layout, 2, Fixed(44))
colgap!(fig.layout, 6)
rowgap!(fig.layout, 4)

# ── Save as PNG ───────────────────────────────────────────────────────────────

figfile = replace(results_file, r"\.jld2$" => "_accuracy3d.png")
save(figfile, fig; px_per_unit = 2)
println("Saved → $figfile")

# ── Write results.md ──────────────────────────────────────────────────────────

md_path = joinpath(results_dir, "results.md")
open(md_path, "w") do io
    println(io, "# 3D PAM Accuracy Validation\n")
    println(io, "**Aperture:** $(round(Int, APERTURE_MM)) mm × $(round(Int, APERTURE_MM)) mm  ")
    n_src = length(AXIAL_MM) * length(LATERAL_Y_MM) * length(LATERAL_Z_MM)
    println(io, "**Sources:** $(length(AXIAL_MM)) axial × $(length(LATERAL_Y_MM)) y × $(length(LATERAL_Z_MM)) z = $n_src total  ")
    println(io, "**Frequency:** $(FREQ/1e6) MHz  ")
    println(io, "**Grid:** $(round(Int, DX*1e6)) µm isotropic  ")
    println(io, "**Skull:** CT slice $SLICE_INDEX, outer surface at $(round(Int, SKULL_DIST*1e3)) mm  \n")
    println(io, "| Method        | Mean error | Std error | n |")
    println(io, "|---------------|-----------|-----------|---|")
    @printf(io, "| Geometric ASA | %.2f mm   | %.2f mm   | %d |\n", mean(geo_all),  std(geo_all),  length(geo_all))
    @printf(io, "| HASA          | %.2f mm   | %.2f mm   | %d |\n", mean(hasa_all), std(hasa_all), length(hasa_all))
    println(io, "\n## Per-depth breakdown\n")
    println(io, "| Depth | Geo mean±std   | HASA mean±std  |")
    println(io, "|-------|----------------|----------------|")
    for (ai, ax_mm) in enumerate(AXIAL_MM)
        gv = filter(!isnan, geo_errors_mm[ai, :, :])
        hv = filter(!isnan, hasa_errors_mm[ai, :, :])
        @printf(io, "| %3.0f mm | %.2f ± %.2f mm | %.2f ± %.2f mm |\n",
                ax_mm, mean(gv), std(gv), mean(hv), std(hv))
    end
    println(io, "\n*Figure: $(basename(figfile))*")
end
println("Written → $md_path")
