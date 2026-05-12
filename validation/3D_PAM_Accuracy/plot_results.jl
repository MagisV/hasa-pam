"""
Plot 3D PAM localization accuracy from saved validation results.

Layout mirrors the 2D validation figure:
  Row 1 (top)    – Geometric ASA  (uncorrected)
  Row 2 (bottom) – HASA           (corrected)
  Columns        – one per lateral-Z slice

Each panel: lateral Y on x-axis, axial depth on y-axis (shallow → deep).
Bubble size and colour both encode radial localisation error.

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

# ── Error colour / size helpers ───────────────────────────────────────────────

MAX_ERR = 6.0
MIN_MS  = 2.5
MAX_MS  = 15.0
errcmap = :plasma

err_to_ms(e) = MIN_MS + (MAX_MS - MIN_MS) * clamp(e, 0.0, MAX_ERR) / MAX_ERR

function plot_error_grid!(ax, errors_slice, axial_mm, lateral_mm)
    for (ai, az) in enumerate(axial_mm), (li, lat) in enumerate(lateral_mm)
        e = errors_slice[ai, li]
        isnan(e) && continue
        scatter!(ax, [lat], [az];
            color       = clamp(e, 0.0, MAX_ERR),
            colorrange  = (0.0, MAX_ERR),
            colormap    = errcmap,
            markersize  = err_to_ms(e),
            strokewidth = 0,
        )
    end
end

ax_lim_dep = (maximum(AXIAL_MM) + 5.0, minimum(AXIAL_MM) - 5.0)

# ── Slice title colours (teal → green → red, extends with grey if needed) ─────

_base_colors = [
    RGBf(0.10, 0.65, 0.70),
    RGBf(0.15, 0.60, 0.15),
    RGBf(0.80, 0.20, 0.15),
]
n_slices    = length(LATERAL_Z_MM)
slice_colors = [_base_colors[mod1(i, length(_base_colors))] for i in 1:n_slices]

# ── Theme ─────────────────────────────────────────────────────────────────────

update_theme!(
    fontsize = 9,
    Axis = (
        xgridvisible      = false,
        ygridvisible      = false,
        topspinevisible   = false,
        rightspinevisible = false,
    ),
)

fig = Figure(size = (300 * n_slices + 100, 420))

# ── Error panels ──────────────────────────────────────────────────────────────

mid_col = ceil(Int, n_slices / 2)

for (zi, lz_mm) in enumerate(LATERAL_Z_MM)
    tcol        = slice_colors[zi]
    show_xlabel = zi == mid_col
    show_ylabel = zi == 1

    # Row 1: Geometric ASA (uncorrected)
    ax_top = Axis(fig[1, zi];
        xlabel             = show_xlabel ? "Lateral Y [mm]" : "",
        ylabel             = show_ylabel ? "Axial Position [mm]" : "",
        yreversed          = true,
        aspect             = DataAspect(),
        title              = "z = $(round(Int, lz_mm)) mm",
        titlecolor         = tcol,
        xticklabelsize     = 7,
        yticklabelsize     = 7,
        yticklabelsvisible = show_ylabel,
    )
    plot_error_grid!(ax_top, geo_errors_mm[:, :, zi], AXIAL_MM, LATERAL_Y_MM)
    ylims!(ax_top, ax_lim_dep...)

    # Row 2: HASA (corrected)
    ax_bot = Axis(fig[2, zi];
        xlabel             = show_xlabel ? "Lateral Y [mm]" : "",
        ylabel             = show_ylabel ? "Axial Position [mm]" : "",
        yreversed          = true,
        aspect             = DataAspect(),
        xticklabelsize     = 7,
        yticklabelsize     = 7,
        yticklabelsvisible = show_ylabel,
    )
    plot_error_grid!(ax_bot, hasa_errors_mm[:, :, zi], AXIAL_MM, LATERAL_Y_MM)
    ylims!(ax_bot, ax_lim_dep...)
end

# Row title labels
Label(fig[0, 1:n_slices], "ASA";
    fontsize = 11, font = :bold,
    color    = RGBf(0.45, 0.0, 0.55),
    halign   = :center,
    valign   = :bottom,
    padding  = (0, 0, 0, 4),
)
Label(fig[2, 1:n_slices, Top()], "HASA";
    fontsize = 11, font = :bold,
    color    = RGBf(0.70, 0.30, 0.05),
    halign   = :center,
    padding  = (0, 0, 6, 0),
)

# ── Shared error colorbar ─────────────────────────────────────────────────────

Colorbar(fig[1:2, n_slices + 1];
    colormap      = errcmap,
    colorrange    = (0.0, MAX_ERR),
    label         = "Error [mm]",
    ticklabelsize = 7,
    labelsize     = 8,
    width         = 12,
    ticks         = 0:1:Int(MAX_ERR),
)

# ── Spacing ───────────────────────────────────────────────────────────────────

colgap!(fig.layout, 5)
rowgap!(fig.layout, 3)

# ── Save ──────────────────────────────────────────────────────────────────────

figfile = replace(results_file, r"\.jld2$" => "_accuracy_slices.png")
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
