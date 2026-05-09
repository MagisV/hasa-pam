"""
Plot 3D PAM localization accuracy from saved validation results.

Produces a two-panel Axis3 figure: left = geometric ASA, right = HASA.
Each of the 27 sources is shown as a scatter point at its true 3D position.
Dot colour and size encode the radial localization error.
The skull cross-section (y-z plane at x = SKULL_DIST) is overlaid as a
semi-transparent heatmap slice for spatial context.

Usage:
    julia --project=. validation/3D_PAM_Accuracy/plot_results.jl [results_file.jld2]
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CairoMakie, JLD2, TranscranialFUS, Statistics, Printf

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

# ── Print summary table ───────────────────────────────────────────────────────

geo_all  = filter(!isnan, geo_errors_mm)
hasa_all = filter(!isnan, hasa_errors_mm)
println()
println("  Method        │ Mean ± Std error")
println("  ──────────────┼──────────────────────")
@printf("  Geometric ASA │ %.2f ± %.2f mm (n=%d)\n", mean(geo_all),  std(geo_all),  length(geo_all))
@printf("  HASA          │ %.2f ± %.2f mm (n=%d)\n", mean(hasa_all), std(hasa_all), length(hasa_all))
println()

# ── Rebuild skull medium for background slice ─────────────────────────────────

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
println("Rebuilding skull medium for background slice…")
c, _, _ = make_pam_medium_3d(
    sim_cfg;
    aberrator           = :skull,
    skull_to_transducer = SKULL_DIST,
    slice_index_z       = SLICE_INDEX,
)

# ── Coordinate arrays ─────────────────────────────────────────────────────────

rr        = receiver_row(sim_cfg)
dx_mm     = DX * 1e3
Ny        = pam_Ny(sim_cfg)
Nz        = pam_Nz(sim_cfg)
ax_coords = [(i - rr) * dx_mm for i in 1:pam_Nx(sim_cfg)]
y_coords  = [(j - (Ny ÷ 2 + 1)) * dx_mm for j in 1:Ny]
z_coords  = [(k - (Nz ÷ 2 + 1)) * dx_mm for k in 1:Nz]

# Skull cross-section at the outer skull face depth
skull_depth_mm = SKULL_DIST * 1e3
skull_row_idx  = argmin(abs.(ax_coords .- skull_depth_mm))
skull_slice_c  = c[skull_row_idx, :, :]   # (Ny, Nz) sound-speed map

# ── Error → visual encoding ───────────────────────────────────────────────────

MAX_ERR = 6.0
MIN_MS  = 6.0
MAX_MS  = 24.0
errcmap = :plasma

err_to_ms(e) = MIN_MS + (MAX_MS - MIN_MS) * clamp(e, 0.0, MAX_ERR) / MAX_ERR

# ── Figure ────────────────────────────────────────────────────────────────────

update_theme!(fontsize = 9)

fig = Figure(size = (900, 440))

panels = [
    ("Geometric ASA", geo_errors_mm),
    ("HASA",          hasa_errors_mm),
]

for (panel_col, (panel_label, errors)) in enumerate(panels)
    ax = Axis3(fig[1, panel_col];
        xlabel          = "Depth [mm]",
        ylabel          = "Lateral Y [mm]",
        zlabel          = "Lateral Z [mm]",
        title           = panel_label,
        aspect          = :data,
        perspectiveness = 0.3,
        azimuth         = 5π / 8,
        elevation       = π / 10,
    )

    # Semi-transparent skull slice at the skull depth
    surface!(ax,
        fill(skull_depth_mm, Ny, Nz),
        repeat(y_coords,              1, Nz),
        repeat(reshape(z_coords, 1, Nz), Ny, 1);
        color      = Float32.(skull_slice_c),
        colormap   = :grays,
        colorrange = (1400.0f0, 2100.0f0),
        alpha      = 0.35,
        shading    = NoShading,
    )

    # Source scatter: collect all 27 points
    xs = Float64[]; ys = Float64[]; zs = Float64[]
    cs = Float64[]; ms = Float64[]
    for (ai, ax_mm) in enumerate(AXIAL_MM),
        (yi, ly_mm) in enumerate(LATERAL_Y_MM),
        (zi, lz_mm) in enumerate(LATERAL_Z_MM)
        e = errors[ai, yi, zi]
        isnan(e) && continue
        push!(xs, Float64(ax_mm))
        push!(ys, Float64(ly_mm))
        push!(zs, Float64(lz_mm))
        push!(cs, clamp(e, 0.0, MAX_ERR))
        push!(ms, err_to_ms(e))
    end

    scatter!(ax, xs, ys, zs;
        color       = cs,
        colorrange  = (0.0, MAX_ERR),
        colormap    = errcmap,
        markersize  = ms,
        strokewidth = 0.5,
        strokecolor = :black,
    )

    # Receiver aperture outline at depth = 0
    half = APERTURE_MM / 2.0
    apts = [(-half, -half), (half, -half), (half, half), (-half, half), (-half, -half)]
    apt_ys = [p[1] for p in apts]
    apt_zs = [p[2] for p in apts]
    lines!(ax, fill(0.0, 5), apt_ys, apt_zs; color = :gray50, linewidth = 0.8)
end

# Shared error colorbar
Colorbar(fig[1, 3];
    colormap      = errcmap,
    colorrange    = (0.0, MAX_ERR),
    label         = "Radial Error [mm]",
    width         = 14,
    ticklabelsize = 8,
    labelsize     = 9,
    ticks         = 0:1:Int(MAX_ERR),
)

colgap!(fig.layout, 8)

# ── Save figure ───────────────────────────────────────────────────────────────

figfile = replace(results_file, r"\.jld2$" => "_accuracy3d.pdf")
save(figfile, fig)
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
