"""
Reproduce Fig. 7 of Schoen & Arvanitis (2020) from saved validation results.

Usage:
    julia --project=. validation/2D_PAM_Accuracy/plot_results.jl [results_file.jld2]
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CairoMakie, JLD2, TranscranialFUS, Statistics, Printf

# ── Load results ──────────────────────────────────────────────────────────────

results_dir = @__DIR__
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
LATERAL_MM     = d["LATERAL_MM"]
APERTURES_MM   = d["APERTURES_MM"]
DX             = d["DX"]
DT             = d["DT"]
T_MAX          = d["T_MAX"]
AXIAL_DIM      = d["AXIAL_DIM"]
TRANS_DIM      = d["TRANS_DIM"]
SKULL_DIST     = d["SKULL_DIST"]
SLICE_INDEX    = d["SLICE_INDEX"]

# ── Rebuild medium for panel (a) ──────────────────────────────────────────────

MAX_APT_MM = maximum(APERTURES_MM)
sim_cfg = PAMConfig(
    dx             = DX,
    dz             = DX,
    axial_dim      = AXIAL_DIM,
    transverse_dim = TRANS_DIM,
    receiver_aperture = MAX_APT_MM * 1e-3,
    t_max          = T_MAX,
    dt             = DT,
)

println("Rebuilding skull medium…")
c, _, _ = make_pam_medium(
    sim_cfg;
    aberrator           = :skull,
    skull_to_transducer = SKULL_DIST,
    slice_index         = SLICE_INDEX,
)

# ── Coordinate arrays ─────────────────────────────────────────────────────────

Nx, Ny = size(c)
R      = receiver_row(sim_cfg)
dx_mm  = DX * 1e3

ax_coords  = [(i - R) * dx_mm for i in 1:Nx]
lat_coords = [(j - (Ny ÷ 2 + 1)) * dx_mm for j in 1:Ny]

# ── Aperture colours (100→red, 75→green, 50→teal) ────────────────────────────

apt_colors = Dict(
    100.0 => RGBf(0.80, 0.20, 0.15),
     75.0 => RGBf(0.15, 0.60, 0.15),
     50.0 => RGBf(0.10, 0.65, 0.70),
)

# ── Error colour / size helpers ───────────────────────────────────────────────

MAX_ERR  = 8.0
MIN_MS   = 2.5
MAX_MS   = 15.0
errcmap  = :plasma

err_to_ms(e) = MIN_MS + (MAX_MS - MIN_MS) * clamp(e, 0.0, MAX_ERR) / MAX_ERR

function plot_error_grid!(ax, errors_pi, axial_mm, lateral_mm)
    for (ai, az) in enumerate(axial_mm), (li, lat) in enumerate(lateral_mm)
        e = errors_pi[ai, li]
        isnan(e) && continue
        scatter!(ax, [lat], [az];
            color      = clamp(e, 0.0, MAX_ERR),
            colorrange = (0.0, MAX_ERR),
            colormap   = errcmap,
            markersize = err_to_ms(e),
            strokewidth = 0,
        )
    end
end

ax_lim_dep = (maximum(AXIAL_MM) + 5.0, minimum(AXIAL_MM) - 5.0)  # yreversed order

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

fig = Figure(size = (920, 420))

# ── Panel (a): skull with source grid ────────────────────────────────────────

# Visible depth range: just above skull to deepest source
depth_top_mm = SKULL_DIST * 1e3 - 8.0   # a bit above the skull
depth_bot_mm = maximum(AXIAL_MM) + 5.0

# Bar y-coords (above skull, negative depth = between receiver and skull)
bar_bot_mm = depth_top_mm - 6.0
bar_top_mm = depth_top_mm - 2.5
bar_mid_mm = (bar_bot_mm + bar_top_mm) / 2.0

ax_a = Axis(fig[1:2, 1];
    xlabel    = "Transverse Position [mm]",
    ylabel    = "Axial Position [mm]",
    aspect    = DataAspect(),
    yreversed = true,
    title     = "a",
    titlealign = :left,
    titlefont  = :bold,
)

# Skull sound-speed heatmap
heatmap!(ax_a, lat_coords, ax_coords, c';
    colormap   = :grays,
    colorrange = (1400, 2100),
)

# Source dots
for az in AXIAL_MM, lat in LATERAL_MM
    scatter!(ax_a, [lat], [az]; color = :black, markersize = 4)
end

# Aperture bars: widest drawn first so narrower bars sit on top
for apt_mm in sort(APERTURES_MM; rev=true)
    half_mm = apt_mm / 2.0
    band!(ax_a, [-half_mm, half_mm], bar_bot_mm, bar_top_mm;
        color = (apt_colors[apt_mm], 0.9))
end

# Labels: placed visually above the bars (smaller y = higher with yreversed)
# bar_bot_mm is the visual top of the bar; go further up (smaller y)
label_y = bar_bot_mm - 2.5
# Place each label in its bar's exclusive outer zone (half-widths: 50, 37.5, 25 mm)
label_xs = [48.0, 31.0, 0.0]   # 100 mm → outer wing; 75 mm → mid ring; 50 mm → centre
for (xi, apt_mm) in zip(label_xs, sort(APERTURES_MM; rev=true))
    text!(ax_a, xi, label_y;
        text     = "$(round(Int, apt_mm)) mm",
        align    = (:center, :center),
        color    = apt_colors[apt_mm],
        fontsize = 7,
        font     = :bold,
    )
end

xlims!(ax_a, -56.0, 56.0)
ylims!(ax_a, depth_bot_mm, label_y - 1.5)   # yreversed: first=visual bottom

# Sound-speed colorbar: small horizontal strip above panel (a)
Colorbar(fig[0, 1];
    colormap      = :grays,
    colorrange    = (1400, 2100),
    label         = "Sound Speed [m/s]",
    vertical      = false,
    height        = 10,
    ticklabelsize = 7,
    labelsize     = 8,
    ticks         = [1400, 1700, 2100],
    tellwidth     = false,
)

# ── Panels (b) and (c) ────────────────────────────────────────────────────────

for (pi, apt_mm) in enumerate(APERTURES_MM)
    tcol = apt_colors[apt_mm]
    gcol = pi + 1   # figure columns 2..4

    show_xlabel = pi == 2
    show_ylabel_b = pi == 1
    show_ylabel_c = pi == 1

    # ---- b: uncorrected ----
    ax_b = Axis(fig[1, gcol];
        xlabel         = show_xlabel ? "Transverse Position [mm]" : "",
        ylabel         = show_ylabel_b ? "Axial Position [mm]" : "",
        yreversed      = true,
        aspect         = DataAspect(),
        title          = "$(round(Int, apt_mm)) mm",
        titlecolor     = tcol,
        xticklabelsize = 7,
        yticklabelsize = 7,
    )
    plot_error_grid!(ax_b, geo_errors_mm[pi, :, :], AXIAL_MM, LATERAL_MM)
    xlims!(ax_b, -25.0, 25.0)
    ylims!(ax_b, ax_lim_dep...)

    # ---- c: corrected ----
    ax_c = Axis(fig[2, gcol];
        xlabel         = show_xlabel ? "Transverse Position [mm]" : "",
        ylabel         = show_ylabel_c ? "Axial Position [mm]" : "",
        yreversed      = true,
        aspect         = DataAspect(),
        xticklabelsize = 7,
        yticklabelsize = 7,
    )
    plot_error_grid!(ax_c, hasa_errors_mm[pi, :, :], AXIAL_MM, LATERAL_MM)
    xlims!(ax_c, -25.0, 25.0)
    ylims!(ax_c, ax_lim_dep...)
end

Label(fig[0, 2:4], "Uncorrected";
    fontsize  = 11, font = :bold,
    color     = RGBf(0.45, 0.0, 0.55),
    halign    = :center,
    valign    = :bottom,
    padding   = (0, 0, 0, 4),
)
Label(fig[2, 2:4, Top()], "Corrected";
    fontsize  = 11, font = :bold,
    color     = RGBf(0.70, 0.30, 0.05),
    halign    = :center,
    padding   = (0, 0, 6, 0),
)

# ── Shared error colorbar ─────────────────────────────────────────────────────

Colorbar(fig[1:2, 5];
    colormap      = errcmap,
    colorrange    = (0.0, MAX_ERR),
    label         = "Error [mm]",
    ticklabelsize = 7,
    labelsize     = 8,
    width         = 12,
    ticks         = 0:2:Int(MAX_ERR),
)

# ── Spacing ───────────────────────────────────────────────────────────────────

colgap!(fig.layout, 5)
rowgap!(fig.layout, 3)

# ── Print table ───────────────────────────────────────────────────────────────

println()
println("  Aperture │ Geo (uncorrected)     │ HASA (corrected)      │ Paper HASA")
println("  ─────────┼──────────────────────┼───────────────────────┼────────────────")
ref = Dict(50.0=>"1.2 ± 0.7", 75.0=>"0.9 ± 0.5", 100.0=>"0.8 ± 0.4")
for (pi, apt_mm) in enumerate(APERTURES_MM)
    gv = filter(!isnan, geo_errors_mm[pi, :, :])
    hv = filter(!isnan, hasa_errors_mm[pi, :, :])
    @printf("  %3.0f mm   │ %4.1f ± %4.1f mm         │ %4.1f ± %4.1f mm          │ %s mm\n",
        apt_mm, mean(gv), std(gv), mean(hv), std(hv), ref[apt_mm])
end

# ── Save ──────────────────────────────────────────────────────────────────────

outfile = replace(results_file, r"\.jld2$" => "_fig7.pdf")
save(outfile, fig)
println("\nSaved → $outfile")
