using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using TranscranialFUS

src = PointSource3D(
    depth      = 30e-3,
    lateral_y  = 2e-3,
    lateral_z  = -1e-3,
    frequency  = 0.5e6,
    amplitude  = 1.0,
    phase      = 0.0,
    delay      = 0.0,
    num_cycles = 5.0,
)

cfg = PAMConfig3D(
    dx = 0.5e-3,
    dy = 0.5e-3,
    dz = 0.5e-3,
    axial_dim        = 60e-3,
    transverse_dim_y = 32e-3,
    transverse_dim_z = 32e-3,
    dt    = 80e-9,
    t_max = 60e-6,
    c0    = 1500.0,
    rho0  = 1000.0,
    tukey_ratio    = 0.25,
    zero_pad_factor = 2,
)

println("Grid: $(pam_Nx(cfg))×$(pam_Ny(cfg))×$(pam_Nz(cfg)), Nt=$(pam_Nt(cfg))")

c, rho, _ = make_pam_medium_3d(cfg)
grid = pam_grid_3d(cfg)
ny, nz, nt = pam_Ny(cfg), pam_Nz(cfg), pam_Nt(cfg)
rf = zeros(Float32, ny, nz, nt)
for iy in 1:ny, iz in 1:nz
    dy_src = grid.y[iy] - src.lateral_y
    dz_src = grid.z[iz] - src.lateral_z
    r  = sqrt(src.depth^2 + dy_src^2 + dz_src^2)
    t0 = r / cfg.c0
    for it in 1:nt
        te = (it - 1) * cfg.dt - t0 - src.delay
        if te >= 0 && te <= src.num_cycles / src.frequency
            rf[iy, iz, it] = Float32(sin(2π * src.frequency * te))
        end
    end
end
println("RF: size=$(size(rf)), max=$(maximum(abs.(rf)))")

rr = receiver_row(cfg)
expected_row = rr + round(Int, src.depth / cfg.dx)
mid_y = argmin(abs.(collect(grid.y) .- src.lateral_y))
mid_z = argmin(abs.(collect(grid.z) .- src.lateral_z))
println("Expected peak: row=$expected_row, col_y=$mid_y ($(round(grid.y[mid_y]*1e3;digits=1))mm), col_z=$mid_z ($(round(grid.z[mid_z]*1e3;digits=1))mm)")

intensity, g, info = reconstruct_pam_3d(
    rf, c, cfg;
    corrected    = true,
    use_gpu      = true,
    show_progress = true,
)
println("Intensity: size=$(size(intensity)), max=$(maximum(intensity))")
println("  at expected peak: $(intensity[expected_row, mid_y, mid_z])")
slice_depth = intensity[expected_row, :, :]
println("  slice at expected depth: max=$(maximum(slice_depth)) at $(Tuple(argmax(slice_depth)))")

stats = analyse_pam_3d(intensity, g, cfg, [src])
println("Peak predicted: $(stats[:predicted_mm])")
println("Truth:          $(stats[:truth_mm])")
println("Radial error:   $(round(stats[:radial_errors_mm][1]; digits=2)) mm  (tolerance=$(cfg.success_tolerance*1e3) mm)")
println("Success: $(stats[:num_success])/$(length([src]))")
