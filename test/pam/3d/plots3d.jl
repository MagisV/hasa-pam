using CairoMakie

@testset "plot projection helpers" begin
    cfg = PAMConfig3D(
        dx=1e-3,
        dy=1e-3,
        dz=1e-3,
        axial_dim=0.004,
        transverse_dim_y=0.003,
        transverse_dim_z=0.005,
    )
    grid = pam_grid_3d(cfg)
    values = reshape(Float64.(1:(pam_Nx(cfg) * pam_Ny(cfg) * pam_Nz(cfg))), pam_Nx(cfg), pam_Ny(cfg), pam_Nz(cfg))

    depth_y = TranscranialFUS._project3d_values(values, :depth_y)
    depth_z = TranscranialFUS._project3d_values(values, :depth_z)
    y_z = TranscranialFUS._project3d_values(values, :y_z)
    @test size(depth_y) == (pam_Nx(cfg), pam_Ny(cfg))
    @test size(depth_z) == (pam_Nx(cfg), pam_Nz(cfg))
    @test size(y_z) == (pam_Ny(cfg), pam_Nz(cfg))
    @test y_z == dropdims(maximum(values; dims=1); dims=1)

    xvals, yvals, xlabel, ylabel = TranscranialFUS._projection_axes_3d(grid, cfg, :depth_z)
    @test length(xvals) == pam_Nz(cfg)
    @test length(yvals) == pam_Nx(cfg)
    @test occursin("Z", xlabel)
    @test occursin("Depth", ylabel)
    @test TranscranialFUS._projection_heatmap_matrix_3d(depth_z, :depth_z) == depth_z'
    @test TranscranialFUS._projection_heatmap_matrix_3d(y_z, :y_z) == y_z

    mask = falses(size(values))
    mask[2, 2, 3] = true
    @test size(TranscranialFUS._project3d_mask(mask, :depth_y)) == (pam_Nx(cfg), pam_Ny(cfg))
    @test TranscranialFUS._project3d_mask(mask, :depth_y)[2, 2]
    @test TranscranialFUS._project3d_mask(mask, :depth_z)[2, 3]
    @test TranscranialFUS._project3d_mask(mask, :y_z)[2, 3]

    c = fill(1500.0, size(values))
    c[2, 2, 3] = 2200.0
    @test TranscranialFUS._c_slice_for_projection(c, :depth_y)[2, 2] == 2200.0
    @test TranscranialFUS._c_slice_for_projection(c, :depth_z)[2, 3] == 2200.0
    @test TranscranialFUS._c_slice_for_projection(c, :y_z)[2, 3] == 2200.0

    sources = [PointSource3D(depth=0.01, lateral_y=0.002, lateral_z=-0.003)]
    @test TranscranialFUS.source_triples_mm(sources) == [(10.0, 2.0, -3.0)]
    @test_throws ErrorException TranscranialFUS._project3d_values(values, :unknown)
    @test_throws ErrorException TranscranialFUS._project3d_mask(mask, :unknown)
    @test_throws ErrorException TranscranialFUS._projection_axes_3d(grid, cfg, :unknown)
end

@testset "3D plot rendering helpers" begin
    cfg = PAMConfig3D(
        dx=1e-3,
        dy=1e-3,
        dz=1e-3,
        axial_dim=0.008,
        transverse_dim_y=0.006,
        transverse_dim_z=0.006,
        success_tolerance=1e-3,
        PML_GUARD=1,
    )
    grid = pam_grid_3d(cfg)
    source = PointSource3D(depth=0.003, lateral_y=0.0, lateral_z=0.0)
    intensity = zeros(Float64, pam_Nx(cfg), pam_Ny(cfg), pam_Nz(cfg))
    intensity[source_grid_index_3d(source, cfg)...] = 1.0
    c = fill(1500.0, size(intensity))
    rho = fill(1000.0, size(intensity))

    points = TranscranialFUS._voxel_points_3d(intensity .> 0, grid, cfg)
    @test length(points.indices) == 1
    @test only(points.depth) ≈ 3.0
    @test only(points.y) ≈ 0.0
    @test only(points.z) ≈ 0.0

    fig = Figure(size=(300, 240))
    ax = Axis(fig[1, 1])
    @test TranscranialFUS.overlay_skull_3d_projection!(ax, c, collect(grid.y) .* 1e3, depth_coordinates_3d(cfg) .* 1e3, :depth_y) === nothing
    @test TranscranialFUS.scatter_sources_3d_projection!(ax, [source], :depth_y) === nothing

    mktempdir() do dir
        boundary_path = joinpath(dir, "boundary3d.png")
        boundary_metrics = TranscranialFUS.save_threshold_boundary_detection_3d(
            boundary_path,
            intensity,
            0.75 .* intensity,
            grid,
            cfg,
            [source];
            threshold_ratios=[0.5, 0.9],
            truth_radius=cfg.success_tolerance,
            c=c,
        )
        @test isfile(boundary_path)
        @test boundary_metrics["selection_metric"] == "source_f1"
        @test boundary_metrics["best_geometric_threshold"] in [0.5, 0.9]

        volume_path = joinpath(dir, "volume3d.png")
        volume_metrics = TranscranialFUS.save_best_threshold_volume_3d(
            volume_path,
            intensity,
            grid,
            cfg,
            [source];
            threshold=0.5,
            truth_radius=cfg.success_tolerance,
        )
        @test isfile(volume_path)
        @test volume_metrics["threshold_ratio"] == 0.5
        @test volume_metrics["predicted_voxels"] == 1
        @test volume_metrics["truth_voxels"] >= 1
    end
end
