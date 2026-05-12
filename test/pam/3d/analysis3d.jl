@testset "detection metrics" begin
    cfg = PAMConfig3D(
        dx=1e-3,
        dy=1e-3,
        dz=1e-3,
        axial_dim=0.012,
        transverse_dim_y=0.012,
        transverse_dim_z=0.012,
        success_tolerance=0.75e-3,
    )
    grid = pam_grid_3d(cfg)
    source = PointSource3D(depth=0.004, lateral_y=0.0, lateral_z=0.0, frequency=1e6)
    truth_mask = pam_truth_mask_3d([source], grid, cfg; radius=cfg.success_tolerance)
    intensity = zeros(Float64, pam_Nx(cfg), pam_Ny(cfg), pam_Nz(cfg))
    truth_idx = source_grid_index_3d(source, cfg)
    intensity[truth_idx...] = 1.0
    intensity[end - 1, 2, 2] = 0.8

    stats = threshold_detection_stats_3d(
        intensity,
        grid,
        cfg,
        [source];
        threshold_ratios=[0.5, 0.9],
        truth_radius=cfg.success_tolerance,
        truth_mask=truth_mask,
    )

    @test length(stats) == 2
    @test stats[1][:predicted_voxels] == 2
    @test stats[1][:precision] ≈ 0.5
    @test stats[1][:source_recall] ≈ 1.0
    @test stats[1][:source_f1] ≈ 2 * 0.5 / 1.5
    @test stats[2][:predicted_voxels] == 1
    @test stats[2][:precision] ≈ 1.0
    @test best_threshold_entry_3d(stats)[:threshold_ratio] == 0.9

    outlines = threshold_outline_entries_3d(stats)
    @test first(outlines).kind == :best_f1
    @test first(outlines).entry[:threshold_ratio] == 0.9
end

@testset "localization metrics" begin
    cfg = PAMConfig3D(
        dx=1e-3,
        dy=1e-3,
        dz=1e-3,
        axial_dim=0.02,
        transverse_dim_y=0.012,
        transverse_dim_z=0.012,
        success_tolerance=1e-3,
        peak_suppression_radius=1.5e-3,
        PML_GUARD=2,
    )
    grid = pam_grid_3d(cfg)
    sources = [
        PointSource3D(depth=0.005, lateral_y=0.0, lateral_z=0.0),
        PointSource3D(depth=0.010, lateral_y=0.002, lateral_z=-0.002),
    ]
    intensity = zeros(Float64, pam_Nx(cfg), pam_Ny(cfg), pam_Nz(cfg))
    for (idx, src) in enumerate(sources)
        intensity[source_grid_index_3d(src, cfg)...] = 2.0 - 0.25 * idx
    end

    stats = analyse_pam_3d(intensity, grid, cfg, sources)
    @test stats[:num_success] == length(sources)
    @test stats[:success_rate] == 1.0
    @test stats[:mean_radial_error_mm] ≈ 0.0 atol=1e-9
    @test length(stats[:peak_indices]) == length(sources)
end
