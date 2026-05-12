@testset "localization metrics" begin
    cfg = PAMConfig(
        dx=0.5e-3,
        dz=0.5e-3,
        axial_dim=0.05,
        transverse_dim=0.04,
        receiver_aperture=nothing,
        PML_GUARD=5,
        peak_suppression_radius=2e-3,
        success_tolerance=1.5e-3,
    )
    kgrid = pam_grid(cfg)
    depth = depth_coordinates(kgrid, cfg)
    lateral = kgrid.y_vec
    sources = [
        PointSource2D(depth=0.015, lateral=-0.004, frequency=0.4e6),
        PointSource2D(depth=0.028, lateral=0.006, frequency=0.4e6),
    ]

    intensity = zeros(Float64, kgrid.Nx, kgrid.Ny)
    σd = 0.8e-3
    σl = 0.8e-3
    for src in sources
        for i in 1:kgrid.Nx, j in 1:kgrid.Ny
            intensity[i, j] += exp(-((depth[i] - src.depth)^2 / (2σd^2) + (lateral[j] - src.lateral)^2 / (2σl^2)))
        end
    end

    stats = analyse_pam_2d(intensity, kgrid, cfg, sources)
    @test stats[:mean_radial_error_mm] < 0.6
    @test stats[:num_success] == 2
    @test length(stats[:axial_fwhm_mm]) == 2
    @test all(stats[:axial_fwhm_mm] .> 0)
    @test all(stats[:lateral_fwhm_mm] .> 0)

    metrics = pam_intensity_metrics(intensity, kgrid, cfg; threshold_ratio=0.5, reference_intensity=2.0)
    @test metrics[:peak_intensity] ≈ maximum(intensity)
    @test metrics[:relative_peak_intensity] ≈ maximum(intensity) / 2.0
    @test metrics[:integrated_intensity_m2] > 0
    @test metrics[:active_area_mm2] > 0
    @test isfinite(metrics[:centroid_depth_mm])
    @test isfinite(metrics[:centroid_lateral_mm])
end

@testset "detection metrics" begin
    cfg = PAMConfig(
        dx=0.5e-3,
        dz=0.5e-3,
        axial_dim=0.05,
        transverse_dim=0.04,
        receiver_aperture=nothing,
        PML_GUARD=5,
        success_tolerance=1.0e-3,
    )
    kgrid = pam_grid(cfg)
    sources = [
        PointSource2D(depth=0.015, lateral=-0.004, frequency=0.4e6),
        PointSource2D(depth=0.028, lateral=0.006, frequency=0.4e6),
    ]
    intensity = zeros(Float64, kgrid.Nx, kgrid.Ny)
    row1, col1 = source_grid_index(sources[1], cfg, kgrid)
    row_false, col_false = source_grid_index(PointSource2D(depth=0.038, lateral=-0.012), cfg, kgrid)
    intensity[row1, col1] = 1.0
    intensity[row_false, col_false] = 0.8

    stats = analyse_pam_detection_2d(
        intensity,
        kgrid,
        cfg,
        sources;
        truth_radius=1.0e-3,
        threshold_ratio=0.5,
        frequencies=[0.4e6],
        psf_axial_fwhm=2.0e-3,
        psf_lateral_fwhm=2.0e-3,
    )

    @test 0 < stats[:precision] < 1
    @test 0 < stats[:recall] < 1
    @test stats[:false_positive_pixels] > 0
    @test stats[:false_negative_pixels] > 0
    @test stats[:spurious_prediction_components] == 1
    @test 0 < stats[:energy_fraction_inside_mask] < 1
    @test stats[:energy_fraction_inside_mask] + stats[:energy_fraction_outside_mask] ≈ 1.0
    @test stats[:energy_fraction_inside_predicted_mask] > stats[:energy_fraction_inside_mask]
    @test stats[:energy_fraction_inside_predicted_mask] + stats[:energy_fraction_outside_predicted_mask] ≈ 1.0
    @test isfinite(stats[:centroid_error_mm])
    @test stats[:axial_spread_mm] > 0
    @test stats[:lateral_spread_mm] > 0
    @test haskey(stats, :psf_target_correlation)
    @test isfinite(stats[:psf_target_normalized_l2_error])

    truth_override = falses(kgrid.Nx, kgrid.Ny)
    truth_override[row_false, col_false] = true
    override_stats = analyse_pam_detection_2d(
        intensity,
        kgrid,
        cfg,
        sources;
        truth_radius=1.0e-3,
        threshold_ratio=0.5,
        truth_mask=truth_override,
        psf_axial_fwhm=2.0e-3,
        psf_lateral_fwhm=2.0e-3,
    )
    @test override_stats[:truth_mask_mode] == :provided
    @test override_stats[:psf_target_mode] == :provided_mask
    @test override_stats[:true_positive_pixels] == 1
    @test override_stats[:false_positive_pixels] == 1
    @test override_stats[:false_negative_pixels] == 0

    source_map = pam_source_map(sources, kgrid, cfg; weights=:uniform)
    @test sum(source_map) == length(sources)
    blurred_truth = pam_psf_blurred_truth_map(
        sources,
        kgrid,
        cfg;
        psf_axial_fwhm=2.0e-3,
        psf_lateral_fwhm=2.0e-3,
        weights=:uniform,
    )
    @test size(blurred_truth) == size(intensity)
    @test sum(blurred_truth) ≈ sum(source_map) atol=1e-8
end
