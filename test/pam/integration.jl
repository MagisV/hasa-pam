@testset "optional k-Wave smoke" begin
    if get(ENV, "TRANSCRANIALFUS_RUN_KWAVE_TESTS", "0") == "1" && kwave_available()
        pam_cfg = PAMConfig(
            dx=0.5e-3,
            dz=0.5e-3,
            axial_dim=0.03,
            transverse_dim=0.03,
            receiver_aperture=0.03,
            PML_GUARD=20,
            t_max=30e-6,
            dt=50e-9,
            zero_pad_factor=2,
            peak_suppression_radius=1.0e-3,
            success_tolerance=1.5e-3,
        )
        c_pam, rho_pam, _ = make_pam_medium(pam_cfg; aberrator=:none)
        sources = [PointSource2D(depth=0.015, lateral=0.003, frequency=0.4e6, amplitude=5e4, num_cycles=4)]
        rf, kgrid_pam, sim_info = simulate_point_sources(c_pam, rho_pam, sources, pam_cfg)
        @test size(rf) == (pam_Ny(pam_cfg), pam_Nt(pam_cfg))
        @test sim_info[:receiver_row] == receiver_row(pam_cfg)
        @test sim_info[:receiver_row] == 1

        pam_map, _, pam_info = reconstruct_pam(rf, c_pam, pam_cfg; frequencies=[0.4e6], corrected=false)
        pam_stats = analyse_pam_2d(pam_map, kgrid_pam, pam_cfg, sources)
        @test pam_info[:corrected] == false
        @test pam_stats[:mean_radial_error_mm] <= 1.5
    else
        @info "Skipping PAM k-Wave smoke test. Set TRANSCRANIALFUS_RUN_KWAVE_TESTS=1 to enable it."
    end
end
