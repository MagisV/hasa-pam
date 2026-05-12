@testset "reconstruction in water" begin
    cfg = PAMConfig(
        dx=0.5e-3,
        dz=0.5e-3,
        axial_dim=0.04,
        transverse_dim=0.03,
        receiver_aperture=nothing,
        t_max=40e-6,
        dt=50e-9,
        PML_GUARD=5,
        zero_pad_factor=2,
        peak_suppression_radius=1e-3,
        success_tolerance=1.0e-3,
    )
    source = PointSource2D(depth=0.015, lateral=0.003, frequency=0.4e6, amplitude=1.0, num_cycles=5)
    c, _, _ = make_pam_medium(cfg; aberrator=:none)
    rf, kgrid = analytic_rf_for_point_source(cfg, source)
    cuda_ok = TranscranialFUS._pam_cuda_functional()

    quiet_result, quiet_progress = capture_stderr_result() do
        reconstruct_pam(rf, c, cfg; frequencies=[source.frequency], corrected=false)
    end
    intensity, _, info = quiet_result
    @test quiet_progress == ""
    stats = analyse_pam_2d(intensity, kgrid, cfg, [source])

    hasa_result, hasa_progress = capture_stderr_result() do
        reconstruct_pam(
            rf,
            c,
            cfg;
            frequencies=[source.frequency],
            corrected=true,
            use_gpu=cuda_ok,
            show_progress=true,
        )
    end
    intensity_hasa, _, info_hasa = hasa_result
    stats_hasa = analyse_pam_2d(intensity_hasa, kgrid, cfg, [source])

    one_window = PAMWindowConfig(
        enabled=true,
        window_duration=pam_Nt(cfg) * cfg.dt,
        hop=pam_Nt(cfg) * cfg.dt,
        taper=:none,
        min_energy_ratio=0.0,
    )
    windowed_result, windowed_progress = capture_stderr_result() do
        reconstruct_pam_windowed(
            rf,
            c,
            cfg;
            frequencies=[source.frequency],
            corrected=false,
            window_config=one_window,
            use_gpu=cuda_ok,
            show_progress=true,
        )
    end
    intensity_windowed, _, info_windowed = windowed_result

    cropped_range = 101:500
    cropped_origin = (first(cropped_range) - 1) * cfg.dt
    intensity_cropped, _, info_cropped = reconstruct_pam(
        rf[:, cropped_range],
        c,
        cfg;
        frequencies=[source.frequency],
        use_gpu=cuda_ok,
        corrected=false,
        time_origin=cropped_origin,
    )
    stats_cropped = analyse_pam_2d(intensity_cropped, kgrid, cfg, [source])

    @test info[:corrected] == false
    @test info[:show_progress] == false
    @test stats[:mean_radial_error_mm] < 1.0
    @test stats[:success_rate] == 1.0
    @test stats[:mean_norm_peak_intensity] > 0.5
    @test info_hasa[:corrected] == true
    @test info_hasa[:use_gpu] == cuda_ok
    @test info_hasa[:backend] == (cuda_ok ? :cuda : :cpu)
    @test info_hasa[:gpu_precision] == (cuda_ok ? Float32 : nothing)
    @test info_hasa[:show_progress] == true
    @test occursin("PAM HASA reconstruction", hasa_progress)
    if cuda_ok
        @test occursin("freq batch march elapsed", hasa_progress)
    else
        @test occursin("frequency 1/", hasa_progress)
    end
    @test occursin("total elapsed", hasa_progress)
    @test stats_hasa[:mean_radial_error_mm] < 1.0
    @test stats_hasa[:success_rate] == 1.0

    if cuda_ok
        @test pam_gpu_maps_approx(intensity_windowed, intensity)
        intensity_hasa_cpu, _, _ = reconstruct_pam(
            rf,
            c,
            cfg;
            frequencies=[source.frequency],
            corrected=true,
            use_gpu=false,
        )
        @test pam_gpu_maps_approx(intensity_hasa, intensity_hasa_cpu)

        c_lens = copy(c)
        c_lens[20:30, 25:35] .= 1700.0
        intensity_lens_cpu, _, _ = reconstruct_pam(
            rf,
            c_lens,
            cfg;
            frequencies=[source.frequency],
            corrected=true,
            use_gpu=false,
        )
        intensity_lens_gpu, _, info_lens_gpu = reconstruct_pam(
            rf,
            c_lens,
            cfg;
            frequencies=[source.frequency],
            corrected=true,
            use_gpu=true,
        )
        @test info_lens_gpu[:backend] == :cuda
        @test pam_gpu_maps_approx(intensity_lens_gpu, intensity_lens_cpu)
    else
        @test intensity_windowed ≈ intensity
        err = try
            reconstruct_pam(rf, c, cfg; frequencies=[source.frequency], corrected=false, use_gpu=true)
            nothing
        catch err
            err
        end
        @test err isa ErrorException
        @test occursin("functional NVIDIA CUDA GPU", sprint(showerror, err))
    end

    @test info_windowed[:use_gpu] == cuda_ok
    @test info_windowed[:backend] == (cuda_ok ? :cuda : :cpu)
    @test info_windowed[:show_progress] == true
    @test occursin("PAM geometric ASA windowed reconstruction", windowed_progress)
    @test occursin("window 1/1", windowed_progress)
    @test occursin("complete", windowed_progress)
    @test info_windowed[:used_window_count] == 1
    @test info_windowed[:skipped_window_count] == 0
    @test info_cropped[:time_origin] ≈ cropped_origin
    @test info_cropped[:use_gpu] == cuda_ok
    @test info_cropped[:backend] == (cuda_ok ? :cuda : :cpu)
    @test stats_cropped[:success_rate] == 1.0
end

@testset "spectral helpers" begin
    p1 = fill(1.0 + 0.0im, 2, 2)
    p2 = fill(-1.0 + 0.0im, 2, 2)
    @test all(abs2.(p1 .+ p2) .== 0.0)
    @test all(abs2.(p1) .+ abs2.(p2) .== 2.0)
end
