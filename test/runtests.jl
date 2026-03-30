using Test
using TranscranialFUS

function synthetic_hu_volume(nslices::Int=5, rows::Int=80, cols::Int=60)
    hu = fill(Float32(-1000), nslices, rows, cols)
    for z in 1:nslices
        hu[z, 24:30, 20:40] .= 1200
    end
    return hu
end

@testset "HU conversion" begin
    hu = Float32[-1000 0 300 1200]
    rho, c = hu_to_rho_c(hu)
    @test rho[1] ≈ 1000
    @test c[2] ≈ 1500
    @test all(rho[3:4] .>= 1000)
    @test all(c[3:4] .>= 1500)
end

@testset "Skull boundary detection" begin
    hu_slice = fill(Float32(-1000), 40, 20)
    hu_slice[12:18, 8:13] .= 900
    inner_row, outer_row = find_skull_boundaries(hu_slice)
    @test inner_row == 12
    @test outer_row == 18
end

@testset "Skull mask construction" begin
    c = fill(1500.0f0, 30, 10)
    c[8:12, :] .= 2200
    c[10, 4] = 1500
    mask = skull_mask_from_c_columnwise(c; mask_outside=true)
    @test !mask[8, 1]
    @test mask[8, 4]
    @test mask[20, 4]
    @test !mask[3, 4]
end

@testset "Medium construction" begin
    hu_vol = synthetic_hu_volume()
    cfg = SimulationConfig(z_focus=0.05, dx=1e-3, dz=1e-3, transverse_dim=0.06, trans_aperture=0.03)
    c, rho, info = make_medium_fixed_transducer(hu_vol, cfg, SKULL_IN_WATER; slice_index=2)
    @test size(c) == (Nx(cfg), Nz(cfg))
    @test size(rho) == size(c)
    @test info[:z_trans_idx] == Nx(cfg) - cfg.PML_GUARD
    @test info[:outer_row] < info[:z_trans_idx]

    cfg2 = SimulationConfig(
        z_focus=0.03,
        dx=1e-3,
        dz=1e-3,
        transverse_dim=0.06,
        trans_aperture=0.03,
        focus_depth_from_inner_skull=0.005,
    )
    c2, rho2, info2 = make_medium_fixed_distance_from_skull(hu_vol, cfg2, SKULL_IN_WATER; slice_index=2)
    @test size(c2, 2) == Nz(cfg2)
    @test info2[:z_trans_idx] - info2[:target_idx] == Nx_hasa(cfg2)
    @test info2[:z_trans_idx] - info2[:outer_row] >= 2
    @test size(rho2) == size(c2)
end

@testset "Geometric delay symmetry" begin
    cfg = SimulationConfig(z_focus=0.03, dx=1e-3, dz=1e-3, transverse_dim=0.06, trans_aperture=0.021, axial_padding=2.0)
    cfg.trans_index = Nx(cfg) - cfg.PML_GUARD
    c = fill(Float32(cfg.c0), Nx(cfg), Nz(cfg))
    rho = fill(Float32(cfg.rho0), Nx(cfg), Nz(cfg))
    _, hasa_info, _ = focus(
        c,
        rho,
        GEOMETRIC,
        cfg,
        SweepSettings();
        animation_settings=AnimationSettings(run_kwave=false, Nt=200),
    )
    tau = hasa_info[:tau]
    @test tau ≈ reverse(tau) atol=1e-9
    @test all(hasa_info[:amplitudes] .≈ 1.0)
end

@testset "Placement resolution" begin
    mode, depth = resolve_placement_mode(:auto, SKULL_IN_WATER)
    @test mode == :fixed_focus_depth
    @test depth ≈ 30e-3

    mode, depth = resolve_placement_mode(:auto, WATER)
    @test mode == :fixed_transducer
    @test isnothing(depth)

    mode, depth = resolve_placement_mode(:fixed_transducer, SKULL_IN_WATER)
    @test mode == :fixed_transducer
    @test isnothing(depth)

    mode, depth = resolve_placement_mode(:fixed_focus_depth, SKULL_IN_WATER)
    @test mode == :fixed_focus_depth
    @test depth ≈ 30e-3

    mode, depth = resolve_placement_mode(:fixed_focus_depth, SKULL_IN_WATER; focus_depth_from_inner_skull=20e-3)
    @test mode == :fixed_focus_depth
    @test depth ≈ 20e-3

    mode, depth = resolve_placement_mode(:auto, SKULL_IN_WATER; focus_depth_from_inner_skull=25e-3)
    @test mode == :fixed_focus_depth
    @test depth ≈ 25e-3

    @test_throws ErrorException resolve_placement_mode(:fixed_transducer, SKULL_IN_WATER; focus_depth_from_inner_skull=20e-3)
    @test_throws ErrorException resolve_placement_mode(:fixed_focus_depth, WATER)
    @test_throws ErrorException resolve_placement_mode("bad_mode", WATER)
end

@testset "Phase unwrapping" begin
    truth = collect(range(0.0, 4π; length=17))
    wrapped = mod.(truth .+ π, 2π) .- π
    @test TranscranialFUS._unwrap_phase(wrapped) ≈ truth atol=1e-10
end

@testset "Focus analysis" begin
    cfg = SimulationConfig(z_focus=0.02, x_focus=0.0, dx=0.5e-3, dz=0.5e-3, transverse_dim=0.03, trans_aperture=0.01, axial_padding=2.5)
    cfg.trans_index = Nx(cfg) - cfg.PML_GUARD
    kgrid = KGrid2D(Nx(cfg), Nz(cfg), cfg.dx, cfg.dz; dt=cfg.dt, Nt=Nt(cfg))
    row_tgt = target_index(cfg)
    col_tgt = fld(length(kgrid.y_vec), 2) + 1

    p = Array{Float64}(undef, kgrid.Nx, kgrid.Ny)
    σ = 1.5e-3
    for i in 1:kgrid.Nx, j in 1:kgrid.Ny
        p[i, j] = exp(-((kgrid.x_vec[i] - kgrid.x_vec[row_tgt])^2 + (kgrid.y_vec[j] - kgrid.y_vec[col_tgt])^2) / (2σ^2))
    end

    stats = analyse_focus_2d(p, kgrid, cfg)
    @test stats[:error_mm] < 0.6
    @test stats[:p_peak] ≈ maximum(p)
    @test stats[:focal_area_mm2] > 0
end

@testset "k-Wave smoke tests" begin
    if get(ENV, "TRANSCRANIALFUS_RUN_KWAVE_TESTS", "0") == "1" && kwave_available()
        cfg = SimulationConfig(
            z_focus=0.01,
            dx=1e-3,
            dz=1e-3,
            transverse_dim=0.02,
            trans_aperture=0.01,
            axial_padding=2.0,
            PML_GUARD=5,
            t_max=8e-6,
            dt=50e-9,
        )
        cfg.trans_index = Nx(cfg) - cfg.PML_GUARD
        c = fill(Float32(cfg.c0), Nx(cfg), Nz(cfg))
        rho = fill(Float32(cfg.rho0), Nx(cfg), Nz(cfg))

        p_rms, _, _ = focus(c, rho, GEOMETRIC, cfg, SweepSettings(record=:p_rms))
        @test size(p_rms) == (Nx(cfg), Nz(cfg))

        p_ts, _, _ = focus(c, rho, GEOMETRIC, cfg, SweepSettings(record=:p))
        @test size(p_ts, 1) == Nx(cfg)
        @test size(p_ts, 2) == Nz(cfg)
        @test size(p_ts, 3) == Nt(cfg)
    else
        @info "Skipping k-Wave smoke tests. Set TRANSCRANIALFUS_RUN_KWAVE_TESTS=1 to enable them."
    end
end

@testset "Optional CT integration" begin
    if isdir(DEFAULT_CT_PATH) && get(ENV, "TRANSCRANIALFUS_RUN_INTEGRATION", "0") == "1" && kwave_available()
        hu_vol, _ = load_default_ct()
        cfg = SimulationConfig(z_focus=0.05, trans_aperture=0.05)
        stats, pressure, _, _, _, _ = run_focus_case(hu_vol, cfg, SKULL_IN_WATER, GEOMETRIC, SweepSettings(); slice_index=250)
        @test pressure !== nothing
        @test stats !== nothing
        @test stats[:p_peak] > 0
    else
        @info "Skipping CT integration test. Requires the local CT dataset, k-Wave availability, and TRANSCRANIALFUS_RUN_INTEGRATION=1."
    end
end
