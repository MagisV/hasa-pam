function run_pam_case_3d(
    c::AbstractArray{<:Real, 3},
    rho::AbstractArray{<:Real, 3},
    sources::AbstractVector{<:EmissionSource3D},
    cfg::PAMConfig3D;
    frequencies::Union{Nothing, AbstractVector{<:Real}}=nothing,
    bandwidth::Real=0.0,
    kwave_use_gpu::Bool=true,
    recon_use_gpu::Bool=true,
    reconstruction_axial_step::Union{Nothing, Real}=nothing,
    reconstruction_mode::Symbol=:full,
    window_config::PAMWindowConfig=PAMWindowConfig(),
    show_progress::Bool=false,
    benchmark::Bool=false,
    window_batch::Int=1,
    simulation_backend::Symbol=:analytic,
    source_phase_mode::Symbol=:coherent,
    rng::Random.AbstractRNG=Random.default_rng(),
    source_variability::SourceVariabilityConfig=SourceVariabilityConfig(),
)
    recon_use_gpu || error("3D PAM reconstruction currently requires --recon-use-gpu=true.")
    recon_freqs = isnothing(frequencies) ? default_recon_frequencies(sources) : Float64.(frequencies)
    phase_mode = _normalize_source_phase_mode(source_phase_mode)
    recon_mode = phase_mode == :random_phase_per_window ?
        :windowed :
        _normalize_reconstruction_mode(reconstruction_mode)
    effective_window_config = PAMWindowConfig(;
        enabled=recon_mode == :windowed,
        window_duration=window_config.window_duration,
        hop=window_config.hop,
        taper=window_config.taper,
        min_energy_ratio=window_config.min_energy_ratio,
        accumulation=window_config.accumulation,
    )
    sim_sources = sources
    n_frames = 1
    if phase_mode == :random_static_phase
        sim_sources = _resample_source_phases_3d(sources, rng)
    elseif phase_mode == :random_phase_per_window
        sim_sources, n_frames = _expand_sources_per_window(
            sources,
            effective_window_config.window_duration,
            effective_window_config.hop,
            cfg.t_max,
            rng;
            variability=source_variability,
        )
    end
    rf, grid, sim_info = if simulation_backend == :kwave
        simulate_point_sources_3d(c, rho, sim_sources, cfg; use_gpu=kwave_use_gpu)
    else
        analytic_rf_for_point_sources_3d(cfg, sim_sources)
    end
    recon_kwargs = (
        frequencies=recon_freqs,
        bandwidth=bandwidth,
        reference_sound_speed=_pam_reference_sound_speed(c, cfg, sources),
        axial_step=reconstruction_axial_step,
        use_gpu=recon_use_gpu,
        show_progress=show_progress,
        benchmark=benchmark,
        window_batch=window_batch,
    )
    pam_geo, _, geo_info = if recon_mode == :windowed
        reconstruct_pam_windowed_3d(
            rf,
            c,
            cfg;
            recon_kwargs...,
            corrected=false,
            window_config=effective_window_config,
        )
    else
        reconstruct_pam_3d(rf, c, cfg; recon_kwargs..., corrected=false)
    end
    pam_hasa, _, hasa_info = if recon_mode == :windowed
        reconstruct_pam_windowed_3d(
            rf,
            c,
            cfg;
            recon_kwargs...,
            corrected=true,
            window_config=effective_window_config,
        )
    else
        reconstruct_pam_3d(rf, c, cfg; recon_kwargs..., corrected=true)
    end

    return Dict{Symbol, Any}(
        :rf => Float64.(rf),
        :kgrid => grid,
        :simulation => sim_info,
        :pam_geo => pam_geo,
        :pam_hasa => pam_hasa,
        :geo_info => geo_info,
        :hasa_info => hasa_info,
        :stats_geo => any(s -> s isa BubbleCluster3D, sources) ? Dict{Symbol,Any}() : analyse_pam_3d(pam_geo, grid, cfg, sources),
        :stats_hasa => any(s -> s isa BubbleCluster3D, sources) ? Dict{Symbol,Any}() : analyse_pam_3d(pam_hasa, grid, cfg, sources),
        :reconstruction_frequencies => recon_freqs,
        :analysis_mode => any(s -> s isa BubbleCluster3D, sources) ? :detection : :localization,
        :analysis_source_count => length(sources),
        :emission_event_count => length(sim_sources),
        :reconstruction_mode => recon_mode,
        :source_phase_mode => phase_mode,
        :n_frames => n_frames,
        :window_config => _window_config_info(effective_window_config),
        :kwave_use_gpu => kwave_use_gpu,
        :recon_use_gpu => recon_use_gpu,
        :show_progress => show_progress,
    )
end
