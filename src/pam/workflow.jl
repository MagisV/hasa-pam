function _default_recon_frequencies(sources::AbstractVector{<:EmissionSource2D})
    all_freqs = Float64[]
    for src in sources
        append!(all_freqs, _emission_frequencies(src))
    end
    return sort(unique(all_freqs))
end

function _pam_reference_sound_speed(
    c::AbstractMatrix{<:Real},
    cfg::PAMConfig,
    sources::AbstractVector{<:EmissionSource2D};
    margin::Real=10e-3,
)
    isempty(sources) && return mean(Float64.(c))
    row_start = clamp(receiver_row(cfg), 1, size(c, 1))
    deepest_source_depth = maximum(src.depth for src in sources)
    row_stop = row_start + ceil(Int, (deepest_source_depth + Float64(margin)) / cfg.dx)
    row_stop = clamp(row_stop, row_start, size(c, 1))
    return mean(Float64.(view(c, row_start:row_stop, :)))
end

function _run_pam_per_window(
    c::AbstractMatrix{<:Real},
    rho::AbstractMatrix{<:Real},
    sources::AbstractVector{<:EmissionSource2D},
    cfg::PAMConfig;
    source_phase_mode::Symbol,
    use_gpu::Bool,
    rng::Random.AbstractRNG,
    recon_kwargs::NamedTuple,
    variability::SourceVariabilityConfig=SourceVariabilityConfig(),
)
    win_cfg = recon_kwargs.window_config
    expanded, n_frames = _expand_sources_per_window(
        sources, win_cfg.window_duration, win_cfg.hop, cfg.t_max, rng;
        variability=variability,
    )
    eff_window_config = PAMWindowConfig(;
        enabled=true,
        window_duration=win_cfg.window_duration,
        hop=win_cfg.hop,
        taper=win_cfg.taper,
        min_energy_ratio=win_cfg.min_energy_ratio,
        accumulation=win_cfg.accumulation,
    )
    eff_recon_kwargs = merge(recon_kwargs, (reconstruction_mode=:windowed, window_config=eff_window_config))
    rf, kgrid, sim_info = simulate_point_sources(c, rho, expanded, cfg; use_gpu=use_gpu)
    results = reconstruct_pam_case(
        rf,
        c,
        expanded,
        cfg;
        simulation_info=sim_info,
        analysis_sources=sources,
        eff_recon_kwargs...,
    )
    results[:kgrid] = kgrid
    results[:source_phase_mode] = source_phase_mode
    results[:n_realizations] = 1
    results[:n_frames] = n_frames
    return results
end

function _run_pam_multirealization(
    c::AbstractMatrix{<:Real},
    rho::AbstractMatrix{<:Real},
    sources::AbstractVector{<:EmissionSource2D},
    cfg::PAMConfig;
    n_realizations::Int,
    source_phase_mode::Symbol,
    use_gpu::Bool,
    rng::Random.AbstractRNG,
    recon_kwargs::NamedTuple,
)
    geo_acc = nothing
    hasa_acc = nothing
    last_kgrid = nothing
    last_results = nothing

    for _ in 1:n_realizations
        resampled = _resample_source_phases(sources, rng)
        rf, kgrid, sim_info = simulate_point_sources(c, rho, resampled, cfg; use_gpu=use_gpu)
        results = reconstruct_pam_case(rf, c, resampled, cfg; simulation_info=sim_info, recon_kwargs...)
        if isnothing(geo_acc)
            geo_acc = Float64.(results[:pam_geo])
            hasa_acc = Float64.(results[:pam_hasa])
        else
            geo_acc .+= results[:pam_geo]
            hasa_acc .+= results[:pam_hasa]
        end
        last_kgrid = kgrid
        last_results = results
    end

    last_results[:pam_geo] = geo_acc ./ n_realizations
    last_results[:pam_hasa] = hasa_acc ./ n_realizations
    last_results[:kgrid] = last_kgrid
    last_results[:source_phase_mode] = source_phase_mode
    last_results[:n_realizations] = n_realizations
    return last_results
end

function run_pam_case(
    c::AbstractMatrix{<:Real},
    rho::AbstractMatrix{<:Real},
    sources::AbstractVector{<:EmissionSource2D},
    cfg::PAMConfig;
    frequencies::Union{Nothing, AbstractVector{<:Real}}=nothing,
    bandwidth::Real=0.0,
    use_gpu::Bool=false,
    analysis_mode::Symbol=:localization,
    peak_method::Symbol=:argmax,
    clean_loop_gain::Real=0.1,
    clean_max_iter::Integer=500,
    clean_threshold_ratio::Real=1e-2,
    clean_psf_axial_fwhm::Union{Nothing, Real}=nothing,
    clean_psf_lateral_fwhm::Union{Nothing, Real}=nothing,
    detection_truth_radius::Real=cfg.success_tolerance,
    detection_threshold_ratio::Real=0.2,
    detection_truth_mask::Union{Nothing, AbstractMatrix{Bool}}=nothing,
    reconstruction_axial_step::Union{Nothing, Real}=50e-6,
    reconstruction_mode::Symbol=:full,
    window_config::PAMWindowConfig=PAMWindowConfig(),
    source_phase_mode::Symbol=:coherent,
    n_realizations::Int=1,
    rng::Random.AbstractRNG=Random.default_rng(),
    source_variability::SourceVariabilityConfig=SourceVariabilityConfig(),
    show_progress::Bool=false,
)
    phase_mode = _normalize_source_phase_mode(source_phase_mode)
    effective_sources = sources
    recon_freqs = isnothing(frequencies) ? _default_recon_frequencies(effective_sources) : Float64.(frequencies)
    recon_kwargs = (
        frequencies=recon_freqs,
        bandwidth=bandwidth,
        analysis_mode=analysis_mode,
        peak_method=peak_method,
        clean_loop_gain=clean_loop_gain,
        clean_max_iter=clean_max_iter,
        clean_threshold_ratio=clean_threshold_ratio,
        clean_psf_axial_fwhm=clean_psf_axial_fwhm,
        clean_psf_lateral_fwhm=clean_psf_lateral_fwhm,
        detection_truth_radius=detection_truth_radius,
        detection_threshold_ratio=detection_threshold_ratio,
        detection_truth_mask=detection_truth_mask,
        reconstruction_axial_step=reconstruction_axial_step,
        reconstruction_mode=reconstruction_mode,
        window_config=window_config,
        use_gpu=use_gpu,
        show_progress=show_progress,
    )
    if phase_mode == :random_phase_per_realization
        n_realizations >= 1 || error("n_realizations must be >= 1.")
        return _run_pam_multirealization(
            c, rho, effective_sources, cfg;
            n_realizations=n_realizations,
            source_phase_mode=phase_mode,
            use_gpu=use_gpu,
            rng=rng,
            recon_kwargs=recon_kwargs,
        )
    elseif phase_mode == :random_phase_per_window
        return _run_pam_per_window(
            c, rho, effective_sources, cfg;
            source_phase_mode=phase_mode,
            use_gpu=use_gpu,
            rng=rng,
            recon_kwargs=recon_kwargs,
            variability=source_variability,
        )
    end
    rf, kgrid, sim_info = simulate_point_sources(c, rho, effective_sources, cfg; use_gpu=use_gpu)
    results = reconstruct_pam_case(rf, c, effective_sources, cfg; simulation_info=sim_info, recon_kwargs...)
    results[:kgrid] = kgrid
    results[:source_phase_mode] = phase_mode
    results[:n_realizations] = 1
    return results
end

function reconstruct_pam_case(
    rf::AbstractMatrix{<:Real},
    c::AbstractMatrix{<:Real},
    sources::AbstractVector{<:EmissionSource2D},
    cfg::PAMConfig;
    simulation_info::AbstractDict=Dict{Symbol, Any}(
        :receiver_row => receiver_row(cfg),
        :receiver_cols => receiver_col_range(cfg),
        :source_indices => Tuple{Int, Int}[],
    ),
    frequencies::Union{Nothing, AbstractVector{<:Real}}=nothing,
    bandwidth::Real=0.0,
    analysis_mode::Symbol=:localization,
    peak_method::Symbol=:argmax,
    clean_loop_gain::Real=0.1,
    clean_max_iter::Integer=500,
    clean_threshold_ratio::Real=1e-2,
    clean_psf_axial_fwhm::Union{Nothing, Real}=nothing,
    clean_psf_lateral_fwhm::Union{Nothing, Real}=nothing,
    detection_truth_radius::Real=cfg.success_tolerance,
    detection_threshold_ratio::Real=0.2,
    detection_truth_mask::Union{Nothing, AbstractMatrix{Bool}}=nothing,
    reconstruction_axial_step::Union{Nothing, Real}=50e-6,
    reconstruction_mode::Symbol=:full,
    window_config::PAMWindowConfig=PAMWindowConfig(),
    use_gpu::Bool=false,
    show_progress::Bool=false,
    analysis_sources::Union{Nothing, AbstractVector{<:EmissionSource2D}}=nothing,
)
    size(c) == (pam_Nx(cfg), pam_Ny(cfg)) ||
        error("Sound-speed map size $(size(c)) does not match PAMConfig size ($(pam_Nx(cfg)), $(pam_Ny(cfg))).")
    size(rf, 1) == pam_Ny(cfg) ||
        error("RF data must have $(pam_Ny(cfg)) receiver rows; got $(size(rf, 1)).")

    recon_freqs = isnothing(frequencies) ? _default_recon_frequencies(sources) : Float64.(frequencies)
    reference_sound_speed = _pam_reference_sound_speed(c, cfg, sources)
    recon_kwargs = (
        frequencies=recon_freqs,
        bandwidth=bandwidth,
        reference_sound_speed=reference_sound_speed,
        axial_step=reconstruction_axial_step,
        use_gpu=use_gpu,
        show_progress=show_progress,
    )
    recon_mode = _normalize_reconstruction_mode(reconstruction_mode)
    effective_window_config = PAMWindowConfig(;
        enabled=recon_mode == :windowed,
        window_duration=window_config.window_duration,
        hop=window_config.hop,
        taper=window_config.taper,
        min_energy_ratio=window_config.min_energy_ratio,
        accumulation=window_config.accumulation,
    )
    pam_geo, kgrid, geo_info = if recon_mode == :windowed
        reconstruct_pam_windowed(
            rf,
            c,
            cfg;
            recon_kwargs...,
            corrected=false,
            window_config=effective_window_config,
        )
    else
        reconstruct_pam(rf, c, cfg; recon_kwargs..., corrected=false)
    end
    pam_hasa, _, hasa_info = if recon_mode == :windowed
        reconstruct_pam_windowed(
            rf,
            c,
            cfg;
            recon_kwargs...,
            corrected=true,
            window_config=effective_window_config,
        )
    else
        reconstruct_pam(rf, c, cfg; recon_kwargs..., corrected=true)
    end

    truth_sources = isnothing(analysis_sources) ? sources : analysis_sources
    isempty(truth_sources) && error("At least one analysis source is required.")

    stats_geo, stats_hasa = if analysis_mode == :localization
        analyse_kwargs = (
            peak_method=peak_method,
            frequencies=recon_freqs,
            clean_loop_gain=clean_loop_gain,
            clean_max_iter=clean_max_iter,
            clean_threshold_ratio=clean_threshold_ratio,
            clean_psf_axial_fwhm=clean_psf_axial_fwhm,
            clean_psf_lateral_fwhm=clean_psf_lateral_fwhm,
        )
        (
            analyse_pam_2d(pam_geo, kgrid, cfg, truth_sources; analyse_kwargs...),
            analyse_pam_2d(pam_hasa, kgrid, cfg, truth_sources; analyse_kwargs...),
        )
    elseif analysis_mode == :detection
        analyse_kwargs = (
            truth_radius=detection_truth_radius,
            threshold_ratio=detection_threshold_ratio,
            truth_mask=detection_truth_mask,
            frequencies=recon_freqs,
            psf_axial_fwhm=clean_psf_axial_fwhm,
            psf_lateral_fwhm=clean_psf_lateral_fwhm,
        )
        (
            analyse_pam_detection_2d(pam_geo, kgrid, cfg, truth_sources; analyse_kwargs...),
            analyse_pam_detection_2d(pam_hasa, kgrid, cfg, truth_sources; analyse_kwargs...),
        )
    else
        error("Unknown analysis_mode: $analysis_mode (expected :localization or :detection).")
    end

    return Dict{Symbol, Any}(
        :rf => Float64.(rf),
        :kgrid => kgrid,
        :simulation => Dict{Symbol, Any}(Symbol(key) => value for (key, value) in simulation_info),
        :pam_geo => pam_geo,
        :pam_hasa => pam_hasa,
        :geo_info => geo_info,
        :hasa_info => hasa_info,
        :stats_geo => stats_geo,
        :stats_hasa => stats_hasa,
        :reconstruction_frequencies => recon_freqs,
        :analysis_mode => analysis_mode,
        :analysis_source_count => length(truth_sources),
        :reconstruction_mode => recon_mode,
        :window_config => _window_config_info(effective_window_config),
        :use_gpu => use_gpu,
        :show_progress => show_progress,
    )
end
