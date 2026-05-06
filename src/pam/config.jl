Base.@kwdef struct PAMConfig
    dx::Float64 = 0.2e-3
    dz::Float64 = 0.2e-3
    axial_dim::Float64 = 90e-3
    transverse_dim::Float64 = 60e-3
    receiver_aperture::Union{Nothing, Float64} = 50e-3
    receiver_row::Union{Nothing, Int} = nothing
    t_max::Float64 = 80e-6
    dt::Float64 = 40e-9
    c0::Float64 = 1500.0
    rho0::Float64 = 1000.0
    PML_GUARD::Int = 20
    zero_pad_factor::Int = 4
    tukey_ratio::Float64 = 0.25
    peak_suppression_radius::Float64 = 2e-3
    success_tolerance::Float64 = 1e-3
end

Base.@kwdef struct PAMWindowConfig
    enabled::Bool = false
    window_duration::Float64 = 10e-6
    hop::Float64 = 5e-6
    taper::Symbol = :hann
    min_energy_ratio::Float64 = 1e-3
    accumulation::Symbol = :intensity
end


function _default_pam_pml_guard(dx::Real)
    # Keep the default physical guard close to 4 mm across PAM resolutions.
    return max(4, round(Int, 4e-3 / Float64(dx)))
end

function _pam_pml_guard(cfg::PAMConfig)
    # `PML_GUARD=20` is the historical default. Interpret it as a placeholder and
    # scale the actual guard with `dx` so coarse grids don't silently lose most
    # of the reconstruction depth.
    if cfg.PML_GUARD == 20
        return _default_pam_pml_guard(cfg.dx)
    end
    return cfg.PML_GUARD
end

function _pam_axial_substeps(dx::Real, axial_step::Real)
    ratio = Float64(dx) / Float64(axial_step)
    nearest = round(Int, ratio)
    if isapprox(ratio, nearest; rtol=1e-9, atol=1e-12)
        return max(1, nearest)
    end
    return max(1, ceil(Int, ratio))
end

function _required_pam_t_max(
    cfg::PAMConfig,
    sources::AbstractVector{<:EmissionSource2D};
    time_margin::Real=10e-6,
)
    isempty(sources) && return 0.0
    kgrid = pam_grid(cfg)
    receiver_laterals = kgrid.y_vec[receiver_col_range(cfg)]
    required_t = 0.0
    for src in sources
        max_receiver_offset = maximum(abs.(receiver_laterals .- src.lateral))
        latest_arrival = src.delay + hypot(src.depth, max_receiver_offset) / cfg.c0
        required_t = max(required_t, latest_arrival + _source_duration(src))
    end
    return required_t + Float64(time_margin)
end

function fit_pam_config(
    cfg::PAMConfig,
    sources::AbstractVector{<:EmissionSource2D};
    min_bottom_margin::Real=10e-3,
    reference_depth::Union{Nothing, Real}=nothing,
    time_margin::Real=10e-6,
)
    deepest_source_depth = isempty(sources) ? 0.0 : maximum(src.depth for src in sources)
    deepest_required = max(
        deepest_source_depth,
        isnothing(reference_depth) ? 0.0 : Float64(reference_depth),
    ) + Float64(min_bottom_margin)
    required_rows = receiver_row(cfg) + round(Int, deepest_required / cfg.dx)
    target_rows = max(pam_Nx(cfg), required_rows)
    target_axial_dim = target_rows * cfg.dx
    target_t_max = max(cfg.t_max, _required_pam_t_max(cfg, sources; time_margin=time_margin))

    target_rows == pam_Nx(cfg) && target_t_max == cfg.t_max && return cfg
    return PAMConfig(
        dx=cfg.dx,
        dz=cfg.dz,
        axial_dim=target_axial_dim,
        transverse_dim=cfg.transverse_dim,
        receiver_aperture=cfg.receiver_aperture,
        receiver_row=cfg.receiver_row,
        t_max=target_t_max,
        dt=cfg.dt,
        c0=cfg.c0,
        rho0=cfg.rho0,
        PML_GUARD=cfg.PML_GUARD,
        zero_pad_factor=cfg.zero_pad_factor,
        tukey_ratio=cfg.tukey_ratio,
        peak_suppression_radius=cfg.peak_suppression_radius,
        success_tolerance=cfg.success_tolerance,
    )
end

pam_Nx(cfg::PAMConfig) = round(Int, cfg.axial_dim / cfg.dx)
pam_Ny(cfg::PAMConfig) = round(Int, cfg.transverse_dim / cfg.dz)
pam_Nt(cfg::PAMConfig) = round(Int, cfg.t_max / cfg.dt)
receiver_row(cfg::PAMConfig) = something(cfg.receiver_row, 1)

function receiver_col_range(cfg::PAMConfig)
    ny = pam_Ny(cfg)
    if isnothing(cfg.receiver_aperture)
        return 1:ny
    end
    n_active = clamp(round(Int, cfg.receiver_aperture / cfg.dz), 1, ny)
    mid = fld(ny, 2) + 1
    half = fld(n_active, 2)
    start_col = mid - half
    end_col = start_col + n_active - 1
    return start_col:end_col
end

function pam_grid(cfg::PAMConfig; Nt::Union{Nothing, Integer}=nothing)
    nt = isnothing(Nt) ? pam_Nt(cfg) : Int(Nt)
    return KGrid2D(pam_Nx(cfg), pam_Ny(cfg), cfg.dx, cfg.dz; dt=cfg.dt, Nt=nt)
end

function depth_coordinates(kgrid::KGrid2D, cfg::PAMConfig)
    return kgrid.x_vec .- kgrid.x_vec[receiver_row(cfg)]
end

function source_grid_index(src::EmissionSource2D, cfg::PAMConfig, kgrid::KGrid2D)
    src.depth >= 0.0 || error("Source depth must be >= 0.")
    row = receiver_row(cfg) + round(Int, src.depth / cfg.dx)
    col = argmin(abs.(kgrid.y_vec .- src.lateral))
    1 <= row <= kgrid.Nx || error("Source depth $(src.depth) m lies outside the computational grid.")
    return row, col
end

function _normalize_reconstruction_mode(reconstruction_mode)
    mode = Symbol(replace(lowercase(string(reconstruction_mode)), "-" => "_"))
    mode in (:full, :windowed) ||
        error("Unknown reconstruction_mode: $reconstruction_mode (expected full or windowed).")
    return mode
end

function pam_reconstruction_mode(reconstruction_mode, cluster_model)
    mode = Symbol(replace(lowercase(string(reconstruction_mode)), "-" => "_"))
    model = Symbol(replace(lowercase(string(cluster_model)), "-" => "_"))
    model in (:point, :vascular) ||
        error("Unknown cluster_model: $cluster_model (expected point or vascular).")
    mode == :auto && return model == :vascular ? :windowed : :full
    return _normalize_reconstruction_mode(mode)
end

function _normalize_window_taper(taper)
    mode = Symbol(replace(lowercase(string(taper)), "-" => "_"))
    mode in (:hann, :none, :rect, :rectangular, :tukey) ||
        error("Unknown PAM window taper: $taper (expected hann, none, rectangular, or tukey).")
    return mode
end

function _validate_window_config(config::PAMWindowConfig)
    config.window_duration > 0 || error("window_duration must be positive.")
    config.hop > 0 || error("hop must be positive.")
    config.min_energy_ratio >= 0 || error("min_energy_ratio must be non-negative.")
    config.accumulation == :intensity ||
        error("Only intensity accumulation is supported for windowed PAM.")
    _normalize_window_taper(config.taper)
    return config
end

function _pam_window_ranges(nt::Integer, dt::Real, config::PAMWindowConfig)
    nt_i = Int(nt)
    nt_i > 0 || error("RF data must contain at least one time sample.")
    _validate_window_config(config)

    win_n = max(1, round(Int, config.window_duration / Float64(dt)))
    hop_n = max(1, round(Int, config.hop / Float64(dt)))
    if nt_i <= win_n
        return [1:nt_i], nt_i, hop_n
    end

    last_start = nt_i - win_n + 1
    starts = collect(1:hop_n:last_start)
    if last(starts) != last_start
        push!(starts, last_start)
    end
    return [start:(start + win_n - 1) for start in starts], win_n, hop_n
end

function _pam_temporal_taper(n::Integer, taper)
    n_i = Int(n)
    n_i > 0 || error("Temporal taper length must be positive.")
    mode = _normalize_window_taper(taper)
    if mode in (:none, :rect, :rectangular)
        return ones(Float64, n_i)
    elseif mode == :tukey
        return _tukey_window(n_i, 0.25)
    end
    n_i == 1 && return ones(Float64, 1)
    return [0.5 * (1 - cos(2π * (idx - 1) / (n_i - 1))) for idx in 1:n_i]
end

function _window_config_info(config::PAMWindowConfig)
    return Dict{Symbol, Any}(
        :enabled => config.enabled,
        :window_duration_s => config.window_duration,
        :hop_s => config.hop,
        :taper => config.taper,
        :min_energy_ratio => config.min_energy_ratio,
        :accumulation => config.accumulation,
    )
end

