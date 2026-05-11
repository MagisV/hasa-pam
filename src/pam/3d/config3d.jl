Base.@kwdef struct PAMConfig3D
    dx::Float64 = 0.2e-3
    dy::Float64 = 0.2e-3
    dz::Float64 = 0.2e-3
    axial_dim::Float64 = 90e-3
    transverse_dim_y::Float64 = 30e-3
    transverse_dim_z::Float64 = 30e-3
    dt::Float64 = 40e-9
    t_max::Float64 = 80e-6
    c0::Float64 = 1500.0
    rho0::Float64 = 1000.0
    tukey_ratio::Float64 = 0.25
    zero_pad_factor::Int = 2
    receiver_row::Union{Nothing, Int} = nothing
    receiver_aperture_y::Union{Nothing, Float64} = nothing
    receiver_aperture_z::Union{Nothing, Float64} = nothing
    PML_GUARD::Int = 20
    success_tolerance::Float64 = 1e-3
    peak_suppression_radius::Float64 = 2e-3
    axial_gain_power::Float64 = 1.5
end

function _pam_pml_guard_3d(cfg::PAMConfig3D)
    cfg.PML_GUARD == 20 && return _default_pam_pml_guard(cfg.dx)
    return cfg.PML_GUARD
end

pam_Nx(cfg::PAMConfig3D) = round(Int, cfg.axial_dim / cfg.dx)
pam_Ny(cfg::PAMConfig3D) = round(Int, cfg.transverse_dim_y / cfg.dy)
pam_Nz(cfg::PAMConfig3D) = round(Int, cfg.transverse_dim_z / cfg.dz)
pam_Nt(cfg::PAMConfig3D) = round(Int, cfg.t_max / cfg.dt)
receiver_row(cfg::PAMConfig3D) = something(cfg.receiver_row, 1)

function receiver_col_range_y(cfg::PAMConfig3D)
    ny = pam_Ny(cfg)
    isnothing(cfg.receiver_aperture_y) && return 1:ny
    n_active = clamp(round(Int, cfg.receiver_aperture_y / cfg.dy), 1, ny)
    mid = fld(ny, 2) + 1
    half = fld(n_active, 2)
    start_col = mid - half
    return start_col:(start_col + n_active - 1)
end

function receiver_col_range_z(cfg::PAMConfig3D)
    nz = pam_Nz(cfg)
    isnothing(cfg.receiver_aperture_z) && return 1:nz
    n_active = clamp(round(Int, cfg.receiver_aperture_z / cfg.dz), 1, nz)
    mid = fld(nz, 2) + 1
    half = fld(n_active, 2)
    start_col = mid - half
    return start_col:(start_col + n_active - 1)
end

function pam_grid_3d(cfg::PAMConfig3D)
    return (
        x = range(0.0; step=cfg.dx, length=pam_Nx(cfg)),
        y = range(-(cfg.transverse_dim_y / 2); step=cfg.dy, length=pam_Ny(cfg)),
        z = range(-(cfg.transverse_dim_z / 2); step=cfg.dz, length=pam_Nz(cfg)),
        t = range(0.0; step=cfg.dt, length=pam_Nt(cfg)),
    )
end

function depth_coordinates_3d(cfg::PAMConfig3D)
    rr = receiver_row(cfg)
    return [(i - rr) * cfg.dx for i in 1:pam_Nx(cfg)]
end

function _required_pam_t_max_3d(
    cfg::PAMConfig3D,
    sources::AbstractVector;
    time_margin::Real=10e-6,
)
    isempty(sources) && return 0.0
    grid = pam_grid_3d(cfg)
    recv_y = grid.y[receiver_col_range_y(cfg)]
    recv_z = grid.z[receiver_col_range_z(cfg)]
    required_t = 0.0
    for src in sources
        max_dy = maximum(abs.(recv_y .- src.lateral_y))
        max_dz = maximum(abs.(recv_z .- src.lateral_z))
        latest_arrival = src.delay + sqrt(src.depth^2 + max_dy^2 + max_dz^2) / cfg.c0
        required_t = max(required_t, latest_arrival + _source_duration(src))
    end
    return required_t + Float64(time_margin)
end

function fit_pam_config_3d(
    cfg::PAMConfig3D,
    sources::AbstractVector;
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
    target_t_max = max(cfg.t_max, _required_pam_t_max_3d(cfg, sources; time_margin=time_margin))

    target_rows == pam_Nx(cfg) && target_t_max == cfg.t_max && return cfg
    return PAMConfig3D(
        dx=cfg.dx,
        dy=cfg.dy,
        dz=cfg.dz,
        axial_dim=target_axial_dim,
        transverse_dim_y=cfg.transverse_dim_y,
        transverse_dim_z=cfg.transverse_dim_z,
        dt=cfg.dt,
        t_max=target_t_max,
        c0=cfg.c0,
        rho0=cfg.rho0,
        tukey_ratio=cfg.tukey_ratio,
        zero_pad_factor=cfg.zero_pad_factor,
        receiver_row=cfg.receiver_row,
        receiver_aperture_y=cfg.receiver_aperture_y,
        receiver_aperture_z=cfg.receiver_aperture_z,
        success_tolerance=cfg.success_tolerance,
        peak_suppression_radius=cfg.peak_suppression_radius,
        axial_gain_power=cfg.axial_gain_power,
    )
end

function source_grid_index_3d(src, cfg::PAMConfig3D)
    src.depth >= 0.0 || error("Source depth must be >= 0.")
    grid = pam_grid_3d(cfg)
    row = receiver_row(cfg) + round(Int, src.depth / cfg.dx)
    col_y = argmin(abs.(grid.y .- src.lateral_y))
    col_z = argmin(abs.(grid.z .- src.lateral_z))
    1 <= row <= pam_Nx(cfg) || error("Source depth $(src.depth) m lies outside the computational grid.")
    return row, col_y, col_z
end
