# Simulation defaults and lightweight analytic helpers for PAM runner scripts.

"""
    default_simulation_info(cfg::PAMConfig)

Return default 2D simulation metadata for workflows without k-Wave output.
"""
function default_simulation_info(cfg::PAMConfig)
    return Dict{Symbol, Any}(
        :receiver_row => receiver_row(cfg),
        :receiver_cols => receiver_col_range(cfg),
        :source_indices => Tuple{Int, Int}[],
    )
end

"""
    default_simulation_info(cfg::PAMConfig3D)

Return default 3D simulation metadata for workflows without k-Wave output.
"""
function default_simulation_info(cfg::PAMConfig3D)
    return Dict{Symbol, Any}(
        :receiver_row => receiver_row(cfg),
        :receiver_cols_y => receiver_col_range_y(cfg),
        :receiver_cols_z => receiver_col_range_z(cfg),
        :source_indices => NTuple{3, Int}[],
    )
end

"""
    default_recon_frequencies(sources)

Return sorted unique emission frequencies in Hz for a source collection.
"""
function default_recon_frequencies(sources)
    freqs = Float64[]
    for src in sources
        append!(freqs, emission_frequencies(src))
    end
    return sort(unique(freqs))
end

"""
    _sample_source_signal(signal, t, dt)

Linearly sample a discrete source signal at physical time `t` seconds.
"""
function _sample_source_signal(signal::AbstractVector{<:Real}, t::Real, dt::Real)
    u = Float64(t) / Float64(dt) + 1.0
    i0 = floor(Int, u)
    i0 < 1 && return 0.0
    i0 > length(signal) && return 0.0
    i0 == length(signal) && return Float64(signal[i0])
    frac = u - i0
    return (1.0 - frac) * Float64(signal[i0]) + frac * Float64(signal[i0 + 1])
end

"""
    analytic_rf_for_point_sources_3d(cfg, sources)

Generate a lightweight analytic 3D receiver RF field for point-like sources.
"""
function analytic_rf_for_point_sources_3d(cfg::PAMConfig3D, sources::AbstractVector{<:EmissionSource3D})
    grid = pam_grid_3d(cfg)
    ny, nz, nt = pam_Ny(cfg), pam_Nz(cfg), pam_Nt(cfg)
    rf = zeros(Float32, ny, nz, nt)
    for src in sources
        source_signal = TranscranialFUS._source_signal(nt, cfg.dt, src)
        for iy in 1:ny, iz in 1:nz
            dy_src = grid.y[iy] - src.lateral_y
            dz_src = grid.z[iz] - src.lateral_z
            r = sqrt(src.depth^2 + dy_src^2 + dz_src^2)
            for it in 1:nt
                emission_t = (it - 1) * cfg.dt - r / cfg.c0
                rf[iy, iz, it] += Float32(_sample_source_signal(source_signal, emission_t, cfg.dt))
            end
        end
    end
    return rf, grid, Dict{Symbol, Any}(
        :receiver_row => receiver_row(cfg),
        :receiver_cols_y => receiver_col_range_y(cfg),
        :receiver_cols_z => receiver_col_range_z(cfg),
        :source_indices => [source_grid_index_3d(src, cfg) for src in sources],
    )
end
