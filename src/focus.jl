mutable struct KGrid2D
    Nx::Int
    Ny::Int
    dx::Float64
    dy::Float64
    dt::Float64
    Nt::Int
    x_vec::Vector{Float64}
    y_vec::Vector{Float64}
end

@enum MediumType WATER SKULL_IN_WATER
@enum Est GEOMETRIC HASA

function KGrid2D(nx::Int, ny::Int, dx::Real, dy::Real; dt::Real, Nt::Integer)
    return KGrid2D(
        nx,
        ny,
        Float64(dx),
        Float64(dy),
        Float64(dt),
        Int(Nt),
        _kwave_axis(nx, Float64(dx)),
        _kwave_axis(ny, Float64(dy)),
    )
end

function _kwave_axis(n::Int, d::Float64)
    if iseven(n)
        return collect((-n ÷ 2):(n ÷ 2 - 1)) .* d
    end
    half = (n - 1) ÷ 2
    return collect(-half:half) .* d
end

mutable struct SimulationConfig
    fc::Float64
    x_focus::Float64
    z_focus::Float64
    trans_aperture::Union{Nothing, Float64}
    trans_index::Int
    trans_skull_dist::Float64
    focus_depth_from_inner_skull::Union{Nothing, Float64}
    dx::Float64
    dz::Float64
    axial_dim::Float64
    transverse_dim::Float64
    axial_padding::Float64
    t_max::Float64
    dt::Float64
    c0::Float64
    rho0::Float64
    PML_GUARD::Int
end

function SimulationConfig(;
    fc::Real=5e5,
    x_focus::Real=0.0,
    z_focus::Real=50e-3,
    trans_aperture::Union{Nothing, Real}=100e-3,
    trans_index::Integer=0,
    trans_skull_dist::Real=5e-3,
    focus_depth_from_inner_skull::Union{Nothing, Real}=nothing,
    dx::Real=0.2e-3,
    dz::Real=0.2e-3,
    transverse_dim::Real=120e-3,
    axial_padding::Real=1.5,
    t_max::Real=220e-6,
    dt::Real=40e-9,
    c0::Real=1500.0,
    rho0::Real=1000.0,
    PML_GUARD::Integer=20,
)
    trans_ap = isnothing(trans_aperture) ? nothing : Float64(trans_aperture)
    transverse_dim_f = Float64(transverse_dim)
    if !isnothing(trans_ap) && trans_ap > transverse_dim_f
        error("Transducer aperture must be <= transverse grid size.")
    end

    focus_depth = isnothing(focus_depth_from_inner_skull) ? nothing : Float64(focus_depth_from_inner_skull)
    return SimulationConfig(
        Float64(fc),
        Float64(x_focus),
        Float64(z_focus),
        trans_ap,
        Int(trans_index),
        Float64(trans_skull_dist),
        focus_depth,
        Float64(dx),
        Float64(dz),
        Float64(axial_padding) * Float64(z_focus),
        transverse_dim_f,
        Float64(axial_padding),
        Float64(t_max),
        Float64(dt),
        Float64(c0),
        Float64(rho0),
        Int(PML_GUARD),
    )
end

Base.@kwdef struct SweepSettings
    use_hasa_phase::Bool = true
    use_hasa_amplitude::Bool = false
    generate_figure::Bool = false
    slice_index_for_skull_ct::Int = 250
    use_gpu::Bool = false
    exclude_skull::Bool = true
    mask_outside::Bool = true
    record::Symbol = :p_rms
end

Base.@kwdef struct AnimationSettings
    num_cycles::Int = 1
    Nt::Int = 200
    run_kwave::Bool = true
end

omega(cfg::SimulationConfig) = 2π * cfg.fc
Nx(cfg::SimulationConfig) = round(Int, cfg.axial_dim / cfg.dx)
Nz(cfg::SimulationConfig) = round(Int, cfg.transverse_dim / cfg.dz)
Nt(cfg::SimulationConfig) = round(Int, cfg.t_max / cfg.dt)
Nx_hasa(cfg::SimulationConfig) = round(Int, cfg.z_focus / cfg.dx)
target_index(cfg::SimulationConfig) = cfg.trans_index - Nx_hasa(cfg)
Nz_active(cfg::SimulationConfig) = isnothing(cfg.trans_aperture) ? Nz(cfg) : round(Int, cfg.trans_aperture / cfg.dz)

function active_col_range(cfg::SimulationConfig)
    mid = fld(Nz(cfg), 2) + 1
    half = fld(Nz_active(cfg), 2)
    start_col = mid - half
    end_col = start_col + Nz_active(cfg) - 1
    return start_col:end_col
end

function set_z_focus!(cfg::SimulationConfig, new_z::Real)
    cfg.z_focus = Float64(new_z)
    cfg.axial_dim = cfg.axial_padding * cfg.z_focus
    return cfg
end

_fftshift(v::AbstractVector) = circshift(v, fld(length(v), 2))
_ifftshift(v::AbstractVector) = circshift(v, -fld(length(v), 2))
_ifftshift_dim2(a::AbstractMatrix) = circshift(a, (0, -fld(size(a, 2), 2)))

function _unwrap_phase(v::AbstractVector{<:Real}; discont::Real=π)
    out = collect(Float64, v)
    offset = 0.0
    for i in 2:length(out)
        delta = out[i] - out[i - 1]
        if delta > discont
            offset -= 2π
        elseif delta < -discont
            offset += 2π
        end
        out[i] += offset
    end
    return out
end

function _unwrap_phase_rows(a::AbstractMatrix{<:Real}; discont::Real=π)
    out = Matrix{Float64}(undef, size(a))
    for row in axes(a, 1)
        out[row, :] = _unwrap_phase(view(a, row, :); discont=discont)
    end
    return out
end

function plot_hasa_results(
    z::AbstractVector{<:Real},
    x::AbstractVector{<:Real},
    cfg::SimulationConfig,
    asamap::AbstractMatrix{<:Real},
    delays::AbstractVector{<:Real},
    delays_h::AbstractVector{<:Real},
    amplitudes::AbstractVector{<:Real},
)
    max_amp = max(maximum(asamap), eps(Float64))
    asa_db = 20 .* log10.(max.(Float64.(asamap)', eps(Float64)) ./ max_amp)

    fig = CairoMakie.Figure(size=(1500, 500))
    ax1 = CairoMakie.Axis(
        fig[1, 1];
        title="HASA outward Propagation",
        xlabel="z [mm]",
        ylabel="x [mm]",
        aspect=CairoMakie.DataAspect(),
    )
    hm = CairoMakie.heatmap!(
        ax1,
        (Float64.(z) .- cfg.z_focus) .* 1e3,
        Float64.(x) .* 1e3,
        asa_db;
        colormap=:viridis,
        colorrange=(-80, 0),
    )
    CairoMakie.Colorbar(fig[1, 2], hm; label="Amplitude [dB]")

    ax2 = CairoMakie.Axis(
        fig[1, 3];
        title="Delays",
        xlabel="Element Position [mm]",
        ylabel="Delay [μs]",
    )
    CairoMakie.lines!(ax2, Float64.(x) .* 1e3, Float64.(delays) .* 1e6; color=:black, linewidth=2.2)
    CairoMakie.lines!(ax2, Float64.(x) .* 1e3, Float64.(delays_h) .* 1e6; color=:black, linestyle=:dash, linewidth=2.0)

    ax3 = CairoMakie.Axis(
        fig[1, 4];
        title="Amplitude",
        xlabel="Element Position [mm]",
        ylabel="Normalized Amplitude",
    )
    norm_amp = Float64.(amplitudes) ./ max(maximum(abs.(amplitudes)), eps(Float64))
    CairoMakie.lines!(ax3, Float64.(x) .* 1e3, norm_amp; color=:black, linewidth=2.2)
    CairoMakie.lines!(ax3, Float64.(x) .* 1e3, ones(length(x)); color=:black, linestyle=:dash, linewidth=2.0)
    return fig
end

function _focus_impl(
    c::AbstractMatrix{<:Real},
    rho::AbstractMatrix{<:Real},
    est_type::Est,
    cfg::SimulationConfig,
    sweep_settings::SweepSettings,
    animation_settings::Union{Nothing, AnimationSettings},
)
    nt = isnothing(animation_settings) ? Nt(cfg) : animation_settings.Nt
    kgrid = KGrid2D(Nx(cfg), Nz(cfg), cfg.dx, cfg.dz; dt=cfg.dt, Nt=nt)

    z_trans_index = cfg.trans_index
    z_start_index = z_trans_index - Nx_hasa(cfg)
    z_start_index >= 1 || error("Transducer index is too shallow for the requested focus depth.")

    z = kgrid.x_vec[z_start_index:(z_trans_index - 1)] .- kgrid.x_vec[z_start_index]
    x = kgrid.y_vec
    dz_local = length(z) > 1 ? z[2] - z[1] : cfg.dx
    dx_local = length(x) > 1 ? x[2] - x[1] : cfg.dz

    asamap = zeros(Float64, length(z), kgrid.Ny)
    dk = 2π / dx_local
    start_val = -fld(length(x), 2)
    end_val = ceil(Int, length(x) / 2) - 1
    k = collect(start_val:end_val) .* dk ./ length(x)

    c_hasa = Float64.(c[z_start_index:z_trans_index, :])
    phase_map = zeros(Float64, length(z), length(x))

    half_width = 15
    x_local = collect(-half_width:half_width) ./ (2 * half_width + 1)
    x_local .-= mean(x_local)
    p_init = (cos.(2π .* x_local) .+ 1.0) .* exp.(-((x_local ./ 0.15) .^ 2))
    p0 = zeros(Float64, length(x))
    x_index = argmin(abs.(x .- cfg.x_focus))
    left = x_index - half_width
    right = x_index + half_width
    source_left = 1
    source_right = length(p_init)
    if left < 1
        source_left += 1 - left
        left = 1
    end
    if right > length(x)
        source_right -= right - length(x)
        right = length(x)
    end
    p0[left:right] .= 1e3 .* p_init[source_left:source_right]

    k0 = omega(cfg) / cfg.c0
    kz = sqrt.(complex.(k0^2 .- k .^ 2, 0.0))
    real_inds = findall(real.(kz ./ k0) .> 0.0)
    x_taper = length(real_inds) == 0 ? Float64[] : collect(range(0.0, 1.0; length=length(real_inds)))
    r = 0.25
    window = similar(x_taper)
    for idx in eachindex(x_taper)
        xi = x_taper[idx]
        if xi < r / 2
            window[idx] = 0.5 * (1 + cos((2π / r) * (xi - r / 2)))
        elseif xi < 1 - r / 2
            window[idx] = 1.0
        else
            window[idx] = 0.5 * (1 + cos((2π / r) * (xi - (1 - r / 2))))
        end
    end
    weighting = zeros(Float64, length(x))
    weighting[real_inds] .= window

    c_prime = c_hasa .- cfg.c0
    mu = (1 .+ c_prime ./ cfg.c0) .^ -2
    lam = (k0^2) .* (1 .- mu)

    p0_hat = _fftshift(fft(p0))
    p_hat = repeat(reshape(p0_hat, 1, :), length(z), 1)
    propagator = exp.(1im .* kz .* dz_local)

    current = copy(p0_hat)
    for zi in 2:length(z)
        pz = ifft(_ifftshift(current))
        pz_hat = _fftshift(fft(lam[zi, :] .* pz))
        p1 = current .* propagator .+ propagator ./ (2im .* kz) .* pz_hat .* dz_local
        p1[setdiff(eachindex(p1), real_inds)] .= 0.0
        p1 .*= weighting
        p_hat[zi, :] .= p1
        current = p1
    end

    p_field = ifft(_ifftshift_dim2(p_hat), 2)
    asamap .+= abs.(p_field) .^ 2
    phase_map .= _unwrap_phase_rows(angle.(p_field); discont=π)
    amp_map = abs.(p_field)
    amp_norm = amp_map ./ max(maximum(amp_map), eps(Float64))
    amplitudes = vec(amp_norm[end, :])

    delays = vec(phase_map[end, :]) ./ omega(cfg)
    tau_kw_h = maximum(delays) .- delays
    delays .= delays .- maximum(delays)

    delays_h = sqrt.((x .- cfg.x_focus) .^ 2 .+ cfg.z_focus^2) ./ cfg.c0
    tau_kw_geo = maximum(delays_h) .- delays_h
    delays_h .= delays_h .- maximum(delays_h)

    fig = sweep_settings.generate_figure ? plot_hasa_results(z, x, cfg, asamap, delays, delays_h, amplitudes) : nothing

    if est_type == GEOMETRIC
        tau = copy(tau_kw_geo)
        amplitudes = ones(length(amplitudes))
    else
        tau = sweep_settings.use_hasa_phase ? copy(tau_kw_h) : copy(tau_kw_geo)
        if !sweep_settings.use_hasa_amplitude
            amplitudes = ones(length(amplitudes))
        end
    end

    col_range = active_col_range(cfg)
    tau = tau[col_range]
    amplitudes = amplitudes[col_range]

    pressure = nothing
    run_kwave = isnothing(animation_settings) || animation_settings.run_kwave
    if run_kwave
        pressure = _simulate_kwave(
            c,
            rho,
            tau,
            amplitudes,
            cfg,
            kgrid,
            col_range;
            record=sweep_settings.record,
            use_gpu=sweep_settings.use_gpu,
            animation_settings=animation_settings,
        )
    end

    hasa_info = Dict{Symbol, Any}(
        :fig => fig,
        :p_hasa => p_field,
        :c_hasa => c_hasa,
        :tau => tau,
        :tau_geo => tau_kw_geo[col_range],
        :amplitudes => amplitudes,
    )
    return pressure, hasa_info, kgrid
end

function focus(
    c::AbstractMatrix{<:Real},
    rho::AbstractMatrix{<:Real},
    est_type::Est,
    cfg::SimulationConfig,
    sweep_settings::SweepSettings,
    animation_settings::Union{Nothing, AnimationSettings},
)
    return _focus_impl(c, rho, est_type, cfg, sweep_settings, animation_settings)
end

function focus(
    c::AbstractMatrix{<:Real},
    rho::AbstractMatrix{<:Real},
    est_type::Est,
    cfg::SimulationConfig,
    sweep_settings::SweepSettings;
    animation_settings::Union{Nothing, AnimationSettings}=nothing,
)
    return _focus_impl(c, rho, est_type, cfg, sweep_settings, animation_settings)
end
