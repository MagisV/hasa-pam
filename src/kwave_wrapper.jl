const _PYTHONCALL_PKGID = Base.PkgId(Base.UUID("6099a3de-0909-46bc-b1f4-468b9a2dfc0d"), "PythonCall")

function _pythoncall_module()
    Base.require(_PYTHONCALL_PKGID)
    return Base.loaded_modules[_PYTHONCALL_PKGID]
end

function _kwave_modules()
    pc = _pythoncall_module()
    CondaPkg.resolve()
    return (
        PythonCall=pc,
        np=pc.pyimport("numpy"),
        copy=pc.pyimport("copy"),
        kgrid=pc.pyimport("kwave.kgrid"),
        kmedium=pc.pyimport("kwave.kmedium"),
        ksource=pc.pyimport("kwave.ksource"),
        ksensor=pc.pyimport("kwave.ksensor"),
        kspace=pc.pyimport("kwave.kspaceFirstOrder2D"),
        simopts=pc.pyimport("kwave.options.simulation_options"),
        execopts=pc.pyimport("kwave.options.simulation_execution_options"),
        signals=pc.pyimport("kwave.utils.signals"),
    )
end

function kwave_available()
    try
        _kwave_modules()
        return true
    catch err
        @info "k-Wave Python backend is not available yet." exception=(err, catch_backtrace())
        return false
    end
end

function _normalize_record(record::Union{Symbol, AbstractString})
    symbol = record isa Symbol ? record : Symbol(record)
    symbol in (:p_rms, :p) || error("Unsupported record mode: $record")
    return symbol
end

function _py_bool_matrix(np, rows::Int, cols::Int)
    pc = _pythoncall_module()
    return np.zeros((rows, cols), dtype=pc.pybuiltins.bool)
end

function _simulate_kwave(
    c::AbstractMatrix{<:Real},
    rho::AbstractMatrix{<:Real},
    tau::AbstractVector{<:Real},
    amplitudes::AbstractVector{<:Real},
    cfg::SimulationConfig,
    kgrid::KGrid2D,
    col_range::UnitRange{Int};
    record::Union{Symbol, AbstractString}=:p_rms,
    use_gpu::Bool=false,
    animation_settings::Union{Nothing, AnimationSettings}=nothing,
)
    mods = _kwave_modules()
    pc = mods.PythonCall
    np = mods.np
    deepcopy_py = mods.copy.deepcopy

    rec_symbol = _normalize_record(record)
    nt = isnothing(animation_settings) ? kgrid.Nt : animation_settings.Nt
    num_cycles = isnothing(animation_settings) ? 40 : animation_settings.num_cycles
    use_envelope = isnothing(animation_settings)
    rise_fall = max(1, floor(Int, 0.1 * num_cycles))

    py_kgrid = mods.kgrid.kWaveGrid(pc.Py([kgrid.Nx, kgrid.Ny]), pc.Py([kgrid.dx, kgrid.dy]))
    py_kgrid.dt = kgrid.dt
    py_kgrid.Nt = nt

    medium = mods.kmedium.kWaveMedium(
        sound_speed=pc.Py(Float32.(c)),
        density=pc.Py(Float32.(rho)),
    )

    source = mods.ksource.kSource()
    src_mask = _py_bool_matrix(np, kgrid.Nx, kgrid.Ny)
    src_mask[cfg.trans_index - 1, (first(col_range) - 1):(last(col_range) - 1)] = true
    source.p_mask = src_mask

    int_delay = floor.(Int, tau ./ cfg.dt)
    frac_delay = Float64.(tau .- int_delay .* cfg.dt)

    if use_envelope
        burst_py = mods.signals.tone_burst(
            sample_freq=1 / cfg.dt,
            signal_freq=cfg.fc,
            num_cycles=num_cycles,
            envelope=pc.Py([rise_fall, rise_fall]),
            signal_offset=pc.Py(int_delay),
        )
    else
        burst_py = mods.signals.tone_burst(
            sample_freq=1 / cfg.dt,
            signal_freq=cfg.fc,
            num_cycles=num_cycles,
            signal_offset=pc.Py(int_delay),
        )
    end
    burst = pc.pyconvert(Matrix{Float64}, burst_py)
    drive_weights = (1e5 .* Float64.(amplitudes) .* exp.(-1im .* omega(cfg) .* frac_delay))
    p_drive = real.(reshape(drive_weights, :, 1) .* burst)
    source.p = pc.Py(p_drive)
    source.p_frequency_ref = cfg.fc
    source.medium = medium

    sensor = mods.ksensor.kSensor()
    sensor.mask = np.ones((kgrid.Nx, kgrid.Ny), dtype=pc.pybuiltins.bool)
    sensor.record = pc.Py([String(rec_symbol)])

    sim_opts = mods.simopts.SimulationOptions(
        pml_inside=true,
        pml_size=10,
        data_recast=false,
        save_to_disk=true,
    )
    exec_opts = mods.execopts.SimulationExecutionOptions(
        is_gpu_simulation=use_gpu,
        delete_data=false,
    )

    data = mods.kspace.kspaceFirstOrder2D(
        kgrid=deepcopy_py(py_kgrid),
        medium=deepcopy_py(medium),
        source=deepcopy_py(source),
        sensor=deepcopy_py(sensor),
        simulation_options=sim_opts,
        execution_options=exec_opts,
    )

    if rec_symbol == :p_rms
        reshaped = np.reshape(data["p_rms"], (kgrid.Nx, kgrid.Ny), order="F")
        return pc.pyconvert(Array{Float64, 2}, reshaped)
    end

    reshaped = np.reshape(data["p"], (kgrid.Nx, kgrid.Ny, nt), order="F")
    return pc.pyconvert(Array{Float64, 3}, reshaped)
end
