function _kwave_modules()
    CondaPkg.resolve()
    py_os = PythonCall.pyimport("os")
    if isfile("/etc/ssl/cert.pem")
        py_os.environ["SSL_CERT_FILE"] = "/etc/ssl/cert.pem"
    end
    return (
        PythonCall=PythonCall,
        np=PythonCall.pyimport("numpy"),
        copy=PythonCall.pyimport("copy"),
        kgrid=PythonCall.pyimport("kwave.kgrid"),
        kmedium=PythonCall.pyimport("kwave.kmedium"),
        ksource=PythonCall.pyimport("kwave.ksource"),
        ksensor=PythonCall.pyimport("kwave.ksensor"),
        kspace=PythonCall.pyimport("kwave.kspaceFirstOrder2D"),
        simopts=PythonCall.pyimport("kwave.options.simulation_options"),
        execopts=PythonCall.pyimport("kwave.options.simulation_execution_options"),
        signals=PythonCall.pyimport("kwave.utils.signals"),
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
    return np.zeros((rows, cols), dtype=PythonCall.pybuiltins.bool)
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

    c_py = np.array(pc.Py(Float32.(c)), dtype=np.float32)
    rho_py = np.array(pc.Py(Float32.(rho)), dtype=np.float32)
    medium = mods.kmedium.kWaveMedium(
        sound_speed=c_py,
        density=rho_py,
    )

    source = mods.ksource.kSource()
    src_mask = _py_bool_matrix(np, kgrid.Nx, kgrid.Ny)
    src_mask[cfg.trans_index - 1, (first(col_range) - 1):(last(col_range) - 1)] = true
    source.p_mask = src_mask

    int_delay = floor.(Int, tau ./ cfg.dt)
    frac_delay = Float64.(tau .- int_delay .* cfg.dt)
    int_delay_py = np.array(pc.Py(Int64.(int_delay)), dtype=np.int64)
    envelope_py = np.array(pc.Py(Int64[rise_fall, rise_fall]), dtype=np.int64)

    if use_envelope
        burst_py = mods.signals.tone_burst(
            sample_freq=1 / cfg.dt,
            signal_freq=cfg.fc,
            num_cycles=num_cycles,
            envelope=envelope_py,
            signal_offset=int_delay_py,
        )
    else
        burst_py = mods.signals.tone_burst(
            sample_freq=1 / cfg.dt,
            signal_freq=cfg.fc,
            num_cycles=num_cycles,
            signal_offset=int_delay_py,
        )
    end
    burst = pc.pyconvert(Matrix{Float64}, burst_py)
    drive_weights = (1e5 .* Float64.(amplitudes) .* exp.(-1im .* omega(cfg) .* frac_delay))
    p_drive = real.(reshape(drive_weights, :, 1) .* burst)
    source.p = np.array(pc.Py(p_drive), dtype=np.float64)
    source.p_frequency_ref = cfg.fc
    source.medium = medium

    sensor = mods.ksensor.kSensor()
    sensor.mask = np.ones((kgrid.Nx, kgrid.Ny), dtype=pc.pybuiltins.bool)
    sensor.record = pc.pybuiltins.list((String(rec_symbol),))

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
