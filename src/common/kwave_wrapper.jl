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
        kspace3d=PythonCall.pyimport("kwave.kspaceFirstOrder3D"),
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

function _py_float_matrix(np, rows::Int, cols::Int)
    return np.zeros((rows, cols), dtype=np.float64)
end

function _as_sensor_matrix(array, expected_rows::Int, expected_cols::Int)
    mat = Float64.(array)
    ndims(mat) == 1 && return reshape(mat, :, 1)
    if size(mat, 1) == expected_rows && size(mat, 2) == expected_cols
        return mat
    elseif size(mat, 1) == expected_cols && size(mat, 2) == expected_rows
        return permutedims(mat)
    end
    error("Unexpected sensor data shape $(size(mat)); expected ($expected_rows, $expected_cols) or ($expected_cols, $expected_rows).")
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
    pml_guard = max(4, cfg.PML_GUARD)

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
        pml_size=pml_guard,
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

function simulate_point_sources(
    c::AbstractMatrix{<:Real},
    rho::AbstractMatrix{<:Real},
    sources::AbstractVector{<:EmissionSource2D},
    cfg::PAMConfig;
    use_gpu::Bool=false,
)
    isempty(sources) && error("At least one emission source is required.")

    nx, ny = size(c)
    size(rho) == size(c) || error("Density map must have the same size as the sound-speed map.")
    nx == pam_Nx(cfg) || error("Sound-speed map height $nx does not match PAMConfig height $(pam_Nx(cfg)).")
    ny == pam_Ny(cfg) || error("Sound-speed map width $ny does not match PAMConfig width $(pam_Ny(cfg)).")

    mods = _kwave_modules()
    pc = mods.PythonCall
    np = mods.np
    deepcopy_py = mods.copy.deepcopy

    nt = pam_Nt(cfg)
    kgrid = pam_grid(cfg)
    pml_guard = _pam_pml_guard(cfg)
    py_kgrid = mods.kgrid.kWaveGrid(pc.Py([kgrid.Nx, kgrid.Ny]), pc.Py([kgrid.dx, kgrid.dy]))
    py_kgrid.dt = kgrid.dt
    py_kgrid.Nt = nt

    c_py = np.array(pc.Py(Float32.(c)), dtype=np.float32)
    rho_py = np.array(pc.Py(Float32.(rho)), dtype=np.float32)
    medium = mods.kmedium.kWaveMedium(sound_speed=c_py, density=rho_py)

    indexed_sources = [(source_grid_index(src, cfg, kgrid), src) for src in sources]
    sort!(indexed_sources; by=entry -> first(entry)[1] + (first(entry)[2] - 1) * nx)

    grouped_sources = Vector{Tuple{Tuple{Int, Int}, Vector{EmissionSource2D}}}()
    for (grid_index, src) in indexed_sources
        if !isempty(grouped_sources) && first(last(grouped_sources)) == grid_index
            push!(last(grouped_sources)[2], src)
        else
            push!(grouped_sources, (grid_index, EmissionSource2D[src]))
        end
    end

    source = mods.ksource.kSource()
    src_mask = _py_bool_matrix(np, nx, ny)
    source_signals = Matrix{Float64}(undef, length(grouped_sources), nt)
    source_indices = Tuple{Int, Int}[]
    for (idx, ((row, col), cell_sources)) in pairs(grouped_sources)
        src_mask[row - 1, col - 1] = true
        source_signals[idx, :] .= 0.0
        for src in cell_sources
            source_signals[idx, :] .+= _source_signal(nt, cfg.dt, src)
        end
        push!(source_indices, (row, col))
    end
    source.p_mask = src_mask
    source.p = np.array(pc.Py(source_signals), dtype=np.float64)
    source.medium = medium
    all_freqs = Float64[]
    for (_, src) in indexed_sources
        append!(all_freqs, _emission_frequencies(src))
    end
    unique_freqs = unique(all_freqs)
    if length(unique_freqs) == 1
        source.p_frequency_ref = unique_freqs[1]
    end

    sensor = mods.ksensor.kSensor()
    sensor_mask = _py_bool_matrix(np, nx, ny)
    row = receiver_row(cfg)
    col_range = receiver_col_range(cfg)
    sensor_mask[row - 1, (first(col_range) - 1):(last(col_range) - 1)] = true
    sensor.mask = sensor_mask
    sensor.record = pc.pybuiltins.list(("p",))

    sim_opts = mods.simopts.SimulationOptions(
        pml_inside=false,
        pml_size=pml_guard,
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

    sensor_data_py = np.array(data["p"], dtype=np.float64)
    sensor_data = _as_sensor_matrix(pc.pyconvert(Array, sensor_data_py), length(col_range), nt)
    rf = zeros(Float64, ny, nt)
    rf[col_range, :] .= sensor_data

    info = Dict{Symbol, Any}(
        :receiver_row => row,
        :receiver_cols => col_range,
        :source_indices => source_indices,
        :num_input_sources => length(sources),
        :num_source_points => length(grouped_sources),
    )
    return rf, kgrid, info
end

function simulate_point_sources_3d(
    c::AbstractArray{<:Real, 3},
    rho::AbstractArray{<:Real, 3},
    sources::AbstractVector{<:EmissionSource3D},
    cfg::PAMConfig3D;
    use_gpu::Bool=false,
)
    isempty(sources) && error("At least one emission source is required.")

    nx, ny, nz = pam_Nx(cfg), pam_Ny(cfg), pam_Nz(cfg)
    size(c) == (nx, ny, nz) || error("Sound-speed map size $(size(c)) does not match PAMConfig3D ($nx, $ny, $nz).")
    size(rho) == (nx, ny, nz) || error("Density map size $(size(rho)) does not match PAMConfig3D ($nx, $ny, $nz).")

    mods = _kwave_modules()
    pc = mods.PythonCall
    np = mods.np
    deepcopy_py = mods.copy.deepcopy

    nt = pam_Nt(cfg)
    kgrid = pam_grid_3d(cfg)
    pml_guard = _pam_pml_guard_3d(cfg)
    py_kgrid = mods.kgrid.kWaveGrid(pc.Py([nx, ny, nz]), pc.Py([cfg.dx, cfg.dy, cfg.dz]))
    py_kgrid.dt = cfg.dt
    py_kgrid.Nt = nt

    c_py = np.array(pc.Py(Float32.(c)), dtype=np.float32)
    rho_py = np.array(pc.Py(Float32.(rho)), dtype=np.float32)
    medium = mods.kmedium.kWaveMedium(sound_speed=c_py, density=rho_py)

    indexed_sources = [(source_grid_index_3d(src, cfg), src) for src in sources]
    sort!(indexed_sources; by=entry -> begin
        (row, cy, cz) = first(entry)
        row + (cy - 1) * nx + (cz - 1) * nx * ny
    end)

    grouped_sources = Vector{Tuple{Tuple{Int,Int,Int}, Vector{EmissionSource3D}}}()
    for (grid_index, src) in indexed_sources
        if !isempty(grouped_sources) && first(last(grouped_sources)) == grid_index
            push!(last(grouped_sources)[2], src)
        else
            push!(grouped_sources, (grid_index, EmissionSource3D[src]))
        end
    end

    source = mods.ksource.kSource()
    src_mask_jl = falses(nx, ny, nz)
    source_signals = Matrix{Float64}(undef, length(grouped_sources), nt)
    source_indices = Tuple{Int,Int,Int}[]
    for (idx, ((row, col_y, col_z), cell_sources)) in pairs(grouped_sources)
        src_mask_jl[row, col_y, col_z] = true
        source_signals[idx, :] .= 0.0
        for src in cell_sources
            source_signals[idx, :] .+= _source_signal(nt, cfg.dt, src)
        end
        push!(source_indices, (row, col_y, col_z))
    end
    source.p_mask = np.array(pc.Py(Array(src_mask_jl)), dtype=pc.pybuiltins.bool)
    source.p = np.array(pc.Py(source_signals), dtype=np.float64)
    source.medium = medium
    all_freqs = Float64[]
    for (_, src) in indexed_sources
        append!(all_freqs, _emission_frequencies(src))
    end
    unique_freqs = unique(all_freqs)
    if length(unique_freqs) == 1
        source.p_frequency_ref = unique_freqs[1]
    end

    sensor = mods.ksensor.kSensor()
    row = receiver_row(cfg)
    col_range_y = receiver_col_range_y(cfg)
    col_range_z = receiver_col_range_z(cfg)
    # Build sensor mask: fill the full ny×nz receiver plane at the given row.
    # Use Julia array then convert, avoiding PythonCall slice translation issues.
    sensor_mask_jl = falses(nx, ny, nz)
    sensor_mask_jl[row, :, :] .= true
    sensor_mask = np.array(pc.Py(Array(sensor_mask_jl)), dtype=pc.pybuiltins.bool)
    sensor.mask = sensor_mask
    sensor.record = pc.pybuiltins.list(("p",))

    sim_opts = mods.simopts.SimulationOptions(
        pml_inside=false,
        pml_size=pml_guard,
        data_recast=false,
        save_to_disk=true,
    )
    exec_opts = mods.execopts.SimulationExecutionOptions(
        is_gpu_simulation=use_gpu,
        delete_data=false,
    )

    data = mods.kspace3d.kspaceFirstOrder3D(
        kgrid=deepcopy_py(py_kgrid),
        medium=deepcopy_py(medium),
        source=deepcopy_py(source),
        sensor=deepcopy_py(sensor),
        simulation_options=sim_opts,
        execution_options=exec_opts,
    )

    sensor_data_py = np.array(data["p"], dtype=np.float64)
    # k-Wave returns (Nt, ny*nz) in Python (row-major). pyconvert gives (ny*nz, Nt) in Julia.
    sensor_data_flat = pc.pyconvert(Matrix{Float64}, sensor_data_py)  # (ny*nz, Nt) or (Nt, ny*nz)
    # Normalise to (ny*nz, Nt) regardless of which axis k-Wave put time on.
    if size(sensor_data_flat, 1) == nt && size(sensor_data_flat, 2) == ny * nz
        sensor_data_flat = permutedims(sensor_data_flat)  # → (ny*nz, Nt)
    elseif size(sensor_data_flat, 1) != ny * nz || size(sensor_data_flat, 2) != nt
        error("Unexpected sensor data shape $(size(sensor_data_flat)); expected ($(ny*nz), $nt) or ($nt, $(ny*nz)).")
    end
    # Julia reshape is column-major: ny varies fastest, matching k-Wave's Fortran-order enumeration.
    rf = reshape(sensor_data_flat, ny, nz, nt)

    info = Dict{Symbol, Any}(
        :receiver_row => row,
        :receiver_cols_y => col_range_y,
        :receiver_cols_z => col_range_z,
        :source_indices => source_indices,
        :num_input_sources => length(sources),
        :num_source_points => length(grouped_sources),
    )
    return rf, kgrid, info
end
