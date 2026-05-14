function pam_gpu_maps_approx(a, b; rtol::Real=1e-2)
    scale = max(maximum(abs.(a)), maximum(abs.(b)), 1.0)
    return isapprox(a, b; rtol=rtol, atol=1e-5 * scale)
end

function synthetic_hu_volume(nslices::Int=5, rows::Int=80, cols::Int=60)
    hu = fill(Float32(-1000), nslices, rows, cols)
    for z in 1:nslices
        hu[z, 24:30, 20:40] .= 1200
    end
    return hu
end

function analytic_rf_for_point_source(cfg::PAMConfig, src::PointSource2D)
    kgrid = pam_grid(cfg)
    nt = pam_Nt(cfg)
    rf = zeros(Float64, kgrid.Ny, nt)
    duration = src.num_cycles / src.frequency
    base_t = collect(0:(nt - 1)) .* cfg.dt

    for j in 1:kgrid.Ny
        distance = hypot(src.depth, kgrid.y_vec[j] - src.lateral)
        arrival = src.delay + distance / cfg.c0
        local_t = base_t .- arrival
        active = findall((local_t .>= 0.0) .& (local_t .<= duration))
        isempty(active) && continue
        env = TranscranialFUS._tukey_window(length(active), 0.25)
        rf[j, active] .= (src.amplitude / sqrt(max(distance, cfg.dx))) .* env .* sin.(2π .* src.frequency .* local_t[active] .+ src.phase)
    end
    return rf, kgrid
end

function capture_stderr_result(f::Function)
    mktemp() do _, io
        result = redirect_stderr(io) do
            f()
        end
        flush(io)
        seekstart(io)
        return result, read(io, String)
    end
end
