abstract type EmissionSource3D end

Base.@kwdef struct PointSource3D <: EmissionSource3D
    depth::Float64
    lateral_y::Float64 = 0.0
    lateral_z::Float64 = 0.0
    frequency::Float64 = 0.5e6
    amplitude::Float64 = 1.0
    phase::Float64 = 0.0
    delay::Float64 = 0.0
    num_cycles::Float64 = 5.0
end

Base.@kwdef struct BubbleCluster3D <: EmissionSource3D
    depth::Float64
    lateral_y::Float64 = 0.0
    lateral_z::Float64 = 0.0
    fundamental::Float64 = 5e5
    amplitude::Float64 = 1.0
    n_bubbles::Float64 = 1.0
    harmonics::Vector{Int} = [2, 3, 4]
    harmonic_amplitudes::Vector{Float64} = [1.0, 0.6, 0.3]
    harmonic_phases::Vector{Float64} = [0.0, 0.0, 0.0]
    gate_duration::Float64 = 50e-6
    taper_ratio::Float64 = 0.25
    delay::Float64 = 0.0
end

_emission_frequencies(src::PointSource3D) = [src.frequency]
_emission_frequencies(src::BubbleCluster3D) = Float64[n * src.fundamental for n in src.harmonics]
emission_frequencies(src::EmissionSource3D) = _emission_frequencies(src)

_source_duration(src::PointSource3D) = src.num_cycles / src.frequency
_source_duration(src::BubbleCluster3D) = src.gate_duration

function _source_signal(nt::Int, dt::Real, src::PointSource3D; taper_ratio::Real=0.25)
    signal = zeros(Float64, nt)
    duration = src.num_cycles / src.frequency
    t = collect(0:(nt - 1)) .* Float64(dt) .- src.delay
    active = findall((t .>= 0.0) .& (t .<= duration))
    isempty(active) && return signal
    envelope = _tukey_window(length(active), taper_ratio)
    signal[active] .= src.amplitude .* envelope .* sin.(2pi .* src.frequency .* t[active] .+ src.phase)
    return signal
end

function _source_signal(nt::Int, dt::Real, src::BubbleCluster3D)
    length(src.harmonics) == length(src.harmonic_amplitudes) ||
        error("BubbleCluster3D: harmonics and harmonic_amplitudes must have equal length.")
    length(src.harmonics) == length(src.harmonic_phases) ||
        error("BubbleCluster3D: harmonics and harmonic_phases must have equal length.")
    signal = zeros(Float64, nt)
    t = collect(0:(nt - 1)) .* Float64(dt) .- src.delay
    active = findall((t .>= 0.0) .& (t .<= src.gate_duration))
    isempty(active) && return signal
    envelope = _tukey_window(length(active), src.taper_ratio)
    total_amp = src.amplitude * src.n_bubbles
    t_active = t[active]
    accumulator = zeros(Float64, length(active))
    @inbounds for i in eachindex(src.harmonics)
        accumulator .+= src.harmonic_amplitudes[i] .* cos.(2pi .* src.harmonics[i] .* src.fundamental .* t_active .+ src.harmonic_phases[i])
    end
    signal[active] .= total_amp .* envelope .* accumulator
    return signal
end

# ── 3D centerline ─────────────────────────────────────────────────────────────

function _squiggle_centerline_3d(
    anchor_depth::Real,
    anchor_y::Real,
    anchor_z::Real;
    root_length::Real,
    squiggle_amplitude_y::Real,
    squiggle_amplitude_x::Real,
    squiggle_wavelength::Real,
    squiggle_phase_x::Real,
    squiggle_slope_x::Real,
    squiggle_slope_y::Real,
    source_spacing::Real,
)
    length_m = Float64(root_length)
    length_m > 0 || error("root_length must be positive.")
    wavelength_m = Float64(squiggle_wavelength)
    wavelength_m > 0 || error("squiggle_wavelength must be positive.")
    spacing = Float64(source_spacing)
    spacing > 0 || error("source_spacing must be positive.")

    n = max(16, ceil(Int, length_m / (spacing / 2)) + 1)
    half = length_m / 2
    points = NTuple{3, Float64}[]
    for s in range(-half, half; length=n)
        z     = Float64(anchor_z) + s
        y     = Float64(anchor_y) + Float64(squiggle_slope_y) * s +
                Float64(squiggle_amplitude_y) * sin(2pi * s / wavelength_m)
        depth = Float64(anchor_depth) + Float64(squiggle_slope_x) * s +
                Float64(squiggle_amplitude_x) * sin(2pi * s / wavelength_m + Float64(squiggle_phase_x))
        push!(points, (depth, y, z))
    end
    return points
end

# ── arc-length interpolation on a 3D polyline ────────────────────────────────

function _interp_centerline_point_3d(centerline::AbstractVector{NTuple{3, Float64}}, distance::Real)
    length(centerline) >= 2 || error("Centerline must contain at least two points.")
    remaining = Float64(distance)
    for idx in 1:(length(centerline) - 1)
        d0, y0, z0 = centerline[idx]
        d1, y1, z1 = centerline[idx + 1]
        seg_len = sqrt((d1 - d0)^2 + (y1 - y0)^2 + (z1 - z0)^2)
        seg_len > eps(Float64) || continue
        if remaining <= seg_len || idx == length(centerline) - 1
            t = clamp(remaining / seg_len, 0.0, 1.0)
            return (
                d0 + t * (d1 - d0),
                y0 + t * (y1 - y0),
                z0 + t * (z1 - z0),
                (d1 - d0) / seg_len,
                (y1 - y0) / seg_len,
                (z1 - z0) / seg_len,
            )
        end
        remaining -= seg_len
    end
    d0, y0, z0 = centerline[end - 1]
    d1, y1, z1 = centerline[end]
    seg_len = max(sqrt((d1 - d0)^2 + (y1 - y0)^2 + (z1 - z0)^2), eps(Float64))
    return (d1, y1, z1, (d1 - d0) / seg_len, (y1 - y0) / seg_len, (z1 - z0) / seg_len)
end

# ── 3D sampler ────────────────────────────────────────────────────────────────

function _within_bounds_3d(point::NTuple{3, Float64}, depth_bounds, y_bounds, z_bounds)
    depth, y, z = point
    return depth_bounds[1] <= depth <= depth_bounds[2] &&
           y_bounds[1] <= y <= y_bounds[2] &&
           z_bounds[1] <= z <= z_bounds[2]
end

function _far_enough_3d(point::NTuple{3, Float64}, points::AbstractVector{NTuple{3, Float64}}, min_separation::Real)
    sep = Float64(min_separation)
    sep <= 0 && return true
    d, y, z = point
    for (od, oy, oz) in points
        sqrt((d - od)^2 + (y - oy)^2 + (z - oz)^2) >= sep || return false
    end
    return true
end

function _sample_squiggle_centerlines_3d(
    centerlines::AbstractVector;
    source_spacing::Real,
    position_jitter::Real,
    min_separation::Real,
    depth_bounds::Tuple{<:Real, <:Real},
    lateral_y_bounds::Tuple{<:Real, <:Real},
    lateral_z_bounds::Tuple{<:Real, <:Real},
    max_sources::Union{Nothing, Integer},
    rng::Random.AbstractRNG,
)
    spacing = Float64(source_spacing)
    spacing > 0 || error("source_spacing must be positive.")
    points = NTuple{3, Float64}[]

    for centerline in centerlines
        length(centerline) >= 2 || continue
        total_len = 0.0
        for idx in 1:(length(centerline) - 1)
            d0, y0, z0 = centerline[idx]
            d1, y1, z1 = centerline[idx + 1]
            total_len += sqrt((d1 - d0)^2 + (y1 - y0)^2 + (z1 - z0)^2)
        end
        total_len > eps(Float64) || continue
        n = max(2, ceil(Int, total_len / spacing) + 1)
        for distance in range(0.0, total_len; length=n)
            depth, y, z, td, ty, tz = _interp_centerline_point_3d(centerline, distance)
            if position_jitter > 0
                # jitter perpendicular to tangent: use two perpendicular directions
                # build an arbitrary perpendicular vector via cross product with (0,1,0) or (1,0,0)
                ref = abs(ty) < 0.9 ? (0.0, 1.0, 0.0) : (1.0, 0.0, 0.0)
                px = ty * ref[3] - tz * ref[2]
                py = tz * ref[1] - td * ref[3]
                pz = td * ref[2] - ty * ref[1]
                plen = sqrt(px^2 + py^2 + pz^2)
                if plen > eps(Float64)
                    px /= plen; py /= plen; pz /= plen
                end
                j1 = randn(rng) * Float64(position_jitter)
                # second perpendicular direction: cross(tangent, perp1)
                qx = ty * pz - tz * py
                qy = tz * px - td * pz
                qz = td * py - ty * px
                j2 = randn(rng) * Float64(position_jitter)
                depth += px * j1 + qx * j2
                y     += py * j1 + qy * j2
                z     += pz * j1 + qz * j2
            end
            point = (depth, y, z)
            _within_bounds_3d(point, depth_bounds, lateral_y_bounds, lateral_z_bounds) || continue
            _far_enough_3d(point, points, min_separation) || continue
            push!(points, point)
        end
    end

    if !isnothing(max_sources) && length(points) > Int(max_sources)
        maxn = Int(max_sources)
        maxn > 0 || error("max_sources must be positive when provided.")
        keep = unique(round.(Int, range(1, length(points); length=maxn)))
        points = points[keep]
    end

    return points
end

# ── geometric phase for 3D ────────────────────────────────────────────────────

function _geometric_drive_phase_3d(
    depth::Real, y::Real, z::Real,
    tx_depth::Real, tx_y::Real, tx_z::Real,
    harmonic::Integer, fundamental::Real, c0::Real,
)
    distance = sqrt((Float64(depth) - Float64(tx_depth))^2 +
                    (Float64(y)     - Float64(tx_y))^2     +
                    (Float64(z)     - Float64(tx_z))^2)
    return -2pi * Int(harmonic) * Float64(fundamental) * distance / Float64(c0)
end

function _cluster_harmonic_phases_3d(
    depth::Real, y::Real, z::Real,
    harmonics::AbstractVector{<:Integer},
    fundamental::Real, c0::Real,
    phase_mode::Symbol,
    tx_depth::Real, tx_y::Real, tx_z::Real,
    phase_jitter::Real,
    rng::Random.AbstractRNG,
)
    phases = Vector{Float64}(undef, length(harmonics))
    for (idx, harmonic) in pairs(harmonics)
        base = if phase_mode in (:geometric, :jittered)
            _geometric_drive_phase_3d(depth, y, z, tx_depth, tx_y, tx_z, harmonic, fundamental, c0)
        elseif phase_mode == :random
            2pi * rand(rng)
        else
            0.0
        end
        if phase_mode == :jittered
            base += randn(rng) * Float64(phase_jitter)
        end
        phases[idx] = base
    end
    return phases
end

# ── phase resampling for per-window mode ─────────────────────────────────────

function _resample_source_phases_3d(
    sources::AbstractVector{<:EmissionSource3D},
    rng::Random.AbstractRNG,
)
    return map(sources) do src
        if src isa BubbleCluster3D
            BubbleCluster3D(
                depth=src.depth, lateral_y=src.lateral_y, lateral_z=src.lateral_z,
                fundamental=src.fundamental, amplitude=src.amplitude,
                n_bubbles=src.n_bubbles, harmonics=copy(src.harmonics),
                harmonic_amplitudes=copy(src.harmonic_amplitudes),
                harmonic_phases=2pi .* rand(rng, length(src.harmonics)),
                gate_duration=src.gate_duration, taper_ratio=src.taper_ratio,
                delay=src.delay,
            )
        elseif src isa PointSource3D
            PointSource3D(
                depth=src.depth, lateral_y=src.lateral_y, lateral_z=src.lateral_z,
                frequency=src.frequency, amplitude=src.amplitude,
                phase=2pi * rand(rng), delay=src.delay, num_cycles=src.num_cycles,
            )
        else
            src
        end
    end
end

function _expand_sources_per_window(
    sources::AbstractVector{<:EmissionSource3D},
    window_duration::Real,
    hop::Real,
    t_max::Real,
    rng::Random.AbstractRNG;
    variability::SourceVariabilityConfig=SourceVariabilityConfig(),
)
    frame_dur = Float64(window_duration)
    hop_s = Float64(hop)
    frame_dur > 0 || error("window_duration must be positive.")
    hop_s > 0 || error("hop must be positive.")
    n_frames = max(1, floor(Int, (Float64(t_max) - frame_dur) / hop_s) + 1)
    expanded = EmissionSource3D[]
    sizehint!(expanded, length(sources) * n_frames)
    fjitter = variability.frequency_jitter_fraction
    fjitter >= 0.0 || error("frequency_jitter_fraction must be non-negative.")
    for src in sources
        for k in 1:n_frames
            d = src.delay + hop_s * (k - 1)
            fscale = fjitter > 0.0 ? max(0.01, 1.0 + fjitter * randn(rng)) : 1.0
            evt = if src isa BubbleCluster3D
                BubbleCluster3D(
                    depth=src.depth, lateral_y=src.lateral_y, lateral_z=src.lateral_z,
                    fundamental=src.fundamental * fscale, amplitude=src.amplitude,
                    n_bubbles=src.n_bubbles, harmonics=copy(src.harmonics),
                    harmonic_amplitudes=copy(src.harmonic_amplitudes),
                    harmonic_phases=2pi .* rand(rng, length(src.harmonics)),
                    gate_duration=min(src.gate_duration, frame_dur), taper_ratio=src.taper_ratio,
                    delay=d,
                )
            elseif src isa PointSource3D
                nc = max(1, round(Int, min(Float64(src.num_cycles) / src.frequency, frame_dur) * src.frequency * fscale))
                PointSource3D(
                    depth=src.depth, lateral_y=src.lateral_y, lateral_z=src.lateral_z,
                    frequency=src.frequency * fscale, amplitude=src.amplitude,
                    phase=2pi * rand(rng), delay=d, num_cycles=Float64(nc),
                )
            else
                src
            end
            push!(expanded, evt)
        end
    end
    isempty(expanded) && error("_expand_sources_per_window: no emissions generated.")
    return expanded, n_frames
end

# ── factory ───────────────────────────────────────────────────────────────────

"""
    make_squiggle_bubble_sources_3d(anchors; kwargs...)

Generate harmonic bubble emitters along one 3D squiggly centerline per
`(depth, lateral_y, lateral_z)` anchor. The vessel runs primarily along Z and
oscillates simultaneously in Y (squiggle_amplitude_y) and depth/X
(squiggle_amplitude_x), with squiggle_phase_x offsetting the two oscillations
so the path has genuine 3D curvature.
"""
function make_squiggle_bubble_sources_3d(
    anchors::AbstractVector;
    root_length::Real = 12e-3,
    squiggle_amplitude_y::Real = 1.5e-3,
    squiggle_amplitude_x::Real = 1.0e-3,
    squiggle_wavelength::Real = 8e-3,
    squiggle_phase_x::Real = pi / 2,
    squiggle_slope_x::Real = 0.0,
    squiggle_slope_y::Real = 0.0,
    source_spacing::Real = 2.0e-3,
    position_jitter::Real = 0.05e-3,
    min_separation::Real = 1.0e-3,
    max_sources_per_anchor::Union{Nothing, Integer} = nothing,
    depth_bounds::Tuple{<:Real, <:Real} = (0.0, Inf),
    lateral_y_bounds::Tuple{<:Real, <:Real} = (-Inf, Inf),
    lateral_z_bounds::Tuple{<:Real, <:Real} = (-Inf, Inf),
    fundamental::Real = 5e5,
    amplitude::Real = 1.0,
    n_bubbles::Real = 1.0,
    harmonics::AbstractVector{<:Integer} = [2, 3, 4],
    harmonic_amplitudes::AbstractVector{<:Real} = [1.0, 0.6, 0.3],
    gate_duration::Real = 50e-6,
    taper_ratio::Real = 0.25,
    delay::Real = 0.0,
    phase_mode = :geometric,
    phase_jitter::Real = 0.2,
    transducer_depth::Real = -30e-3,
    transducer_y::Real = 0.0,
    transducer_z::Real = 0.0,
    c0::Real = 1500.0,
    rng::Random.AbstractRNG = Random.default_rng(),
)
    isempty(anchors) && error("At least one squiggle anchor is required.")
    harmonics_i = Int.(harmonics)
    isempty(harmonics_i) && error("harmonics must be non-empty.")
    harmonic_amplitudes_f = Float64.(harmonic_amplitudes)
    length(harmonic_amplitudes_f) == length(harmonics_i) ||
        error("harmonic_amplitudes must have the same length as harmonics.")
    mode = _normalize_cluster_phase_mode(phase_mode)

    clusters = EmissionSource3D[]
    all_centerlines = Vector{NTuple{3, Float64}}[]
    source_count_by_anchor = Int[]
    anchor_triples = [(Float64(a[1]), Float64(a[2]), Float64(a[3])) for a in anchors]

    for (anchor_depth, anchor_y, anchor_z) in anchor_triples
        centerline = _squiggle_centerline_3d(
            anchor_depth, anchor_y, anchor_z;
            root_length = root_length,
            squiggle_amplitude_y = squiggle_amplitude_y,
            squiggle_amplitude_x = squiggle_amplitude_x,
            squiggle_wavelength = squiggle_wavelength,
            squiggle_phase_x = squiggle_phase_x,
            squiggle_slope_x = squiggle_slope_x,
            squiggle_slope_y = squiggle_slope_y,
            source_spacing = source_spacing,
        )
        push!(all_centerlines, centerline)
        points = _sample_squiggle_centerlines_3d(
            [centerline];
            source_spacing = source_spacing,
            position_jitter = position_jitter,
            min_separation = min_separation,
            depth_bounds = (Float64(depth_bounds[1]), Float64(depth_bounds[2])),
            lateral_y_bounds = (Float64(lateral_y_bounds[1]), Float64(lateral_y_bounds[2])),
            lateral_z_bounds = (Float64(lateral_z_bounds[1]), Float64(lateral_z_bounds[2])),
            max_sources = max_sources_per_anchor,
            rng = rng,
        )
        push!(source_count_by_anchor, length(points))

        for (depth, y, z) in points
            phases = _cluster_harmonic_phases_3d(
                depth, y, z,
                harmonics_i, fundamental, c0,
                mode,
                transducer_depth, transducer_y, transducer_z,
                phase_jitter, rng,
            )
            push!(clusters, BubbleCluster3D(
                depth = depth,
                lateral_y = y,
                lateral_z = z,
                fundamental = Float64(fundamental),
                amplitude = Float64(amplitude),
                n_bubbles = Float64(n_bubbles),
                harmonics = copy(harmonics_i),
                harmonic_amplitudes = copy(harmonic_amplitudes_f),
                harmonic_phases = phases,
                gate_duration = Float64(gate_duration),
                taper_ratio = Float64(taper_ratio),
                delay = Float64(delay),
            ))
        end
    end

    isempty(clusters) && error("Squiggle 3D source generation produced no sources inside the requested bounds.")
    meta = Dict{Symbol, Any}(
        :source_model          => :squiggle,
        :anchors               => anchor_triples,
        :root_length           => Float64(root_length),
        :squiggle_amplitude_y  => Float64(squiggle_amplitude_y),
        :squiggle_amplitude_x  => Float64(squiggle_amplitude_x),
        :squiggle_wavelength   => Float64(squiggle_wavelength),
        :squiggle_phase_x      => Float64(squiggle_phase_x),
        :squiggle_slope_x      => Float64(squiggle_slope_x),
        :squiggle_slope_y      => Float64(squiggle_slope_y),
        :source_spacing        => Float64(source_spacing),
        :position_jitter       => Float64(position_jitter),
        :min_separation        => Float64(min_separation),
        :max_sources_per_anchor => isnothing(max_sources_per_anchor) ? nothing : Int(max_sources_per_anchor),
        :source_count_by_anchor => source_count_by_anchor,
        :centerlines           => all_centerlines,
        :phase_mode            => mode,
    )
    return clusters, meta
end
