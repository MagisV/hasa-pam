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
    t_active = t[active]
    accumulator = zeros(Float64, length(active))
    @inbounds for i in eachindex(src.harmonics)
        accumulator .+= src.harmonic_amplitudes[i] .* cos.(2pi .* src.harmonics[i] .* src.fundamental .* t_active .+ src.harmonic_phases[i])
    end
    signal[active] .= src.amplitude .* envelope .* accumulator
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
                harmonics=copy(src.harmonics),
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
                    harmonics=copy(src.harmonics),
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

# ── Synthetic 3D vascular network ─────────────────────────────────────────────

function _random_unit_vector_3d(rng::Random.AbstractRNG)
    v = randn(rng, 3)
    n = sqrt(v[1]^2 + v[2]^2 + v[3]^2)
    n > eps(Float64) || return (1.0, 0.0, 0.0)
    return (v[1] / n, v[2] / n, v[3] / n)
end

function _random_ellipsoid_direction_3d(rng::Random.AbstractRNG, radii::NTuple{3, Float64})
    v = _random_unit_vector_3d(rng)
    return _normalize3((radii[1] * v[1], radii[2] * v[2], radii[3] * v[3]))
end

function _random_horizontal_direction_3d(rng::Random.AbstractRNG)
    return _normalize3((0.2 * randn(rng), randn(rng), randn(rng)))
end

function _normalize3(v::NTuple{3, Float64})
    n = sqrt(v[1]^2 + v[2]^2 + v[3]^2)
    n > eps(Float64) || return (1.0, 0.0, 0.0)
    return (v[1] / n, v[2] / n, v[3] / n)
end

function _normalize_network_orientation(orientation)
    value = Symbol(replace(lowercase(string(orientation)), "-" => "_"))
    value == :lateral && return :horizontal
    value == :ellipsoid && return :axial
    value in (:horizontal, :isotropic, :axial) ||
        error("network_orientation must be horizontal, isotropic, or axial.")
    return value
end

function _random_network_direction_3d(
    rng::Random.AbstractRNG,
    radii::NTuple{3, Float64},
    orientation::Symbol,
)
    orientation == :horizontal && return _random_horizontal_direction_3d(rng)
    orientation == :axial && return _random_ellipsoid_direction_3d(rng, radii)
    return _random_unit_vector_3d(rng)
end

function _orthogonal_unit3(v::NTuple{3, Float64}, rng::Random.AbstractRNG)
    ref = _random_unit_vector_3d(rng)
    cross = (
        v[2] * ref[3] - v[3] * ref[2],
        v[3] * ref[1] - v[1] * ref[3],
        v[1] * ref[2] - v[2] * ref[1],
    )
    if sqrt(cross[1]^2 + cross[2]^2 + cross[3]^2) <= eps(Float64)
        ref = abs(v[1]) < 0.9 ? (1.0, 0.0, 0.0) : (0.0, 1.0, 0.0)
        cross = (
            v[2] * ref[3] - v[3] * ref[2],
            v[3] * ref[1] - v[1] * ref[3],
            v[1] * ref[2] - v[2] * ref[1],
        )
    end
    return _normalize3((Float64(cross[1]), Float64(cross[2]), Float64(cross[3])))
end

function _blend_direction3(
    direction::NTuple{3, Float64},
    perturbation::NTuple{3, Float64},
    amount::Real,
)
    a = clamp(Float64(amount), 0.0, 1.0)
    return _normalize3((
        (1 - a) * direction[1] + a * perturbation[1],
        (1 - a) * direction[2] + a * perturbation[2],
        (1 - a) * direction[3] + a * perturbation[3],
    ))
end

function _within_network_ellipsoid(
    point::NTuple{3, Float64},
    center::NTuple{3, Float64},
    radii::NTuple{3, Float64},
)
    dx = point[1] - center[1]
    dy = point[2] - center[2]
    dz = point[3] - center[3]
    return (dx / radii[1])^2 + (dy / radii[2])^2 + (dz / radii[3])^2 <= 1.0
end

function _network_child_direction(
    parent_dir::NTuple{3, Float64},
    angle_rad::Real,
    sign::Real,
    rng::Random.AbstractRNG,
)
    perp = _orthogonal_unit3(parent_dir, rng)
    a = Float64(angle_rad) * Float64(sign)
    return _normalize3((
        cos(a) * parent_dir[1] + sin(a) * perp[1],
        cos(a) * parent_dir[2] + sin(a) * perp[2],
        cos(a) * parent_dir[3] + sin(a) * perp[3],
    ))
end

function _grow_network_branch_3d(
    start::NTuple{3, Float64},
    direction::NTuple{3, Float64},
    center::NTuple{3, Float64};
    ellipsoid_radii::NTuple{3, Float64},
    segment_length::Real,
    step::Real,
    tortuosity::Real,
    network_orientation::Symbol,
    depth_bounds::Tuple{<:Real, <:Real},
    lateral_y_bounds::Tuple{<:Real, <:Real},
    lateral_z_bounds::Tuple{<:Real, <:Real},
    rng::Random.AbstractRNG,
)
    spacing = Float64(step)
    spacing > 0 || error("network step must be positive.")
    n_steps = max(1, ceil(Int, Float64(segment_length) / spacing))
    points = NTuple{3, Float64}[start]
    current = start
    dir = _normalize3(direction)
    for _ in 1:n_steps
        if tortuosity > 0
            dir = _blend_direction3(
                dir,
                _random_network_direction_3d(rng, ellipsoid_radii, network_orientation),
                tortuosity,
            )
        end
        next_point = (
            current[1] + spacing * dir[1],
            current[2] + spacing * dir[2],
            current[3] + spacing * dir[3],
        )
        _within_network_ellipsoid(next_point, center, ellipsoid_radii) || break
        _within_bounds_3d(next_point, depth_bounds, lateral_y_bounds, lateral_z_bounds) || break
        push!(points, next_point)
        current = next_point
    end
    return points, dir
end

function _generate_ellipsoid_network_centerlines_3d(
    center::NTuple{3, Float64};
    ellipsoid_radii::NTuple{3, Float64},
    root_count::Integer,
    generations::Integer,
    branch_length::Real,
    step::Real,
    branch_angle::Real,
    tortuosity::Real,
    network_orientation::Symbol,
    depth_bounds::Tuple{<:Real, <:Real},
    lateral_y_bounds::Tuple{<:Real, <:Real},
    lateral_z_bounds::Tuple{<:Real, <:Real},
    rng::Random.AbstractRNG,
)
    root_count > 0 || error("network root count must be positive.")
    generations > 0 || error("network generations must be positive.")
    centerlines = Vector{NTuple{3, Float64}}[]
    tips = [(center, _random_network_direction_3d(rng, ellipsoid_radii, network_orientation))]
    while length(tips) < Int(root_count)
        push!(tips, (center, _random_network_direction_3d(rng, ellipsoid_radii, network_orientation)))
    end

    for gen in 1:Int(generations)
        next_tips = Tuple{NTuple{3, Float64}, NTuple{3, Float64}}[]
        length_scale = Float64(branch_length) * (0.72 ^ (gen - 1))
        for (start, dir) in tips
            seg_len = length_scale * (0.75 + 0.5 * rand(rng))
            line, end_dir = _grow_network_branch_3d(
                start,
                dir,
                center;
                ellipsoid_radii=ellipsoid_radii,
                segment_length=seg_len,
                step=step,
                tortuosity=tortuosity,
                network_orientation=network_orientation,
                depth_bounds=depth_bounds,
                lateral_y_bounds=lateral_y_bounds,
                lateral_z_bounds=lateral_z_bounds,
                rng=rng,
            )
            length(line) >= 2 || continue
            push!(centerlines, line)
            tip = line[end]
            gen == Int(generations) && continue
            push!(next_tips, (tip, _network_child_direction(end_dir, branch_angle, 1.0, rng)))
            push!(next_tips, (tip, _network_child_direction(end_dir, branch_angle, -1.0, rng)))
        end
        tips = next_tips
        isempty(tips) && break
    end
    isempty(centerlines) && error("Network generation produced no centerline segments inside bounds.")
    return centerlines
end

function _sample_network_sources_3d(
    centerlines::AbstractVector;
    center::NTuple{3, Float64},
    source_spacing::Real,
    density_sigmas::NTuple{3, Float64},
    min_separation::Real,
    max_sources::Union{Nothing, Integer},
    rng::Random.AbstractRNG,
)
    spacing = Float64(source_spacing)
    spacing > 0 || error("source_spacing must be positive.")
    all(s -> s > 0, density_sigmas) || error("network density sigmas must be positive.")
    points = NTuple{3, Float64}[]
    weights = Float64[]

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
            depth, y, z, _, _, _ = _interp_centerline_point_3d(centerline, distance)
            normalized_r2 =
                ((depth - center[1]) / density_sigmas[1])^2 +
                ((y - center[2]) / density_sigmas[2])^2 +
                ((z - center[3]) / density_sigmas[3])^2
            weight = exp(-0.5 * normalized_r2)
            rand(rng) <= weight || continue
            point = (depth, y, z)
            _far_enough_3d(point, points, min_separation) || continue
            push!(points, point)
            push!(weights, weight)
        end
    end

    if !isnothing(max_sources) && length(points) > Int(max_sources)
        maxn = Int(max_sources)
        maxn > 0 || error("max_sources must be positive when provided.")
        score_order = sortperm(weights; rev=true)
        keep = sort(score_order[1:maxn])
        points = points[keep]
        weights = weights[keep]
    end
    isempty(points) && error("Network bubble sampling produced no sources; increase density sigma or lower spacing/min separation.")
    return points, weights
end

"""
    make_network_bubble_sources_3d(centers; kwargs...)

Generate a synthetic branching vascular network inside an ellipsoid around each
`(depth, lateral_y, lateral_z)` center. Bubble emitters are sampled along the
network centerlines with anisotropic Gaussian density around the center, so the
source population is highest at the geometric focus and tapers toward the focal
volume edge.
"""
function make_network_bubble_sources_3d(
    centers::AbstractVector;
    axial_radius::Real = 10e-3,
    lateral_y_radius::Real = 1.5e-3,
    lateral_z_radius::Real = 1.5e-3,
    root_count::Integer = 12,
    generations::Integer = 3,
    branch_length::Real = 5.0e-3,
    branch_step::Real = 0.4e-3,
    branch_angle::Real = pi / 5,
    tortuosity::Real = 0.18,
    network_orientation = :isotropic,
    source_spacing::Real = 0.4e-3,
    density_sigma::Real = 0.0,
    density_sigma_depth::Real = 10.0e-3,
    density_sigma_y::Real = 1.5e-3,
    density_sigma_z::Real = 1.5e-3,
    min_separation::Real = 0.25e-3,
    max_sources_per_center::Union{Nothing, Integer} = 80,
    depth_bounds::Tuple{<:Real, <:Real} = (0.0, Inf),
    lateral_y_bounds::Tuple{<:Real, <:Real} = (-Inf, Inf),
    lateral_z_bounds::Tuple{<:Real, <:Real} = (-Inf, Inf),
    fundamental::Real = 5e5,
    amplitude::Real = 1.0,
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
    isempty(centers) && error("At least one network center is required.")
    harmonics_i = Int.(harmonics)
    isempty(harmonics_i) && error("harmonics must be non-empty.")
    harmonic_amplitudes_f = Float64.(harmonic_amplitudes)
    length(harmonic_amplitudes_f) == length(harmonics_i) ||
        error("harmonic_amplitudes must have the same length as harmonics.")
    mode = _normalize_cluster_phase_mode(phase_mode)
    ellipsoid_radii = (Float64(axial_radius), Float64(lateral_y_radius), Float64(lateral_z_radius))
    all(r -> r > 0, ellipsoid_radii) || error("network ellipsoid radii must be positive.")
    legacy_density_sigma = Float64(density_sigma)
    density_sigmas = legacy_density_sigma > 0 ?
        (legacy_density_sigma, legacy_density_sigma, legacy_density_sigma) :
        (Float64(density_sigma_depth), Float64(density_sigma_y), Float64(density_sigma_z))
    all(s -> s > 0, density_sigmas) || error("network density sigmas must be positive.")
    orientation = _normalize_network_orientation(network_orientation)

    clusters = EmissionSource3D[]
    all_centerlines = Vector{NTuple{3, Float64}}[]
    center_meta = Dict{Symbol, Any}[]
    center_triples = [(Float64(c[1]), Float64(c[2]), Float64(c[3])) for c in centers]

    for center in center_triples
        centerlines = _generate_ellipsoid_network_centerlines_3d(
            center;
            ellipsoid_radii=ellipsoid_radii,
            root_count=root_count,
            generations=generations,
            branch_length=branch_length,
            step=branch_step,
            branch_angle=branch_angle,
            tortuosity=tortuosity,
            network_orientation=orientation,
            depth_bounds=depth_bounds,
            lateral_y_bounds=lateral_y_bounds,
            lateral_z_bounds=lateral_z_bounds,
            rng=rng,
        )
        append!(all_centerlines, centerlines)
        points, weights = _sample_network_sources_3d(
            centerlines;
            center=center,
            source_spacing=source_spacing,
            density_sigmas=density_sigmas,
            min_separation=min_separation,
            max_sources=max_sources_per_center,
            rng=rng,
        )
        source_start = length(clusters) + 1
        for (depth, y, z) in points
            phases = _cluster_harmonic_phases_3d(
                depth, y, z,
                harmonics_i, fundamental, c0,
                mode,
                transducer_depth, transducer_y, transducer_z,
                phase_jitter, rng,
            )
            push!(clusters, BubbleCluster3D(
                depth=depth,
                lateral_y=y,
                lateral_z=z,
                fundamental=Float64(fundamental),
                amplitude=Float64(amplitude),
                harmonics=copy(harmonics_i),
                harmonic_amplitudes=copy(harmonic_amplitudes_f),
                harmonic_phases=phases,
                gate_duration=Float64(gate_duration),
                taper_ratio=Float64(taper_ratio),
                delay=Float64(delay),
            ))
        end
        push!(center_meta, Dict{Symbol, Any}(
            :center => center,
            :centerline_count => length(centerlines),
            :source_count => length(points),
            :source_range => (source_start, length(clusters)),
            :density_weights => weights,
        ))
    end

    isempty(clusters) && error("Network source generation produced no sources.")
    meta = Dict{Symbol, Any}(
        :source_model => :network,
        :centers => center_triples,
        :ellipsoid_radii => ellipsoid_radii,
        :axial_radius => ellipsoid_radii[1],
        :lateral_y_radius => ellipsoid_radii[2],
        :lateral_z_radius => ellipsoid_radii[3],
        :root_count => Int(root_count),
        :generations => Int(generations),
        :branch_length => Float64(branch_length),
        :branch_step => Float64(branch_step),
        :branch_angle => Float64(branch_angle),
        :tortuosity => Float64(tortuosity),
        :network_orientation => orientation,
        :source_spacing => Float64(source_spacing),
        :density_sigma => legacy_density_sigma > 0 ? legacy_density_sigma : nothing,
        :density_sigmas => density_sigmas,
        :min_separation => Float64(min_separation),
        :max_sources_per_center => isnothing(max_sources_per_center) ? nothing : Int(max_sources_per_center),
        :centerlines => all_centerlines,
        :centers_meta => center_meta,
        :phase_mode => mode,
    )
    return clusters, meta
end
