abstract type EmissionSource2D end

Base.@kwdef struct PointSource2D <: EmissionSource2D
    depth::Float64
    lateral::Float64
    frequency::Float64 = 8e5
    amplitude::Float64 = 1.0
    phase::Float64 = 0.0
    delay::Float64 = 0.0
    num_cycles::Int = 5
end

Base.@kwdef struct BubbleCluster2D <: EmissionSource2D
    depth::Float64
    lateral::Float64
    fundamental::Float64 = 5e5
    amplitude::Float64 = 1.0
    n_bubbles::Float64 = 1.0
    harmonics::Vector{Int} = [2, 3]
    harmonic_amplitudes::Vector{Float64} = [1.0, 0.6]
    harmonic_phases::Vector{Float64} = [0.0, 0.0]
    gate_duration::Float64 = 50e-6
    taper_ratio::Float64 = 0.25
    delay::Float64 = 0.0
end

Base.@kwdef struct GaussianPulseCluster2D <: EmissionSource2D
    depth::Float64
    lateral::Float64
    fundamental::Float64 = 5e5
    amplitude::Float64 = 1.0
    n_bubbles::Float64 = 1.0
    harmonics::Vector{Int} = [2, 3]
    harmonic_amplitudes::Vector{Float64} = [1.0, 0.6]
    harmonic_phases::Vector{Float64} = [0.0, 0.0]
    gate_duration::Float64 = 10e-6
    taper_ratio::Float64 = 0.25
    delay::Float64 = 0.0
end

Base.@kwdef struct SourceVariabilityConfig
    frequency_jitter_fraction::Float64 = 0.0
end

_emission_frequencies(src::PointSource2D) = Float64[src.frequency]
_emission_frequencies(src::BubbleCluster2D) = Float64[n * src.fundamental for n in src.harmonics]
_emission_frequencies(src::GaussianPulseCluster2D) = Float64[n * src.fundamental for n in src.harmonics]

emission_frequencies(src::EmissionSource2D) = _emission_frequencies(src)
cavitation_model(::BubbleCluster2D) = :harmonic_cos
cavitation_model(::GaussianPulseCluster2D) = :gaussian_pulse

function _normalize_cavitation_model(cavitation_model)
    model = Symbol(replace(lowercase(string(cavitation_model)), "-" => "_"))
    model in (:harmonic_cos, :gaussian_pulse) ||
        error("Unknown cavitation_model: $cavitation_model (expected harmonic-cos or gaussian-pulse).")
    return model
end

function _normalize_cluster_phase_mode(phase_mode)
    mode = Symbol(replace(lowercase(string(phase_mode)), "-" => "_"))
    mode == :random_static_phase && return :random
    mode in (:coherent, :geometric, :random, :jittered) ||
        error("Unknown phase_mode: $phase_mode (expected coherent, geometric, random, random_static_phase, or jittered).")
    return mode
end

function _normalize_source_phase_mode(source_phase_mode)
    mode = Symbol(replace(lowercase(string(source_phase_mode)), "-" => "_"))
    mode in (:coherent, :random_static_phase, :random_phase_per_window, :random_phase_per_realization) ||
        error(
            "Unknown source_phase_mode: $source_phase_mode. " *
            "Expected: coherent, random_static_phase, random_phase_per_window, or random_phase_per_realization.",
        )
    return mode
end

function _geometric_drive_phase(
    depth::Real,
    lateral::Real,
    tx_depth::Real,
    tx_lateral::Real,
    harmonic::Integer,
    fundamental::Real,
    c0::Real,
)
    distance = hypot(Float64(depth) - Float64(tx_depth), Float64(lateral) - Float64(tx_lateral))
    return -2pi * Int(harmonic) * Float64(fundamental) * distance / Float64(c0)
end

function _cluster_harmonic_phases(
    depth::Real,
    lateral::Real,
    harmonics::AbstractVector{<:Integer},
    fundamental::Real,
    c0::Real,
    phase_mode::Symbol,
    tx_depth::Real,
    tx_lateral::Real,
    phase_jitter::Real,
    rng::Random.AbstractRNG,
)
    phases = Vector{Float64}(undef, length(harmonics))
    for (idx, harmonic) in pairs(harmonics)
        base = if phase_mode in (:geometric, :jittered)
            _geometric_drive_phase(depth, lateral, tx_depth, tx_lateral, harmonic, fundamental, c0)
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

function _within_bounds(point::Tuple{Float64, Float64}, depth_bounds, lateral_bounds)
    depth, lateral = point
    return depth_bounds[1] <= depth <= depth_bounds[2] &&
           lateral_bounds[1] <= lateral <= lateral_bounds[2]
end

function _far_enough(point::Tuple{Float64, Float64}, points::AbstractVector{<:Tuple}, min_separation::Real)
    sep = Float64(min_separation)
    sep <= 0 && return true
    for other in points
        hypot(point[1] - other[1], point[2] - other[2]) >= sep || return false
    end
    return true
end

function _squiggle_centerline(
    anchor_depth::Real,
    anchor_lateral::Real;
    root_length::Real,
    amplitude::Real,
    wavelength::Real,
    slope::Real,
    source_spacing::Real=0.8e-3,
)
    length_m = Float64(root_length)
    length_m > 0 || error("root_length must be positive.")
    wavelength_m = Float64(wavelength)
    wavelength_m > 0 || error("squiggle_wavelength must be positive.")
    spacing = Float64(source_spacing)
    spacing > 0 || error("source_spacing must be positive.")

    n = max(16, ceil(Int, length_m / (spacing / 2)) + 1)
    half = length_m / 2
    points = Tuple{Float64, Float64}[]
    for x in range(-half, half; length=n)
        lateral = Float64(anchor_lateral) + x
        depth = Float64(anchor_depth) + Float64(slope) * x + Float64(amplitude) * sin(2pi * x / wavelength_m)
        push!(points, (depth, lateral))
    end
    return points
end

function _interp_centerline_point(centerline::AbstractVector{<:Tuple}, distance::Real)
    length(centerline) >= 2 || error("Centerline must contain at least two points.")
    remaining = Float64(distance)
    for idx in 1:(length(centerline) - 1)
        d0, l0 = centerline[idx]
        d1, l1 = centerline[idx + 1]
        sd = d1 - d0
        sl = l1 - l0
        seg_len = hypot(sd, sl)
        seg_len > eps(Float64) || continue
        if remaining <= seg_len || idx == length(centerline) - 1
            t = clamp(remaining / seg_len, 0.0, 1.0)
            depth = d0 + t * sd
            lateral = l0 + t * sl
            return depth, lateral, sd / seg_len, sl / seg_len
        end
        remaining -= seg_len
    end
    d0, l0 = centerline[end - 1]
    d1, l1 = centerline[end]
    seg_len = max(hypot(d1 - d0, l1 - l0), eps(Float64))
    return d1, l1, (d1 - d0) / seg_len, (l1 - l0) / seg_len
end

function _sample_squiggle_centerlines(
    centerlines::AbstractVector;
    source_spacing::Real,
    position_jitter::Real,
    min_separation::Real,
    depth_bounds::Tuple{<:Real, <:Real},
    lateral_bounds::Tuple{<:Real, <:Real},
    max_sources::Union{Nothing, Integer},
    rng::Random.AbstractRNG,
)
    spacing = Float64(source_spacing)
    spacing > 0 || error("source_spacing must be positive.")
    points = Tuple{Float64, Float64}[]

    for centerline in centerlines
        length(centerline) >= 2 || continue
        total_len = 0.0
        for idx in 1:(length(centerline) - 1)
            d0, l0 = centerline[idx]
            d1, l1 = centerline[idx + 1]
            total_len += hypot(d1 - d0, l1 - l0)
        end
        total_len > eps(Float64) || continue
        n = max(2, ceil(Int, total_len / spacing) + 1)
        for distance in range(0.0, total_len; length=n)
            depth, lateral, tangent_d, tangent_l = _interp_centerline_point(centerline, distance)
            if position_jitter > 0
                jitter = randn(rng) * Float64(position_jitter)
                depth += -tangent_l * jitter
                lateral += tangent_d * jitter
            end
            point = (depth, lateral)
            _within_bounds(point, depth_bounds, lateral_bounds) || continue
            _far_enough(point, points, min_separation) || continue
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

"""
    make_squiggle_bubble_sources(anchors; kwargs...)

Generate harmonic bubble emitters along one deterministic squiggly centerline
per `(depth, lateral)` anchor.
"""
function make_squiggle_bubble_sources(
    anchors::AbstractVector{<:Tuple};
    root_length::Real=12e-3,
    squiggle_amplitude::Real=1.5e-3,
    squiggle_wavelength::Real=8e-3,
    squiggle_slope::Real=0.0,
    source_spacing::Real=2.0e-3,
    position_jitter::Real=0.05e-3,
    min_separation::Real=1.0e-3,
    max_sources_per_anchor::Union{Nothing, Integer}=nothing,
    depth_bounds::Tuple{<:Real, <:Real}=(0.0, Inf),
    lateral_bounds::Tuple{<:Real, <:Real}=(-Inf, Inf),
    fundamental::Real=5e5,
    amplitude::Real=1.0,
    n_bubbles::Real=1.0,
    harmonics::AbstractVector{<:Integer}=[2, 3, 4],
    harmonic_amplitudes::AbstractVector{<:Real}=[1.0, 0.6, 0.3],
    cavitation_model=:harmonic_cos,
    gate_duration::Real=50e-6,
    taper_ratio::Real=0.25,
    delay::Real=0.0,
    phase_mode=:geometric,
    phase_jitter::Real=0.2,
    transducer_depth::Real=-30e-3,
    transducer_lateral::Real=0.0,
    c0::Real=1500.0,
    rng::Random.AbstractRNG=Random.default_rng(),
)
    isempty(anchors) && error("At least one squiggle anchor is required.")
    harmonics_i = Int.(harmonics)
    isempty(harmonics_i) && error("harmonics must be non-empty.")
    harmonic_amplitudes_f = Float64.(harmonic_amplitudes)
    length(harmonic_amplitudes_f) == length(harmonics_i) ||
        error("harmonic_amplitudes must have the same length as harmonics.")
    mode = _normalize_cluster_phase_mode(phase_mode)
    source_model = _normalize_cavitation_model(cavitation_model)

    clusters = EmissionSource2D[]
    all_centerlines = Vector{Tuple{Float64, Float64}}[]
    source_count_by_anchor = Int[]
    anchor_pairs = [(Float64(anchor[1]), Float64(anchor[2])) for anchor in anchors]

    for (anchor_depth, anchor_lateral) in anchor_pairs
        centerline = _squiggle_centerline(
            anchor_depth,
            anchor_lateral;
            root_length=root_length,
            amplitude=squiggle_amplitude,
            wavelength=squiggle_wavelength,
            slope=squiggle_slope,
            source_spacing=source_spacing,
        )
        push!(all_centerlines, centerline)
        points = _sample_squiggle_centerlines(
            [centerline];
            source_spacing=source_spacing,
            position_jitter=position_jitter,
            min_separation=min_separation,
            depth_bounds=(Float64(depth_bounds[1]), Float64(depth_bounds[2])),
            lateral_bounds=(Float64(lateral_bounds[1]), Float64(lateral_bounds[2])),
            max_sources=max_sources_per_anchor,
            rng=rng,
        )
        push!(source_count_by_anchor, length(points))

        for (depth, lateral) in points
            phases = _cluster_harmonic_phases(
                depth,
                lateral,
                harmonics_i,
                fundamental,
                c0,
                mode,
                transducer_depth,
                transducer_lateral,
                phase_jitter,
                rng,
            )
            kwargs = (
                depth=depth,
                lateral=lateral,
                fundamental=Float64(fundamental),
                amplitude=Float64(amplitude),
                n_bubbles=Float64(n_bubbles),
                harmonics=copy(harmonics_i),
                harmonic_amplitudes=copy(harmonic_amplitudes_f),
                harmonic_phases=phases,
                gate_duration=Float64(gate_duration),
                taper_ratio=Float64(taper_ratio),
                delay=Float64(delay),
            )
            push!(clusters, source_model == :gaussian_pulse ? GaussianPulseCluster2D(; kwargs...) : BubbleCluster2D(; kwargs...))
        end
    end

    isempty(clusters) && error("Squiggle source generation produced no sources inside the requested bounds.")
    meta = Dict{Symbol, Any}(
        :source_model => :squiggle,
        :anchors => anchor_pairs,
        :root_length => Float64(root_length),
        :squiggle_amplitude => Float64(squiggle_amplitude),
        :squiggle_wavelength => Float64(squiggle_wavelength),
        :squiggle_slope => Float64(squiggle_slope),
        :source_spacing => Float64(source_spacing),
        :position_jitter => Float64(position_jitter),
        :min_separation => Float64(min_separation),
        :max_sources_per_anchor => isnothing(max_sources_per_anchor) ? nothing : Int(max_sources_per_anchor),
        :source_count_by_anchor => source_count_by_anchor,
        :centerlines => all_centerlines,
        :phase_mode => mode,
        :cavitation_model => source_model,
    )
    return clusters, meta
end

_source_duration(src::PointSource2D) = src.num_cycles / src.frequency
_source_duration(src::BubbleCluster2D) = src.gate_duration
_source_duration(src::GaussianPulseCluster2D) = src.gate_duration

function _tukey_window(n::Int, ratio::Real)
    n <= 0 && return Float64[]
    n == 1 && return ones(Float64, 1)
    r = clamp(Float64(ratio), 0.0, 1.0)
    r == 0.0 && return ones(Float64, n)

    x = collect(range(0.0, 1.0; length=n))
    w = ones(Float64, n)
    if r == 1.0
        return 0.5 .* (1 .- cos.(2pi .* x))
    end

    left_edge = r / 2
    right_edge = 1 - left_edge
    @inbounds for idx in eachindex(x)
        xi = x[idx]
        if xi < left_edge
            w[idx] = 0.5 * (1 + cos((2pi / r) * (xi - left_edge)))
        elseif xi > right_edge
            w[idx] = 0.5 * (1 + cos((2pi / r) * (xi - right_edge)))
        end
    end
    return w
end

function _tone_burst_signal(nt::Int, dt::Real, src::PointSource2D; taper_ratio::Real=0.25)
    signal = zeros(Float64, nt)
    duration = src.num_cycles / src.frequency
    samples = collect(0:(nt - 1))
    t = samples .* Float64(dt) .- src.delay
    active = findall((t .>= 0.0) .& (t .<= duration))
    isempty(active) && return signal

    envelope = _tukey_window(length(active), taper_ratio)
    signal[active] .= src.amplitude .* envelope .* sin.(2pi .* src.frequency .* t[active] .+ src.phase)
    return signal
end

function _cluster_emission_signal(nt::Int, dt::Real, src::BubbleCluster2D)
    length(src.harmonics) == length(src.harmonic_amplitudes) ||
        error("BubbleCluster2D: harmonics and harmonic_amplitudes must have equal length.")
    length(src.harmonics) == length(src.harmonic_phases) ||
        error("BubbleCluster2D: harmonics and harmonic_phases must have equal length.")

    signal = zeros(Float64, nt)
    samples = collect(0:(nt - 1))
    t = samples .* Float64(dt) .- src.delay
    active = findall((t .>= 0.0) .& (t .<= src.gate_duration))
    isempty(active) && return signal

    envelope = _tukey_window(length(active), src.taper_ratio)
    total_amp = src.amplitude * src.n_bubbles
    t_active = t[active]

    accumulator = zeros(Float64, length(active))
    @inbounds for i in eachindex(src.harmonics)
        n = src.harmonics[i]
        alpha_n = src.harmonic_amplitudes[i]
        phi_n = src.harmonic_phases[i]
        accumulator .+= alpha_n .* cos.(2pi .* n .* src.fundamental .* t_active .+ phi_n)
    end
    signal[active] .= total_amp .* envelope .* accumulator
    return signal
end

function _cluster_emission_signal(nt::Int, dt::Real, src::GaussianPulseCluster2D)
    length(src.harmonics) == length(src.harmonic_amplitudes) ||
        error("GaussianPulseCluster2D: harmonics and harmonic_amplitudes must have equal length.")
    length(src.harmonics) == length(src.harmonic_phases) ||
        error("GaussianPulseCluster2D: harmonics and harmonic_phases must have equal length.")

    signal = zeros(Float64, nt)
    samples = collect(0:(nt - 1))
    t = samples .* Float64(dt) .- src.delay
    active = findall((t .>= 0.0) .& (t .<= src.gate_duration))
    isempty(active) && return signal

    duration = Float64(src.gate_duration)
    duration > 0 || return signal
    center = duration / 2
    sigma = duration / 6
    t_active = t[active]
    envelope = exp.(-0.5 .* ((t_active .- center) ./ sigma) .^ 2)
    total_amp = src.amplitude * src.n_bubbles

    accumulator = zeros(Float64, length(active))
    @inbounds for i in eachindex(src.harmonics)
        n = src.harmonics[i]
        alpha_n = src.harmonic_amplitudes[i]
        phi_n = src.harmonic_phases[i]
        accumulator .+= alpha_n .* cos.(2pi .* n .* src.fundamental .* (t_active .- center) .+ phi_n)
    end
    signal[active] .= total_amp .* envelope .* accumulator
    return signal
end

_source_signal(nt::Int, dt::Real, src::PointSource2D) = _tone_burst_signal(nt, dt, src)
_source_signal(nt::Int, dt::Real, src::BubbleCluster2D) = _cluster_emission_signal(nt, dt, src)
_source_signal(nt::Int, dt::Real, src::GaussianPulseCluster2D) = _cluster_emission_signal(nt, dt, src)

function _resample_source_phases(
    sources::AbstractVector{<:EmissionSource2D},
    rng::Random.AbstractRNG,
)
    return map(sources) do src
        if src isa BubbleCluster2D
            BubbleCluster2D(
                depth=src.depth, lateral=src.lateral,
                fundamental=src.fundamental, amplitude=src.amplitude,
                n_bubbles=src.n_bubbles, harmonics=copy(src.harmonics),
                harmonic_amplitudes=copy(src.harmonic_amplitudes),
                harmonic_phases=2pi .* rand(rng, length(src.harmonics)),
                gate_duration=src.gate_duration, taper_ratio=src.taper_ratio,
                delay=src.delay,
            )
        elseif src isa GaussianPulseCluster2D
            GaussianPulseCluster2D(
                depth=src.depth, lateral=src.lateral,
                fundamental=src.fundamental, amplitude=src.amplitude,
                n_bubbles=src.n_bubbles, harmonics=copy(src.harmonics),
                harmonic_amplitudes=copy(src.harmonic_amplitudes),
                harmonic_phases=2pi .* rand(rng, length(src.harmonics)),
                gate_duration=src.gate_duration, taper_ratio=src.taper_ratio,
                delay=src.delay,
            )
        elseif src isa PointSource2D
            PointSource2D(
                depth=src.depth, lateral=src.lateral,
                frequency=src.frequency, amplitude=src.amplitude,
                phase=2pi * rand(rng), delay=src.delay, num_cycles=src.num_cycles,
            )
        else
            src
        end
    end
end

function _expand_sources_per_window(
    sources::AbstractVector{<:EmissionSource2D},
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
    expanded = EmissionSource2D[]
    sizehint!(expanded, length(sources) * n_frames)
    fjitter = variability.frequency_jitter_fraction
    fjitter >= 0.0 || error("frequency_jitter_fraction must be non-negative.")
    for src in sources
        for k in 1:n_frames
            d = src.delay + hop_s * (k - 1)
            fscale = fjitter > 0.0 ? max(0.01, 1.0 + fjitter * randn(rng)) : 1.0
            evt = if src isa BubbleCluster2D
                BubbleCluster2D(
                    depth=src.depth, lateral=src.lateral,
                    fundamental=src.fundamental * fscale, amplitude=src.amplitude,
                    n_bubbles=src.n_bubbles, harmonics=copy(src.harmonics),
                    harmonic_amplitudes=copy(src.harmonic_amplitudes),
                    harmonic_phases=2pi .* rand(rng, length(src.harmonics)),
                    gate_duration=min(src.gate_duration, frame_dur), taper_ratio=src.taper_ratio,
                    delay=d,
                )
            elseif src isa GaussianPulseCluster2D
                GaussianPulseCluster2D(
                    depth=src.depth, lateral=src.lateral,
                    fundamental=src.fundamental * fscale, amplitude=src.amplitude,
                    n_bubbles=src.n_bubbles, harmonics=copy(src.harmonics),
                    harmonic_amplitudes=copy(src.harmonic_amplitudes),
                    harmonic_phases=2pi .* rand(rng, length(src.harmonics)),
                    gate_duration=min(src.gate_duration, frame_dur), taper_ratio=src.taper_ratio,
                    delay=d,
                )
            elseif src isa PointSource2D
                nc = max(1, round(Int, min(Float64(src.num_cycles) / src.frequency, frame_dur) * src.frequency * fscale))
                PointSource2D(
                    depth=src.depth, lateral=src.lateral,
                    frequency=src.frequency * fscale, amplitude=src.amplitude,
                    phase=2pi * rand(rng), delay=d, num_cycles=nc,
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
