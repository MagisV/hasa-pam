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

_emission_frequencies(src::PointSource2D) = Float64[src.frequency]
_emission_frequencies(src::BubbleCluster2D) = Float64[n * src.fundamental for n in src.harmonics]

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
    return -2π * Int(harmonic) * Float64(fundamental) * distance / Float64(c0)
end

function _normalize_cluster_phase_mode(phase_mode)
    mode = Symbol(lowercase(string(phase_mode)))
    mode in (:coherent, :geometric, :random, :jittered) ||
        error("Unknown phase_mode: $phase_mode (expected coherent, geometric, random, or jittered).")
    return mode
end

function _normalize_vascular_topology(topology)
    mode = Symbol(lowercase(string(topology)))
    mode in (:squiggle, :bundle, :tree) ||
        error("Unknown vascular topology: $topology (expected squiggle, bundle, or tree).")
    return mode
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
            2π * rand(rng)
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

function _vascular_tree_segments(
    anchor_depth::Real,
    anchor_lateral::Real;
    root_length::Real,
    branch_levels::Integer,
    branch_angle::Real,
    branch_scale::Real,
)
    root_half = Float64(root_length) / 2
    root = (
        Float64(anchor_depth) - root_half,
        Float64(anchor_lateral),
        Float64(anchor_depth) + root_half,
        Float64(anchor_lateral),
    )
    segments = [root]
    active = [root]

    for _ in 1:Int(branch_levels)
        next_active = typeof(root)[]
        for segment in active
            d0, l0, d1, l1 = segment
            vd = d1 - d0
            vl = l1 - l0
            parent_len = hypot(vd, vl)
            parent_len > eps(Float64) || continue
            base_angle = atan(vl, vd)
            child_len = parent_len * Float64(branch_scale)

            for (side, t) in zip((-1.0, 1.0), (0.35, 0.65))
                branch_depth = d0 + t * vd
                branch_lateral = l0 + t * vl
                angle = base_angle + side * Float64(branch_angle)
                child = (
                    branch_depth,
                    branch_lateral,
                    branch_depth + child_len * cos(angle),
                    branch_lateral + child_len * sin(angle),
                )
                push!(segments, child)
                push!(next_active, child)
            end
        end
        active = next_active
    end

    return segments
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

function _sample_vascular_segments(
    segments::AbstractVector{<:Tuple};
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
    for segment in segments
        d0, l0, d1, l1 = segment
        seg_len = hypot(d1 - d0, l1 - l0)
        seg_len > eps(Float64) || continue
        n = max(2, ceil(Int, seg_len / spacing) + 1)
        for idx in 1:n
            t = n == 1 ? 0.0 : (idx - 1) / (n - 1)
            depth = d0 + t * (d1 - d0)
            lateral = l0 + t * (l1 - l0)
            if position_jitter > 0
                jitter = Float64(position_jitter)
                depth += randn(rng) * jitter
                lateral += randn(rng) * jitter
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

function _squiggle_centerline(
    anchor_depth::Real,
    anchor_lateral::Real;
    root_length::Real,
    amplitude::Real,
    wavelength::Real,
    slope::Real,
    axial_offset::Real=0.0,
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
        depth = Float64(anchor_depth) + Float64(axial_offset) +
                Float64(slope) * x +
                Float64(amplitude) * sin(2π * x / wavelength_m)
        push!(points, (depth, lateral))
    end
    return points
end

function _vascular_centerlines(
    anchor_depth::Real,
    anchor_lateral::Real;
    topology::Symbol,
    root_length::Real,
    squiggle_amplitude::Real,
    squiggle_wavelength::Real,
    squiggle_slope::Real,
    bundle_count::Integer,
    bundle_spacing::Real,
    source_spacing::Real,
)
    if topology == :squiggle
        return [
            _squiggle_centerline(
                anchor_depth,
                anchor_lateral;
                root_length=root_length,
                amplitude=squiggle_amplitude,
                wavelength=squiggle_wavelength,
                slope=squiggle_slope,
                source_spacing=source_spacing,
            ),
        ]
    elseif topology == :bundle
        count = Int(bundle_count)
        count > 0 || error("bundle_count must be positive.")
        spacing = Float64(bundle_spacing)
        spacing >= 0 || error("bundle_spacing must be non-negative.")
        center = (count + 1) / 2
        return [
            _squiggle_centerline(
                anchor_depth,
                anchor_lateral;
                root_length=root_length,
                amplitude=squiggle_amplitude,
                wavelength=squiggle_wavelength,
                slope=squiggle_slope,
                axial_offset=(idx - center) * spacing,
                source_spacing=source_spacing,
            )
            for idx in 1:count
        ]
    end
    error("_vascular_centerlines only supports squiggle and bundle topologies.")
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

function _sample_vascular_centerlines(
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
    make_vascular_bubble_clusters(anchors; kwargs...)

Generate many small `BubbleCluster2D` emitters along a deterministic 2D
vessel-like topology rooted at each `(depth, lateral)` anchor. The default
topology is a paper-like squiggly horizontal vessel; the original branching
tree remains available via `topology=:tree`.
"""
function make_vascular_bubble_clusters(
    anchors::AbstractVector{<:Tuple};
    topology=:squiggle,
    root_length::Real=12e-3,
    branch_levels::Integer=2,
    branch_angle::Real=30π / 180,
    branch_scale::Real=0.65,
    squiggle_amplitude::Real=1.5e-3,
    squiggle_wavelength::Real=8e-3,
    squiggle_slope::Real=0.0,
    bundle_count::Integer=3,
    bundle_spacing::Real=2e-3,
    source_spacing::Real=0.8e-3,
    position_jitter::Real=0.15e-3,
    min_separation::Real=0.3e-3,
    max_sources_per_anchor::Union{Nothing, Integer}=nothing,
    depth_bounds::Tuple{<:Real, <:Real}=(0.0, Inf),
    lateral_bounds::Tuple{<:Real, <:Real}=(-Inf, Inf),
    fundamental::Real=5e5,
    amplitude::Real=1.0,
    n_bubbles::Real=1.0,
    harmonics::AbstractVector{<:Integer}=[2, 3],
    harmonic_amplitudes::AbstractVector{<:Real}=[1.0, 0.6],
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
    isempty(anchors) && error("At least one vascular anchor is required.")
    harmonics_i = Int.(harmonics)
    isempty(harmonics_i) && error("harmonics must be non-empty.")
    harmonic_amplitudes_f = Float64.(harmonic_amplitudes)
    length(harmonic_amplitudes_f) == length(harmonics_i) ||
        error("harmonic_amplitudes must have the same length as harmonics.")
    mode = _normalize_cluster_phase_mode(phase_mode)
    vessel_topology = _normalize_vascular_topology(topology)

    clusters = BubbleCluster2D[]
    all_segments = Tuple{Float64, Float64, Float64, Float64}[]
    all_centerlines = Vector{Tuple{Float64, Float64}}[]
    source_count_by_anchor = Int[]
    anchor_pairs = [(Float64(anchor[1]), Float64(anchor[2])) for anchor in anchors]

    for (anchor_depth, anchor_lateral) in anchor_pairs
        points = if vessel_topology == :tree
            segments = _vascular_tree_segments(
                anchor_depth,
                anchor_lateral;
                root_length=root_length,
                branch_levels=branch_levels,
                branch_angle=branch_angle,
                branch_scale=branch_scale,
            )
            append!(all_segments, segments)
            _sample_vascular_segments(
                segments;
                source_spacing=source_spacing,
                position_jitter=position_jitter,
                min_separation=min_separation,
                depth_bounds=(Float64(depth_bounds[1]), Float64(depth_bounds[2])),
                lateral_bounds=(Float64(lateral_bounds[1]), Float64(lateral_bounds[2])),
                max_sources=max_sources_per_anchor,
                rng=rng,
            )
        else
            centerlines = _vascular_centerlines(
                anchor_depth,
                anchor_lateral;
                topology=vessel_topology,
                root_length=root_length,
                squiggle_amplitude=squiggle_amplitude,
                squiggle_wavelength=squiggle_wavelength,
                squiggle_slope=squiggle_slope,
                bundle_count=bundle_count,
                bundle_spacing=bundle_spacing,
                source_spacing=source_spacing,
            )
            append!(all_centerlines, centerlines)
            _sample_vascular_centerlines(
                centerlines;
                source_spacing=source_spacing,
                position_jitter=position_jitter,
                min_separation=min_separation,
                depth_bounds=(Float64(depth_bounds[1]), Float64(depth_bounds[2])),
                lateral_bounds=(Float64(lateral_bounds[1]), Float64(lateral_bounds[2])),
                max_sources=max_sources_per_anchor,
                rng=rng,
            )
        end
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
            push!(clusters, BubbleCluster2D(
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
            ))
        end
    end

    isempty(clusters) && error("Vascular cluster generation produced no sources inside the requested bounds.")
    meta = Dict{Symbol, Any}(
        :cluster_model => :vascular,
        :topology => vessel_topology,
        :anchors => anchor_pairs,
        :root_length => Float64(root_length),
        :branch_levels => Int(branch_levels),
        :branch_angle => Float64(branch_angle),
        :branch_scale => Float64(branch_scale),
        :squiggle_amplitude => Float64(squiggle_amplitude),
        :squiggle_wavelength => Float64(squiggle_wavelength),
        :squiggle_slope => Float64(squiggle_slope),
        :bundle_count => Int(bundle_count),
        :bundle_spacing => Float64(bundle_spacing),
        :source_spacing => Float64(source_spacing),
        :position_jitter => Float64(position_jitter),
        :min_separation => Float64(min_separation),
        :max_sources_per_anchor => isnothing(max_sources_per_anchor) ? nothing : Int(max_sources_per_anchor),
        :source_count_by_anchor => source_count_by_anchor,
        :segments => all_segments,
        :centerlines => all_centerlines,
        :phase_mode => mode,
    )
    return clusters, meta
end

Base.@kwdef struct PAMConfig
    dx::Float64 = 0.2e-3
    dz::Float64 = 0.2e-3
    axial_dim::Float64 = 90e-3
    transverse_dim::Float64 = 60e-3
    receiver_aperture::Union{Nothing, Float64} = 50e-3
    receiver_row::Union{Nothing, Int} = nothing
    t_max::Float64 = 80e-6
    dt::Float64 = 40e-9
    c0::Float64 = 1500.0
    rho0::Float64 = 1000.0
    PML_GUARD::Int = 20
    zero_pad_factor::Int = 4
    tukey_ratio::Float64 = 0.25
    peak_suppression_radius::Float64 = 2e-3
    success_tolerance::Float64 = 1e-3
end

function _default_pam_pml_guard(dx::Real)
    # Keep the default physical guard close to 4 mm across PAM resolutions.
    return max(4, round(Int, 4e-3 / Float64(dx)))
end

function _pam_pml_guard(cfg::PAMConfig)
    # `PML_GUARD=20` is the historical default. Interpret it as a placeholder and
    # scale the actual guard with `dx` so coarse grids don't silently lose most
    # of the reconstruction depth.
    if cfg.PML_GUARD == 20
        return _default_pam_pml_guard(cfg.dx)
    end
    return cfg.PML_GUARD
end

_source_duration(src::PointSource2D) = src.num_cycles / src.frequency
_source_duration(src::BubbleCluster2D) = src.gate_duration

function _pam_axial_substeps(dx::Real, axial_step::Real)
    ratio = Float64(dx) / Float64(axial_step)
    nearest = round(Int, ratio)
    if isapprox(ratio, nearest; rtol=1e-9, atol=1e-12)
        return max(1, nearest)
    end
    return max(1, ceil(Int, ratio))
end

function _required_pam_t_max(
    cfg::PAMConfig,
    sources::AbstractVector{<:EmissionSource2D};
    time_margin::Real=10e-6,
)
    isempty(sources) && return 0.0
    kgrid = pam_grid(cfg)
    receiver_laterals = kgrid.y_vec[receiver_col_range(cfg)]
    required_t = 0.0
    for src in sources
        max_receiver_offset = maximum(abs.(receiver_laterals .- src.lateral))
        latest_arrival = src.delay + hypot(src.depth, max_receiver_offset) / cfg.c0
        required_t = max(required_t, latest_arrival + _source_duration(src))
    end
    return required_t + Float64(time_margin)
end

function fit_pam_config(
    cfg::PAMConfig,
    sources::AbstractVector{<:EmissionSource2D};
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
    target_t_max = max(cfg.t_max, _required_pam_t_max(cfg, sources; time_margin=time_margin))

    target_rows == pam_Nx(cfg) && target_t_max == cfg.t_max && return cfg
    return PAMConfig(
        dx=cfg.dx,
        dz=cfg.dz,
        axial_dim=target_axial_dim,
        transverse_dim=cfg.transverse_dim,
        receiver_aperture=cfg.receiver_aperture,
        receiver_row=cfg.receiver_row,
        t_max=target_t_max,
        dt=cfg.dt,
        c0=cfg.c0,
        rho0=cfg.rho0,
        PML_GUARD=cfg.PML_GUARD,
        zero_pad_factor=cfg.zero_pad_factor,
        tukey_ratio=cfg.tukey_ratio,
        peak_suppression_radius=cfg.peak_suppression_radius,
        success_tolerance=cfg.success_tolerance,
    )
end

pam_Nx(cfg::PAMConfig) = round(Int, cfg.axial_dim / cfg.dx)
pam_Ny(cfg::PAMConfig) = round(Int, cfg.transverse_dim / cfg.dz)
pam_Nt(cfg::PAMConfig) = round(Int, cfg.t_max / cfg.dt)
receiver_row(cfg::PAMConfig) = something(cfg.receiver_row, 1)

function receiver_col_range(cfg::PAMConfig)
    ny = pam_Ny(cfg)
    if isnothing(cfg.receiver_aperture)
        return 1:ny
    end
    n_active = clamp(round(Int, cfg.receiver_aperture / cfg.dz), 1, ny)
    mid = fld(ny, 2) + 1
    half = fld(n_active, 2)
    start_col = mid - half
    end_col = start_col + n_active - 1
    return start_col:end_col
end

function pam_grid(cfg::PAMConfig; Nt::Union{Nothing, Integer}=nothing)
    nt = isnothing(Nt) ? pam_Nt(cfg) : Int(Nt)
    return KGrid2D(pam_Nx(cfg), pam_Ny(cfg), cfg.dx, cfg.dz; dt=cfg.dt, Nt=nt)
end

function depth_coordinates(kgrid::KGrid2D, cfg::PAMConfig)
    return kgrid.x_vec .- kgrid.x_vec[receiver_row(cfg)]
end

function _resample_pam_slice(
    slice::AbstractMatrix{<:Real},
    spacing_row_mm::Float64,
    spacing_col_mm::Float64,
    new_row_mm::Float64,
    new_col_mm::Float64,
)
    out_rows = round(Int, size(slice, 1) * spacing_row_mm / new_row_mm)
    out_cols = round(Int, size(slice, 2) * spacing_col_mm / new_col_mm)
    row_coords = 1 .+ (0:(out_rows - 1)) .* (new_row_mm / spacing_row_mm)
    col_coords = 1 .+ (0:(out_cols - 1)) .* (new_col_mm / spacing_col_mm)
    itp = extrapolate(interpolate(Float32.(slice), BSpline(Linear())), Flat())

    out = Matrix{Float32}(undef, out_rows, out_cols)
    @inbounds for row in 1:out_rows
        row_coord = row_coords[row]
        for col in 1:out_cols
            out[row, col] = Float32(itp(row_coord, col_coords[col]))
        end
    end
    return out
end

function _load_pam_ct(
    hu_vol::Union{Nothing, AbstractArray{<:Real, 3}},
    spacing_m::Union{Nothing, NTuple{3, <:Real}},
    ct_path::AbstractString,
)
    if isnothing(hu_vol)
        isnothing(spacing_m) || error("Pass both hu_vol and spacing_m, or neither.")
        return load_default_ct(ct_path=ct_path)
    end
    isnothing(spacing_m) && error("spacing_m is required when supplying hu_vol for PAM skull medium construction.")
    return hu_vol, spacing_m
end

function make_pam_medium(
    cfg::PAMConfig;
    aberrator::Symbol=:none,
    lens_center_depth::Real=20e-3,
    lens_center_lateral::Real=0.0,
    lens_axial_radius::Real=4e-3,
    lens_lateral_radius::Real=12e-3,
    c_aberrator::Real=1700.0,
    rho_aberrator::Real=1150.0,
    hu_vol::Union{Nothing, AbstractArray{<:Real, 3}}=nothing,
    spacing_m::Union{Nothing, NTuple{3, <:Real}}=nothing,
    ct_path::AbstractString=DEFAULT_CT_PATH,
    slice_index::Integer=250,
    skull_to_transducer::Real=30e-3,
    hu_bone_thr::Integer=200,
)
    kgrid = pam_grid(cfg)
    c = fill(Float32(cfg.c0), kgrid.Nx, kgrid.Ny)
    rho = fill(Float32(cfg.rho0), kgrid.Nx, kgrid.Ny)

    if aberrator == :none
        return c, rho, Dict{Symbol, Any}(:aberrator => :none)
    elseif aberrator == :skull
        hu_local, spacing_local = _load_pam_ct(hu_vol, spacing_m, ct_path)
        slice0 = Int(slice_index)
        0 <= slice0 < size(hu_local, 1) || error("slice_index=$slice0 is out of bounds for $(size(hu_local, 1)) CT slices.")

        hu_slice = Float32.(hu_local[slice0 + 1, :, :])
        row_spacing_mm = Float64(spacing_local[2]) * 1e3
        col_spacing_mm = Float64(spacing_local[1]) * 1e3
        target_row_mm = cfg.dx * 1e3
        target_col_mm = cfg.dz * 1e3
        if !isapprox(row_spacing_mm, target_row_mm; atol=1e-9) || !isapprox(col_spacing_mm, target_col_mm; atol=1e-9)
            hu_slice = _resample_pam_slice(
                hu_slice,
                row_spacing_mm,
                col_spacing_mm,
                target_row_mm,
                target_col_mm,
            )
        end
        hu_slice = _adjust_lateral_size(hu_slice, kgrid.Ny)

        outer_row_rel, inner_row_rel = find_skull_boundaries(
            hu_slice;
            hu_bone_thr=hu_bone_thr,
            num_cols=10,
            expand_if_empty=true,
        )

        outer_row_target = receiver_row(cfg) + round(Int, Float64(skull_to_transducer) / cfg.dx)
        shift = outer_row_target - outer_row_rel
        if shift > 0
            padded = fill(Float32(-1000), size(hu_slice, 1) + shift, size(hu_slice, 2))
            padded[(shift + 1):end, :] .= hu_slice
            hu_slice = padded
        elseif shift < 0
            crop_start = 1 - shift
            crop_start <= size(hu_slice, 1) || error("Skull alignment would crop away the entire CT slice.")
            hu_slice = hu_slice[crop_start:end, :]
        end

        desired_rows = kgrid.Nx
        if size(hu_slice, 1) > desired_rows
            hu_slice = hu_slice[1:desired_rows, :]
        elseif size(hu_slice, 1) < desired_rows
            padded = fill(Float32(-1000), desired_rows, size(hu_slice, 2))
            padded[1:size(hu_slice, 1), :] .= hu_slice
            hu_slice = padded
        end

        outer_row_rel, inner_row_rel = find_skull_boundaries(
            hu_slice;
            hu_bone_thr=hu_bone_thr,
            num_cols=10,
            expand_if_empty=true,
        )
        outer_row_rel == outer_row_target || error("Failed to align the skull to the requested PAM outer row.")

        rho_slice, c_slice = hu_to_rho_c(
            hu_slice;
            hu_bone_thr=hu_bone_thr,
            rho_water=cfg.rho0,
            rho_bone=2100.0,
            c_water=cfg.c0,
            c_bone=2500.0,
        )
        c .= c_slice
        rho .= rho_slice
        return c, rho, Dict{Symbol, Any}(
            :aberrator => :skull,
            :slice_index => slice0,
            :outer_row => outer_row_rel,
            :inner_row => inner_row_rel,
            :receiver_row => receiver_row(cfg),
            :skull_to_transducer => Float64(skull_to_transducer),
            :hu_bone_thr => Int(hu_bone_thr),
            :ct_path => ct_path,
        )
    elseif aberrator != :lens
        error("Unknown PAM medium aberrator: $aberrator")
    end

    depth = depth_coordinates(kgrid, cfg)
    lateral = kgrid.y_vec
    mask = falses(kgrid.Nx, kgrid.Ny)
    @inbounds for i in 1:kgrid.Nx, j in 1:kgrid.Ny
        value = ((depth[i] - lens_center_depth) / lens_axial_radius)^2 +
                ((lateral[j] - lens_center_lateral) / lens_lateral_radius)^2
        if value <= 1.0
            mask[i, j] = true
            c[i, j] = Float32(c_aberrator)
            rho[i, j] = Float32(rho_aberrator)
        end
    end

    return c, rho, Dict{Symbol, Any}(
        :aberrator => :lens,
        :mask => mask,
        :lens_center_depth => Float64(lens_center_depth),
        :lens_center_lateral => Float64(lens_center_lateral),
        :lens_axial_radius => Float64(lens_axial_radius),
        :lens_lateral_radius => Float64(lens_lateral_radius),
        :c_aberrator => Float64(c_aberrator),
        :rho_aberrator => Float64(rho_aberrator),
    )
end

function _tukey_window(n::Int, ratio::Real)
    n <= 0 && return Float64[]
    n == 1 && return ones(Float64, 1)
    r = clamp(Float64(ratio), 0.0, 1.0)
    r == 0.0 && return ones(Float64, n)

    x = collect(range(0.0, 1.0; length=n))
    w = ones(Float64, n)
    if r == 1.0
        return 0.5 .* (1 .- cos.(2π .* x))
    end

    left_edge = r / 2
    right_edge = 1 - left_edge
    @inbounds for idx in eachindex(x)
        xi = x[idx]
        if xi < left_edge
            w[idx] = 0.5 * (1 + cos((2π / r) * (xi - left_edge)))
        elseif xi > right_edge
            w[idx] = 0.5 * (1 + cos((2π / r) * (xi - right_edge)))
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
    signal[active] .= src.amplitude .* envelope .* sin.(2π .* src.frequency .* t[active] .+ src.phase)
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
        αn = src.harmonic_amplitudes[i]
        φn = src.harmonic_phases[i]
        accumulator .+= αn .* cos.(2π .* n .* src.fundamental .* t_active .+ φn)
    end
    signal[active] .= total_amp .* envelope .* accumulator
    return signal
end

_source_signal(nt::Int, dt::Real, src::PointSource2D) = _tone_burst_signal(nt, dt, src)
_source_signal(nt::Int, dt::Real, src::BubbleCluster2D) = _cluster_emission_signal(nt, dt, src)

function source_grid_index(src::EmissionSource2D, cfg::PAMConfig, kgrid::KGrid2D)
    src.depth >= 0.0 || error("Source depth must be >= 0.")
    row = receiver_row(cfg) + round(Int, src.depth / cfg.dx)
    col = argmin(abs.(kgrid.y_vec .- src.lateral))
    1 <= row <= kgrid.Nx || error("Source depth $(src.depth) m lies outside the computational grid.")
    return row, col
end

function _zero_pad_receiver_rf(rf::AbstractMatrix, target_ny::Int)
    ny, nt = size(rf)
    target_ny >= ny || error("target_ny must be >= current Ny.")
    extra = target_ny - ny
    left = fld(extra, 2)
    range = (left + 1):(left + ny)
    out = zeros(promote_type(Float64, eltype(rf)), target_ny, nt)
    out[range, :] .= rf
    return out, range
end

function _edge_pad_lateral(a::AbstractMatrix{<:Real}, target_ny::Int)
    nx, ny = size(a)
    target_ny >= ny || error("target_ny must be >= current Ny.")
    extra = target_ny - ny
    left = fld(extra, 2)
    range = (left + 1):(left + ny)

    out = Matrix{Float64}(undef, nx, target_ny)
    out[:, range] .= Float64.(a)
    if left > 0
        out[:, 1:left] .= reshape(Float64.(a[:, 1]), :, 1)
    end
    right = target_ny - last(range)
    if right > 0
        out[:, (last(range) + 1):end] .= reshape(Float64.(a[:, end]), :, 1)
    end
    return out, range
end

function _fft_wavenumbers(n::Int, spacing::Real)
    dk = 2π / Float64(spacing)
    start_val = -fld(n, 2)
    end_val = ceil(Int, n / 2) - 1
    return collect(start_val:end_val) .* dk ./ n
end

function _select_frequency_bins(
    rf::AbstractMatrix{<:Real},
    dt::Real,
    frequencies;
    bandwidth::Real=0.0,
)
    nt = size(rf, 2)
    freq_axis = collect(0:(nt - 1)) ./ (nt * Float64(dt))
    pos_bins = 2:(fld(nt, 2) + 1)  # positive frequencies, excluding DC

    if isnothing(frequencies)
        spectrum = fft(rf, 2)
        mean_mag = vec(mean(abs.(spectrum[:, pos_bins]); dims=1))
        idx = argmax(mean_mag)
        return [freq_axis[pos_bins[idx]]], [pos_bins[idx]]
    end

    bins = Int[]
    resolved_freqs = Float64[]
    half_bw = Float64(bandwidth) / 2
    for freq in frequencies
        f = Float64(freq)
        if half_bw > 0
            for bin in pos_bins
                fb = freq_axis[bin]
                if fb >= f - half_bw && fb <= f + half_bw && bin ∉ bins
                    push!(bins, bin)
                    push!(resolved_freqs, fb)
                end
            end
        else
            idx = argmin(abs.(freq_axis[pos_bins] .- f))
            bin = pos_bins[idx]
            if bin ∉ bins
                push!(bins, bin)
                push!(resolved_freqs, freq_axis[bin])
            end
        end
    end
    return resolved_freqs, bins
end

function _connected_component(mask::BitMatrix, seed::Tuple{Int, Int})
    mask[seed...] || return Tuple{Int, Int}[]
    rows, cols = size(mask)
    visited = falses(rows, cols)
    queue = [seed]
    visited[seed...] = true
    component = Tuple{Int, Int}[]

    while !isempty(queue)
        current = popfirst!(queue)
        push!(component, current)
        i, j = current
        for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
            ii = i + di
            jj = j + dj
            if 1 <= ii <= rows && 1 <= jj <= cols && mask[ii, jj] && !visited[ii, jj]
                visited[ii, jj] = true
                push!(queue, (ii, jj))
            end
        end
    end

    return component
end

function _peak_fwhm_mm(intensity::AbstractMatrix{<:Real}, kgrid::KGrid2D, cfg::PAMConfig, idx::Tuple{Int, Int})
    peak = Float64(intensity[idx...])
    peak <= 0.0 && return 0.0, 0.0

    mask = Float64.(intensity) .>= peak / 2
    component = _connected_component(mask, idx)
    isempty(component) && return 0.0, 0.0

    depth = depth_coordinates(kgrid, cfg)
    lateral = kgrid.y_vec
    rows = first.(component)
    cols = last.(component)
    axial_fwhm = (maximum(depth[rows]) - minimum(depth[rows])) * 1e3
    lateral_fwhm = (maximum(lateral[cols]) - minimum(lateral[cols])) * 1e3
    return axial_fwhm, lateral_fwhm
end

function _best_assignment(cost::AbstractMatrix{<:Real})
    n_rows, n_cols = size(cost)
    n_rows == n_cols || error("Cost matrix must be square for assignment.")

    best_cost = Ref(Inf)
    best_perm = collect(1:n_cols)
    used = falses(n_cols)
    current = Vector{Int}(undef, n_rows)

    function recurse(row::Int, running::Float64)
        if row > n_rows
            if running < best_cost[]
                best_cost[] = running
                best_perm .= current
            end
            return
        end

        for col in 1:n_cols
            used[col] && continue
            new_cost = running + Float64(cost[row, col])
            new_cost < best_cost[] || continue
            used[col] = true
            current[row] = col
            recurse(row + 1, new_cost)
            used[col] = false
        end
    end

    recurse(1, 0.0)
    return best_perm, best_cost[]
end

function find_pam_peaks(
    intensity::AbstractMatrix{<:Real},
    kgrid::KGrid2D,
    cfg::PAMConfig;
    n_peaks::Integer,
    suppression_radius::Real=cfg.peak_suppression_radius,
)
    work = copy(Float64.(intensity))
    row_start = receiver_row(cfg) + 1
    row_stop = size(work, 1)
    row_start <= row_stop || error("No valid reconstruction rows remain after excluding the receiver row and PML.")
    work[1:(row_start - 1), :] .= -Inf
    if row_stop < size(work, 1)
        work[(row_stop + 1):end, :] .= -Inf
    end

    rad_rows = max(1, round(Int, suppression_radius / cfg.dx))
    rad_cols = max(1, round(Int, suppression_radius / cfg.dz))
    peaks = Tuple{Int, Int}[]

    for _ in 1:Int(n_peaks)
        idx = Tuple(argmax(work))
        isfinite(work[idx...]) || break
        push!(peaks, idx)
        r0, c0 = idx
        r1 = max(1, r0 - rad_rows)
        r2 = min(size(work, 1), r0 + rad_rows)
        c1 = max(1, c0 - rad_cols)
        c2 = min(size(work, 2), c0 + rad_cols)
        work[r1:r2, c1:c2] .= -Inf
    end

    return peaks
end

function _default_psf_widths(
    cfg::PAMConfig,
    kgrid::KGrid2D,
    frequencies::Union{Nothing, AbstractVector{<:Real}};
    characteristic_depth::Real=30e-3,
)
    aperture = something(cfg.receiver_aperture, kgrid.Ny * cfg.dz)
    depth = max(Float64(characteristic_depth), 2 * cfg.dx)

    freqs = isnothing(frequencies) || isempty(frequencies) ? nothing : Float64.(frequencies)
    if isnothing(freqs)
        # Fallback: assume one wavelength worth of structure
        lambda = cfg.c0 / 5e5
        lateral = lambda * depth / aperture
        axial = 2 * lambda * (depth / aperture)^2
        return max(axial, 2 * cfg.dx), max(lateral, 2 * cfg.dz)
    end

    f_max = maximum(freqs)
    f_min = minimum(freqs)
    lambda_min = cfg.c0 / f_max
    bw = f_max - f_min

    lateral = lambda_min * depth / aperture
    axial = if bw > 0
        cfg.c0 / (2 * bw)
    else
        2 * lambda_min * (depth / aperture)^2
    end
    return max(axial, 2 * cfg.dx), max(lateral, 2 * cfg.dz)
end

"""
    find_pam_peaks_clean(intensity, kgrid, cfg; n_peaks, frequencies=nothing,
                         psf_axial_fwhm=nothing, psf_lateral_fwhm=nothing,
                         loop_gain=0.1, max_iter=500, threshold_ratio=1e-2,
                         suppression_radius=nothing)

Iterative CLEAN (Högbom) peak detector for PAM intensity maps. Each iteration
finds the brightest residual pixel, adds `loop_gain * peak` to the accumulator,
and subtracts a scaled Gaussian PSF from the residual. The `n_peaks` brightest
maxima in the accumulator are returned. If `suppression_radius` is not given,
it defaults to the lateral PSF FWHM, which lets sources as close as one PSF
width apart be resolved distinctly.
"""
function find_pam_peaks_clean(
    intensity::AbstractMatrix{<:Real},
    kgrid::KGrid2D,
    cfg::PAMConfig;
    n_peaks::Integer,
    frequencies::Union{Nothing, AbstractVector{<:Real}}=nothing,
    psf_axial_fwhm::Union{Nothing, Real}=nothing,
    psf_lateral_fwhm::Union{Nothing, Real}=nothing,
    loop_gain::Real=0.1,
    max_iter::Integer=500,
    threshold_ratio::Real=1e-2,
    suppression_radius::Union{Nothing, Real}=nothing,
)
    0 < loop_gain <= 1 || error("loop_gain must lie in (0, 1].")
    n_peaks > 0 || error("n_peaks must be positive.")

    residual = copy(Float64.(intensity))
    row_start = receiver_row(cfg) + 1
    row_stop = size(residual, 1)
    row_start <= row_stop || error("No valid reconstruction rows remain.")
    residual[1:(row_start - 1), :] .= -Inf
    if row_stop < size(residual, 1)
        residual[(row_stop + 1):end, :] .= -Inf
    end

    ax_fwhm, lat_fwhm = if isnothing(psf_axial_fwhm) || isnothing(psf_lateral_fwhm)
        _default_psf_widths(cfg, kgrid, frequencies)
    else
        Float64(psf_axial_fwhm), Float64(psf_lateral_fwhm)
    end
    ax_fwhm = something(psf_axial_fwhm, ax_fwhm)
    lat_fwhm = something(psf_lateral_fwhm, lat_fwhm)

    σ_ax_cells = max(1.0, Float64(ax_fwhm) / (cfg.dx * 2.3548))
    σ_lat_cells = max(1.0, Float64(lat_fwhm) / (cfg.dz * 2.3548))
    half_ax = max(1, ceil(Int, 3 * σ_ax_cells))
    half_lat = max(1, ceil(Int, 3 * σ_lat_cells))

    finite_mask = isfinite.(residual)
    any(finite_mask) || return Tuple{Int, Int}[]
    peak_init = maximum(residual[finite_mask])
    peak_init > 0 || return Tuple{Int, Int}[]
    threshold = peak_init * Float64(threshold_ratio)

    accum = zeros(Float64, size(residual))
    nx, ny = size(residual)

    for _ in 1:Int(max_iter)
        idx = Tuple(argmax(residual))
        pv = residual[idx...]
        (!isfinite(pv) || pv < threshold) && break

        scale = Float64(loop_gain) * pv
        r0, c0 = idx
        accum[r0, c0] += scale

        r1 = max(1, r0 - half_ax)
        r2 = min(nx, r0 + half_ax)
        c1 = max(1, c0 - half_lat)
        c2 = min(ny, c0 + half_lat)
        @inbounds for r in r1:r2
            dr = (r - r0) / σ_ax_cells
            for c in c1:c2
                dc = (c - c0) / σ_lat_cells
                weight = exp(-0.5 * (dr^2 + dc^2))
                residual[r, c] -= scale * weight
            end
        end
    end

    sup_radius = isnothing(suppression_radius) ? Float64(lat_fwhm) : Float64(suppression_radius)
    return find_pam_peaks(accum, kgrid, cfg; n_peaks=n_peaks, suppression_radius=sup_radius)
end

function pam_truth_mask(
    sources::AbstractVector{<:EmissionSource2D},
    kgrid::KGrid2D,
    cfg::PAMConfig;
    radius::Real=cfg.success_tolerance,
)
    radius_m = Float64(radius)
    radius_m >= 0 || error("truth-mask radius must be non-negative.")

    mask = falses(kgrid.Nx, kgrid.Ny)
    depth = depth_coordinates(kgrid, cfg)
    lateral = kgrid.y_vec
    radius2 = radius_m^2
    row_radius = ceil(Int, radius_m / cfg.dx)
    col_radius = ceil(Int, radius_m / cfg.dz)

    for src in sources
        row0, col0 = source_grid_index(src, cfg, kgrid)
        row_start = max(receiver_row(cfg) + 1, row0 - row_radius)
        row_stop = min(kgrid.Nx, row0 + row_radius)
        col_start = max(1, col0 - col_radius)
        col_stop = min(kgrid.Ny, col0 + col_radius)
        @inbounds for row in row_start:row_stop
            dd = depth[row] - src.depth
            for col in col_start:col_stop
                dl = lateral[col] - src.lateral
                if dd^2 + dl^2 <= radius2
                    mask[row, col] = true
                end
            end
        end
    end

    return mask
end

function _mark_centerline_segment!(
    mask::BitMatrix,
    depth::AbstractVector{<:Real},
    lateral::AbstractVector{<:Real},
    cfg::PAMConfig,
    d0::Real,
    l0::Real,
    d1::Real,
    l1::Real,
    radius_m::Real,
)
    dd = Float64(d1) - Float64(d0)
    dl = Float64(l1) - Float64(l0)
    seg_len2 = dd^2 + dl^2
    radius2 = Float64(radius_m)^2
    row_pad = ceil(Int, Float64(radius_m) / cfg.dx) + 1
    col_pad = ceil(Int, Float64(radius_m) / cfg.dz) + 1
    row_min = clamp(searchsortedfirst(depth, min(Float64(d0), Float64(d1)) - Float64(radius_m)) - row_pad, receiver_row(cfg) + 1, length(depth))
    row_max = clamp(searchsortedlast(depth, max(Float64(d0), Float64(d1)) + Float64(radius_m)) + row_pad, receiver_row(cfg) + 1, length(depth))
    col_min = clamp(searchsortedfirst(lateral, min(Float64(l0), Float64(l1)) - Float64(radius_m)) - col_pad, 1, length(lateral))
    col_max = clamp(searchsortedlast(lateral, max(Float64(l0), Float64(l1)) + Float64(radius_m)) + col_pad, 1, length(lateral))

    @inbounds for row in row_min:row_max
        pd = Float64(depth[row])
        for col in col_min:col_max
            pl = Float64(lateral[col])
            t = seg_len2 <= eps(Float64) ? 0.0 : clamp(((pd - Float64(d0)) * dd + (pl - Float64(l0)) * dl) / seg_len2, 0.0, 1.0)
            nearest_d = Float64(d0) + t * dd
            nearest_l = Float64(l0) + t * dl
            if (pd - nearest_d)^2 + (pl - nearest_l)^2 <= radius2
                mask[row, col] = true
            end
        end
    end
    return mask
end

function pam_centerline_truth_mask(
    centerlines::AbstractVector,
    kgrid::KGrid2D,
    cfg::PAMConfig;
    radius::Real=cfg.success_tolerance,
)
    radius_m = Float64(radius)
    radius_m >= 0 || error("centerline truth-mask radius must be non-negative.")
    mask = falses(kgrid.Nx, kgrid.Ny)
    depth = depth_coordinates(kgrid, cfg)
    lateral = kgrid.y_vec

    for centerline in centerlines
        length(centerline) >= 2 || continue
        for idx in 1:(length(centerline) - 1)
            d0, l0 = centerline[idx]
            d1, l1 = centerline[idx + 1]
            _mark_centerline_segment!(mask, depth, lateral, cfg, d0, l0, d1, l1, radius_m)
        end
    end
    return mask
end

function threshold_pam_map(
    intensity::AbstractMatrix{<:Real},
    cfg::PAMConfig;
    threshold_ratio::Real=0.2,
)
    ratio = Float64(threshold_ratio)
    ratio > 0 || error("threshold_ratio must be positive.")
    work = Float64.(intensity)
    ref = maximum(work)
    ref > 0 || return falses(size(work))

    mask = work .>= (ratio * ref)
    mask[1:receiver_row(cfg), :] .= false
    return mask
end

function pam_intensity_metrics(
    intensity::AbstractMatrix{<:Real},
    kgrid::KGrid2D,
    cfg::PAMConfig;
    threshold_ratio::Real=0.2,
    reference_intensity::Union{Nothing, Real}=nothing,
)
    size(intensity) == (kgrid.Nx, kgrid.Ny) ||
        error("PAM intensity size $(size(intensity)) does not match kgrid size ($(kgrid.Nx), $(kgrid.Ny)).")
    ratio = Float64(threshold_ratio)
    ratio > 0 || error("threshold_ratio must be positive.")

    work = Float64.(intensity)
    peak = maximum(work)
    ref = isnothing(reference_intensity) ? peak : Float64(reference_intensity)
    ref = max(ref, eps(Float64))
    local_threshold = ratio * max(peak, eps(Float64))
    shared_threshold = ratio * ref

    active = work .>= local_threshold
    active[1:receiver_row(cfg), :] .= false
    shared_active = work .>= shared_threshold
    shared_active[1:receiver_row(cfg), :] .= false

    depth = depth_coordinates(kgrid, cfg)
    lateral = kgrid.y_vec
    active_idxs = findall(active)
    centroid_depth_mm = NaN
    centroid_lateral_mm = NaN
    if !isempty(active_idxs)
        weight_sum = sum(work[idx] for idx in active_idxs)
        if weight_sum > 0
            centroid_depth_mm = sum(work[idx] * depth[idx[1]] for idx in active_idxs) / weight_sum * 1e3
            centroid_lateral_mm = sum(work[idx] * lateral[idx[2]] for idx in active_idxs) / weight_sum * 1e3
        end
    end

    pixel_area_mm2 = cfg.dx * cfg.dz * 1e6
    max_idx = Tuple(argmax(work))
    return Dict{Symbol, Any}(
        :peak_intensity => peak,
        :relative_peak_intensity => peak / ref,
        :integrated_intensity_m2 => sum(work) * cfg.dx * cfg.dz,
        :threshold_ratio => ratio,
        :active_area_mm2 => count(active) * pixel_area_mm2,
        :shared_scale_active_area_mm2 => count(shared_active) * pixel_area_mm2,
        :centroid_depth_mm => centroid_depth_mm,
        :centroid_lateral_mm => centroid_lateral_mm,
        :peak_depth_mm => depth[max_idx[1]] * 1e3,
        :peak_lateral_mm => lateral[max_idx[2]] * 1e3,
    )
end

function _component_overlap_counts(mask::BitMatrix, reference::BitMatrix)
    size(mask) == size(reference) || error("Component masks must have the same size.")
    rows, cols = size(mask)
    visited = falses(rows, cols)
    total = 0
    overlapping = 0

    for row in 1:rows, col in 1:cols
        (mask[row, col] && !visited[row, col]) || continue
        total += 1
        touches_reference = false
        queue = [(row, col)]
        visited[row, col] = true

        while !isempty(queue)
            current = popfirst!(queue)
            i, j = current
            touches_reference |= reference[i, j]
            for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
                ii = i + di
                jj = j + dj
                if 1 <= ii <= rows && 1 <= jj <= cols && mask[ii, jj] && !visited[ii, jj]
                    visited[ii, jj] = true
                    push!(queue, (ii, jj))
                end
            end
        end

        overlapping += touches_reference ? 1 : 0
    end

    return total, overlapping, total - overlapping
end

_safe_fraction(num::Real, den::Real) = den > 0 ? Float64(num) / Float64(den) : 0.0

function analyse_pam_detection_2d(
    intensity::AbstractMatrix{<:Real},
    kgrid::KGrid2D,
    cfg::PAMConfig,
    sources::AbstractVector{<:EmissionSource2D};
    truth_radius::Real=cfg.success_tolerance,
    threshold_ratio::Real=0.2,
    truth_mask::Union{Nothing, AbstractMatrix{Bool}}=nothing,
)
    isempty(sources) && error("At least one emission source is required for PAM detection analysis.")
    truth = if isnothing(truth_mask)
        pam_truth_mask(sources, kgrid, cfg; radius=truth_radius)
    else
        size(truth_mask) == (kgrid.Nx, kgrid.Ny) ||
            error("truth_mask size $(size(truth_mask)) does not match kgrid size ($(kgrid.Nx), $(kgrid.Ny)).")
        BitMatrix(truth_mask)
    end
    predicted = threshold_pam_map(intensity, cfg; threshold_ratio=threshold_ratio)

    tp = count(predicted .& truth)
    fp = count(predicted .& (.!truth))
    fn = count((.!predicted) .& truth)
    tn = length(predicted) - tp - fp - fn

    precision = _safe_fraction(tp, tp + fp)
    recall = _safe_fraction(tp, tp + fn)
    f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0
    dice = _safe_fraction(2 * tp, 2 * tp + fp + fn)
    jaccard = _safe_fraction(tp, tp + fp + fn)

    prediction_components, matched_prediction_components, spurious_prediction_components =
        _component_overlap_counts(predicted, truth)
    truth_components, recovered_truth_components, missed_truth_components =
        _component_overlap_counts(truth, predicted)

    pixel_area_mm2 = cfg.dx * cfg.dz * 1e6
    max_idx = Tuple(argmax(Float64.(intensity)))
    depth = depth_coordinates(kgrid, cfg)
    lateral = kgrid.y_vec
    max_intensity = maximum(Float64.(intensity))

    return Dict{Symbol, Any}(
        :truth_mm => [(src.depth * 1e3, src.lateral * 1e3) for src in sources],
        :num_truth_sources => length(sources),
        :truth_radius_mm => Float64(truth_radius) * 1e3,
        :truth_mask_mode => isnothing(truth_mask) ? :source_disks : :provided,
        :threshold_ratio => Float64(threshold_ratio),
        :threshold_db => 10 * log10(Float64(threshold_ratio)),
        :true_positive_pixels => tp,
        :false_positive_pixels => fp,
        :false_negative_pixels => fn,
        :true_negative_pixels => tn,
        :precision => precision,
        :recall => recall,
        :f1 => f1,
        :dice => dice,
        :jaccard => jaccard,
        :truth_area_mm2 => count(truth) * pixel_area_mm2,
        :predicted_area_mm2 => count(predicted) * pixel_area_mm2,
        :overlap_area_mm2 => tp * pixel_area_mm2,
        :false_positive_area_mm2 => fp * pixel_area_mm2,
        :false_negative_area_mm2 => fn * pixel_area_mm2,
        :prediction_components => prediction_components,
        :matched_prediction_components => matched_prediction_components,
        :spurious_prediction_components => spurious_prediction_components,
        :truth_components => truth_components,
        :recovered_truth_components => recovered_truth_components,
        :missed_truth_components => missed_truth_components,
        :peak_mm => (depth[max_idx[1]] * 1e3, lateral[max_idx[2]] * 1e3),
        :peak_intensity => Float64(intensity[max_idx...]),
        :max_intensity => max_intensity,
    )
end

function analyse_pam_2d(
    intensity::AbstractMatrix{<:Real},
    kgrid::KGrid2D,
    cfg::PAMConfig,
    sources::AbstractVector{<:EmissionSource2D};
    n_peaks::Union{Nothing, Integer}=nothing,
    success_tolerance::Real=cfg.success_tolerance,
    suppression_radius::Real=cfg.peak_suppression_radius,
    peak_method::Symbol=:argmax,
    frequencies::Union{Nothing, AbstractVector{<:Real}}=nothing,
    clean_loop_gain::Real=0.1,
    clean_max_iter::Integer=500,
    clean_threshold_ratio::Real=1e-2,
    clean_psf_axial_fwhm::Union{Nothing, Real}=nothing,
    clean_psf_lateral_fwhm::Union{Nothing, Real}=nothing,
)
    n_truth = length(sources)
    n_truth > 0 || error("At least one emission source is required for PAM analysis.")
    n_find = isnothing(n_peaks) ? n_truth : Int(n_peaks)
    peaks = if peak_method == :clean
        find_pam_peaks_clean(
            intensity, kgrid, cfg;
            n_peaks=n_find,
            frequencies=frequencies,
            psf_axial_fwhm=clean_psf_axial_fwhm,
            psf_lateral_fwhm=clean_psf_lateral_fwhm,
            loop_gain=clean_loop_gain,
            max_iter=clean_max_iter,
            threshold_ratio=clean_threshold_ratio,
            suppression_radius=nothing,
        )
    elseif peak_method == :argmax
        find_pam_peaks(intensity, kgrid, cfg; n_peaks=n_find, suppression_radius=suppression_radius)
    else
        error("Unknown peak_method: $peak_method (expected :argmax or :clean).")
    end
    length(peaks) == n_truth || error("Expected to recover $n_truth peaks, found $(length(peaks)).")

    depth = depth_coordinates(kgrid, cfg)
    lateral = kgrid.y_vec
    truth_mm = [(src.depth * 1e3, src.lateral * 1e3) for src in sources]
    pred_mm = [(depth[idx[1]] * 1e3, lateral[idx[2]] * 1e3) for idx in peaks]

    cost = Matrix{Float64}(undef, n_truth, n_truth)
    for i in 1:n_truth, j in 1:n_truth
        d_ax = truth_mm[i][1] - pred_mm[j][1]
        d_lat = truth_mm[i][2] - pred_mm[j][2]
        cost[i, j] = hypot(d_ax, d_lat)
    end
    assignment, _ = _best_assignment(cost)

    matched_pred_mm = [pred_mm[assignment[i]] for i in 1:n_truth]
    matched_indices = [peaks[assignment[i]] for i in 1:n_truth]
    axial_errors_mm = [truth_mm[i][1] - matched_pred_mm[i][1] for i in 1:n_truth]
    lateral_errors_mm = [truth_mm[i][2] - matched_pred_mm[i][2] for i in 1:n_truth]
    radial_errors_mm = [hypot(axial_errors_mm[i], lateral_errors_mm[i]) for i in 1:n_truth]

    raw_peak_intensities = [Float64(intensity[idx...]) for idx in matched_indices]
    max_intensity = max(maximum(Float64.(intensity)), eps(Float64))
    norm_peak_intensities = raw_peak_intensities ./ max_intensity

    axial_fwhm_mm = Float64[]
    lateral_fwhm_mm = Float64[]
    for idx in matched_indices
        axial_fwhm, lateral_fwhm = _peak_fwhm_mm(intensity, kgrid, cfg, idx)
        push!(axial_fwhm_mm, axial_fwhm)
        push!(lateral_fwhm_mm, lateral_fwhm)
    end

    tol_mm = Float64(success_tolerance) * 1e3
    successes = radial_errors_mm .<= tol_mm

    num_success = count(identity, successes)
    return Dict{Symbol, Any}(
        :truth_mm => truth_mm,
        :predicted_mm => matched_pred_mm,
        :peak_indices => matched_indices,
        :axial_errors_mm => axial_errors_mm,
        :lateral_errors_mm => lateral_errors_mm,
        :radial_errors_mm => radial_errors_mm,
        :mean_axial_error_mm => mean(abs.(axial_errors_mm)),
        :mean_lateral_error_mm => mean(abs.(lateral_errors_mm)),
        :mean_radial_error_mm => mean(radial_errors_mm),
        :max_radial_error_mm => maximum(radial_errors_mm),
        :success_tolerance_mm => tol_mm,
        :success_rate => num_success / n_truth,
        :num_success => num_success,
        :raw_peak_intensities => raw_peak_intensities,
        :norm_peak_intensities => norm_peak_intensities,
        :mean_norm_peak_intensity => mean(norm_peak_intensities),
        :axial_fwhm_mm => axial_fwhm_mm,
        :lateral_fwhm_mm => lateral_fwhm_mm,
        :mean_axial_fwhm_mm => mean(axial_fwhm_mm),
        :mean_lateral_fwhm_mm => mean(lateral_fwhm_mm),
    )
end

function reconstruct_pam(
    rf::AbstractMatrix{<:Real},
    c::AbstractMatrix{<:Real},
    cfg::PAMConfig;
    frequencies::Union{Nothing, AbstractVector{<:Real}}=nothing,
    bandwidth::Real=0.0,
    corrected::Bool=true,
    reference_sound_speed::Union{Nothing, Real}=nothing,
    axial_step::Union{Nothing, Real}=nothing,
)
    nx, ny = size(c)
    size(rf, 1) == ny || error("RF data must have size (Ny, Nt); expected Ny=$ny, got $(size(rf, 1)).")
    nt = size(rf, 2)
    kgrid = KGrid2D(nx, ny, cfg.dx, cfg.dz; dt=cfg.dt, Nt=nt)
    rr = receiver_row(cfg)
    rr <= nx || error("Receiver row lies outside the computational grid.")

    selected_freqs, selected_bins = _select_frequency_bins(rf, cfg.dt, frequencies; bandwidth=bandwidth)
    rf_fft = fft(Float64.(rf), 2)
    padded_ny = cfg.zero_pad_factor > 1 ? cfg.zero_pad_factor * ny : ny
    _, crop_range = _zero_pad_receiver_rf(rf, padded_ny)
    c_padded, _ = _edge_pad_lateral(c, padded_ny)
    c0 = isnothing(reference_sound_speed) ? mean(c_padded) : Float64(reference_sound_speed)
    c0 > 0 || error("reference_sound_speed must be positive.")
    target_axial_step = isnothing(axial_step) ? cfg.dx : Float64(axial_step)
    0 < target_axial_step <= cfg.dx || error("axial_step must lie in (0, cfg.dx].")
    axial_substeps = _pam_axial_substeps(cfg.dx, target_axial_step)
    effective_axial_step = cfg.dx / axial_substeps
    intensity_padded = zeros(Float64, nx, padded_ny)
    row_stop = nx
    row_stop > rr || error("No valid reconstruction rows remain below the receiver row.")

    for (freq, bin) in zip(selected_freqs, selected_bins)
        p0 = rf_fft[:, bin]
        p0_padded, _ = _zero_pad_receiver_rf(reshape(p0, ny, 1), padded_ny)
        p0_vec = vec(p0_padded[:, 1])

        k0 = 2π * freq / c0
        k = _fft_wavenumbers(padded_ny, cfg.dz)
        kz = sqrt.(complex.(k0^2 .- k .^ 2, 0.0))
        propagator = exp.(1im .* kz .* effective_axial_step)

        real_inds = findall(real.(kz ./ k0) .> 0.0)
        propagating = falses(padded_ny)
        propagating[real_inds] .= true
        evanescent_inds = findall(x -> !x, propagating)
        weighting = zeros(Float64, padded_ny)
        weighting[real_inds] .= _tukey_window(length(real_inds), cfg.tukey_ratio)

        current = _fftshift(fft(p0_vec))
        current .*= weighting

        mu = (c0 ./ c_padded) .^ 2
        lambda = (k0^2) .* (1 .- mu)

        correction = zeros(ComplexF64, padded_ny)
        for idx in real_inds
            abs(kz[idx]) > sqrt(eps(Float64)) || continue
            correction[idx] = propagator[idx] * effective_axial_step / (2im * kz[idx])
        end

        for row in (rr + 1):row_stop
            for _ in 1:axial_substeps
                if corrected
                    p_space = ifft(_ifftshift(current))
                    conv_term = _fftshift(fft(lambda[row, :] .* p_space))
                    next = current .* propagator
                    next .+= correction .* conv_term
                else
                    next = current .* propagator
                end
                next[evanescent_inds] .= 0.0
                current = next
            end
            # Taper once per reconstruction row to suppress long-range numerical
            # growth without making damping depend on substep count.
            current .*= weighting

            p_row = ifft(_ifftshift(current))
            intensity_padded[row, :] .+= abs2.(p_row)
        end
    end

    intensity = intensity_padded[:, crop_range]
    info = Dict{Symbol, Any}(
        :frequencies => selected_freqs,
        :frequency_bins => selected_bins,
        :bandwidth => Float64(bandwidth),
        :corrected => corrected,
        :receiver_row => rr,
        :crop_range => crop_range,
        :reference_sound_speed => c0,
        :axial_step => effective_axial_step,
        :axial_substeps_per_cell => axial_substeps,
    )
    return intensity, kgrid, info
end

function _default_recon_frequencies(sources::AbstractVector{<:EmissionSource2D})
    all_freqs = Float64[]
    for src in sources
        append!(all_freqs, _emission_frequencies(src))
    end
    return sort(unique(all_freqs))
end

function _pam_reference_sound_speed(
    c::AbstractMatrix{<:Real},
    cfg::PAMConfig,
    sources::AbstractVector{<:EmissionSource2D};
    margin::Real=10e-3,
)
    isempty(sources) && return mean(Float64.(c))
    row_start = clamp(receiver_row(cfg), 1, size(c, 1))
    deepest_source_depth = maximum(src.depth for src in sources)
    row_stop = row_start + ceil(Int, (deepest_source_depth + Float64(margin)) / cfg.dx)
    row_stop = clamp(row_stop, row_start, size(c, 1))
    return mean(Float64.(view(c, row_start:row_stop, :)))
end

function run_pam_case(
    c::AbstractMatrix{<:Real},
    rho::AbstractMatrix{<:Real},
    sources::AbstractVector{<:EmissionSource2D},
    cfg::PAMConfig;
    frequencies::Union{Nothing, AbstractVector{<:Real}}=nothing,
    bandwidth::Real=0.0,
    use_gpu::Bool=false,
    analysis_mode::Symbol=:localization,
    peak_method::Symbol=:argmax,
    clean_loop_gain::Real=0.1,
    clean_max_iter::Integer=500,
    clean_threshold_ratio::Real=1e-2,
    clean_psf_axial_fwhm::Union{Nothing, Real}=nothing,
    clean_psf_lateral_fwhm::Union{Nothing, Real}=nothing,
    detection_truth_radius::Real=cfg.success_tolerance,
    detection_threshold_ratio::Real=0.2,
    detection_truth_mask::Union{Nothing, AbstractMatrix{Bool}}=nothing,
    reconstruction_axial_step::Union{Nothing, Real}=50e-6,
)
    recon_freqs = isnothing(frequencies) ? _default_recon_frequencies(sources) : Float64.(frequencies)
    rf, kgrid, sim_info = simulate_point_sources(c, rho, sources, cfg; use_gpu=use_gpu)
    results = reconstruct_pam_case(
        rf,
        c,
        sources,
        cfg;
        simulation_info=sim_info,
        frequencies=recon_freqs,
        bandwidth=bandwidth,
        analysis_mode=analysis_mode,
        peak_method=peak_method,
        clean_loop_gain=clean_loop_gain,
        clean_max_iter=clean_max_iter,
        clean_threshold_ratio=clean_threshold_ratio,
        clean_psf_axial_fwhm=clean_psf_axial_fwhm,
        clean_psf_lateral_fwhm=clean_psf_lateral_fwhm,
        detection_truth_radius=detection_truth_radius,
        detection_threshold_ratio=detection_threshold_ratio,
        detection_truth_mask=detection_truth_mask,
        reconstruction_axial_step=reconstruction_axial_step,
    )
    results[:kgrid] = kgrid
    return results
end

function reconstruct_pam_case(
    rf::AbstractMatrix{<:Real},
    c::AbstractMatrix{<:Real},
    sources::AbstractVector{<:EmissionSource2D},
    cfg::PAMConfig;
    simulation_info::AbstractDict=Dict{Symbol, Any}(
        :receiver_row => receiver_row(cfg),
        :receiver_cols => receiver_col_range(cfg),
        :source_indices => Tuple{Int, Int}[],
    ),
    frequencies::Union{Nothing, AbstractVector{<:Real}}=nothing,
    bandwidth::Real=0.0,
    analysis_mode::Symbol=:localization,
    peak_method::Symbol=:argmax,
    clean_loop_gain::Real=0.1,
    clean_max_iter::Integer=500,
    clean_threshold_ratio::Real=1e-2,
    clean_psf_axial_fwhm::Union{Nothing, Real}=nothing,
    clean_psf_lateral_fwhm::Union{Nothing, Real}=nothing,
    detection_truth_radius::Real=cfg.success_tolerance,
    detection_threshold_ratio::Real=0.2,
    detection_truth_mask::Union{Nothing, AbstractMatrix{Bool}}=nothing,
    reconstruction_axial_step::Union{Nothing, Real}=50e-6,
)
    size(c) == (pam_Nx(cfg), pam_Ny(cfg)) ||
        error("Sound-speed map size $(size(c)) does not match PAMConfig size ($(pam_Nx(cfg)), $(pam_Ny(cfg))).")
    size(rf, 1) == pam_Ny(cfg) ||
        error("RF data must have $(pam_Ny(cfg)) receiver rows; got $(size(rf, 1)).")

    recon_freqs = isnothing(frequencies) ? _default_recon_frequencies(sources) : Float64.(frequencies)
    reference_sound_speed = _pam_reference_sound_speed(c, cfg, sources)
    recon_kwargs = (
        frequencies=recon_freqs,
        bandwidth=bandwidth,
        reference_sound_speed=reference_sound_speed,
        axial_step=reconstruction_axial_step,
    )
    pam_geo, kgrid, geo_info = reconstruct_pam(rf, c, cfg; recon_kwargs..., corrected=false)
    pam_hasa, _, hasa_info = reconstruct_pam(rf, c, cfg; recon_kwargs..., corrected=true)

    stats_geo, stats_hasa = if analysis_mode == :localization
        analyse_kwargs = (
            peak_method=peak_method,
            frequencies=recon_freqs,
            clean_loop_gain=clean_loop_gain,
            clean_max_iter=clean_max_iter,
            clean_threshold_ratio=clean_threshold_ratio,
            clean_psf_axial_fwhm=clean_psf_axial_fwhm,
            clean_psf_lateral_fwhm=clean_psf_lateral_fwhm,
        )
        (
            analyse_pam_2d(pam_geo, kgrid, cfg, sources; analyse_kwargs...),
            analyse_pam_2d(pam_hasa, kgrid, cfg, sources; analyse_kwargs...),
        )
    elseif analysis_mode == :detection
        analyse_kwargs = (
            truth_radius=detection_truth_radius,
            threshold_ratio=detection_threshold_ratio,
            truth_mask=detection_truth_mask,
        )
        (
            analyse_pam_detection_2d(pam_geo, kgrid, cfg, sources; analyse_kwargs...),
            analyse_pam_detection_2d(pam_hasa, kgrid, cfg, sources; analyse_kwargs...),
        )
    else
        error("Unknown analysis_mode: $analysis_mode (expected :localization or :detection).")
    end

    return Dict{Symbol, Any}(
        :rf => Float64.(rf),
        :kgrid => kgrid,
        :simulation => Dict{Symbol, Any}(Symbol(key) => value for (key, value) in simulation_info),
        :pam_geo => pam_geo,
        :pam_hasa => pam_hasa,
        :geo_info => geo_info,
        :hasa_info => hasa_info,
        :stats_geo => stats_geo,
        :stats_hasa => stats_hasa,
        :reconstruction_frequencies => recon_freqs,
        :analysis_mode => analysis_mode,
    )
end

function _pam_mm_key(depth_mm::Real, lateral_mm::Real)
    return (round(Float64(depth_mm); digits=6), round(Float64(lateral_mm); digits=6))
end

function _pam_mm_key(src::PointSource2D)
    return _pam_mm_key(src.depth * 1e3, src.lateral * 1e3)
end

function _resolve_pam_sweep_targets(
    preset::Union{Symbol, AbstractString};
    axial_targets_mm::Union{Nothing, AbstractVector{<:Real}}=nothing,
    lateral_targets_mm::Union{Nothing, AbstractVector{<:Real}}=nothing,
)
    explicit_targets = !isnothing(axial_targets_mm) || !isnothing(lateral_targets_mm)
    if explicit_targets
        isnothing(axial_targets_mm) && error("Custom PAM sweep requires explicit axial target positions.")
        isnothing(lateral_targets_mm) && error("Custom PAM sweep requires explicit lateral target positions.")
        axial = sort(unique(Float64.(axial_targets_mm)))
        lateral = sort(unique(Float64.(lateral_targets_mm)))
        isempty(axial) && error("At least one axial target is required for a PAM sweep.")
        isempty(lateral) && error("At least one lateral target is required for a PAM sweep.")
        return :custom, axial, lateral
    end

    mode = preset isa Symbol ? preset : Symbol(lowercase(strip(preset)))
    if mode == :paper
        return :paper, [30.0, 40.0, 50.0, 60.0, 70.0, 80.0], [-20.0, -10.0, 0.0, 10.0, 20.0]
    elseif mode == :quick
        return :quick, [40.0, 60.0, 80.0], [-10.0, 0.0, 10.0]
    elseif mode == :custom
        error("Custom PAM sweep requires both --axial-targets-mm and --lateral-targets-mm.")
    end
    error("Unknown PAM sweep preset: $preset")
end

function _default_pam_sweep_examples(targets::AbstractVector{PointSource2D})
    isempty(targets) && error("At least one target is required to choose PAM sweep examples.")

    depth_values = sort(unique(Float64[src.depth * 1e3 for src in targets]))
    num_examples = min(3, length(depth_values))
    selected_depth_indices = unique(round.(Int, collect(range(1, length(depth_values); length=num_examples))))

    examples = Tuple{Float64, Float64}[]
    for depth_idx in selected_depth_indices
        depth_mm = depth_values[depth_idx]
        candidates = [src for src in targets if isapprox(src.depth * 1e3, depth_mm; atol=1e-6)]
        isempty(candidates) && continue
        best = candidates[argmin(abs.([src.lateral for src in candidates]))]
        push!(examples, _pam_mm_key(best))
    end
    return examples
end

function _normalize_pam_sweep_examples(
    targets::AbstractVector{PointSource2D},
    example_targets_mm::Union{Nothing, AbstractVector{<:Tuple{<:Real, <:Real}}},
)
    if isnothing(example_targets_mm)
        return _default_pam_sweep_examples(targets)
    end

    1 <= length(example_targets_mm) <= 3 || error("Provide between 1 and 3 PAM sweep example targets.")
    available = Set(_pam_mm_key(src) for src in targets)
    examples = Tuple{Float64, Float64}[]
    for target in example_targets_mm
        key = _pam_mm_key(target[1], target[2])
        key in available || error("Example target $(target[1]) mm, $(target[2]) mm is not part of the PAM sweep.")
        push!(examples, key)
    end
    return sort(unique(examples))
end

function _pam_skull_cavity_start_rows(
    c::AbstractMatrix{<:Real};
    c_water::Real=1500.0,
    tol::Real=5.0,
    min_thick_rows::Integer=2,
)
    skull_mask = skull_mask_from_c_columnwise(
        c;
        c_water=c_water,
        tol=tol,
        min_thick_rows=min_thick_rows,
        dilate_rows=1,
        close_iters=1,
        mask_outside=false,
    )

    ny = size(c, 2)
    start_rows = zeros(Int, ny)
    has_skull = falses(ny)
    for col in 1:ny
        rows = findall(skull_mask[:, col])
        isempty(rows) && continue
        has_skull[col] = true
        start_rows[col] = last(rows) + 1
    end
    return start_rows, has_skull
end

function _filter_pam_targets_in_skull_cavity(
    c::AbstractMatrix{<:Real},
    cfg::PAMConfig,
    targets::AbstractVector{PointSource2D};
    min_margin::Real=1e-3,
    c_water::Real=cfg.c0,
    tol::Real=5.0,
    min_thick_rows::Integer=2,
)
    kgrid = pam_grid(cfg)
    cavity_start_rows, has_skull = _pam_skull_cavity_start_rows(
        c;
        c_water=c_water,
        tol=tol,
        min_thick_rows=min_thick_rows,
    )
    margin_rows = max(0, ceil(Int, Float64(min_margin) / cfg.dx))

    valid_targets = PointSource2D[]
    dropped_targets = Dict{Symbol, Any}[]

    for src in targets
        row, col = source_grid_index(src, cfg, kgrid)
        truth_mm = (src.depth * 1e3, src.lateral * 1e3)
        if !has_skull[col]
            push!(dropped_targets, Dict{Symbol, Any}(
                :truth_mm => truth_mm,
                :row => row,
                :col => col,
                :reason => :no_skull_above,
            ))
            continue
        end

        required_row = cavity_start_rows[col] + margin_rows
        if row < required_row
            push!(dropped_targets, Dict{Symbol, Any}(
                :truth_mm => truth_mm,
                :row => row,
                :col => col,
                :required_row => required_row,
                :reason => :too_shallow_for_cavity,
            ))
            continue
        end

        if abs(Float64(c[row, col]) - Float64(c_water)) > Float64(tol)
            push!(dropped_targets, Dict{Symbol, Any}(
                :truth_mm => truth_mm,
                :row => row,
                :col => col,
                :reason => :non_fluid_target_cell,
            ))
            continue
        end

        push!(valid_targets, src)
    end

    return valid_targets, dropped_targets, cavity_start_rows
end

function run_pam_sweep(
    c::AbstractMatrix{<:Real},
    rho::AbstractMatrix{<:Real},
    targets::AbstractVector{PointSource2D},
    cfg::PAMConfig;
    frequencies::Union{Nothing, AbstractVector{<:Real}}=nothing,
    example_targets_mm::Union{Nothing, AbstractVector{<:Tuple{<:Real, <:Real}}}=nothing,
    use_gpu::Bool=false,
    runner::Function=run_pam_case,
    case_callback::Union{Nothing, Function}=nothing,
)
    isempty(targets) && error("At least one PAM sweep target is required.")

    sorted_targets = sort(collect(targets); by=src -> _pam_mm_key(src))
    axial_targets_mm = sort(unique(Float64[src.depth * 1e3 for src in sorted_targets]))
    lateral_targets_mm = sort(unique(Float64[src.lateral * 1e3 for src in sorted_targets]))
    axial_index = Dict(_pam_mm_key(depth_mm, 0.0)[1] => idx for (idx, depth_mm) in pairs(axial_targets_mm))
    lateral_index = Dict(_pam_mm_key(0.0, lateral_mm)[2] => idx for (idx, lateral_mm) in pairs(lateral_targets_mm))

    geo_error_mm = fill(NaN, length(axial_targets_mm), length(lateral_targets_mm))
    hasa_error_mm = similar(geo_error_mm)
    geo_peak_intensity = similar(geo_error_mm)
    hasa_peak_intensity = similar(geo_error_mm)

    example_keys = Set(_normalize_pam_sweep_examples(sorted_targets, example_targets_mm))
    cases = Dict{Symbol, Any}[]
    example_cases = Dict{Symbol, Any}[]

    for src in sorted_targets
        results = runner(
            c,
            rho,
            PointSource2D[src],
            cfg;
            frequencies=frequencies,
            use_gpu=use_gpu,
        )
        stats_geo = results[:stats_geo]
        stats_hasa = results[:stats_hasa]

        target_key = _pam_mm_key(src)
        row = axial_index[target_key[1]]
        col = lateral_index[target_key[2]]
        geo_error_mm[row, col] = Float64(stats_geo[:mean_radial_error_mm])
        hasa_error_mm[row, col] = Float64(stats_hasa[:mean_radial_error_mm])
        geo_peak_intensity[row, col] = Float64(stats_geo[:mean_norm_peak_intensity])
        hasa_peak_intensity[row, col] = Float64(stats_hasa[:mean_norm_peak_intensity])

        case_result = Dict{Symbol, Any}(
            :source => src,
            :truth_mm => (src.depth * 1e3, src.lateral * 1e3),
            :stats_geo => stats_geo,
            :stats_hasa => stats_hasa,
            :geo_predicted_mm => only(stats_geo[:predicted_mm]),
            :hasa_predicted_mm => only(stats_hasa[:predicted_mm]),
            :reconstruction_frequencies => results[:reconstruction_frequencies],
            :simulation => results[:simulation],
        )
        push!(cases, case_result)

        if !isnothing(case_callback)
            case_callback(case_result, results)
        end

        if target_key in example_keys
            example_result = copy(case_result)
            example_result[:rf] = results[:rf]
            example_result[:pam_geo] = results[:pam_geo]
            example_result[:pam_hasa] = results[:pam_hasa]
            example_result[:kgrid] = results[:kgrid]
            push!(example_cases, example_result)
        end
    end

    sort!(example_cases; by=case -> _pam_mm_key(case[:truth_mm]...))
    return Dict{Symbol, Any}(
        :cases => cases,
        :axial_targets_mm => axial_targets_mm,
        :lateral_targets_mm => lateral_targets_mm,
        :geo_error_mm => geo_error_mm,
        :hasa_error_mm => hasa_error_mm,
        :geo_peak_intensity => geo_peak_intensity,
        :hasa_peak_intensity => hasa_peak_intensity,
        :example_cases => example_cases,
        :example_targets_mm => [case[:truth_mm] for case in example_cases],
    )
end
