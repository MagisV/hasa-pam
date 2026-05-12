# Source parsing and source metadata helpers for PAM runner scripts.

"""
    point_pairs_m(points)

Convert 2D point tuples in meters to JSON-friendly nested arrays.
"""
point_pairs_m(points) = [[point[1], point[2]] for point in points]

"""
    centerlines_m(centerlines)

Convert 2D centerline tuples in meters to JSON-friendly nested arrays.
"""
centerlines_m(centerlines) = [point_pairs_m(line) for line in centerlines]

"""
    expand_source_values(values, n, default)

Expand an empty, scalar, or per-source CLI value list to length `n`.
"""
function expand_source_values(values::Vector{Float64}, n::Int, default::Float64)
    isempty(values) && return fill(default, n)
    length(values) == 1 && return fill(values[1], n)
    length(values) == n && return values
    error("Per-source parameter list must have length 1 or match the number of sources ($n).")
end

"""
    parse_coordinate_pairs_mm(spec, option_name)

Parse comma-separated `depth:lateral` coordinates in millimeters into meters.
"""
function parse_coordinate_pairs_mm(spec::AbstractString, option_name::AbstractString)
    coord_tokens = [strip(token) for token in split(spec, ",") if !isempty(strip(token))]
    1 <= length(coord_tokens) <= 20 || error("Provide between 1 and 20 coordinates via --$option_name=depth:lateral,...")
    pairs = Tuple{Float64, Float64}[]
    for token in coord_tokens
        parts = split(token, ":"; limit=2)
        length(parts) == 2 || error("Each coordinate must be depth_mm:lateral_mm, got: $token")
        push!(pairs, (parse(Float64, strip(parts[1])) * 1e-3, parse(Float64, strip(parts[2])) * 1e-3))
    end
    return pairs
end

"""
    parse_coordinate_triples_mm(spec, option_name)

Parse comma-separated `depth:y:z` coordinates in millimeters into meters.
"""
function parse_coordinate_triples_mm(spec::AbstractString, option_name::AbstractString)
    coord_tokens = [strip(token) for token in split(spec, ",") if !isempty(strip(token))]
    1 <= length(coord_tokens) <= 20 || error("Provide between 1 and 20 coordinates via --$option_name=depth:y:z,...")
    triples = NTuple{3, Float64}[]
    for token in coord_tokens
        parts = split(token, ":"; limit=3)
        length(parts) == 3 || error("3D coordinates must be depth:y:z in mm, got: $token")
        push!(triples, (
            parse(Float64, strip(parts[1])) * 1e-3,
            parse(Float64, strip(parts[2])) * 1e-3,
            parse(Float64, strip(parts[3])) * 1e-3,
        ))
    end
    return triples
end

"""
    parse_point_sources(opts)

Parse 2D point-source CLI options into sources and emission metadata.
"""
function parse_point_sources(opts)
    coordinates = parse_coordinate_pairs_mm(opts["sources-mm"], "sources-mm")
    n_sources = length(coordinates)
    frequencies_mhz = expand_source_values(
        parse_float_list(opts["source-frequencies-mhz"]),
        n_sources,
        parse(Float64, opts["frequency-mhz"]),
    )
    amplitudes_pa = expand_source_values(
        parse_float_list(opts["source-amplitudes-pa"]),
        n_sources,
        parse(Float64, opts["amplitude-pa"]),
    )
    phases_deg = expand_source_values(parse_float_list(opts["phases-deg"]), n_sources, 0.0)
    delays_us = expand_source_values(parse_float_list(opts["delays-us"]), n_sources, 0.0)
    num_cycles = parse(Int, opts["num-cycles"])

    phase_mode = lowercase(strip(opts["phase-mode"]))
    phase_mode in ("coherent", "random", "jittered") ||
        error("Point --phase-mode must be coherent, random, or jittered, got: $phase_mode")
    rng = Random.MersenneTwister(parse(Int, opts["random-seed"]))
    if phase_mode == "random"
        phases_deg = rand(rng, n_sources) .* 360.0
    elseif phase_mode == "jittered"
        phases_deg = phases_deg .+ randn(rng, n_sources) .* (parse(Float64, opts["phase-jitter-rad"]) * 180 / pi)
    end

    sources = EmissionSource2D[]
    for (idx, (depth_m, lateral_m)) in pairs(coordinates)
        push!(sources, PointSource2D(
            depth=depth_m,
            lateral=lateral_m,
            frequency=frequencies_mhz[idx] * 1e6,
            amplitude=amplitudes_pa[idx],
            phase=phases_deg[idx] * pi / 180,
            delay=delays_us[idx] * 1e-6,
            num_cycles=num_cycles,
        ))
    end
    return sources, Dict{String, Any}(
        "source_model" => "point",
        "coordinates_m" => [collect(coord) for coord in coordinates],
        "n_coordinate_sources" => n_sources,
        "n_emission_sources" => length(sources),
        "physical_source_count" => length(sources),
        "emission_event_count" => length(sources),
        "activity_model" => Dict("activity_mode" => "point_tone_burst"),
        "phase_mode" => phase_mode,
        "frequencies_hz" => frequencies_mhz .* 1e6,
        "amplitudes_pa" => amplitudes_pa,
        "phases_rad" => phases_deg .* pi ./ 180,
        "delays_s" => delays_us .* 1e-6,
        "num_cycles" => num_cycles,
        "random_seed" => parse(Int, opts["random-seed"]),
    )
end

"""
    parse_point_sources_3d(opts)

Parse 3D point-source CLI options into sources and emission metadata.
"""
function parse_point_sources_3d(opts)
    coordinates = parse_coordinate_triples_mm(opts["sources-mm"], "sources-mm")
    n_sources = length(coordinates)
    frequencies_mhz = expand_source_values(
        parse_float_list(opts["source-frequencies-mhz"]),
        n_sources,
        parse(Float64, opts["frequency-mhz"]),
    )
    amplitudes_pa = expand_source_values(
        parse_float_list(opts["source-amplitudes-pa"]),
        n_sources,
        parse(Float64, opts["amplitude-pa"]),
    )
    phases_deg = expand_source_values(parse_float_list(opts["phases-deg"]), n_sources, 0.0)
    delays_us = expand_source_values(parse_float_list(opts["delays-us"]), n_sources, 0.0)
    num_cycles = parse(Int, opts["num-cycles"])

    phase_mode = lowercase(strip(opts["phase-mode"]))
    phase_mode in ("coherent", "random", "jittered") ||
        error("Point --phase-mode must be coherent, random, or jittered, got: $phase_mode")
    rng = Random.MersenneTwister(parse(Int, opts["random-seed"]))
    if phase_mode == "random"
        phases_deg = rand(rng, n_sources) .* 360.0
    elseif phase_mode == "jittered"
        phases_deg = phases_deg .+ randn(rng, n_sources) .* (parse(Float64, opts["phase-jitter-rad"]) * 180 / pi)
    end

    sources = EmissionSource3D[]
    for (idx, (depth_m, lateral_y_m, lateral_z_m)) in pairs(coordinates)
        push!(sources, PointSource3D(
            depth=depth_m,
            lateral_y=lateral_y_m,
            lateral_z=lateral_z_m,
            frequency=frequencies_mhz[idx] * 1e6,
            amplitude=amplitudes_pa[idx],
            phase=phases_deg[idx] * pi / 180,
            delay=delays_us[idx] * 1e-6,
            num_cycles=num_cycles,
        ))
    end
    return sources, Dict{String, Any}(
        "source_model" => "point3d",
        "coordinates_m" => [collect(coord) for coord in coordinates],
        "n_coordinate_sources" => n_sources,
        "n_emission_sources" => length(sources),
        "physical_source_count" => length(sources),
        "emission_event_count" => length(sources),
        "activity_model" => Dict("activity_mode" => "point_tone_burst_3d"),
        "phase_mode" => phase_mode,
        "frequencies_hz" => frequencies_mhz .* 1e6,
        "amplitudes_pa" => amplitudes_pa,
        "phases_rad" => phases_deg .* pi ./ 180,
        "delays_s" => delays_us .* 1e-6,
        "num_cycles" => num_cycles,
        "random_seed" => parse(Int, opts["random-seed"]),
    )
end

"""
    centerlines_m_3d(centerlines)

Convert 3D centerline tuples in meters to JSON-friendly nested arrays.
"""
centerlines_m_3d(centerlines) = [[[p[1], p[2], p[3]] for p in line] for line in centerlines]

"""
    parse_squiggle_sources_3d(opts, cfg)

Parse 3D squiggle-source CLI options into sources and emission metadata.
"""
function parse_squiggle_sources_3d(opts, cfg::PAMConfig3D)
    anchors = parse_coordinate_triples_mm(opts["anchors-mm"], "anchors-mm")
    f0 = parse(Float64, opts["fundamental-mhz"]) * 1e6
    harmonics = parse_int_list(opts["harmonics"])
    isempty(harmonics) && error("--harmonics must be a non-empty integer list.")
    harmonic_amplitudes = parse_float_list(opts["harmonic-amplitudes"])
    length(harmonic_amplitudes) == length(harmonics) ||
        error("--harmonic-amplitudes must have the same length as --harmonics ($(length(harmonics))).")

    n_anchors = length(anchors)
    delays_us = expand_source_values(parse_float_list(opts["delays-us"]), n_anchors, 0.0)
    max_sources_raw = parse(Int, opts["vascular-max-sources-per-anchor"])
    max_sources = max_sources_raw <= 0 ? nothing : max_sources_raw
    phase_mode = Symbol(replace(lowercase(strip(opts["phase-mode"])), "-" => "_"))

    sources = EmissionSource3D[]
    all_centerlines_m = Any[]
    anchors_meta = Dict{String, Any}[]
    rng = Random.MersenneTwister(parse(Int, opts["random-seed"]))
    half_y = cfg.transverse_dim_y / 2
    half_z = cfg.transverse_dim_z / 2

    for (idx, anchor) in pairs(anchors)
        anchor_sources, anchor_meta = make_squiggle_bubble_sources_3d(
            [anchor];
            root_length = parse(Float64, opts["vascular-length-mm"]) * 1e-3,
            squiggle_amplitude_y = parse(Float64, opts["vascular-squiggle-amplitude-mm"]) * 1e-3,
            squiggle_amplitude_x = parse(Float64, opts["vascular-squiggle-amplitude-x-mm"]) * 1e-3,
            squiggle_wavelength = parse(Float64, opts["vascular-squiggle-wavelength-mm"]) * 1e-3,
            squiggle_phase_x = parse(Float64, opts["squiggle-phase-x-deg"]) * pi / 180,
            squiggle_slope_x = parse(Float64, opts["vascular-squiggle-slope"]),
            squiggle_slope_y = 0.0,
            source_spacing = parse(Float64, opts["vascular-source-spacing-mm"]) * 1e-3,
            position_jitter = parse(Float64, opts["vascular-position-jitter-mm"]) * 1e-3,
            min_separation = parse(Float64, opts["vascular-min-separation-mm"]) * 1e-3,
            max_sources_per_anchor = max_sources,
            depth_bounds = (0.0, Inf),
            lateral_y_bounds = (-half_y, half_y),
            lateral_z_bounds = (-half_z, half_z),
            fundamental = f0,
            amplitude = parse(Float64, opts["amplitude-pa"]),
            harmonics = harmonics,
            harmonic_amplitudes = harmonic_amplitudes,
            gate_duration = parse(Float64, opts["gate-us"]) * 1e-6,
            taper_ratio = parse(Float64, opts["taper-ratio"]),
            delay = delays_us[idx] * 1e-6,
            phase_mode = phase_mode,
            phase_jitter = parse(Float64, opts["phase-jitter-rad"]),
            transducer_depth = -parse(Float64, opts["skull-transducer-distance-mm"]) * 1e-3,
            transducer_y = 0.0,
            transducer_z = 0.0,
            c0 = cfg.c0,
            rng = rng,
        )
        append!(sources, anchor_sources)
        anchor_cls = centerlines_m_3d(anchor_meta[:centerlines])
        append!(all_centerlines_m, anchor_cls)
        push!(anchors_meta, Dict(
            "anchor_m" => collect(anchor),
            "source_count" => length(anchor_sources),
            "centerlines_m" => anchor_cls,
        ))
    end

    return sources, Dict{String, Any}(
        "source_model" => "squiggle3d",
        "anchor_clusters_m" => [collect(anchor) for anchor in anchors],
        "n_anchor_clusters" => length(anchors),
        "n_emission_sources" => length(sources),
        "physical_source_count" => length(sources),
        "emission_event_count" => length(sources),
        "activity_model" => Dict("activity_mode" => "random_phase_per_window"),
        "phase_mode" => String(phase_mode),
        "fundamental_hz" => f0,
        "harmonics" => harmonics,
        "harmonic_amplitudes" => harmonic_amplitudes,
        "gate_duration_s" => parse(Float64, opts["gate-us"]) * 1e-6,
        "phase_jitter_rad" => parse(Float64, opts["phase-jitter-rad"]),
        "random_seed" => parse(Int, opts["random-seed"]),
        "delays_s" => delays_us .* 1e-6,
        "squiggle" => Dict(
            "length_m" => parse(Float64, opts["vascular-length-mm"]) * 1e-3,
            "squiggle_amplitude_y_m" => parse(Float64, opts["vascular-squiggle-amplitude-mm"]) * 1e-3,
            "squiggle_amplitude_x_m" => parse(Float64, opts["vascular-squiggle-amplitude-x-mm"]) * 1e-3,
            "squiggle_wavelength_m" => parse(Float64, opts["vascular-squiggle-wavelength-mm"]) * 1e-3,
            "squiggle_phase_x_deg" => parse(Float64, opts["squiggle-phase-x-deg"]),
            "squiggle_slope" => parse(Float64, opts["vascular-squiggle-slope"]),
            "source_spacing_m" => parse(Float64, opts["vascular-source-spacing-mm"]) * 1e-3,
            "position_jitter_m" => parse(Float64, opts["vascular-position-jitter-mm"]) * 1e-3,
            "min_separation_m" => parse(Float64, opts["vascular-min-separation-mm"]) * 1e-3,
            "max_sources_per_anchor" => max_sources_raw,
            "truth_radius_m" => parse(Float64, opts["vascular-radius-mm"]) * 1e-3,
            "centerlines_m" => all_centerlines_m,
            "anchors" => anchors_meta,
        ),
    )
end

"""
    parse_network_sources_3d(opts, cfg)

Parse 3D network-source CLI options into sources and emission metadata.
"""
function parse_network_sources_3d(opts, cfg::PAMConfig3D)
    centers = parse_coordinate_triples_mm(opts["anchors-mm"], "anchors-mm")
    f0 = parse(Float64, opts["fundamental-mhz"]) * 1e6
    harmonics = parse_int_list(opts["harmonics"])
    isempty(harmonics) && error("--harmonics must be a non-empty integer list.")
    harmonic_amplitudes = parse_float_list(opts["harmonic-amplitudes"])
    length(harmonic_amplitudes) == length(harmonics) ||
        error("--harmonic-amplitudes must have the same length as --harmonics ($(length(harmonics))).")

    n_centers = length(centers)
    delays_us = expand_source_values(parse_float_list(opts["delays-us"]), n_centers, 0.0)
    max_sources_raw = parse(Int, opts["network-max-sources-per-center"])
    max_sources = max_sources_raw <= 0 ? nothing : max_sources_raw
    phase_mode = Symbol(replace(lowercase(strip(opts["phase-mode"])), "-" => "_"))
    axial_radius_m = parse(Float64, opts["network-axial-radius-mm"]) * 1e-3
    lateral_y_radius_m = parse(Float64, opts["network-lateral-y-radius-mm"]) * 1e-3
    lateral_z_radius_m = parse(Float64, opts["network-lateral-z-radius-mm"]) * 1e-3
    ellipsoid_radii_m = [axial_radius_m, lateral_y_radius_m, lateral_z_radius_m]
    density_sigma_m = parse(Float64, opts["network-density-sigma-mm"]) * 1e-3
    density_axial_sigma_m = parse(Float64, opts["network-density-axial-sigma-mm"]) * 1e-3
    density_lateral_y_sigma_m = parse(Float64, opts["network-density-lateral-y-sigma-mm"]) * 1e-3
    density_lateral_z_sigma_m = parse(Float64, opts["network-density-lateral-z-sigma-mm"]) * 1e-3
    density_sigmas_m = density_sigma_m > 0 ?
        [density_sigma_m, density_sigma_m, density_sigma_m] :
        [density_axial_sigma_m, density_lateral_y_sigma_m, density_lateral_z_sigma_m]

    sources = EmissionSource3D[]
    all_centerlines_m = Any[]
    centers_meta = Dict{String, Any}[]
    rng = Random.MersenneTwister(parse(Int, opts["random-seed"]))
    half_y = cfg.transverse_dim_y / 2
    half_z = cfg.transverse_dim_z / 2

    for (idx, center) in pairs(centers)
        center_sources, network_meta = make_network_bubble_sources_3d(
            [center];
            axial_radius = axial_radius_m,
            lateral_y_radius = lateral_y_radius_m,
            lateral_z_radius = lateral_z_radius_m,
            root_count = parse(Int, opts["network-root-count"]),
            generations = parse(Int, opts["network-generations"]),
            branch_length = parse(Float64, opts["network-branch-length-mm"]) * 1e-3,
            branch_step = parse(Float64, opts["network-branch-step-mm"]) * 1e-3,
            branch_angle = parse(Float64, opts["network-branch-angle-deg"]) * pi / 180,
            tortuosity = parse(Float64, opts["network-tortuosity"]),
            network_orientation = Symbol(replace(lowercase(strip(opts["network-orientation"])), "-" => "_")),
            source_spacing = parse(Float64, opts["vascular-source-spacing-mm"]) * 1e-3,
            density_sigma = density_sigma_m,
            density_sigma_depth = density_axial_sigma_m,
            density_sigma_y = density_lateral_y_sigma_m,
            density_sigma_z = density_lateral_z_sigma_m,
            min_separation = parse(Float64, opts["vascular-min-separation-mm"]) * 1e-3,
            max_sources_per_center = max_sources,
            depth_bounds = (0.0, Inf),
            lateral_y_bounds = (-half_y, half_y),
            lateral_z_bounds = (-half_z, half_z),
            fundamental = f0,
            amplitude = parse(Float64, opts["amplitude-pa"]),
            harmonics = harmonics,
            harmonic_amplitudes = harmonic_amplitudes,
            gate_duration = parse(Float64, opts["gate-us"]) * 1e-6,
            taper_ratio = parse(Float64, opts["taper-ratio"]),
            delay = delays_us[idx] * 1e-6,
            phase_mode = phase_mode,
            phase_jitter = parse(Float64, opts["phase-jitter-rad"]),
            transducer_depth = -parse(Float64, opts["skull-transducer-distance-mm"]) * 1e-3,
            transducer_y = 0.0,
            transducer_z = 0.0,
            c0 = cfg.c0,
            rng = rng,
        )
        append!(sources, center_sources)
        center_cls = centerlines_m_3d(network_meta[:centerlines])
        append!(all_centerlines_m, center_cls)
        push!(centers_meta, Dict(
            "center_m" => collect(center),
            "source_count" => length(center_sources),
            "centerline_count" => length(network_meta[:centerlines]),
            "centerlines_m" => center_cls,
        ))
    end

    return sources, Dict{String, Any}(
        "source_model" => "network3d",
        "network_centers_m" => [collect(center) for center in centers],
        "n_network_centers" => length(centers),
        "n_emission_sources" => length(sources),
        "physical_source_count" => length(sources),
        "emission_event_count" => length(sources),
        "activity_model" => Dict("activity_mode" => "random_phase_per_window"),
        "phase_mode" => String(phase_mode),
        "fundamental_hz" => f0,
        "harmonics" => harmonics,
        "harmonic_amplitudes" => harmonic_amplitudes,
        "gate_duration_s" => parse(Float64, opts["gate-us"]) * 1e-6,
        "phase_jitter_rad" => parse(Float64, opts["phase-jitter-rad"]),
        "random_seed" => parse(Int, opts["random-seed"]),
        "delays_s" => delays_us .* 1e-6,
        "network" => Dict(
            "axial_radius_m" => axial_radius_m,
            "lateral_y_radius_m" => lateral_y_radius_m,
            "lateral_z_radius_m" => lateral_z_radius_m,
            "ellipsoid_radii_m" => ellipsoid_radii_m,
            "root_count" => parse(Int, opts["network-root-count"]),
            "generations" => parse(Int, opts["network-generations"]),
            "branch_length_m" => parse(Float64, opts["network-branch-length-mm"]) * 1e-3,
            "branch_step_m" => parse(Float64, opts["network-branch-step-mm"]) * 1e-3,
            "branch_angle_deg" => parse(Float64, opts["network-branch-angle-deg"]),
            "tortuosity" => parse(Float64, opts["network-tortuosity"]),
            "orientation" => opts["network-orientation"],
            "density_sigma_m" => density_sigma_m,
            "density_sigmas_m" => density_sigmas_m,
            "source_spacing_m" => parse(Float64, opts["vascular-source-spacing-mm"]) * 1e-3,
            "min_separation_m" => parse(Float64, opts["vascular-min-separation-mm"]) * 1e-3,
            "max_sources_per_center" => max_sources_raw,
            "truth_radius_m" => parse(Float64, opts["vascular-radius-mm"]) * 1e-3,
            "centerlines_m" => all_centerlines_m,
            "centers" => centers_meta,
        ),
    )
end

"""
    parse_squiggle_sources(opts, cfg)

Parse 2D squiggle-source CLI options into sources and emission metadata.
"""
function parse_squiggle_sources(opts, cfg::PAMConfig)
    anchors = parse_coordinate_pairs_mm(opts["anchors-mm"], "anchors-mm")
    f0 = parse(Float64, opts["fundamental-mhz"]) * 1e6
    harmonics = parse_int_list(opts["harmonics"])
    isempty(harmonics) && error("--harmonics must be a non-empty integer list.")
    harmonic_amplitudes = parse_float_list(opts["harmonic-amplitudes"])
    length(harmonic_amplitudes) == length(harmonics) ||
        error("--harmonic-amplitudes must have the same length as --harmonics ($(length(harmonics))).")

    n_anchors = length(anchors)
    delays_us = expand_source_values(parse_float_list(opts["delays-us"]), n_anchors, 0.0)
    tx_depth, tx_lateral = parse_transducer_mm(opts["transducer-mm"])
    max_sources_raw = parse(Int, opts["vascular-max-sources-per-anchor"])
    max_sources = max_sources_raw <= 0 ? nothing : max_sources_raw
    phase_mode = Symbol(replace(lowercase(strip(opts["phase-mode"])), "-" => "_"))

    sources = EmissionSource2D[]
    all_centerlines_m = Any[]
    anchors_meta = Dict{String, Any}[]
    rng = Random.MersenneTwister(parse(Int, opts["random-seed"]))
    for (idx, anchor) in pairs(anchors)
        anchor_sources, anchor_meta = make_squiggle_bubble_sources(
            [anchor];
            root_length=parse(Float64, opts["vascular-length-mm"]) * 1e-3,
            squiggle_amplitude=parse(Float64, opts["vascular-squiggle-amplitude-mm"]) * 1e-3,
            squiggle_wavelength=parse(Float64, opts["vascular-squiggle-wavelength-mm"]) * 1e-3,
            squiggle_slope=parse(Float64, opts["vascular-squiggle-slope"]),
            source_spacing=parse(Float64, opts["vascular-source-spacing-mm"]) * 1e-3,
            position_jitter=parse(Float64, opts["vascular-position-jitter-mm"]) * 1e-3,
            min_separation=parse(Float64, opts["vascular-min-separation-mm"]) * 1e-3,
            max_sources_per_anchor=max_sources,
            depth_bounds=(0.0, Inf),
            lateral_bounds=(-cfg.transverse_dim / 2, cfg.transverse_dim / 2),
            fundamental=f0,
            amplitude=parse(Float64, opts["amplitude-pa"]),
            harmonics=harmonics,
            harmonic_amplitudes=harmonic_amplitudes,
            gate_duration=parse(Float64, opts["gate-us"]) * 1e-6,
            taper_ratio=parse(Float64, opts["taper-ratio"]),
            delay=delays_us[idx] * 1e-6,
            phase_mode=phase_mode,
            phase_jitter=parse(Float64, opts["phase-jitter-rad"]),
            transducer_depth=tx_depth,
            transducer_lateral=tx_lateral,
            c0=cfg.c0,
            rng=rng,
        )
        append!(sources, anchor_sources)
        anchor_centerlines = centerlines_m(anchor_meta[:centerlines])
        append!(all_centerlines_m, anchor_centerlines)
        push!(anchors_meta, Dict(
            "anchor_m" => collect(anchor),
            "source_count" => length(anchor_sources),
            "centerlines_m" => anchor_centerlines,
        ))
    end

    return sources, Dict{String, Any}(
        "source_model" => "squiggle",
        "anchor_clusters_m" => [collect(anchor) for anchor in anchors],
        "n_anchor_clusters" => length(anchors),
        "n_emission_sources" => length(sources),
        "physical_source_count" => length(sources),
        "emission_event_count" => length(sources),
        "activity_model" => Dict("activity_mode" => "random_phase_per_window"),
        "phase_mode" => String(phase_mode),
        "fundamental_hz" => f0,
        "harmonics" => harmonics,
        "harmonic_amplitudes" => harmonic_amplitudes,
        "gate_duration_s" => parse(Float64, opts["gate-us"]) * 1e-6,
        "transducer_m" => [tx_depth, tx_lateral],
        "phase_jitter_rad" => parse(Float64, opts["phase-jitter-rad"]),
        "random_seed" => parse(Int, opts["random-seed"]),
        "delays_s" => delays_us .* 1e-6,
        "squiggle" => Dict(
            "length_m" => parse(Float64, opts["vascular-length-mm"]) * 1e-3,
            "squiggle_amplitude_m" => parse(Float64, opts["vascular-squiggle-amplitude-mm"]) * 1e-3,
            "squiggle_wavelength_m" => parse(Float64, opts["vascular-squiggle-wavelength-mm"]) * 1e-3,
            "squiggle_slope" => parse(Float64, opts["vascular-squiggle-slope"]),
            "source_spacing_m" => parse(Float64, opts["vascular-source-spacing-mm"]) * 1e-3,
            "position_jitter_m" => parse(Float64, opts["vascular-position-jitter-mm"]) * 1e-3,
            "min_separation_m" => parse(Float64, opts["vascular-min-separation-mm"]) * 1e-3,
            "max_sources_per_anchor" => max_sources_raw,
            "truth_radius_m" => parse(Float64, opts["vascular-radius-mm"]) * 1e-3,
            "centerlines_m" => all_centerlines_m,
            "anchors" => anchors_meta,
        ),
    )
end

"""
    parse_sources(opts, cfg)

Dispatch 2D PAM CLI source parsing based on `--source-model`.
"""
function parse_sources(opts, cfg::PAMConfig)
    source_model = parse_source_model(opts["source-model"])
    source_model == :point && return parse_point_sources(opts)
    source_model == :squiggle && return parse_squiggle_sources(opts, cfg)
    error("2D PAM CLI supports --source-model=point or --source-model=squiggle.")
end
