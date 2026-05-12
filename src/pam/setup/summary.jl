function json3_to_any(x)
    if x isa JSON3.Object
        return Dict{String, Any}(String(k) => json3_to_any(v) for (k, v) in pairs(x))
    elseif x isa JSON3.Array
        return Any[json3_to_any(v) for v in x]
    else
        return x
    end
end

function source_model_from_meta(meta, sources)
    if haskey(meta, "source_model")
        model = Symbol(String(meta["source_model"]))
        model == :vascular && return :squiggle
        model == :squiggle3d && return :squiggle
        model == :network3d && return :network
        return model
    end
    if haskey(meta, "cluster_model")
        old = Symbol(String(meta["cluster_model"]))
        return old == :vascular ? :squiggle : old
    end
    return any(src -> src isa Union{BubbleCluster2D, BubbleCluster3D}, sources) ? :squiggle : :point
end

function centerlines_from_emission_meta(meta)
    key = haskey(meta, "squiggle") ? "squiggle" : (haskey(meta, "network") ? "network" : (haskey(meta, "vascular") ? "vascular" : ""))
    isempty(key) && return nothing
    block = meta[key]
    haskey(block, "centerlines_m") || return nothing
    centerlines = Vector{Tuple{Float64, Float64}}[]
    for raw_line in block["centerlines_m"]
        line = Tuple{Float64, Float64}[]
        for point in raw_line
            push!(line, (Float64(point[1]), Float64(point[2])))
        end
        length(line) >= 2 && push!(centerlines, line)
    end
    return isempty(centerlines) ? nothing : centerlines
end

function detection_truth_mask_from_meta(meta, kgrid, cfg, radius::Real)
    centerlines = centerlines_from_emission_meta(meta)
    isnothing(centerlines) && return nothing
    return pam_centerline_truth_mask(centerlines, kgrid, cfg; radius=radius)
end

string_key_dict(d::AbstractDict) = Dict(String(k) => v for (k, v) in d)

function compact_window_info(info)
    haskey(info, :used_window_count) || return nothing
    range_pairs(ranges) = [[first(range), last(range)] for range in ranges]
    return Dict(
        "total_window_count" => info[:total_window_count],
        "used_window_count" => info[:used_window_count],
        "skipped_window_count" => info[:skipped_window_count],
        "window_samples" => info[:window_samples],
        "hop_samples" => info[:hop_samples],
        "effective_window_duration_s" => get(info, :effective_window_duration_s, nothing),
        "effective_hop_s" => get(info, :effective_hop_s, nothing),
        "energy_threshold" => info[:energy_threshold],
        "used_window_ranges" => range_pairs(info[:used_window_ranges]),
        "skipped_window_ranges" => range_pairs(info[:skipped_window_ranges]),
        "accumulation" => haskey(info, :accumulation) ? String(info[:accumulation]) : nothing,
    )
end

function source_summary(src::PointSource2D)
    return Dict(
        "kind" => "point",
        "depth_m" => src.depth,
        "lateral_m" => src.lateral,
        "frequency_hz" => src.frequency,
        "amplitude_pa" => src.amplitude,
        "phase_rad" => src.phase,
        "delay_s" => src.delay,
        "num_cycles" => src.num_cycles,
    )
end

function source_summary(src::BubbleCluster2D)
    return Dict(
        "kind" => "bubble_cluster",
        "depth_m" => src.depth,
        "lateral_m" => src.lateral,
        "fundamental_hz" => src.fundamental,
        "amplitude_pa" => src.amplitude,
        "harmonics" => src.harmonics,
        "harmonic_amplitudes" => src.harmonic_amplitudes,
        "harmonic_phases_rad" => src.harmonic_phases,
        "gate_duration_s" => src.gate_duration,
        "delay_s" => src.delay,
    )
end

function source_summary(src::PointSource3D)
    return Dict(
        "kind" => "point3d",
        "depth_m" => src.depth,
        "lateral_y_m" => src.lateral_y,
        "lateral_z_m" => src.lateral_z,
        "frequency_hz" => src.frequency,
        "amplitude_pa" => src.amplitude,
        "phase_rad" => src.phase,
        "delay_s" => src.delay,
        "num_cycles" => src.num_cycles,
    )
end

function source_summary(src::BubbleCluster3D)
    return Dict(
        "kind" => "bubble3d",
        "depth_m" => src.depth,
        "lateral_y_m" => src.lateral_y,
        "lateral_z_m" => src.lateral_z,
        "fundamental_hz" => src.fundamental,
        "amplitude_pa" => src.amplitude,
        "harmonics" => src.harmonics,
        "harmonic_amplitudes" => src.harmonic_amplitudes,
        "harmonic_phases_rad" => src.harmonic_phases,
        "gate_duration_s" => src.gate_duration,
        "delay_s" => src.delay,
    )
end
