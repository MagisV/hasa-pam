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

