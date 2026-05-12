# Deterministic helpers for the Python k-Wave bridge. These are kept separate
# from live backend execution so default coverage can exercise them in CI.

function _normalize_record(record::Union{Symbol, AbstractString})
    symbol = record isa Symbol ? record : Symbol(record)
    symbol in (:p_rms, :p) || error("Unsupported record mode: $record")
    return symbol
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

function _validate_point_source_inputs(
    c::AbstractMatrix{<:Real},
    rho::AbstractMatrix{<:Real},
    sources::AbstractVector{<:EmissionSource2D},
    cfg::PAMConfig,
)
    isempty(sources) && error("At least one emission source is required.")
    nx, ny = size(c)
    size(rho) == size(c) || error("Density map must have the same size as the sound-speed map.")
    nx == pam_Nx(cfg) || error("Sound-speed map height $nx does not match PAMConfig height $(pam_Nx(cfg)).")
    ny == pam_Ny(cfg) || error("Sound-speed map width $ny does not match PAMConfig width $(pam_Ny(cfg)).")
    return nx, ny
end

function _validate_point_source_inputs_3d(
    c::AbstractArray{<:Real, 3},
    rho::AbstractArray{<:Real, 3},
    sources::AbstractVector{<:EmissionSource3D},
    cfg::PAMConfig3D,
)
    isempty(sources) && error("At least one emission source is required.")
    nx, ny, nz = pam_Nx(cfg), pam_Ny(cfg), pam_Nz(cfg)
    size(c) == (nx, ny, nz) || error("Sound-speed map size $(size(c)) does not match PAMConfig3D ($nx, $ny, $nz).")
    size(rho) == (nx, ny, nz) || error("Density map size $(size(rho)) does not match PAMConfig3D ($nx, $ny, $nz).")
    return nx, ny, nz
end

function _indexed_sources_2d(sources::AbstractVector{<:EmissionSource2D}, cfg::PAMConfig, kgrid::KGrid2D, nx::Int)
    indexed_sources = [(source_grid_index(src, cfg, kgrid), src) for src in sources]
    sort!(indexed_sources; by=entry -> first(entry)[1] + (first(entry)[2] - 1) * nx)
    return indexed_sources
end

function _indexed_sources_3d(sources::AbstractVector{<:EmissionSource3D}, cfg::PAMConfig3D, nx::Int, ny::Int)
    indexed_sources = [(source_grid_index_3d(src, cfg), src) for src in sources]
    sort!(indexed_sources; by=entry -> begin
        row, cy, cz = first(entry)
        row + (cy - 1) * nx + (cz - 1) * nx * ny
    end)
    return indexed_sources
end

function _group_sources_2d(indexed_sources)
    grouped_sources = Vector{Tuple{Tuple{Int, Int}, Vector{EmissionSource2D}}}()
    for (grid_index, src) in indexed_sources
        if !isempty(grouped_sources) && first(last(grouped_sources)) == grid_index
            push!(last(grouped_sources)[2], src)
        else
            push!(grouped_sources, (grid_index, EmissionSource2D[src]))
        end
    end
    return grouped_sources
end

function _group_sources_3d(indexed_sources)
    grouped_sources = Vector{Tuple{Tuple{Int, Int, Int}, Vector{EmissionSource3D}}}()
    for (grid_index, src) in indexed_sources
        if !isempty(grouped_sources) && first(last(grouped_sources)) == grid_index
            push!(last(grouped_sources)[2], src)
        else
            push!(grouped_sources, (grid_index, EmissionSource3D[src]))
        end
    end
    return grouped_sources
end

function _unique_emission_frequencies(indexed_sources)
    all_freqs = Float64[]
    for (_, src) in indexed_sources
        append!(all_freqs, _emission_frequencies(src))
    end
    return unique(all_freqs)
end

function _kwave_info_2d(row::Int, col_range::UnitRange{Int}, source_indices, sources, grouped_sources)
    return Dict{Symbol, Any}(
        :receiver_row => row,
        :receiver_cols => col_range,
        :source_indices => source_indices,
        :num_input_sources => length(sources),
        :num_source_points => length(grouped_sources),
    )
end

function _kwave_info_3d(row::Int, col_range_y::UnitRange{Int}, col_range_z::UnitRange{Int}, source_indices, sources, grouped_sources)
    return Dict{Symbol, Any}(
        :receiver_row => row,
        :receiver_cols_y => col_range_y,
        :receiver_cols_z => col_range_z,
        :source_indices => source_indices,
        :num_input_sources => length(sources),
        :num_source_points => length(grouped_sources),
    )
end
