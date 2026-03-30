Base.@kwdef struct CTInfo
    dx::Float64
    dy::Float64
    dz::Float64
    Nx::Int
    Ny::Int
    Nz::Int
end

struct DICOMSliceMeta
    path::String
    series_uid::String
    z_position_mm::Float64
    instance_number::Int
    rows::Int
    cols::Int
    spacing_y_mm::Float64
    spacing_x_mm::Float64
end

_to_float(x) = x isa AbstractString ? parse(Float64, x) : Float64(x)
_to_int(x) = x isa AbstractString ? parse(Int, x) : Int(x)

function _series_uid(meta::DICOM.DICOMData)
    string(meta[(0x0020, 0x000e)])
end

function _slice_z_position_mm(meta::DICOM.DICOMData)
    if haskey(meta, (0x0020, 0x0032))
        pos = meta[(0x0020, 0x0032)]
        return _to_float(pos[3])
    end
    if haskey(meta, (0x0020, 0x1041))
        return _to_float(meta[(0x0020, 0x1041)])
    end
    error("DICOM slice is missing Image Position (Patient) / Slice Location.")
end

function _slice_instance_number(meta::DICOM.DICOMData)
    haskey(meta, (0x0020, 0x0013)) || return 0
    _to_int(meta[(0x0020, 0x0013)])
end

function _scan_dicom_series(dicom_dir::AbstractString)
    isdir(dicom_dir) || error("DICOM directory does not exist: $dicom_dir")
    files = sort(filter(f -> isfile(f), readdir(dicom_dir; join=true)))
    isempty(files) && error("No files found in DICOM directory: $dicom_dir")

    metas = DICOMSliceMeta[]
    uid_counts = Dict{String, Int}()
    for path in files
        meta = dcm_parse(path)
        rows = _to_int(meta[(0x0028, 0x0010)])
        cols = _to_int(meta[(0x0028, 0x0011)])
        spacing = meta[(0x0028, 0x0030)]
        spacing_y_mm = _to_float(spacing[1])
        spacing_x_mm = _to_float(spacing[2])
        uid = _series_uid(meta)
        push!(
            metas,
            DICOMSliceMeta(
                path,
                uid,
                _slice_z_position_mm(meta),
                _slice_instance_number(meta),
                rows,
                cols,
                spacing_y_mm,
                spacing_x_mm,
            ),
        )
        uid_counts[uid] = get(uid_counts, uid, 0) + 1
    end

    series_uid = first(sort!(collect(keys(uid_counts)); by=uid -> uid_counts[uid], rev=true))
    selected = filter(meta -> meta.series_uid == series_uid, metas)
    sort!(selected; by=meta -> (meta.instance_number, meta.z_position_mm))
    return selected
end

function _z_spacing_mm(selected::AbstractVector{DICOMSliceMeta})
    length(selected) == 1 && return 1.0
    z_positions = sort(unique(meta.z_position_mm for meta in selected))
    diffs = diff(z_positions)
    positive_diffs = abs.(diffs[abs.(diffs) .> sqrt(eps(Float64))])
    isempty(positive_diffs) && return abs(diffs[1])
    return median(positive_diffs)
end

function _crop_range(index0::Integer, size_n::Integer, max_n::Integer, label::AbstractString)
    index0 >= 0 || error("$label ROI index must be >= 0, got $index0")
    size_n > 0 || error("$label ROI size must be > 0, got $size_n")
    stop0 = index0 + size_n
    stop0 <= max_n || error("$label ROI exceeds source extent ($stop0 > $max_n)")
    (index0 + 1):stop0
end

function _resample_xy_slice(
    slice::AbstractMatrix{<:Real},
    spacing_y_mm::Float64,
    spacing_x_mm::Float64,
    new_spacing_xy_mm::Float64,
    out_y::Int,
    out_x::Int,
)
    y_coords = 1 .+ (0:(out_y - 1)) .* (new_spacing_xy_mm / spacing_y_mm)
    x_coords = 1 .+ (0:(out_x - 1)) .* (new_spacing_xy_mm / spacing_x_mm)
    itp = extrapolate(interpolate(Float32.(slice), BSpline(Linear())), Flat())

    out = Matrix{Float32}(undef, out_y, out_x)
    @inbounds for iy in 1:out_y
        yc = y_coords[iy]
        for ix in 1:out_x
            out[iy, ix] = Float32(itp(yc, x_coords[ix]))
        end
    end
    return out
end

function load_roi_resample_xy(
    dicom_dir::AbstractString,
    index_xyz::NTuple{3, <:Integer},
    size_xyz::NTuple{3, <:Integer};
    new_spacing_xy_mm::Real=0.20,
)
    @info "reading DICOM series" dicom_dir
    selected = _scan_dicom_series(dicom_dir)
    first_meta = first(selected)

    x_range = _crop_range(index_xyz[1], size_xyz[1], first_meta.cols, "x")
    y_range = _crop_range(index_xyz[2], size_xyz[2], first_meta.rows, "y")
    z_range = _crop_range(index_xyz[3], size_xyz[3], length(selected), "z")

    spacing_x_mm = first_meta.spacing_x_mm
    spacing_y_mm = first_meta.spacing_y_mm
    spacing_z_mm = _z_spacing_mm(selected)

    out_x = round(Int, size_xyz[1] * spacing_x_mm / new_spacing_xy_mm)
    out_y = round(Int, size_xyz[2] * spacing_y_mm / new_spacing_xy_mm)
    out_z = length(z_range)

    @info "cropping ROI" index_xyz size_xyz
    @info "resampling x/y only" spacing_x_mm spacing_y_mm new_spacing_xy_mm

    hu = Array{Float32}(undef, out_z, out_y, out_x)
    for (out_idx, slice_idx) in enumerate(z_range)
        meta = dcm_parse(selected[slice_idx].path)
        raw = meta[(0x7fe0, 0x0010)]
        slope = haskey(meta, (0x0028, 0x1053)) ? _to_float(meta[(0x0028, 0x1053)]) : 1.0
        intercept = haskey(meta, (0x0028, 0x1052)) ? _to_float(meta[(0x0028, 0x1052)]) : 0.0

        cropped = Float32.(raw[y_range, x_range]) .* Float32(slope) .+ Float32(intercept)
        hu[out_idx, :, :] = _resample_xy_slice(
            cropped,
            spacing_y_mm,
            spacing_x_mm,
            Float64(new_spacing_xy_mm),
            out_y,
            out_x,
        )
    end

    spacing_m = (
        Float64(new_spacing_xy_mm) * 1e-3,
        Float64(new_spacing_xy_mm) * 1e-3,
        spacing_z_mm * 1e-3,
    )
    return hu, spacing_m
end

function load_roi_resample_xy(
    dicom_dir::AbstractString,
    index_xyz::AbstractVector{<:Integer},
    size_xyz::AbstractVector{<:Integer};
    new_spacing_xy_mm::Real=0.20,
)
    return load_roi_resample_xy(
        dicom_dir,
        Tuple(index_xyz),
        Tuple(size_xyz);
        new_spacing_xy_mm=new_spacing_xy_mm,
    )
end

function load_default_ct(;
    ct_path::AbstractString=DEFAULT_CT_PATH,
    index_xyz::NTuple{3, Int}=DEFAULT_ROI_INDEX_XYZ,
    size_xyz::NTuple{3, Int}=DEFAULT_ROI_SIZE_XYZ,
    new_spacing_xy_mm::Real=0.20,
)
    return load_roi_resample_xy(
        ct_path,
        index_xyz,
        size_xyz;
        new_spacing_xy_mm=new_spacing_xy_mm,
    )
end

function CTInfo(hu_vol::AbstractArray{<:Real, 3}, spacing_m::NTuple{3, <:Real})
    dx, dy, dz = spacing_m
    return CTInfo(
        dx=Float64(dx),
        dy=Float64(dy),
        dz=Float64(dz),
        Nx=size(hu_vol, 3),
        Ny=size(hu_vol, 2),
        Nz=size(hu_vol, 1),
    )
end
