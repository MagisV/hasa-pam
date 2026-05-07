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

_emission_frequencies(src::PointSource3D) = [src.frequency]
_source_duration(src::PointSource3D) = src.num_cycles / src.frequency
