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
emission_frequencies(src::EmissionSource3D) = _emission_frequencies(src)
_source_duration(src::PointSource3D) = src.num_cycles / src.frequency

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
