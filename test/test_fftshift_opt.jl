"""
Verification that removing per-step fftshift/ifftshift from the PAM propagation
loop is mathematically equivalent to the original shifted convention.

Tests the validation checklist:
  - corrected=false
  - corrected=true, homogeneous c
  - corrected=true, heterogeneous c
  - zero_pad_factor=1 (odd-ish padded_ny)
  - zero_pad_factor=2 (even padded_ny)

Run with:  julia --project test/test_fftshift_opt.jl
"""

using FFTW
using LinearAlgebra
using Statistics
using Test

_fftshift(v::AbstractVector)  = circshift(v, fld(length(v), 2))
_ifftshift(v::AbstractVector) = circshift(v, -fld(length(v), 2))

function _fft_wavenumbers(n::Int, spacing::Real)
    dk = 2π / Float64(spacing)
    start_val = -fld(n, 2)
    end_val = ceil(Int, n / 2) - 1
    return collect(start_val:end_val) .* dk ./ n
end

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

"""
OLD approach: fftshift after initial FFT, ifftshift before each IFFT,
fftshift after each convolution FFT. Everything stays in centered order.
"""
function propagate_old(p0_vec, lambda, propagator, weighting, correction,
                       evanescent_inds, n_rows; corrected=true, axial_substeps=1)
    current = _fftshift(fft(p0_vec))
    current .*= weighting

    out = zeros(ComplexF64, n_rows, length(p0_vec))
    for row in 1:n_rows
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
        current .*= weighting
        p_row = ifft(_ifftshift(current))
        out[row, :] .= p_row
    end
    return out
end

"""
NEW approach: compute all spectral arrays in centered order, then _ifftshift
once. The propagation loop uses no fftshift/ifftshift at all.
"""
function propagate_new(p0_vec, lambda, propagator_c, weighting_c, correction_c,
                       propagating, n_rows; corrected=true, axial_substeps=1)
    propagator   = _ifftshift(propagator_c)
    weighting    = _ifftshift(weighting_c)
    correction   = _ifftshift(correction_c)
    evanescent_inds = findall(_ifftshift(.!propagating))

    current = fft(p0_vec)
    current .*= weighting

    out = zeros(ComplexF64, n_rows, length(p0_vec))
    for row in 1:n_rows
        for _ in 1:axial_substeps
            if corrected
                p_space = ifft(current)
                conv_term = fft(lambda[row, :] .* p_space)
                next = current .* propagator
                next .+= correction .* conv_term
            else
                next = current .* propagator
            end
            next[evanescent_inds] .= 0.0
            current = next
        end
        current .*= weighting
        p_row = ifft(current)
        out[row, :] .= p_row
    end
    return out
end

function make_test_arrays(n, freq, c0, dz, tukey_ratio)
    k = _fft_wavenumbers(n, dz)
    k0 = 2π * freq / c0
    kz = sqrt.(complex.(k0^2 .- k .^ 2, 0.0))
    propagator = exp.(1im .* kz .* dz)

    real_inds = findall(real.(kz ./ k0) .> 0.0)
    propagating = falses(n)
    propagating[real_inds] .= true
    evanescent_inds = findall(x -> !x, propagating)
    weighting = zeros(Float64, n)
    weighting[real_inds] .= _tukey_window(length(real_inds), tukey_ratio)

    correction = zeros(ComplexF64, n)
    for idx in real_inds
        abs(kz[idx]) > sqrt(eps(Float64)) || continue
        correction[idx] = propagator[idx] * dz / (2im * kz[idx])
    end

    return propagator, weighting, correction, real_inds, evanescent_inds, propagating
end

@testset "fftshift optimisation equivalence" begin
    c0 = 1500.0
    dz = 0.5e-3
    freq = 0.4e6
    n_rows = 5
    axial_substeps_list = [1, 3]
    tukey_ratio = 0.25

    for n in [60, 120, 61]  # even, even (doubled), odd
        @testset "n=$n" begin
            p0_vec = randn(ComplexF64, n)
            propagator, weighting, correction, _, evanescent_inds, propagating =
                make_test_arrays(n, freq, c0, dz, tukey_ratio)

            # Homogeneous c: lambda = 0 everywhere (no correction term matters)
            lambda_hom = zeros(Float64, n_rows, n)
            # Heterogeneous c: random perturbation
            lambda_het = 0.1 .* randn(Float64, n_rows, n)

            for (label, lambda) in [("homogeneous", lambda_hom), ("heterogeneous", lambda_het)]
                for corrected in [false, true]
                    for axial_substeps in axial_substeps_list
                        tag = "n=$n, $label, corrected=$corrected, substeps=$axial_substeps"

                        out_old = propagate_old(
                            copy(p0_vec), lambda, propagator, weighting, correction,
                            evanescent_inds, n_rows;
                            corrected=corrected, axial_substeps=axial_substeps,
                        )
                        out_new = propagate_new(
                            copy(p0_vec), lambda, propagator, weighting, correction,
                            propagating, n_rows;
                            corrected=corrected, axial_substeps=axial_substeps,
                        )

                        relerr = norm(out_old .- out_new) / max(norm(out_old), eps())
                        @test relerr < 1e-12
                    end
                end
            end
        end
    end
end

println("All tests passed.")
