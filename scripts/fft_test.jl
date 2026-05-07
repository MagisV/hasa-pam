using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using CUDA, CUDA.CUFFT

pny, pnz, nf = 128, 128, 1

# Test 1: FFT of a delta at center (65,65) in a batched 3D array
test_cpu = zeros(ComplexF32, pny, pnz, nf)
test_cpu[65, 65, 1] = 1.0f0
test_d = CUDA.CuArray(test_cpu)
plan_fwd = plan_fft!(test_d, (1,2))
plan_fwd * test_d
result = Array(test_d[:,:,1])
println("FFT of delta at (65,65):")
println("  max=", maximum(abs.(result)), " at ", Tuple(argmax(abs.(result))))
println("  result[1,1]=", result[1,1], "  result[65,65]=", result[65,65])
# Expected: uniform magnitude 1.0 (unnormalized FFT of delta = 1 everywhere)

# Test 2: IFFT of ones should give delta at (1,1) with value 1.0
test2_d = CUDA.ones(ComplexF32, pny, pnz, nf)
plan_bwd = plan_ifft!(similar(test2_d), (1,2))
plan_bwd * test2_d
r2 = Array(test2_d[:,:,1])
println("IFFT of ones:")
println("  max=", maximum(abs.(r2)), " at ", Tuple(argmax(abs.(r2))))
println("  r2[1,1]=", r2[1,1])
# Expected: delta at (1,1) with value 1.0 (normalized IFFT)

# Test 3: round-trip
test3_d = CUDA.zeros(ComplexF32, pny, pnz, nf)
test3_d[33:96, 33:96, 1] .= 1.0f0  # put a box at center
orig = Array(test3_d[:,:,1])
plan_f2 = plan_fft!(test3_d, (1,2))
plan_b2 = plan_ifft!(similar(test3_d), (1,2))
plan_f2 * test3_d
plan_b2 * test3_d
rt = Array(test3_d[:,:,1])
println("Round-trip error: ", maximum(abs.(rt .- orig)))
