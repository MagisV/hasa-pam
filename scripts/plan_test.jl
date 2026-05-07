using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using CUDA, CUDA.CUFFT, FFTW

# Test: does plan_ifft! created AFTER plan_fft! give the right IFFT?
pny, pnz, nf = 128, 128, 1

# Create a spatial signal: delta at (65,65) (center, where DC will be after ifftshift)
spatial_cpu = zeros(ComplexF32, pny, pnz, nf)
spatial_cpu[65, 65, 1] = 1.0f0  # DC after ifftshift → should FFT to constant
test_d = CUDA.CuArray(spatial_cpu)

# Create plan_fwd, apply it
plan_fwd = plan_fft!(test_d, (1,2))
plan_fwd * test_d
after_fft = Array(test_d[:,:,1])
println("After FFT of delta at (65,65):")
println("  max=$(maximum(abs.(after_fft))) at $(Tuple(argmax(abs.(after_fft))))")
println("  after_fft[1,1]=$(after_fft[1,1])  (should be 1.0)")

# Now create plan_bwd from similar of the NOW-transformed array
plan_bwd = plan_ifft!(similar(test_d), (1,2))

# Apply plan_bwd to a constant array (should give delta at (1,1))
const_d = CUDA.ones(ComplexF32, pny, pnz, nf)
plan_bwd * const_d
after_ifft = Array(const_d[:,:,1])
println("After IFFT of ones:")
println("  max=$(maximum(abs.(after_ifft))) at $(Tuple(argmax(abs.(after_ifft))))")
println("  after_ifft[1,1]=$(after_ifft[1,1])  (should be 1.0 = delta)")
println("  after_ifft[65,65]=$(after_ifft[65,65])  (should be ~0)")
println("  Is it uniform? allclose=$(all(abs.(after_ifft) .≈ abs.(after_ifft[1,1])))")

# Compare with CPU IFFT
cpu_ifft_ones = ifft(ones(ComplexF32, pny, pnz))
println("CPU IFFT of ones: max=$(maximum(abs.(cpu_ifft_ones))) at $(Tuple(argmax(abs.(cpu_ifft_ones))))")
println("  cpu_ifft_ones[1,1]=$(cpu_ifft_ones[1,1])")
