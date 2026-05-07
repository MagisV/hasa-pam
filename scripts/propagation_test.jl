using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using CUDA, CUDA.CUFFT, FFTW

# Minimal test: propagate a single spherical wave to its source depth
# and check that the peak appears at the correct lateral position.

c0 = 1500.0f0
f0 = 0.5f6
k0 = 2f0 * π * f0 / c0
dy = 0.5f-3
dz = 0.5f-3
dx = 0.5f-3

ny, nz = 64, 64
padded_ny, padded_nz = 128, 128

# Source position
src_y = 2f-3
src_z = -1f-3
src_depth = 30f-3

# Build synthetic RF at receiver plane (y, z) for frequency f0
# Phase = -k0 * r (spherical wave from source)
y_vec = Float32.([-(ny/2) * dy + (i-1)*dy for i in 1:ny])
z_vec = Float32.([-(nz/2) * dz + (i-1)*dz for i in 1:nz])
rf_freq = zeros(ComplexF32, ny, nz)
for iy in 1:ny, iz in 1:nz
    r = sqrt(src_depth^2 + (y_vec[iy]-src_y)^2 + (z_vec[iz]-src_z)^2)
    rf_freq[iy, iz] = cis(-k0 * r)
end

println("RF phase range: min=", minimum(angle.(rf_freq)), " max=", maximum(angle.(rf_freq)))

# Zero-pad
left_y = (padded_ny - ny) ÷ 2
left_z = (padded_nz - nz) ÷ 2
crop_y = (left_y+1):(left_y+ny)
crop_z = (left_z+1):(left_z+nz)

# Place in padded array and 2D FFT
p0 = zeros(ComplexF32, padded_ny, padded_nz)
p0[crop_y, crop_z] = rf_freq

# CPU reference propagation
p0_fft = fft(p0)  # 2D FFT

k_y_cpu = [(-padded_ny÷2 + i - 1) * (2π / (padded_ny * dy)) for i in 1:padded_ny]
k_z_cpu = [(-padded_nz÷2 + j - 1) * (2π / (padded_nz * dz)) for j in 1:padded_nz]
KY = reshape(k_y_cpu, :, 1) .* ones(1, padded_nz)
KZ = ones(padded_ny, 1) .* reshape(k_z_cpu, 1, :)
k_lat2 = KY.^2 .+ KZ.^2
k_axial_cpu = sqrt.(complex.(Float32(k0)^2 .- k_lat2))
propagating = real.(k_axial_cpu ./ k0) .> 0

# Propagate to depth src_depth
n_steps = round(Int, src_depth / dx)
propagator_centered = exp.(1im .* k_axial_cpu .* dx) .* propagating
propagator_centered_n = propagator_centered .^ n_steps

# Apply in k-space (after ifftshift, in FFT order)
prop_fft_order = circshift(propagator_centered_n, (-padded_ny÷2, -padded_nz÷2))
p_at_depth = p0_fft .* prop_fft_order

# Back to physical space (ifft2)
p_phys = ifft(p_at_depth)

# Check peak
intensity = abs2.(p_phys)
peak_idx = Tuple(argmax(intensity))
println("CPU reference: peak at padded index ", peak_idx, " expected around (", left_y+argmin(abs.(y_vec.-src_y)), ",", left_z+argmin(abs.(z_vec.-src_z)), ")")
println("  y_peak=", (peak_idx[1] - left_y - 1)*dy*1e3, " mm  expected=", src_y*1e3, " mm")
println("  z_peak=", (peak_idx[2] - left_z - 1)*dz*1e3, " mm  expected=", src_z*1e3, " mm")

# GPU test with same approach
p0_d = CUDA.CuArray(p0)
plan_fwd = plan_fft!(p0_d, (1,2))
plan_bwd = plan_ifft!(similar(p0_d), (1,2))
plan_fwd * p0_d

prop_d = CUDA.CuArray(ComplexF32.(prop_fft_order))
p0_d .*= reshape(prop_d, padded_ny, padded_nz, 1)  # broadcast over nf=1 dim... wait, p0_d is 2D

# Actually replicate GPU path: p0_d is (ny, nz, nf)
p0_3d_d = CUDA.CuArray(reshape(p0, padded_ny, padded_nz, 1))
plan_fwd3 = plan_fft!(p0_3d_d, (1,2))
plan_bwd3 = plan_ifft!(similar(p0_3d_d), (1,2))
plan_fwd3 * p0_3d_d
p0_3d_d .*= reshape(prop_d, padded_ny, padded_nz, 1)
plan_bwd3 * p0_3d_d
r3d = Array(p0_3d_d[:,:,1])
peak3d = Tuple(argmax(abs2.(r3d)))
println("GPU 3D-plan: peak at padded index ", peak3d)
println("  y_peak=", (peak3d[1] - left_y - 1)*dy*1e3, " mm  expected=", src_y*1e3, " mm")
