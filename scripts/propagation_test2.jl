using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using CUDA, CUDA.CUFFT, FFTW, Statistics

# Replicate exactly what reconstruction3d.jl does, step by step.

c0 = 1500.0
f0 = 0.5e6
k0 = 2π * f0 / c0
dy = 0.5e-3
dz = 0.5e-3
dx = 0.5e-3

ny, nz = 64, 64
padded_ny, padded_nz = 128, 128
nt = 750
dt = 80e-9
rr = 1
n_rows = 120  # nx
src_depth = 30e-3
src_y = 2e-3
src_z = -1e-3
target_row = rr + round(Int, src_depth / dx)  # = 61
axial_substeps = 1

T  = Float32
CT = ComplexF32

# Build y/z grids (centered)
y_vec = [-(ny/2)*dy + (i-1)*dy for i in 1:ny]
z_vec = [-(nz/2)*dz + (j-1)*dz for j in 1:nz]

# Build synthetic RF (ny, nz, nt)
rf = zeros(T, ny, nz, nt)
for iy in 1:ny, iz in 1:nz
    r = sqrt(src_depth^2 + (y_vec[iy]-src_y)^2 + (z_vec[iz]-src_z)^2)
    t0 = r / c0
    for it in 1:nt
        te = (it-1)*dt - t0
        if te >= 0 && te <= 5/f0
            rf[iy, iz, it] = T(sin(2π*f0*te))
        end
    end
end

# Bin for f0
freq_axis = collect(0:(nt-1)) / (nt * dt)
bin = argmin(abs.(freq_axis[2:fld(nt,2)+1] .- f0)) + 1
println("Frequency bin: $bin, freq=$(freq_axis[bin]) Hz")

# Crop ranges
left_y = (padded_ny - ny) ÷ 2
left_z = (padded_nz - nz) ÷ 2
crop_y = (left_y+1):(left_y+ny)
crop_z = (left_z+1):(left_z+nz)

# Build padded sound speed (uniform water)
c_padded = fill(T(c0), n_rows, padded_ny, padded_nz)

# Compute eta
eta_cpu = permutedims(1 .- (c0 ./ c_padded).^2, (2,3,1))  # (padded_ny, padded_nz, nx)
eta_d = CUDA.CuArray(T.(eta_cpu))
println("eta max: $(maximum(abs.(eta_cpu))) (should be ~0 for water)")

# Wavenumber grids
function fft_wavenumbers(n, spacing)
    dk = 2π / spacing
    start_val = -n ÷ 2
    end_val = ceil(Int, n/2) - 1
    return collect(start_val:end_val) .* dk ./ n
end

k_y = fft_wavenumbers(padded_ny, dy)
k_z = fft_wavenumbers(padded_nz, dz)
KY  = reshape(k_y, :, 1) .* ones(1, padded_nz)
KZ  = ones(padded_ny, 1) .* reshape(k_z, 1, :)
k_lat2 = KY.^2 .+ KZ.^2
k_axial = sqrt.(complex.(k0^2 .- k_lat2))
propagating = real.(k_axial ./ k0) .> 0.0
propagator  = exp.(1im .* k_axial .* dx)

n_prop = count(propagating)
println("Propagating modes: $n_prop / $(padded_ny*padded_nz) = $(round(100*n_prop/(padded_ny*padded_nz);digits=1))%")

k_radii = sqrt.(k_lat2)
k_max_prop = maximum(k_radii[propagating]; init=0.0)
println("k_max_prop=$(round(k_max_prop;digits=1)) rad/m, k0=$(round(k0;digits=1)) rad/m")

function tukey_radial(k_radii, k_max, ratio)
    w = ones(size(k_radii))
    k_max > 0 || return w
    tr = k_max * (1 - ratio)
    for i in eachindex(k_radii)
        r = k_radii[i]
        if r > k_max; w[i] = 0.0
        elseif r >= tr && ratio > 0
            t = (r - tr) / (k_max - tr)
            w[i] = 0.5 * (1 + cos(π*t))
        end
    end
    return w
end

weighting = tukey_radial(k_radii, k_max_prop, 0.25) .* propagating

ifftshift2d(a) = circshift(a, (-size(a,1)÷2, -size(a,2)÷2))

prop_fft = ifftshift2d(propagating .* propagator)
weight_fft = ifftshift2d(weighting)
prop_n_weight_fft = ifftshift2d(propagating .* propagator .^ axial_substeps .* weighting)

# GPU arrays (shape: padded_ny, padded_nz, nfreq=1)
prop_d          = CUDA.CuArray(CT.(reshape(prop_fft,          padded_ny, padded_nz, 1)))
weight_d        = CUDA.CuArray(CT.(reshape(weight_fft,        padded_ny, padded_nz, 1)))
prop_n_weight_d = CUDA.CuArray(CT.(reshape(prop_n_weight_fft, padded_ny, padded_nz, 1)))

# Initial conditions from RF
rf_fft = fft(rf, 3)  # time FFT → (ny, nz, nt)
rf_fft_d = CUDA.CuArray(CT.(rf_fft))

p0_d = CUDA.zeros(CT, padded_ny, padded_nz, 1)
# Place receiver data at crop positions
t0_w = 0.0
phase = CT(cis(-T(2π) * T(f0) * T(t0_w)))
CUDA.@allowscalar nothing  # suppress warning
p0_d_slice = CUDA.zeros(CT, ny, nz)
p0_d_slice .= rf_fft_d[:, :, bin] .* phase
# Need to place p0_d_slice into p0_d[crop_y, crop_z, 1]
tmp_cpu = zeros(CT, padded_ny, padded_nz, 1)
tmp_cpu[crop_y, crop_z, 1] = Array(p0_d_slice)
p0_d = CUDA.CuArray(tmp_cpu)

println("p0 max before FFT: $(maximum(abs.(Array(p0_d))))")

plan_fwd = plan_fft!(p0_d, (1,2))
plan_bwd = plan_ifft!(similar(p0_d), (1,2))
plan_fwd * p0_d
println("p0 max after FFT (k-space): $(maximum(abs.(Array(p0_d))))")

current_d = p0_d
current_d .*= weight_d
println("After weight: max=$(maximum(abs.(Array(current_d))))")

p_row_d   = similar(current_d)

# March to target row and check intensity
intensity_at_rows = Float32[]
for row in (rr+1):n_rows
    current_d .*= prop_n_weight_d
    p_row_d .= current_d
    plan_bwd * p_row_d
    # Max intensity at this row
    row_arr = Array(p_row_d[:,:,1])
    push!(intensity_at_rows, maximum(abs2.(row_arr)))
end

println("Max intensity at rows 58-65: ", intensity_at_rows[57:64])
println("Overall max at row: $(argmax(intensity_at_rows) + rr) (expected $target_row)")

# Check the spatial field at target row
current_d2 = CUDA.CuArray(tmp_cpu)
plan_fwd2 = plan_fft!(current_d2, (1,2))
plan_bwd2 = plan_ifft!(similar(current_d2), (1,2))
plan_fwd2 * current_d2
current_d2 .*= weight_d
for row in (rr+1):target_row
    current_d2 .*= prop_n_weight_d
end
p_target = similar(current_d2)
p_target .= current_d2
plan_bwd2 * p_target
field_at_target = Array(p_target[:,:,1])
println("Field at target depth (row=$target_row):")
println("  max at: $(Tuple(argmax(abs2.(field_at_target))))")
phys_idx = Tuple(argmax(abs2.(field_at_target[crop_y, crop_z])))
println("  cropped field max at physical index: $phys_idx (expected ≈($(round(Int,(src_y+ny*dy/2)/dy+1)), $(round(Int,(src_z+nz*dz/2)/dz+1))))")
println("  y_peak=$(y_vec[phys_idx[1]]*1e3) mm, z_peak=$(z_vec[phys_idx[2]]*1e3) mm")
