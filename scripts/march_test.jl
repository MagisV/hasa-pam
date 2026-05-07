using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using CUDA, CUDA.CUFFT, FFTW

# Reproduce the exact GPU march loop from reconstruction3d.jl and check output.
c0 = 1500.0; f0 = 0.5e6; k0 = 2π*f0/c0
dy = dz = dx = 0.5e-3
ny, nz, padded_ny, padded_nz, n_rows = 64, 64, 128, 128, 120
dt = 80e-9; nt = 750; rr = 1; W = 1; nfreq = 1; nfreq_W = 1
T = Float32; CT = ComplexF32

y_vec = [-(ny/2)*dy + (i-1)*dy for i in 1:ny]
z_vec = [-(nz/2)*dz + (j-1)*dz for j in 1:nz]
left_y = (padded_ny-ny)÷2; left_z = (padded_nz-nz)÷2
crop_y = (left_y+1):(left_y+ny); crop_z = (left_z+1):(left_z+nz)

src_depth = 30e-3; src_y = 2e-3; src_z = -1e-3
target_row = 1 + round(Int, src_depth/dx)

# Build RF (ny,nz,nt)
rf = zeros(T, ny, nz, nt)
for iy in 1:ny, iz in 1:nz
    r = sqrt(src_depth^2+(y_vec[iy]-src_y)^2+(z_vec[iz]-src_z)^2)
    t0 = r/c0
    for it in 1:nt
        te = (it-1)*dt - t0
        if te >= 0 && te <= 5/f0
            rf[iy,iz,it] = T(sin(2π*f0*te))
        end
    end
end
freq_axis = collect(0:(nt-1))/(nt*dt)
bin = argmin(abs.(freq_axis[2:fld(nt,2)+1] .- f0)) + 1

# Propagator setup
function fft_wn(n,sp); dk=2π/sp; s=-n÷2; return collect(s:s+n-1) .* dk ./ n; end
k_y = fft_wn(padded_ny,dy); k_z = fft_wn(padded_nz,dz)
KY=reshape(k_y,:,1).*ones(1,padded_nz); KZ=ones(padded_ny,1).*reshape(k_z,1,:)
k_lat2=KY.^2 .+KZ.^2; k_axial=sqrt.(complex.(k0^2 .-k_lat2))
propagating=real.(k_axial./k0).>0.0
propagator=exp.(1im.*k_axial.*dx).*propagating
k_radii=sqrt.(k_lat2); k_max=maximum(k_radii[propagating]; init=0.0)
function tukey_r(kr,km,r); w=ones(size(kr)); tr=km*(1-r)
    for i in eachindex(kr); ri=kr[i]
        if ri>km; w[i]=0.0; elseif ri>=tr && r>0; t=(ri-tr)/(km-tr); w[i]=0.5*(1+cos(π*t)); end
    end; return w; end
weighting=tukey_r(k_radii,k_max,0.25).*propagating
ifs2d(a)=circshift(a,(-size(a,1)÷2,-size(a,2)÷2))
prop_n_weight_cpu = ifs2d(propagating.*propagator.*weighting)
weight_cpu = ifs2d(weighting)

prop_n_weight_d = CUDA.CuArray(CT.(reshape(prop_n_weight_cpu,padded_ny,padded_nz,1)))
weight_d        = CUDA.CuArray(CT.(reshape(weight_cpu,padded_ny,padded_nz,1)))

# Initial condition
rf_fft = fft(rf, 3)
phase = CT(cis(-T(2π)*T(f0)*0.0f0))
p0_cpu = zeros(CT, padded_ny, padded_nz, 1)
p0_cpu[crop_y, crop_z, 1] = rf_fft[:, :, bin] .* phase
p0_d = CUDA.CuArray(p0_cpu)

plan_fwd = plan_fft!(p0_d, (1,2))
plan_bwd = plan_ifft!(similar(p0_d), (1,2))
plan_fwd * p0_d
current_d = p0_d
current_d .*= weight_d

# Accumulator (padded_ny, padded_nz, W, nx) = (128,128,1,120)
intensity_yzWx_d = CUDA.zeros(T, padded_ny, padded_nz, W, n_rows)
p_row_d = similar(current_d)

function _accum!(dst, src, nfpw)
    i = CUDA.threadIdx().x + (CUDA.blockIdx().x-1)*CUDA.blockDim().x
    nynz = size(src,1)*size(src,2)
    i > nynz && return
    iy = (i-1)%size(src,1)+1
    iz = (i-1)÷size(src,1)+1
    W_ = size(dst,3)
    for w in 1:W_
        acc = zero(real(eltype(src)))
        base = (w-1)*nfpw
        for j in 1:nfpw
            v = src[iy,iz,base+j]
            acc += real(v)^2+imag(v)^2
        end
        dst[iy,iz,w] += acc
    end
    return
end

ny_nz = padded_ny*padded_nz
nthreads = min(ny_nz, 512); nblocks = cld(ny_nz, nthreads)

for row in (rr+1):n_rows
    current_d .*= prop_n_weight_d
    p_row_d .= current_d
    plan_bwd * p_row_d
    CUDA.@cuda threads=nthreads blocks=nblocks _accum!(view(intensity_yzWx_d,:,:,:,row), p_row_d, nfreq)
end

# Download and check
raw_all = permutedims(Array(intensity_yzWx_d),(1,2,4,3))  # (128,128,120,1)
raws = [Float64.(permutedims(raw_all[:,:,:,w],(3,1,2))) for w in 1:W]  # (120,128,128)
intensity = raws[1][:, crop_y, crop_z]  # (120,64,64)

mid_y = argmin(abs.(y_vec.-src_y)); mid_z = argmin(abs.(z_vec.-src_z))
println("intensity at expected peak ($target_row,$mid_y,$mid_z): $(intensity[target_row,mid_y,mid_z])")
slice = intensity[target_row, :, :]
println("Slice at depth $target_row: max=$(maximum(slice)) at $(Tuple(argmax(slice)))")
println("slice[1,1]=$(slice[1,1])  slice[$mid_y,$mid_z]=$(slice[mid_y,mid_z])")
println("Overall peak: $(Tuple(argmax(intensity))), value=$(maximum(intensity))")
println("Are all slice values equal? $(all(slice .≈ slice[1,1]))")
