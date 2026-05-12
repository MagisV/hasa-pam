using Test
using Random
using Statistics
using FFTW
using TranscranialFUS

isdefined(Main, :synthetic_hu_volume) || include("helpers.jl")

@testset "PAM" begin
    @testset "2D" begin
        include("2d/config.jl")
        include("2d/sources.jl")
        include("2d/medium.jl")
        include("2d/reconstruction.jl")
        include("2d/analysis.jl")
        include("2d/workflow.jl")
        include("2d/plots.jl")
    end

    @testset "3D" begin
        include("3d/config3d.jl")
        include("3d/sources3d.jl")
        include("3d/medium3d.jl")
        include("3d/reconstruction3d.jl")
        include("3d/analysis3d.jl")
        include("3d/workflow3d.jl")
        include("3d/plots3d.jl")
    end

    @testset "Setup" begin
        include("setup/config.jl")
        include("setup/sources.jl")
        include("setup/medium.jl")
        include("setup/summary.jl")
    end

    include("integration.jl")
end
