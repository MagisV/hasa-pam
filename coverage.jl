#!/usr/bin/env julia

# Generate LCOV output from Julia .cov files produced by `Pkg.test(; coverage=true)`.
import Pkg

const ROOT = @__DIR__
const COVERAGE_DIR = joinpath(ROOT, "coverage")
const LCOV_PATH = joinpath(COVERAGE_DIR, "lcov.info")
const SUMMARY_PATH = joinpath(COVERAGE_DIR, "summary.txt")

Pkg.activate(; temp=true)
Pkg.add("Coverage")

using Coverage

const EXCLUDED_FILES = Set(
    normpath.([
        joinpath(ROOT, "src", "common", "kwave_wrapper.jl"),
        joinpath(ROOT, "scripts", "run_pam.jl"),
    ]),
)

const EXCLUDED_BLOCK_STARTS = Dict(
    normpath(joinpath(ROOT, "src", "pam", "2d", "reconstruction.jl")) => [
        r"^const _PAM_CUDA_",
        r"^function _pam_cuda_functional\(",
        r"^function _assert_pam_cuda_available\(",
        r"^function _accum_abs2_sum",
        r"^struct PAMCUDASetup\b",
        r"^function _pam_cuda_setup\(",
        r"^function _reconstruct_pam_cuda\(",
    ],
    normpath(joinpath(ROOT, "src", "pam", "3d", "reconstruction3d.jl")) => [
        r"^struct PAMCUDASetup3D\b",
        r"^function _pam_cuda_setup_3d\(",
        r"^function _accum_abs2_sum_batched_3d!\(",
        r"^function _reconstruct_pam_cuda_3d\(",
        r"^function reconstruct_pam_windowed_3d\(",
    ],
)

function _block_end(source, start_idx::Int)
    lines = split(source, '\n'; keepempty=true)
    line = lines[start_idx]
    starts_block = occursin(r"^(function|struct)\b", line)
    starts_block || return start_idx

    depth = 0
    for idx in start_idx:length(lines)
        stripped = strip(lines[idx])
        if occursin(r"^(function|struct|if|for|while|let|begin|try|macro)\b", stripped)
            depth += 1
        end
        if stripped == "end"
            depth -= 1
            depth == 0 && return idx
        end
    end
    return length(source)
end

function _exclude_block_coverage!(fc::Coverage.FileCoverage)
    path = normpath(fc.filename)
    starts = get(EXCLUDED_BLOCK_STARTS, path, nothing)
    isnothing(starts) && return fc

    lines = split(fc.source, '\n'; keepempty=true)
    idx = 1
    while idx <= min(length(lines), length(fc.coverage))
        line = lines[idx]
        if any(pattern -> occursin(pattern, line), starts)
            stop = _block_end(fc.source, idx)
            stop = min(stop, length(fc.coverage))
            fc.coverage[idx:stop] .= nothing
            idx = stop + 1
        else
            idx += 1
        end
    end
    return fc
end

function _apply_coverage_policy!(coverage)
    filter!(fc -> normpath(fc.filename) ∉ EXCLUDED_FILES, coverage)
    foreach(_exclude_block_coverage!, coverage)
    return coverage
end

mkpath(COVERAGE_DIR)

coverage = Coverage.FileCoverage[]
append!(coverage, process_folder(joinpath(ROOT, "src", "pam")))
append!(coverage, process_folder(joinpath(ROOT, "src", "common")))
push!(coverage, process_file(joinpath(ROOT, "scripts", "run_pam.jl"))) # filtered: covered by src/pam/setup/runner.jl
_apply_coverage_policy!(coverage)

covered_lines, total_lines = get_summary(coverage)
coverage_percent = total_lines == 0 ? 100.0 : 100.0 * covered_lines / total_lines

LCOV.writefile(LCOV_PATH, coverage)

open(SUMMARY_PATH, "w") do io
    println(io, "covered_lines=", covered_lines)
    println(io, "total_lines=", total_lines)
    println(io, "coverage_percent=", round(coverage_percent; digits=2))
end

println("Coverage: $(round(coverage_percent; digits=2))% ($(covered_lines)/$(total_lines) lines)")
println("Wrote $(relpath(LCOV_PATH, ROOT))")
