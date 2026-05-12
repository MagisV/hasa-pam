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

mkpath(COVERAGE_DIR)

coverage = Coverage.FileCoverage[]
append!(coverage, process_folder(joinpath(ROOT, "src", "pam")))
append!(coverage, process_folder(joinpath(ROOT, "src", "common")))
push!(coverage, process_file(joinpath(ROOT, "scripts", "run_pam.jl")))

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
