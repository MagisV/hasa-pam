# TranscranialFUS Documentation

`TranscranialFUS` is a Julia project for transcranial ultrasound focusing and passive acoustic mapping (PAM). The repository is intended to be used primarily through the scripts in `scripts/`, especially `scripts/run_pam.jl`.

The documentation is organized around the command-line workflow:

- [Getting Started](@ref) covers setup, data assumptions, and first smoke runs.
- [CLI Overview](@ref) explains common command syntax and runtime behavior.
- [Running PAM](@ref) is the main user guide for `scripts/run_pam.jl`.
- [Running Focus](@ref) documents the focusing scripts kept from the earlier workflow.
- [PAM CLI Parameters](@ref) is generated from CLI option metadata in the package.
- [High-Level Workflow](@ref) explains what happens under the hood without going into every internal function.

The code is research-oriented and assumes access to local CT data for skull-backed examples. Homogeneous water examples do not require CT data.
