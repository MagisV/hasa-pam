# CLI Overview

The maintained entry point is:

```bash
julia --project=. scripts/run_pam.jl --option=value
```

The focusing scripts are also available:

```bash
julia --project=. scripts/run_focus.jl --option=value
julia --project=. scripts/compare_focus_estimators.jl --option=value
```

All scripts use `--name=value` arguments. Positional arguments and space-separated forms such as `--name value` are not supported.

## Output Directories

Runs write to an automatically named directory under `outputs/`. The name includes the date, main run type, medium, source model, grid size, and other identifying parameters. Use `--out-dir=/path/to/output` when a fixed location is needed.

Typical outputs include:

- `summary.json`: run settings, source metadata, metrics, and output paths.
- `result.jld2`: numerical arrays and structured data for later analysis.
- `overview.png`: PAM source/reconstruction overview where applicable.
- `activity_boundaries.png`: detection-threshold visualization for activity sources.
- `pressure.png` or `comparison.png`: focusing figures.

## GPU Flags

PAM separates the forward simulator and reconstruction GPU switches:

- `--kwave-use-gpu=true` controls k-Wave forward simulation.
- `--recon-use-gpu=true` controls CUDA.jl ASA/HASA reconstruction.

Older notes may mention `--use-gpu`; the current PAM runner uses the two explicit flags above.

## Coordinate Conventions

PAM coordinates are specified in millimeters relative to the receiver/transducer plane:

- 2D point or anchor coordinates use `depth:lateral`.
- 3D point or anchor coordinates use `depth:y:z`.
- Comma-separated values specify multiple coordinates.

Examples:

```bash
--sources-mm=30:0
--sources-mm=25:-6,32:0,40:8
--sources-mm=30:2:-1
--anchors-mm=45:0:0
```

Depth is positive into the medium. Lateral coordinates are centered around the transducer/receiver axis.
