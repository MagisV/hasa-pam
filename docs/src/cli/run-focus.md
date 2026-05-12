# Running Focus

The focusing scripts are retained from the earlier transcranial focusing workflow. They are useful for comparing geometric and HASA focusing corrections through water or a CT-backed skull medium.

## Single Focusing Case

Default skull-backed HASA case:

```bash
julia --project=. scripts/run_focus.jl \
  --estimator=hasa \
  --medium=skull_in_water \
  --slice-index=250
```

Centered target at 60 mm below the transducer:

```bash
julia --project=. scripts/run_focus.jl \
  --estimator=hasa \
  --medium=skull_in_water \
  --placement=fixed_transducer \
  --slice-index=250 \
  --lateral-cm=0.0 \
  --focal-cm=6.0
```

Target 30 mm below the inner skull:

```bash
julia --project=. scripts/run_focus.jl \
  --estimator=hasa \
  --medium=skull_in_water \
  --placement=fixed_focus_depth \
  --slice-index=250 \
  --focal-cm=6.0 \
  --focus-depth-from-inner-skull-mm=30
```

## Compare Estimators

```bash
julia --project=. scripts/compare_focus_estimators.jl \
  --medium=skull_in_water \
  --slice-index=250
```

## Main Options

- `--ct-path`: DICOM folder for CT-backed skull runs.
- `--slice-index`: CT slice used for the 2D focusing medium.
- `--frequency-mhz`: transmit frequency.
- `--focal-cm`: transducer-to-focus distance for fixed-transducer placement, or transducer distance to the resolved target for fixed-depth placement.
- `--lateral-cm`: lateral target offset.
- `--aperture-cm`: transducer aperture width.
- `--estimator`: `geometric` or `hasa`.
- `--medium`: `water` or `skull_in_water`.
- `--placement`: `auto`, `fixed_transducer`, or `fixed_focus_depth`.
- `--focus-depth-from-inner-skull-mm`: target depth below the inner skull for fixed-depth placement.
- `--out-dir`: override the automatically generated output directory.

`scripts/run_focus.jl` writes `summary.json`, `result.jld2`, and `pressure.png`. `scripts/compare_focus_estimators.jl` writes `summary.json` and `comparison.png`.
