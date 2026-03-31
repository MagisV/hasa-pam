# TranscranialFUS

Notebook-shaped Julia port of the core focusing pipeline from `../Ultrasound/transcranial_fus_hasa_aberration_correction/focus.ipynb`.

## Scope

- Julia-native CT loading and preprocessing
- Julia port of the medium construction, HASA/geometric focusing, and focus analysis code
- Thin Python wrapper around `k-wave-python` for synthetic pressure simulation
- Two runnable scripts for single-run execution and estimator comparison

Benchmark sweeps, pickle-driven figure sections, and animation sections are intentionally out of scope for v1.

## Layout

- `src/ct.jl`: DICOM loading, ROI crop, XY resampling, `CTInfo`
- `src/medium.jl`: HU conversion, skull masking, and medium construction
- `src/focus.jl`: configs, enums, plotting, and `focus`
- `src/pam.jl`: passive acoustic mapping configs, source models, reconstruction, and metrics
- `src/kwave_wrapper.jl`: thin `k-wave-python` bridge
- `src/analysis.jl`: `analyse_focus_2d` and `run_focus_case`
- `scripts/run_focus_case.jl`: run one water or skull case
- `scripts/compare_estimators.jl`: compare geometric and HASA
- `scripts/run_pam_case.jl`: simulate point emitters and reconstruct them with geometric ASA and HASA PAM

## Environment

Instantiate the Julia project:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

`PythonCall.jl` uses `CondaPkg.toml` to manage the Python-side `k-wave-python` dependency when the wrapper is used.

## CT Data

The default loader path matches the local notebook setup:

`../Ultrasound/DIRU_20240404_human_skull_phase_correction_1_2_(skull_Normal)/DICOM/PAT_0000/STD_0000/SER_0002/OBJ_0001`

The loader mirrors the notebook behavior:

- select the dominant DICOM series in the folder
- crop the ROI using the notebook's `(x, y, z)` indices and sizes
- resample `x/y` only to `0.20 mm`
- keep `z` spacing unchanged

## Running

Single run:

```bash
julia --project=. scripts/run_focus_case.jl --estimator=hasa --medium=skull_in_water --slice-index=250
```

The scripts expose the two placement modes from the original notebook pipeline:

- `--placement=fixed_transducer`
  Skull placement and focal target placement are both defined relative to the transducer.
  `--focal-cm` sets the transducer-to-focus distance.
- `--placement=fixed_focus_depth`
  The focal target is placed at a depth relative to the inner skull with `--focus-depth-from-inner-skull-mm`.
  `--focal-cm` then determines how far away the transducer plane is placed from that target.

If you leave `--placement=auto`, the skull example defaults to `fixed_focus_depth` with `30 mm` from the inner skull, matching the notebook example.

Adjust the focal target position from the command line in either mode:

- Fixed-transducer placement, centered target, 60 mm below the transducer plane:

```bash
julia --project=. scripts/run_focus_case.jl \
  --estimator=hasa \
  --medium=skull_in_water \
  --placement=fixed_transducer \
  --slice-index=250 \
  --lateral-cm=0.0 \
  --focal-cm=6.0
```

- Fixed-transducer placement, shift the target 10 mm to the right while keeping the same axial distance from the transducer:

```bash
julia --project=. scripts/run_focus_case.jl \
  --estimator=hasa \
  --medium=skull_in_water \
  --placement=fixed_transducer \
  --slice-index=250 \
  --lateral-cm=1.0 \
  --focal-cm=6.0
```

- Fixed-transducer placement, shift the target 10 mm to the left and move it deeper to 80 mm:

```bash
julia --project=. scripts/run_focus_case.jl \
  --estimator=hasa \
  --medium=skull_in_water \
  --placement=fixed_transducer \
  --slice-index=250 \
  --lateral-cm=-1.0 \
  --focal-cm=8.0
```

- Fixed-focus-depth placement, keep a 60 mm transducer-to-focus distance but place the target 20 mm below the inner skull:

```bash
julia --project=. scripts/run_focus_case.jl \
  --estimator=hasa \
  --medium=skull_in_water \
  --placement=fixed_focus_depth \
  --slice-index=250 \
  --lateral-cm=0.0 \
  --focal-cm=6.0 \
  --focus-depth-from-inner-skull-mm=20
```

- Fixed-focus-depth placement, move the target 10 mm to the left while keeping it 30 mm below the inner skull:

```bash
julia --project=. scripts/run_focus_case.jl \
  --estimator=hasa \
  --medium=skull_in_water \
  --placement=fixed_focus_depth \
  --slice-index=250 \
  --lateral-cm=-1.0 \
  --focal-cm=6.0 \
  --focus-depth-from-inner-skull-mm=30
```

Geometric vs HASA comparison:

```bash
julia --project=. scripts/compare_estimators.jl --medium=skull_in_water --slice-index=250
```

Both scripts write results into an `outputs/` subdirectory by default. The default directory name is generated from the placement mode, main run parameters, and a timestamp; `--out-dir=...` overrides it.

## Passive Acoustic Mapping

The project also includes a 2D passive acoustic mapping workflow built around a simple point-emitter model:

- up to 5 point sources specified by `depth_mm:lateral_mm`
- outward propagation simulated with `k-wave-python`
- geometric ASA reconstruction and corrected HASA reconstruction on the same RF data
- localization and image-quality metrics based on the HASA paper:
  axial, lateral, and radial localization error; axial and lateral FWHM; normalized peak intensity

For now the propagation medium is deliberately simple:

- `--aberrator=none` for homogeneous water
- `--aberrator=lens` for a simple elliptical speed perturbation between the array and the sources
- `--aberrator=skull` to place a CT-derived cranium into the PAM grid

The default PAM grid is intentionally finer than the fast focus smoke tests. In practice the passive reconstruction is sensitive to spatial resolution; the default `0.2 mm` spacing is the intended starting point for meaningful localization.

When `--aberrator=skull` is used:

- the receiver/transducer stays at the top of the physical grid
- the outer skull surface is aligned to `--skull-transducer-distance-mm` below the transducer, default `30 mm`
- source coordinates in `--sources-mm=depth:lateral,...` remain defined relative to the transducer, not relative to the skull
- the axial grid is automatically extended to fit the deepest source plus `--bottom-margin-mm`

Single-source example with a simple aberrator:

```bash
julia --project=. scripts/run_pam_case.jl \
  --sources-mm=30:0 \
  --aberrator=lens
```

Three simultaneous sources:

```bash
julia --project=. scripts/run_pam_case.jl \
  --sources-mm=25:-6,32:0,40:8 \
  --aberrator=lens
```

Homogeneous-water baseline:

```bash
julia --project=. scripts/run_pam_case.jl \
  --sources-mm=30:0 \
  --aberrator=none
```

Single-source transcranial example with the cranium placed `30 mm` below the transducer:

```bash
julia --project=. scripts/run_pam_case.jl \
  --sources-mm=50:0 \
  --aberrator=skull \
  --slice-index=250 \
  --skull-transducer-distance-mm=30
```

Shift the skull slice and move the emitter laterally:

```bash
julia --project=. scripts/run_pam_case.jl \
  --sources-mm=45:6 \
  --aberrator=skull \
  --slice-index=230 \
  --skull-transducer-distance-mm=25
```

You can also perturb the emitter timing and phase:

```bash
julia --project=. scripts/run_pam_case.jl \
  --sources-mm=25:-6,32:0,40:8 \
  --phases-deg=0,90,180 \
  --delays-us=0,2,4 \
  --aberrator=lens
```

Each PAM run writes:

- `overview.png`: medium, RF data, and side-by-side geometric/HASA reconstructions
- `summary.json`: reconstruction metrics and run metadata
- `result.jld2`: raw arrays and Julia structs for follow-up analysis

## Tests

Run the unit tests:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The Python k-Wave smoke tests are opt-in:

```bash
TRANSCRANIALFUS_RUN_KWAVE_TESTS=1 julia --project=. -e 'using Pkg; Pkg.test()'
```

The heavier CT integration test is also opt-in:

```bash
TRANSCRANIALFUS_RUN_INTEGRATION=1 TRANSCRANIALFUS_RUN_KWAVE_TESTS=1 julia --project=. -e 'using Pkg; Pkg.test()'
```
