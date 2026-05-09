## TranscranialFUS

This repository contains the Spring 2026 project for the ETH Zurich course [Solving PDEs in parallel on GPUs with Julia II](https://pde-on-gpu.vaw.ethz.ch/part2/) (`ETHZ 101-0250-01`).

The project builds on earlier Python code for 2D transcranial ultrasound focusing based on the heterogeneous angular spectrum method (HASA) of Schoen and Arvanitis. The first goal was to port that focusing workflow to Julia while preserving the original end-to-end notebook pipeline. The goal of this project is to extend that foundation toward a GPU-accelerated passive acoustic mapping (PAM) workflow in Julia. 

### Relation to the papers

The two main papers play different roles in this repository.

* **Schoen and Arvanitis (2020)** provides the computational foundation: a heterogeneous angular spectrum formulation for trans-skull aberration correction and passive acoustic mapping/source localization. That is the direct methodological basis for the focusing code and for the PAM reconstruction implemented here. 
* **Ozdas et al. (2020)** provides the therapeutic motivation: focused ultrasound can be used to localize drug delivery in the brain by aggregating and uncaging ultrasound-controlled carriers with millimeter precision while keeping the blood-brain barrier intact. This repository does **not** reproduce that biological protocol directly, but it is motivated by the same broader setting of transcranial targeting and monitoring. 

### Current scope

The repository currently has two main components:

* **Focusing**: a Julia implementation of the transcranial focusing workflow, kept close to the structure of a Python notebook implemented for a research project within the Neurotechnology Group at ETH.
* **PAM**: a 2D passive acoustic mapping pipeline in Julia, including geometric ASA and heterogeneous ASA-based reconstruction

At the moment, the codebase includes Julia-native CT loading and preprocessing, medium construction for transcranial simulations and focusing setup (which are all part of the previous project and provide the relevant scaffolding), a thin Python bridge to `k-wave-python` for forward simulations, and Julia implementations of the passive mapping pipeline, including scripts for single cases, estimator comparisons, and parameter sweeps. 

### Why this project

A central challenge in transcranial ultrasound is that the skull distorts both transmitted and received wavefields. For therapy, this means the transmit field must be corrected so that acoustic energy reaches the intended intracranial target. For monitoring, signals emitted inside the brain must be mapped back through the skull so that their source locations can be estimated accurately. Schoen and Arvanitis address both problems with a computationally efficient heterogeneous ASA model for trans-skull focusing and passive source localization. 

In parallel, Ozdas et al. shows why this matters in a biomedical setting: focused ultrasound can be used to spatially target drug carriers, first by aggregating them and then by uncaging their payload locally, enabling focal delivery with much lower systemic dose and without detectable BBB opening. That paper does not implement passive acoustic mapping, but it motivates the need for reliable transcranial focusing and, more broadly, for methods that could support localization and monitoring of acoustically active agents in the brain. 

### Computational idea

The focusing workflow follows the standard aberration-correction idea used in simulation-based transcranial focusing. A virtual source is placed at the desired target, and its field is propagated outward through a CT-derived heterogeneous skull model to the transducer plane. The phase, and optionally amplitude, predicted at the array are then conjugated and used as the transmit drive. In this repository, that correction step is modeled with HASA, while `k-wave-python` is used as the higher-fidelity forward model for the emitted field. 

The PAM workflow uses the complementary problem. Here the sources are real emitters inside the brain, or point sources in a forward simulation. Their signals propagate outward to the array, producing RF data. Reconstruction then back-propagates those measurements through the same heterogeneous medium model to estimate the source location. In the language of Schoen and Arvanitis, this is passive acoustic mapping or point-source localization through the skull. 



## Repository Layout

- `src/ct.jl`: DICOM loading, ROI crop, and XY resampling
- `src/medium.jl`: HU conversion, skull masking, and focusing-medium construction
- `src/focus.jl`: focusing configs, placement handling, HASA/geometric delays, and plotting helpers
- `src/pam.jl`: compatibility include for the split PAM implementation under `src/pam/`
- `src/pam/sources.jl`: PAM source models, squiggle source construction, source signals, and phase variability
- `src/pam/config.jl`: PAM configs, grid helpers, fitting, and source indexing
- `src/pam/medium.jl`: skull/lens PAM medium generation
- `src/pam/reconstruction.jl`: geometric ASA/HASA propagation and windowed reconstruction loop
- `src/pam/analysis.jl`: PAM peaks, masks, PSF helpers, localization metrics, and detection metrics
- `src/pam/workflow.jl`: case-level simulation/reconstruction orchestration
- `src/pam/sweep.jl`: PAM sweep helpers and aggregation
- `src/kwave_wrapper.jl`: Julia-to-`k-wave-python` bridge
- `src/analysis.jl`: focusing analysis helpers and `run_focus_case`
- `scripts/run_focus_case.jl`: run one focusing case
- `scripts/compare_estimators.jl`: compare geometric and HASA focusing
- `scripts/run_pam.jl`: unified PAM runner for coordinate-placed point sources and squiggle activity
- `scripts/run_pam_sweep.jl`: run a single-source localization sweep and summarize corrected vs uncorrected error
- `test/runtests.jl`: unit tests, smoke tests, and optional integration tests

## Setup

Instantiate the Julia environment:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Python-side dependencies are managed through [CondaPkg.toml](/Users/vm/INI_code/Julia%20II/CondaPkg.toml). The current forward-model backend uses:

- Python `3.11` to `3.12`
- `k-wave-python==0.4.0`
- `numpy`
- `scipy`

The first call into the k-Wave wrapper may need to resolve Python packages and k-Wave resources.

## CT Data

The default CT path points to the local DICOM folder:

`C:\Users\AU-FUS-Valentin\Desktop\OBJ_0001`

You can override it in the scripts with `--ct-path=...`.

The CT scan data used in this project is private and is not distributed with the repository. If you need access to the dataset, please reach out to the author.

The loader is designed to mirror the working notebook behavior:

- select the dominant DICOM series in the folder
- crop the ROI using the notebook defaults
  - `index_xyz = (170, 190, 400)`
  - `size_xyz = (705, 360, 450)`
- resample `x` and `y` to `0.20 mm`
- keep the original `z` spacing

## Focusing Workflow

The focusing path keeps the original notebook-style naming and behavior where practical:

- `load_roi_resample_xy`
- `hu_to_rho_c`
- `find_skull_boundaries`
- `skull_mask_from_c_columnwise`
- `make_medium`
- `focus`
- `analyse_focus_2d`

Supported focusing media:

- `water`
- `skull_in_water`

Supported estimators:

- `geometric`
- `hasa`

### Placement Modes

The focusing scripts support the two placement mechanisms from the original pipeline:

- `--placement=fixed_transducer`
  Skull placement and focal target placement are both defined relative to the transducer. `--focal-cm` is the transducer-to-focus distance.
- `--placement=fixed_focus_depth`
  The target is placed at `--focus-depth-from-inner-skull-mm` below the inner skull, and `--focal-cm` determines how far away the transducer is from that target.
- `--placement=auto`
  For skull runs this defaults to `fixed_focus_depth` with `30 mm` below the inner skull. For water runs it defaults to `fixed_transducer`.

### Run One Focusing Case

Default skull example:

```bash
julia --project=. scripts/run_focus_case.jl \
  --estimator=hasa \
  --medium=skull_in_water \
  --slice-index=250
```

Centered target at `60 mm` below the transducer:

```bash
julia --project=. scripts/run_focus_case.jl \
  --estimator=hasa \
  --medium=skull_in_water \
  --placement=fixed_transducer \
  --slice-index=250 \
  --lateral-cm=0.0 \
  --focal-cm=6.0
```

Target shifted `10 mm` to the right:

```bash
julia --project=. scripts/run_focus_case.jl \
  --estimator=hasa \
  --medium=skull_in_water \
  --placement=fixed_transducer \
  --slice-index=250 \
  --lateral-cm=1.0 \
  --focal-cm=6.0
```

Target `30 mm` below the inner skull:

```bash
julia --project=. scripts/run_focus_case.jl \
  --estimator=hasa \
  --medium=skull_in_water \
  --placement=fixed_focus_depth \
  --slice-index=250 \
  --lateral-cm=0.0 \
  --focal-cm=6.0 \
  --focus-depth-from-inner-skull-mm=30
```

### Compare Geometric vs HASA Focusing

```bash
julia --project=. scripts/compare_estimators.jl \
  --medium=skull_in_water \
  --slice-index=250
```

### Focusing Outputs

`run_focus_case.jl` writes:

- `summary.json`
- `result.jld2`
- `pressure.png`

`compare_estimators.jl` writes:

- `summary.json`
- `comparison.png`

Both scripts generate an `outputs/...` directory name automatically from the main run parameters and a timestamp. `--out-dir=...` overrides that behavior.

## Passive Acoustic Mapping

The PAM path implements a 2D passive reconstruction workflow based on simple point emitters and denser bubble-cluster activity:

- forward propagation simulated with `k-wave-python`
- geometric ASA reconstruction
- corrected HASA reconstruction
- localization and image-quality metrics such as axial, lateral, and radial error, FWHM, peak intensity, and success rate
- thresholded detection metrics for dense aggregate cases, including precision, recall, F1, false-positive area, and false-negative area

Core types:

- `PointSource2D`
- `BubbleCluster2D`
- `GaussianPulseCluster2D`
- `PAMConfig`
- `PAMWindowConfig`
- `SourceVariabilityConfig`

Key helpers:

- `fit_pam_config`
- `make_squiggle_bubble_sources`
- `make_pam_medium`
- `simulate_point_sources`
- `reconstruct_pam`
- `reconstruct_pam_windowed`
- `find_pam_peaks`
- `pam_centerline_truth_mask`
- `analyse_pam_2d`
- `analyse_pam_detection_2d`
- `run_pam_case`
- `run_pam_sweep`

### PAM Medium Options

Supported aberrators:

- `--aberrator=none`: homogeneous water control. Corrected and uncorrected PAM should match because there is no heterogeneous phase error to correct.
- `--aberrator=lens`: simple elliptical speed perturbation
- `--aberrator=skull`: CT-derived skull inserted into the PAM domain

For `--aberrator=skull`:

- the receiver plane stays at the top of the physical domain
- the outer skull surface is placed `--skull-transducer-distance-mm` below the receiver, default `30 mm`
- source coordinates stay defined relative to the transducer/receiver, not relative to the skull
- `fit_pam_config` extends the axial domain automatically to fit the deepest source plus `--bottom-margin-mm`
- `fit_pam_config` also extends `t_max` when a deep or long-gated source would otherwise be truncated
- the run scripts default to `--recon-step-um=50` for HASA/ASA axial integration, matching the trans-skull PAM paper setup

### Run One PAM Case

The maintained PAM entrypoint is `scripts/run_pam.jl`. It supports coordinate-placed point sources and squiggle activity sources.

Single-source homogeneous-water baseline:

```bash
julia --project=. scripts/run_pam.jl \
  --source-model=point \
  --sources-mm=30:0 \
  --aberrator=none
```

Simple 3D homogeneous-water point-source run (requires a CUDA-capable GPU):

```bash
julia --project=. scripts/run_pam.jl \
  --dimension=3 \
  --source-model=point \
  --sources-mm=30:2:-1 \
  --aberrator=none \
  --use-gpu=true
```

Multiple point emitters with explicit phase and delay control:

```bash
julia --project=. scripts/run_pam.jl \
  --source-model=point \
  --sources-mm=25:-6,32:0,40:8 \
  --phases-deg=0,90,180 \
  --delays-us=0,2,4 \
  --aberrator=lens
```

Squiggle activity uses `--anchors-mm=depth:lateral,...`; each anchor expands into harmonic bubble emitters sampled along one squiggly centerline. In `--recon-mode=auto`, `--source-model=squiggle` selects windowed incoherent reconstruction:

```text
I_total(x,z) = mean_windows sum_f |p_hasa(x,z,f,window)|^2
```

Current leading squiggle setting:

```bash
julia --project=. scripts/run_pam.jl `
  --source-model=squiggle `
  --anchors-mm=45:0 `
  --aberrator=skull `
  --boundary-threshold-ratios=0.6,0.65,0.7 `
  --cavitation-model=harmonic-cos `
  --frequency-jitter-percent=1 `
  --harmonic-amplitudes=1.0,0.6,0.3 `
  --harmonics=2,3,4 `
  --random-seed=45 `
  --receiver-aperture-mm=full `
  --recon-bandwidth-khz=500 `
  --recon-hop-us=10 `
  --recon-min-window-energy-ratio=0.001 `
  --recon-window-us=20 `
  --skull-transducer-distance-mm=30 `
  --slice-index=250 `
  --source-phase-mode=random_phase_per_window `
  --t-max-us=500 `
  --transverse-mm=102.4 `
  --vascular-length-mm=12 `
  --recon-progress=true `
  --use-gpu=true `
  --benchmark=true
```

Running reconstruction only with previous k-wave simulation as data:

```bash
julia --project=. scripts/run_pam.jl `
  --from-run-dir=outputs/20260507_102449_run_pam_skull_squiggle_1anchors_21src_ax80p0mm_lat102p0mm_f0p5mhz_h234_slice250_st30p0mm_randomphaseperwindow `
  --boundary-threshold-ratios=0.6,0.65,0.7 `
  --recon-bandwidth-khz=500 `
  --recon-hop-us=10 `
  --recon-min-window-energy-ratio=0.001 `
  --recon-window-us=20 `
  --recon-progress=true `
  --use-gpu=true `
  --window-batch=8 `
  --benchmark=false
```

3D
```bash
julia --project=. scripts/run_pam.jl `
  --dimension=3 `
  --source-model=point `
  --sources-mm=30:2:-1 `
  --aberrator=none `
  --use-gpu=true `
```

```bash
julia --project=. scripts/run_pam.jl `
  --dimension=3 `
  --source-model=point `
  --sources-mm=30:2:-1 `
  --frequency-mhz=0.5 `
  --num-cycles=5 `
  --aberrator=none `
  --axial-mm=60 `
  --transverse-mm=32 `
  --t-max-us=60 `
  --receiver-aperture-mm=full `
  --use-gpu=true `
  --recon-progress=true

```

3D squiggle vascular source (homogeneous water):

```bash
julia --project=. scripts/run_pam.jl `
  --dimension=3 `
  --source-model=squiggle `
  --anchors-mm=55:0:0 `
  --vascular-length-mm=12 `
  --vascular-squiggle-amplitude-mm=1.5 `
  --vascular-squiggle-amplitude-x-mm=1.0 `
  --squiggle-phase-x-deg=90 `
  --harmonics=2,3,4 `
  --harmonic-amplitudes=1.0,0.6,0.3 `
  --aberrator=none `
  --axial-mm=80 `
  --transverse-mm=64 `
  --dx-mm=0.2 `
  --dy-mm=0.5 `
  --dz-mm=0.5 `
  --t-max-us=250 `
  --sim-mode=kwave `
  --use-gpu=true `
  --recon-progress=true
```

3D squiggle vascular source with CT skull:

```bash
julia --project=. scripts/run_pam.jl `
  --dimension=3 `
  --source-model=squiggle `
  --gate-us=45 `
  --anchors-mm=42:0:0 `
  --vascular-length-mm=12 `
  --vascular-squiggle-amplitude-mm=0.3 `
  --vascular-squiggle-amplitude-x-mm=0.2 `
  --vascular-squiggle-wavelength-mm=8 `
  --squiggle-phase-x-deg=90 `
  --vascular-source-spacing-mm=0.5 `
  --vascular-min-separation-mm=0.25 `
  --harmonics=2,3,4 `
  --harmonic-amplitudes=1.0,0.6,0.3 `
  --aberrator=skull `
  --skull-transducer-distance-mm=20 `
  --slice-index=250 `
  --axial-mm=70 `
  --transverse-mm=64 `
  --dx-mm=0.2 `
  --dy-mm=0.5 `
  --dz-mm=0.5 `
  --t-max-us=250 `
  --frequency-mhz=0.5 `
  --receiver-aperture-mm=full `
  --source-phase-mode=random_phase_per_window `
  --frequency-jitter-percent=1 `
  --recon-window-us=40 `
  --recon-hop-us=20 `
  --recon-bandwidth-khz=40 `
  --auto-threshold-search=true `
  --auto-threshold-min=0.10 `
  --auto-threshold-max=0.95 `
  --auto-threshold-step=0.01 `
  --sim-mode=kwave `
  --use-gpu=true `
  --window-batch=2 `
  --recon-progress=true
```

3D synthetic vascular network at the transducer focus:

```bash
julia --project=. scripts/run_pam.jl `
  --dimension=3 `
  --source-model=network `
  --anchors-mm=42:0:0 `
  --network-radius-mm=5 `
  --network-root-count=5 `
  --network-generations=3 `
  --network-branch-length-mm=2.5 `
  --network-branch-step-mm=0.4 `
  --network-branch-angle-deg=36 `
  --network-tortuosity=0.18 `
  --network-density-sigma-mm=2.0 `
  --network-max-sources-per-center=80 `
  --vascular-source-spacing-mm=0.5 `
  --vascular-min-separation-mm=0.25 `
  --fundamental-mhz=0.5 `
  --harmonics=2,3,4 `
  --harmonic-amplitudes=1.0,0.6,0.3 `
  --aberrator=skull `
  --skull-transducer-distance-mm=20 `
  --slice-index=250 `
  --axial-mm=70 `
  --transverse-mm=64 `
  --dx-mm=0.2 `
  --dy-mm=0.5 `
  --dz-mm=0.5 `
  --t-max-us=250 `
  --receiver-aperture-mm=full `
  --source-phase-mode=random_phase_per_window `
  --frequency-jitter-percent=1 `
  --recon-window-us=40 `
  --recon-hop-us=20 `
  --recon-bandwidth-khz=40 `
  --auto-threshold-search=true `
  --auto-threshold-min=0.10 `
  --auto-threshold-max=0.95 `
  --auto-threshold-step=0.01 `
  --sim-mode=kwave `
  --use-gpu=true `
  --window-batch=2 `
  --recon-progress=true
```

`--source-model=network` grows a random branching 3D centerline structure inside
a sphere around each `--anchors-mm=depth:y:z` center, then samples bubble
emitters along those branches with a Gaussian radial density. The default
network radius is `5 mm`, the density sigma is `2 mm`, and the source cap is
`80` bubbles per center.

For sparse 3D squiggle and network sources, the threshold summary reports both voxel overlap
metrics and source-aware metrics. `source_f1` combines voxel precision with the
fraction of simulated bubble centers hit within the source detection radius, and
is used to select the best 3D threshold. By default, 3D analysis runs a dense
post-reconstruction threshold search and `activity_boundaries.png` shows the
precision, recall, and F1 curves plus three readable outlines: best F1, a
recall-biased threshold, and a precision-biased threshold.

The reconstruction bandwidth is an important runtime knob: tighter bandwidths
select fewer FFT frequency bins for ASA/HASA, reducing reconstruction time
roughly in proportion to the frequency-bin count. In the focused 3D skull
sweeps, `40 kHz` bandwidth with `1%` frequency jitter was the best runtime
default: it stayed in the same F1 range as wider bands while reducing the HASA
march time substantially. `60 kHz` with `2%` jitter recovered a little more F1
at extra cost and is a useful accuracy check.

3D heterogeneous skull medium (CT-backed):

```bash
julia --project=. scripts/run_pam.jl `
  --dimension=3 `
  --transverse-mm=64 `
  --axial-mm=80 `
  --dx-mm=0.2 `
  --dy-mm=0.5 `
  --dz-mm=0.5 `
  --t-max-us=80 `
  --source-model=point `
  --sources-mm=50:2:-1 `
  --num-cycles=5 `
  --aberrator=skull `
  --skull-transducer-distance-mm=30 `
  --frequency-mhz=0.5 `
  --slice-index=250 `
  --receiver-aperture-mm=full `
  --use-gpu=true `
  --recon-progress=true
```

The PAM run scripts write:

- `overview.png`
- `activity_boundaries.png`, with threshold-dependent active-region boundaries overlaid on the heatmaps, precision/recall/F1 threshold curves, and selected-threshold metrics
- `summary.json`
- `result.jld2`

To rerun only reconstruction and figure generation from saved RF data, pass an existing output folder:

```bash
julia --project=. scripts/run_pam.jl \
  --from-run-dir=outputs/previous_pam_run \
  --recon-bandwidth-khz=20
```

`--from-run-dir` loads the previous `result.jld2`, reuses its RF data, medium, grid, and sources, and writes a fresh `outputs/<timestamp>_reconstruct_<old-folder>/` directory. Simulation-specific options such as source locations, medium/skull settings, grid size, and time step are rejected in this mode; reconstruction and analysis options such as `--use-gpu`, `--recon-bandwidth-khz`, `--recon-step-um`, `--recon-frequencies-mhz`, `--peak-method`, and detection thresholds remain adjustable.

### CUDA PAM Reconstruction

`--use-gpu=true` enables the CUDA.jl PAM reconstruction backend. It requires a functional NVIDIA CUDA GPU; if CUDA.jl cannot see one, reconstruction errors clearly instead of silently falling back to CPU. The first CUDA path uses Float32 device arithmetic, keeps the existing shifted FFT convention (`fftshift`/`ifftshift`) for CPU/GPU parity, and still marches corrected HASA rows serially while running the lateral FFTs and per-row vector operations on the GPU.

Small case for fast CUDA iteration (no CT data required, ~10 s end-to-end):

```bash
julia --project=. scripts/run_pam.jl `
  --source-model=point `
  --sources-mm=30:0 `
  --aberrator=none `
  --axial-mm=40 `
  --transverse-mm=51.2 `
  --t-max-us=100 `
  --use-gpu=true `
  --recon-progress=true
```

### Source Phase Modes

`--source-phase-mode` controls the physical regime being simulated and is reported in `summary.json`.

| Mode | Physical meaning |
|---|---|
| `coherent` | All sources share the same phase relation. Contributions add constructively/destructively by geometry. |
| `random_static_phase` | Each source draws a random phase once at setup and keeps it for the full simulation. |
| `random_phase_per_window` | Each source emits once per reconstruction window with fresh random phases. A **single** k-Wave simulation spans all windows; windowed reconstruction is forced automatically. |
| `random_phase_per_realization` | Each of `--n-realizations` k-Wave runs draws fresh random phases; intensity maps are averaged across runs. |

**Coherent baseline** — sources lock in phase, single simulation:

```bash
julia --project=. scripts/run_pam.jl \
  --source-model=point \
  --sources-mm=30:0 \
  --source-phase-mode=coherent \
  --phase-mode=coherent \
  --aberrator=none
```

**Random static phase** — fixed random phases, single simulation:

```bash
julia --project=. scripts/run_pam.jl \
  --source-model=squiggle \
  --anchors-mm=30:0 \
  --vascular-length-mm=12 \
  --source-phase-mode=random_static_phase \
  --phase-mode=random \
  --random-seed=42 \
  --aberrator=none
```

**Incoherent averaging over realizations** — 20 independent phase draws:

```bash
julia --project=. scripts/run_pam.jl \
  --source-model=squiggle \
  --anchors-mm=30:0 \
  --vascular-length-mm=12 \
  --source-phase-mode=random_phase_per_realization \
  --n-realizations=20 \
  --random-seed=42 \
  --aberrator=none
```

**Incoherent averaging per window** — single k-Wave run; each source gets fresh random phases per window:

```bash
julia --project=. scripts/run_pam.jl \
  --source-model=squiggle \
  --anchors-mm=30:0 \
  --vascular-length-mm=12 \
  --source-phase-mode=random_phase_per_window \
  --recon-window-us=10 \
  --recon-hop-us=5 \
  --random-seed=42 \
  --aberrator=none
```

For per-window random phase, `--frequency-jitter-percent` applies a multiplicative jitter to each source fundamental frequency.

```bash
julia --project=. scripts/run_pam.jl \
  --source-model=squiggle \
  --anchors-mm=30:0 \
  --vascular-length-mm=12 \
  --source-phase-mode=random_phase_per_window \
  --recon-window-us=10 \
  --recon-hop-us=5 \
  --frequency-jitter-percent=5 \
  --t-max-us=200 \
  --random-seed=42 \
  --aberrator=none
```

Supported amplitude distributions are `fixed`, `uniform`, `lognormal`, and `gaussian`. For `uniform`, `--amplitude-sigma` is the relative half-width around each source amplitude. For `lognormal`, it is the log-space standard deviation. For `gaussian`, it is the relative standard deviation and sampled amplitudes are clipped at zero. `--frequency-jitter-percent` applies a multiplicative jitter to each source fundamental frequency, so harmonic frequencies shift with it. The selected settings are written to `summary.json` under `source_variability`.

**Stochastic broadband** — each source emits independent noise centred on its harmonic frequencies:

```bash
julia --project=. scripts/run_pam.jl \
  --source-model=squiggle \
  --anchors-mm=30:0 \
  --vascular-length-mm=12 \
  --source-phase-mode=random_phase_per_window \
  --random-seed=42 \
  --aberrator=none
```

### Run a PAM Sweep

The sweep script runs **single-source** reconstructions over a target grid and compares uncorrected vs corrected localization.

Default paper-style sweep:

```bash
julia --project=. scripts/run_pam_sweep.jl
```

This defaults to:

- `--aberrator=skull`
- `--frequency-mhz=1.0`
- `--receiver-aperture-mm=50`
- axial targets `30,40,50,60,70,80 mm`
- lateral targets `-20,-10,0,10,20 mm`
- `--slice-index=250`
- `--skull-transducer-distance-mm=30`

Quick sweep for fast figure generation:

```bash
julia --project=. scripts/run_pam_sweep.jl --sweep-preset=quick
```

The quick preset uses:

- axial targets `40,60,80 mm`
- lateral targets `-10,0,10 mm`

Custom sweep:

```bash
julia --project=. scripts/run_pam_sweep.jl \
  --sweep-preset=custom \
  --axial-targets-mm=40,50,60 \
  --lateral-targets-mm=-10,0,10 \
  --aberrator=skull
```

Custom example cases for the overview panel:

```bash
julia --project=. scripts/run_pam_sweep.jl \
  --sweep-preset=quick \
  --example-targets-mm=40:0,60:0,80:10
```

For skull sweeps, requested targets are filtered so that only points inside the cranial cavity are retained. The margin from the inner skull can be adjusted with `--skull-cavity-margin-mm`.

`run_pam_sweep.jl` writes:

- `overview.png`
- `summary.json`
- `result.jld2`
- `cases/`: one figure per retained target, each showing uncorrected vs corrected reconstruction

## Tests

Run the standard test suite:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Enable the k-Wave smoke tests:

```bash
TRANSCRANIALFUS_RUN_KWAVE_TESTS=1 julia --project=. -e 'using Pkg; Pkg.test()'
```

Enable the heavier CT-backed integration test as well:

```bash
TRANSCRANIALFUS_RUN_INTEGRATION=1 TRANSCRANIALFUS_RUN_KWAVE_TESTS=1 julia --project=. -e 'using Pkg; Pkg.test()'
```

The tests currently cover:

- HU-to-medium conversion
- skull boundary detection and masking
- focusing placement resolution
- squiggle PAM source generation and detection metrics
- PAM medium fitting and skull placement
- PAM sweep aggregation and target filtering
- source phase mode normalisation and per-window phase/frequency resampling
- opt-in k-Wave smoke tests

## AI Usage

This project was developed with AI assistance. OpenAI Codex was used for code generation, debugging, refactoring, testing support, and documentation updates. All generated code and text remain the responsibility of the project authors and should be reviewed critically before use.
