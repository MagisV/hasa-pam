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
- `src/pam.jl`: PAM configs, point-source models, skull/lens medium generation, reconstruction, metrics, and sweeps
- `src/kwave_wrapper.jl`: Julia-to-`k-wave-python` bridge
- `src/analysis.jl`: focusing analysis helpers and `run_focus_case`
- `scripts/run_focus_case.jl`: run one focusing case
- `scripts/compare_estimators.jl`: compare geometric and HASA focusing
- `scripts/run_pam_case.jl`: simulate point emitters and reconstruct them with PAM
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

The default CT path mirrors the local notebook setup:

`../Ultrasound/DIRU_20240404_human_skull_phase_correction_1_2_(skull_Normal)/DICOM/PAT_0000/STD_0000/SER_0002/OBJ_0001`

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
- `StochasticSource2D`
- `PAMConfig`
- `PAMWindowConfig`
- `SourceVariabilityConfig`

Key helpers:

- `fit_pam_config`
- `make_vascular_bubble_clusters`
- `make_burst_train_sources`
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

Default single-source example:

```bash
julia --project=. scripts/run_pam_case.jl \
  --sources-mm=30:0 \
  --aberrator=lens
```

Homogeneous-water baseline:

```bash
julia --project=. scripts/run_pam_case.jl \
  --sources-mm=30:0 \
  --aberrator=none
```

Single-source transcranial case:

```bash
julia --project=. scripts/run_pam_case.jl \
  --sources-mm=50:0 \
  --aberrator=skull \
  --slice-index=250 \
  --skull-transducer-distance-mm=30
```

Multiple point emitters with explicit phase and delay control:

```bash
julia --project=. scripts/run_pam_case.jl \
  --sources-mm=25:-6,32:0,40:8 \
  --phases-deg=0,90,180 \
  --delays-us=0,2,4 \
  --aberrator=lens
```

Vascular-like bubble aggregate with skull correction. `--clusters-mm` gives one or more vascular anchors; each anchor expands into many small harmonic bubble emitters along a paper-like squiggly vessel by default. The default aggregate analysis mode is detection, so `summary.json` reports precision/recall-style map recovery instead of one distance error per cluster. Squiggle and bundle topologies use a continuous centerline tube truth mask for detection; `--vascular-topology=tree` keeps the older branching stress case.
Cluster runs default to an 80 mm axial domain so activity below the skull is not clipped near 60 mm; override with `--axial-mm=...` if needed. The reconstruction reference speed is averaged over the receiver-to-source region, so extra trailing axial padding does not change the correction.

For vascular PAM, the default cluster workflow is an activity-area reconstruction rather than single-bubble localization. In `--recon-mode=auto`, `--cluster-model=vascular` selects windowed incoherent reconstruction, while point-source runs keep the original full-record reconstruction. The RF record is split into short tapered windows, low-energy windows are skipped, each remaining window is reconstructed with the existing geometric ASA and HASA code, and the output maps are accumulated as intensity:

```text
I_total(x,z) = mean_windows sum_f |p_hasa(x,z,f,window)|^2
```

The important detail is that windows are combined after `abs2`, not by summing complex pressure. This avoids preserving arbitrary phase interference between simultaneously active bubbles and instead estimates where acoustic activity occurred over the vascular region. The resulting vascular figures and detection metrics should be interpreted as accumulated activity or dose maps, not individual bubble position estimates.

The synthetic vascular source model also has a time-varying mode. `--activity-mode=burst-train` expands each physical vascular emitter into short delayed Gaussian pulse events within the gate. By default, every physical emitter can be active in each activity frame, with small amplitude and phase jitter across frames to decorrelate the windows. `summary.json` records both `physical_source_count` and `emission_event_count`.

```bash
julia --project=. scripts/run_pam_clusters.jl \
  --clusters-mm=54:0 \
  --cluster-model=vascular \
  --vascular-topology=squiggle \
  --vascular-length-mm=12 \
  --vascular-squiggle-amplitude-mm=1.5 \
  --vascular-squiggle-wavelength-mm=8 \
  --vascular-source-spacing-mm=0.8 \
  --vascular-radius-mm=1.0 \
  --detection-threshold-ratio=0.2 \
  --fundamental-mhz=0.5 \
  --harmonics=2,3 \
  --harmonic-amplitudes=1.0,0.6 \
  --gate-us=50 \
  --phase-mode=geometric \
  --aberrator=skull
```

Longer squiggle example in homogeneous water, using the windowed activity model explicitly:

```bash
julia --project=. scripts/run_pam_clusters.jl \
  --clusters-mm=45:0 \
  --cluster-model=vascular \
  --vascular-topology=squiggle \
  --vascular-length-mm=30 \
  --vascular-squiggle-amplitude-mm=1 \
  --vascular-squiggle-wavelength-mm=30 \
  --vascular-source-spacing-mm=0.2 \
  --vascular-min-separation-mm=0.15 \
  --vascular-max-sources-per-anchor=150 \
  --vascular-position-jitter-mm=0.5 \
  --vascular-radius-mm=1.0 \
  --fundamental-mhz=0.2 \
  --harmonics=2,4,6 \
  --harmonic-amplitudes=1.0,0.6,0.4 \
  --cavitation-model=gaussian-pulse \
  --gate-us=30 \
  --n-bubbles=1 \
  --phase-mode=geometric \
  --random-seed=1 \
  --recon-mode=windowed \
  --recon-window-us=10 \
  --recon-hop-us=5 \
  --recon-window-taper=hann \
  --recon-min-window-energy-ratio=0.001 \
  --recon-bandwidth-khz=150 \
  --activity-mode=burst-train \
  --activity-frame-us=10 \
  --activity-hop-us=5 \
  --activity-phase-jitter-rad=0.3 \
  --activity-amplitude-jitter=0.5 \
  --activity-active-probability=1.0 \
  --aberrator=none
```

Use `--vascular-topology=bundle --vascular-bundle-count=3 --vascular-bundle-spacing-mm=2.0` for a small set of parallel squiggly vessels, or `--vascular-topology=tree --vascular-branch-levels=2` for the previous branching model.

The one-emitter-per-aggregate behavior is available with `--cluster-model=point --analysis-mode=localization`.

The PAM run scripts write:

- `overview.png`
- `activity_boundaries.png` for vascular cluster runs, with threshold-dependent active-region boundaries overlaid on the heatmaps plus a quantitative metrics table
- `summary.json`
- `result.jld2`

To rerun only reconstruction and figure generation from saved RF data, pass an existing output folder:

```bash
julia --project=. scripts/run_pam_clusters.jl \
  --from-run-dir=outputs/20260504_135027_run_pam_clusters_skull_vascular_1anchors_68src_f0p5mhz_h23_geometric_ax200p0mm_slice250_st30p0mm \
  --recon-bandwidth-khz=20
```

`--from-run-dir` loads the previous `result.jld2`, reuses its RF data, medium, grid, and sources/clusters, and writes a fresh `outputs/<timestamp>_reconstruct_<old-folder>/` directory. Simulation-specific options such as source locations, medium/skull settings, grid size, time step, and GPU simulation are rejected in this mode; reconstruction and analysis options such as `--recon-bandwidth-khz`, `--recon-step-um`, `--recon-frequencies-mhz`, `--peak-method`, and cluster detection thresholds remain adjustable.

### Source Phase Modes

`--source-phase-mode` controls the physical regime being simulated and is reported in `summary.json`.

| Mode | Physical meaning |
|---|---|
| `coherent` | All sources share the same phase relation. Contributions add constructively/destructively by geometry. |
| `random_static_phase` | Each source draws a random phase once at setup and keeps it for the full simulation. |
| `random_phase_per_window` | Each source emits once per reconstruction window with fresh random phases. A **single** k-Wave simulation spans all windows; windowed reconstruction is forced automatically. |
| `random_phase_per_realization` | Each of `--n-realizations` k-Wave runs draws fresh random phases; intensity maps are averaged across runs. |
| `stochastic_broadband` | Each source is replaced with a `StochasticSource2D` that emits independent bandlimited noise centred on the cluster harmonics. |

**Coherent baseline** — sources lock in phase, single simulation:

```bash
julia --project=. scripts/run_pam_clusters.jl \
  --clusters-mm=30:0 \
  --cluster-model=point \
  --source-phase-mode=coherent \
  --phase-mode=coherent \
  --aberrator=none
```

**Random static phase** — fixed random phases, single simulation:

```bash
julia --project=. scripts/run_pam_clusters.jl \
  --clusters-mm=30:0 \
  --cluster-model=vascular \
  --vascular-topology=squiggle \
  --vascular-length-mm=12 \
  --source-phase-mode=random_static_phase \
  --phase-mode=random \
  --random-seed=42 \
  --aberrator=none
```

**Incoherent averaging over realizations** — 20 independent phase draws:

```bash
julia --project=. scripts/run_pam_clusters.jl \
  --clusters-mm=30:0 \
  --cluster-model=vascular \
  --vascular-topology=squiggle \
  --vascular-length-mm=12 \
  --source-phase-mode=random_phase_per_realization \
  --n-realizations=20 \
  --random-seed=42 \
  --aberrator=none
```

**Incoherent averaging per window** — single k-Wave run; each source gets fresh random phases per window:

```bash
julia --project=. scripts/run_pam_clusters.jl \
  --clusters-mm=30:0 \
  --cluster-model=vascular \
  --vascular-topology=squiggle \
  --vascular-length-mm=12 \
  --source-phase-mode=random_phase_per_window \
  --recon-window-us=10 \
  --recon-hop-us=5 \
  --random-seed=42 \
  --aberrator=none
```

Per-window source variability can be added to the same mode. Dropout is off by default with `--dropout-probability=0.0`; amplitudes are fixed by default with `--amplitude-distribution=fixed`; frequency jitter is off by default with `--frequency-jitter-percent=0.0`.

```bash
julia --project=. scripts/run_pam_clusters.jl \
  --clusters-mm=30:0 \
  --cluster-model=vascular \
  --vascular-topology=squiggle \
  --vascular-length-mm=12 \
  --source-phase-mode=random_phase_per_window \
  --recon-window-us=10 \
  --recon-hop-us=5 \
  --amplitude-distribution=lognormal \
  --amplitude-sigma=0.5 \
  --frequency-jitter-percent=5 \
  --dropout-probability=0.5 \
  --t-max-us=200 \
  --random-seed=42 \
  --aberrator=none
```
1
```bash
julia --project=. scripts/run_pam_clusters.jl \
  --clusters-mm=30:0 \
  --cluster-model=vascular \
  --vascular-topology=squiggle \
  --vascular-length-mm=12 \
  --source-phase-mode=random_phase_per_window \
  --recon-window-us=10 \
  --recon-hop-us=5 \
  --amplitude-distribution=lognormal \
  --amplitude-sigma=0.5 \
  --frequency-jitter-percent=5 \
  --dropout-probability=0 \
  --t-max-us=500 \
  --random-seed=42 \
  --aberrator=none
```

2

3
```bash
julia --project=. scripts/run_pam_clusters.jl \
  --clusters-mm=30:0 \
  --cluster-model=vascular \
  --vascular-topology=squiggle \
  --vascular-length-mm=12 \
  --source-phase-mode=random_phase_per_window \
  --recon-window-us=10 \
  --recon-hop-us=5 \
  --amplitude-distribution=fixed \
  --amplitude-sigma=0.5 \
  --frequency-jitter-percent=5 \
  --dropout-probability=0 \
  --t-max-us=500 \
  --random-seed=42 \
  --aberrator=none
```

4
```bash
julia --project=. scripts/run_pam_clusters.jl \
  --clusters-mm=30:0 \
  --cluster-model=vascular \
  --vascular-topology=squiggle \
  --vascular-length-mm=12 \
  --source-phase-mode=random_phase_per_window \
  --recon-window-us=10 \
  --recon-hop-us=5 \
  --amplitude-distribution=fixed \
  --amplitude-sigma=0.5 \
  --frequency-jitter-percent=0 \
  --dropout-probability=0 \
  --t-max-us=500 \
  --random-seed=42 \
  --aberrator=none
```

Supported amplitude distributions are `fixed`, `uniform`, `lognormal`, and `gaussian`. For `uniform`, `--amplitude-sigma` is the relative half-width around each source amplitude. For `lognormal`, it is the log-space standard deviation. For `gaussian`, it is the relative standard deviation and sampled amplitudes are clipped at zero. `--frequency-jitter-percent` applies a multiplicative jitter to each source fundamental frequency, so harmonic frequencies shift with it. The selected settings are written to `summary.json` under `source_variability`.

**Stochastic broadband** — each source emits independent noise centred on its harmonic frequencies:

```bash
julia --project=. scripts/run_pam_clusters.jl \
  --clusters-mm=30:0 \
  --cluster-model=vascular \
  --vascular-topology=squiggle \
  --vascular-length-mm=12 \
  --source-phase-mode=stochastic_broadband \
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
- vascular PAM source generation and detection metrics
- PAM medium fitting and skull placement
- PAM sweep aggregation and target filtering
- source phase mode normalisation, `StochasticSource2D` signal generation, phase resampling
- opt-in k-Wave smoke tests

## AI Usage

This project was developed with AI assistance. OpenAI Codex was used for code generation, debugging, refactoring, testing support, and documentation updates. All generated code and text remain the responsibility of the project authors and should be reviewed critically before use.
