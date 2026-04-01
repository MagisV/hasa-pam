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

The PAM path implements a 2D passive reconstruction workflow based on simple point emitters:

- forward propagation simulated with `k-wave-python`
- geometric ASA reconstruction
- corrected HASA reconstruction
- localization and image-quality metrics such as axial, lateral, and radial error, FWHM, peak intensity, and success rate

Core types:

- `PointSource2D`
- `PAMConfig`

Key helpers:

- `fit_pam_config`
- `make_pam_medium`
- `simulate_point_sources`
- `reconstruct_pam`
- `find_pam_peaks`
- `analyse_pam_2d`
- `run_pam_case`
- `run_pam_sweep`

### PAM Medium Options

Supported aberrators:

- `--aberrator=none`: homogeneous water
- `--aberrator=lens`: simple elliptical speed perturbation
- `--aberrator=skull`: CT-derived skull inserted into the PAM domain

For `--aberrator=skull`:

- the receiver plane stays at the top of the physical domain
- the outer skull surface is placed `--skull-transducer-distance-mm` below the receiver, default `30 mm`
- source coordinates stay defined relative to the transducer/receiver, not relative to the skull
- `fit_pam_config` extends the axial domain automatically to fit the deepest source plus `--bottom-margin-mm`

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

`run_pam_case.jl` writes:

- `overview.png`
- `summary.json`
- `result.jld2`

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
- PAM medium fitting and skull placement
- PAM sweep aggregation and target filtering
- opt-in k-Wave smoke tests

## AI Usage

This project was developed with AI assistance. OpenAI Codex was used for code generation, debugging, refactoring, testing support, and documentation updates. All generated code and text remain the responsibility of the project authors and should be reviewed critically before use.
