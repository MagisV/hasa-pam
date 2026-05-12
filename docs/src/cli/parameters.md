# PAM CLI Parameters

This page is generated from the PAM CLI option metadata in `src/pam/setup/config.jl`.
Use options as `--name=value`; positional arguments are not supported.

The listed defaults are the base defaults. `scripts/run_pam.jl` applies a few model-aware overrides after parsing:

- `--dimension=3` defaults to `--source-model=point`, 3D coordinates, coarser `dy/dz`, and shorter `t-max-us` unless those options are provided.
- `--source-model=point` defaults to coherent phase, narrower receiver aperture, shorter duration, and `--recon-bandwidth-khz=0`.
- 3D `squiggle` and `network` runs default to windowed-friendly reconstruction settings such as `--recon-bandwidth-khz=40`, `--recon-window-us=40`, and `--recon-hop-us=20`.

For practical guidance, start with [Running PAM](@ref) before tuning individual parameters.

## General

| Option | Default | Value | Applies to | Choices | Description |
|---|---:|---|---|---|---|
| `--dimension` | `2` | 2\|3 | PAM | `2`, `3` | Selects the 2D or 3D PAM workflow. |
| `--source-model` | `squiggle` | point\|squiggle\|network | PAM | `point`, `squiggle`, `network` | Selects explicit point sources, a squiggly vascular source, or a synthetic 3D network. |
| `--from-run-dir` |   | path | 2D reconstruction only |   | Loads RF data, medium, grid, and sources from a previous output directory and reruns reconstruction/analysis only. |
| `--random-seed` | `42` | integer | PAM |   | Seed used for stochastic phases, source placement jitter, and generated vascular/network geometry. |
| `--benchmark` | `false` | bool | PAM |   | Prints additional timing information from simulation and reconstruction. |

## Source geometry

| Option | Default | Value | Applies to | Choices | Description |
|---|---:|---|---|---|---|
| `--sources-mm` | `30:0` | depth:lateral[,depth:lateral] or depth:y:z | point |   | Point source coordinates in millimeters. 2D uses depth:lateral; 3D uses depth:y:z. |
| `--anchors-mm` | `45:0` | depth:lateral[,depth:lateral] or depth:y:z | squiggle, network |   | Anchor coordinates for generated vascular or network activity. 2D uses depth:lateral; 3D uses depth:y:z. |
| `--transducer-mm` | `-30:0` | depth:lateral | 2D squiggle |   | Reference transducer position used when computing geometric source phases in 2D. |

## Source signal

| Option | Default | Value | Applies to | Choices | Description |
|---|---:|---|---|---|---|
| `--frequency-mhz` | `0.4` | MHz | point |   | Tone-burst frequency for point sources unless per-source frequencies are supplied. |
| `--fundamental-mhz` | `0.5` | MHz | squiggle, network |   | Fundamental activity frequency. Harmonic frequencies are integer multiples of this value. |
| `--amplitude-pa` | `1.0` | pressure | PAM |   | Default pressure amplitude for generated sources. |
| `--source-amplitudes-pa` |   | comma list | point |   | Optional per-point-source amplitudes. Use one value for all sources or one value per source. |
| `--source-frequencies-mhz` |   | comma list | point |   | Optional per-point-source frequencies in MHz. Use one value for all sources or one value per source. |
| `--phases-deg` |   | comma list | point |   | Optional per-point-source phases in degrees before phase-mode randomization. |
| `--delays-us` | `0` | comma list | PAM |   | Emission delays in microseconds. Use one value for all sources or one value per coordinate/anchor. |
| `--num-cycles` | `4` | integer | point |   | Number of cycles in each point-source tone burst. |
| `--harmonics` | `2,3,4` | comma list | squiggle, network |   | Harmonic orders emitted by generated bubble activity. |
| `--harmonic-amplitudes` | `1.0,0.6,0.3` | comma list | squiggle, network |   | Relative amplitude for each harmonic listed in --harmonics. |
| `--gate-us` | `50` | microseconds | squiggle, network |   | Duration of each activity emission gate. |
| `--taper-ratio` | `0.25` | fraction | squiggle, network |   | Tukey taper fraction applied to generated activity gates. |
| `--phase-mode` | `geometric` | coherent\|random\|jittered\|geometric | PAM | `coherent`, `random`, `jittered`, `geometric` | Controls initial source phases. Point sources accept coherent, random, and jittered; generated activity also uses geometric travel-time phases. |
| `--phase-jitter-rad` | `0.2` | radians | PAM |   | Standard deviation for jittered source phases. |
| `--source-phase-mode` | `random_phase_per_window` | coherent\|random_static_phase\|random_phase_per_window | PAM | `coherent`, `random_static_phase`, `random_phase_per_window` | Controls whether source phases are fixed or redrawn across reconstruction windows. |
| `--frequency-jitter-percent` | `1` | percent | squiggle, network |   | Multiplicative jitter applied to generated source fundamentals before harmonics are formed. |

## Vascular source

| Option | Default | Value | Applies to | Choices | Description |
|---|---:|---|---|---|---|
| `--vascular-length-mm` | `12` | mm | squiggle |   | Length of the generated squiggle centerline for each anchor. |
| `--vascular-squiggle-amplitude-mm` | `1.5` | mm | squiggle |   | Lateral squiggle amplitude in 2D, or y-amplitude in 3D. |
| `--vascular-squiggle-amplitude-x-mm` | `1.0` | mm | 3D squiggle |   | Depth-direction squiggle amplitude for 3D vascular sources. |
| `--vascular-squiggle-wavelength-mm` | `8` | mm | squiggle |   | Spatial wavelength of the generated squiggle path. |
| `--vascular-squiggle-slope` | `0.0` | slope | squiggle |   | Linear slope added to the generated squiggle path. |
| `--squiggle-phase-x-deg` | `90` | degrees | 3D squiggle |   | Phase offset for the 3D depth-direction squiggle component. |
| `--vascular-source-spacing-mm` | `0.5` | mm | squiggle, network |   | Approximate spacing between sampled bubble emitters along generated centerlines. |
| `--vascular-position-jitter-mm` | `0.05` | mm | squiggle |   | Random position jitter applied when sampling vascular sources. |
| `--vascular-min-separation-mm` | `0.25` | mm | squiggle, network |   | Minimum allowed distance between generated bubble emitters. |
| `--vascular-max-sources-per-anchor` | `0` | integer | squiggle |   | Caps generated sources per anchor. A value of 0 disables the cap. |

## Analysis

| Option | Default | Value | Applies to | Choices | Description |
|---|---:|---|---|---|---|
| `--vascular-radius-mm` | `1.0` | mm | squiggle, network |   | Truth radius used when scoring activity detection around generated sources. |
| `--peak-suppression-radius-mm` | `8.0` | mm | PAM |   | Radius used to suppress neighboring peaks during localization analysis. |
| `--success-tolerance-mm` | `1.5` | mm | PAM |   | Localization error threshold used when reporting success. |
| `--axial-gain-power` | `1.5` | power | 3D |   | Depth-gain exponent applied in 3D analysis/visualization. |
| `--analysis-mode` | `auto` | auto\|localization\|detection | PAM | `auto`, `localization`, `detection` | Selects localization or activity-detection metrics. Auto uses detection for squiggle/network sources. |
| `--detection-threshold-ratio` | `0.2` | ratio | detection |   | Single threshold ratio used by basic detection analysis. |
| `--boundary-threshold-ratios` | `0.5,0.55,0.6,0.65,0.7,0.75` | comma list | detection |   | Threshold ratios used for boundary overlays and threshold sweeps. |
| `--auto-threshold-search` | `true` | bool | detection |   | Searches a dense threshold range and selects representative detection thresholds. |
| `--auto-threshold-min` | `0.10` | ratio | detection |   | Minimum threshold ratio for automatic threshold search. |
| `--auto-threshold-max` | `0.95` | ratio | detection |   | Maximum threshold ratio for automatic threshold search. |
| `--auto-threshold-step` | `0.01` | ratio | detection |   | Threshold ratio spacing for automatic threshold search. |

## Network source

| Option | Default | Value | Applies to | Choices | Description |
|---|---:|---|---|---|---|
| `--network-axial-radius-mm` | `10.0` | mm | 3D network |   | Axial radius of the ellipsoid used to clip generated network activity. |
| `--network-lateral-y-radius-mm` | `1.5` | mm | 3D network |   | Y radius of the generated network ellipsoid. |
| `--network-lateral-z-radius-mm` | `1.5` | mm | 3D network |   | Z radius of the generated network ellipsoid. |
| `--network-root-count` | `12` | integer | 3D network |   | Number of root branches grown for each network center. |
| `--network-generations` | `3` | integer | 3D network |   | Number of branching generations in the synthetic network. |
| `--network-branch-length-mm` | `5.0` | mm | 3D network |   | Nominal length of each generated branch segment. |
| `--network-branch-step-mm` | `0.4` | mm | 3D network |   | Sampling step along generated network branches. |
| `--network-branch-angle-deg` | `36` | degrees | 3D network |   | Nominal branching angle for synthetic network growth. |
| `--network-tortuosity` | `0.18` | fraction | 3D network |   | Strength of random branch curvature in the synthetic network. |
| `--network-orientation` | `isotropic` | isotropic\|horizontal\|axial | 3D network | `isotropic`, `horizontal`, `axial` | Orientation prior for generated network branches. |
| `--network-density-sigma-mm` | `0` | mm | 3D network |   | Optional isotropic Gaussian density sigma. A value of 0 uses the anisotropic sigma options. |
| `--network-density-axial-sigma-mm` | `10.0` | mm | 3D network |   | Axial Gaussian density sigma for network source sampling. |
| `--network-density-lateral-y-sigma-mm` | `1.5` | mm | 3D network |   | Y Gaussian density sigma for network source sampling. |
| `--network-density-lateral-z-sigma-mm` | `1.5` | mm | 3D network |   | Z Gaussian density sigma for network source sampling. |
| `--network-max-sources-per-center` | `80` | integer | 3D network |   | Caps generated sources per network center. Values <= 0 disable the cap. |

## Grid

| Option | Default | Value | Applies to | Choices | Description |
|---|---:|---|---|---|---|
| `--axial-mm` | `80` | mm | PAM |   | Requested axial domain depth. The runner may extend this to fit sources and time of flight. |
| `--transverse-mm` | `102.4` | mm | PAM |   | Default lateral domain width. In 3D this seeds y and z widths unless overridden. |
| `--transverse-y-mm` |   | mm | 3D |   | Overrides the 3D y-width when set. |
| `--transverse-z-mm` |   | mm | 3D |   | Overrides the 3D z-width when set. |
| `--dx-mm` | `0.2` | mm | PAM |   | Axial grid spacing. |
| `--dy-mm` |   | mm | 3D |   | 3D y grid spacing. Defaults to --dz-mm when omitted. |
| `--dz-mm` | `0.2` | mm | PAM |   | 2D lateral spacing or 3D z spacing. |
| `--t-max-us` | `500` | microseconds | PAM |   | Requested simulation duration. The runner may extend this when needed to capture source arrivals. |
| `--dt-ns` | `20` | nanoseconds | PAM |   | Simulation time step. |
| `--zero-pad-factor` | `4` | integer | PAM |   | Lateral FFT zero-padding factor used by ASA/HASA reconstruction. |
| `--bottom-margin-mm` | `10` | mm | PAM |   | Minimum margin below the deepest source when auto-fitting the PAM domain. |

## Receiver

| Option | Default | Value | Applies to | Choices | Description |
|---|---:|---|---|---|---|
| `--receiver-aperture-mm` | `full` | mm\|full | PAM |   | Receiver aperture width. Use full, all, or none to use the whole receiver plane. |
| `--receiver-aperture-y-mm` |   | mm\|full | 3D |   | Overrides the 3D receiver aperture in y. |
| `--receiver-aperture-z-mm` |   | mm\|full | 3D |   | Overrides the 3D receiver aperture in z. |

## Medium

| Option | Default | Value | Applies to | Choices | Description |
|---|---:|---|---|---|---|
| `--aberrator` | `none` | none\|water\|skull | PAM | `none`, `water`, `skull` | Selects homogeneous water/no aberrator or a CT-derived skull medium. |
| `--ct-path` | `/Users/vm/Desktop/OBJ_0001` | path | skull |   | Path to the private DICOM folder used for CT-backed skull media. |
| `--slice-index` | `250` | integer | skull |   | CT slice index used when building the skull medium. |
| `--skull-transducer-distance-mm` | `30` | mm | skull |   | Distance from the receiver/transducer plane to the outer skull surface. |
| `--hu-bone-thr` | `200` | HU | skull |   | Hounsfield-unit threshold used to identify bone in CT data. |

## Simulation

| Option | Default | Value | Applies to | Choices | Description |
|---|---:|---|---|---|---|
| `--simulation-backend` | `kwave` | kwave\|analytic | PAM | `kwave`, `analytic` | Forward model backend. CT skull runs require k-Wave. |
| `--kwave-use-gpu` | `true` | bool | k-Wave |   | Passes GPU execution to k-Wave where supported. |

## Reconstruction

| Option | Default | Value | Applies to | Choices | Description |
|---|---:|---|---|---|---|
| `--recon-use-gpu` | `true` | bool | PAM |   | Uses the CUDA.jl reconstruction backend. 3D reconstruction currently requires this to be true. |
| `--recon-bandwidth-khz` | `500` | kHz | PAM |   | Half-width bandwidth used to select frequency bins around reconstruction frequencies. Use 0 to keep only the target bins. |
| `--recon-step-um` | `50` | micrometers | PAM |   | Axial integration step used by ASA/HASA reconstruction. |
| `--recon-mode` | `auto` | auto\|full\|windowed | PAM | `auto`, `full`, `windowed` | Reconstruction mode. Auto uses full for point sources and windowed for squiggle/network activity. |
| `--recon-window-us` | `20` | microseconds | windowed |   | Window duration for windowed incoherent reconstruction. |
| `--recon-hop-us` | `10` | microseconds | windowed |   | Hop between consecutive reconstruction windows. |
| `--recon-window-taper` | `hann` | hann\|none\|rectangular\|tukey | windowed | `hann`, `none`, `rectangular`, `tukey` | Taper applied to each reconstruction window. |
| `--recon-min-window-energy-ratio` | `0.001` | ratio | windowed |   | Skips windows whose energy is below this fraction of the maximum window energy. |
| `--recon-progress` | `false` | bool | PAM |   | Prints reconstruction progress updates. |
| `--window-batch` | `1` | integer | windowed GPU |   | Number of reconstruction windows batched together on the GPU. |

