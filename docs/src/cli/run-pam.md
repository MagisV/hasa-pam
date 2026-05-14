# Running PAM

`scripts/run_pam.jl` simulates passive acoustic emissions and reconstructs source activity from receiver RF data. It supports 2D and 3D domains, homogeneous water controls, CT-backed skull media, sparse point sources, and generated vascular activity.

## Homogeneous Point Source

Use this when checking the installation or testing reconstruction behavior without CT data:

```bash
julia --project=. scripts/run_pam.jl \
  --source-model=point \
  --sources-mm=30:0 \
  --aberrator=none \
  --recon-use-gpu=false
```

For multiple 2D point emitters:

```bash
julia --project=. scripts/run_pam.jl \
  --source-model=point \
  --sources-mm=25:-6,32:0,40:8 \
  --phases-deg=0,90,180 \
  --delays-us=0,2,4 \
  --aberrator=none
```

## 3D Point Source

3D reconstruction currently uses the CUDA backend:

```bash
julia --project=. scripts/run_pam.jl \
  --dimension=3 \
  --source-model=point \
  --sources-mm=30:2:-1 \
  --aberrator=none \
  --recon-use-gpu=true
```

## Squiggle Activity

`--source-model=squiggle` expands each anchor into a generated vascular centerline and samples bubble emitters along it. In `--recon-mode=auto`, squiggle runs use windowed incoherent reconstruction.

```bash
julia --project=. scripts/run_pam.jl \
  --source-model=squiggle \
  --anchors-mm=45:0 \
  --aberrator=skull \
  --skull-transducer-distance-mm=30 \
  --slice-index=250 \
  --source-phase-mode=random_phase_per_window \
  --frequency-jitter-percent=1 \
  --recon-bandwidth-khz=500 \
  --recon-window-us=20 \
  --recon-hop-us=10 \
  --recon-progress=true
```

## 3D Network Activity

`--source-model=network` creates a random branching 3D centerline structure around each anchor and samples emitters inside an ellipsoid.

```bash
julia --project=. scripts/run_pam.jl \
  --dimension=3 \
  --source-model=network \
  --anchors-mm=45:0:0 \
  --network-root-count=12 \
  --network-generations=3 \
  --aberrator=skull \
  --skull-transducer-distance-mm=20 \
  --slice-index=250 \
  --recon-bandwidth-khz=40 \
  --recon-window-us=40 \
  --recon-hop-us=20 \
  --recon-use-gpu=true \
  --recon-progress=true
```

## Reconstruction-Only Mode

For 2D runs, `--from-run-dir` reuses a previous `result.jld2` and reruns reconstruction and analysis without rerunning k-Wave:

```bash
julia --project=. scripts/run_pam.jl \
  --from-run-dir=outputs/previous_pam_run \
  --recon-bandwidth-khz=20 \
  --recon-use-gpu=true
```

In this mode, simulation-specific options such as source locations, medium/skull settings, grid size, and time step are rejected. Reconstruction and analysis options remain adjustable.

## Choosing Important Parameters

Start with the source model and medium:

- Use `--source-model=point` for localization tests.
- Use `--source-model=squiggle` for generated vascular activity.
- Use `--source-model=network` for 3D branching activity.
- Use `--aberrator=none` for homogeneous controls.
- Use `--aberrator=skull` with `--ct-path`, `--slice-index`, and `--skull-transducer-distance-mm` for CT-backed transcranial cases.

Then tune reconstruction:

- `--recon-mode=auto` is usually the right default.
- `--recon-bandwidth-khz` is a major runtime/accuracy knob.
- `--recon-window-us` and `--recon-hop-us` control windowed incoherent reconstruction.
- `--window-batch` can improve GPU throughput for windowed reconstruction.

See [PAM CLI Parameters](@ref) for the generated option reference.
