# Troubleshooting

## The CLI Rejects My Argument

Arguments must use `--name=value`. Space-separated forms such as `--name value` are not supported.

For PAM, see [PAM CLI Parameters](@ref) for supported options. Unknown options are accepted by the low-level parser, but they only matter if the runner reads them.

## CT Data Is Missing

Skull-backed runs need private DICOM data. Use homogeneous water controls while testing setup:

```bash
julia --project=. scripts/run_pam.jl \
  --source-model=point \
  --sources-mm=30:0 \
  --aberrator=none
```

For skull runs, pass the data location explicitly:

```bash
--ct-path=/path/to/dicom-folder
```

## CUDA Is Not Available

If CUDA.jl cannot see a GPU, set `--recon-use-gpu=false` for supported 2D runs. 3D reconstruction currently requires the CUDA path.

k-Wave simulation GPU usage is controlled separately with `--kwave-use-gpu`.

## k-Wave Or Python Setup Fails

Python dependencies are managed through `CondaPkg.toml`. The first k-Wave run may take longer because Python packages and k-Wave resources are resolved.

If setup fails, try instantiating the Julia environment first:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Then run a homogeneous point-source example before attempting CT-backed skull simulations.

## A Run Takes Too Long

The largest runtime controls are grid size, dimensionality, reconstruction bandwidth, window count, and GPU usage.

For faster iteration:

- Start with `--aberrator=none`.
- Use `--source-model=point`.
- Reduce `--axial-mm` and `--transverse-mm`.
- Use tighter `--recon-bandwidth-khz`.
- Use `--recon-progress=true` to confirm reconstruction is moving.

## Reconstruction-Only Reruns Fail

`--from-run-dir` reuses the previous simulation data. Remove options that would change the simulation, source geometry, medium, or grid. Keep only reconstruction and analysis options such as `--recon-bandwidth-khz`, `--recon-step-um`, `--recon-use-gpu`, and threshold settings.
