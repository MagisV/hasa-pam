# Getting Started

## Requirements

Use Julia 1.12 with the project environment at the repository root.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The forward simulation path uses Python packages managed by `CondaPkg.toml`, including `k-wave-python`, `numpy`, and `scipy`. The first call into the k-Wave wrapper may resolve Python packages and k-Wave resources.

CUDA-backed reconstruction requires an NVIDIA GPU visible to CUDA.jl. 3D PAM reconstruction currently requires `--recon-use-gpu=true`.

## CT Data

The private CT data used during development is not distributed with this repository. By default, scripts look for a DICOM folder under:

```text
~/Desktop/OBJ_0001
```

Override this with `--ct-path=/path/to/dicom-folder` for skull-backed runs. Homogeneous runs with `--aberrator=none` do not need CT data.

## First PAM Run

A small homogeneous point-source run is the fastest way to confirm the CLI works:

```bash
julia --project=. scripts/run_pam.jl \
  --source-model=point \
  --sources-mm=30:0 \
  --aberrator=none \
  --recon-use-gpu=false
```

For 3D, use a CUDA-capable machine:

```bash
julia --project=. scripts/run_pam.jl \
  --dimension=3 \
  --source-model=point \
  --sources-mm=30:2:-1 \
  --aberrator=none \
  --recon-use-gpu=true
```

## Tests

Run the standard test suite from the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Optional k-Wave and CT-backed tests are controlled by environment variables documented in the repository README.
