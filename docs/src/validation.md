# Validation

Validation scripts live under `validation/`. They are separate from the standard unit test suite and are intended to reproduce or sanity-check larger workflows.

## 2D PAM Accuracy

`validation/2D_PAM_Accuracy/run_validation.jl` reproduces a point-source localization sweep inspired by Schoen and Arvanitis. It simulates a grid of sources through a skull medium with k-Wave and reconstructs with homogeneous ASA and heterogeneous HASA.

```bash
julia --project=. validation/2D_PAM_Accuracy/run_validation.jl
```

Use GPU reconstruction with:

```bash
julia --project=. validation/2D_PAM_Accuracy/run_validation.jl --recon-use-gpu
```

## 3D PAM Accuracy

`validation/3D_PAM_Accuracy/run_validation.jl` runs the 3D validation workflow. It is heavier than the unit tests and assumes the runtime environment has the required GPU and data dependencies.

```bash
julia --project=. validation/3D_PAM_Accuracy/run_validation.jl
```

## Tests Versus Validation

The standard test suite checks implementation behavior and smoke paths:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Validation scripts are larger reproducibility runs. They may require private CT data, a CUDA-capable GPU, and longer wall time.
