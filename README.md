[![Tests](https://github.com/MagisV/hasa-pam/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/MagisV/hasa-pam/actions/workflows/tests.yml)

# TranscranialFUS

Julia project for transcranial ultrasound focusing and passive acoustic mapping (PAM), developed for the ETH Zurich course "Solving PDEs in parallel on GPUs with Julia II".

The repository is meant to be used primarily from the command line. The maintained PAM entry point is `scripts/run_pam.jl`; focusing scripts are kept for the earlier transcranial focusing workflow.

## Documentation

The detailed user documentation lives in `docs/` and is built with Documenter.jl.

- Start with `docs/src/getting-started.md`.
- Use `docs/src/cli/run-pam.md` for PAM examples.
- Use `docs/src/cli/parameters.md` for the generated PAM parameter reference after building the docs.
- Use `docs/src/workflow.md` for the high-level source, medium, simulation, reconstruction, and analysis flow.

Build the documentation locally with:

```bash
julia --project=docs docs/make.jl
```

### Hosted docs (GitHub Pages)

The site is built and deployed automatically when changes land on `main`
([Deploy Documentation](https://github.com/MagisV/hasa-pam/actions/workflows/deploy-docs.yml)).

Enable it once in the GitHub repository: **Settings → Pages → Build and deployment → Source → GitHub Actions**.

For a public repository under `MagisV`, the published URL is typically:

`https://magisv.github.io/hasa-pam/`

GitHub Pages is free on public repositories. With [GitHub Education](https://education.github.com/), you may also get Pro benefits that help with private repositories or higher usage limits.

## Setup

Instantiate the Julia environment:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Python-side dependencies for k-Wave are managed through `CondaPkg.toml`. The first k-Wave run may resolve Python packages and k-Wave resources.

## Data

The CT scan data used for skull-backed examples is private and is not distributed with the repository. Homogeneous water runs with `--aberrator=none` do not need CT data. For skull-backed runs, pass:

```bash
--ct-path=/path/to/dicom-folder
```

## Quick PAM Example

Small homogeneous 2D point-source run:

```bash
julia --project=. scripts/run_pam.jl \
  --source-model=point \
  --sources-mm=30:0 \
  --aberrator=none \
  --recon-use-gpu=false
```

Simple 3D point-source run on CUDA:

```bash
julia --project=. scripts/run_pam.jl \
  --dimension=3 \
  --source-model=point \
  --sources-mm=30:2:-1 \
  --aberrator=none \
  --recon-use-gpu=true
```

## Quick Focusing Example

```bash
julia --project=. scripts/run_focus.jl \
  --estimator=hasa \
  --medium=skull_in_water \
  --slice-index=250
```

## Tests

Run the standard test suite:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Enable optional k-Wave smoke tests:

```bash
TRANSCRANIALFUS_RUN_KWAVE_TESTS=1 julia --project=. -e 'using Pkg; Pkg.test()'
```

Enable the heavier CT-backed integration tests as well:

```bash
TRANSCRANIALFUS_RUN_INTEGRATION=1 TRANSCRANIALFUS_RUN_KWAVE_TESTS=1 julia --project=. -e 'using Pkg; Pkg.test()'
```

## AI Usage

This project was developed with AI assistance. Generated code and text should be reviewed critically before use.
