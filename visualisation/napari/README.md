# napari PAM convergence viewer

## Dependencies

```
pip install napari[all] h5py imageio imageio-ffmpeg
```

## Usage

### 1. Export (once per run)

```powershell
julia --project=. visualisation/napari/export_for_napari.jl `
  --run-dir=visualisation/outputs/20260509_170538_window_convergence/rerender_2
```

Reads `data.jld2` and `recons/window_XXXX.jld2` from the run folder,
writes `napari_data.h5`.

### 2. Interactive viewer (slider to scrub)

```powershell
python visualisation/napari/show_convergence.py `
  --run-dir=visualisation/outputs/20260509_170538_window_convergence/rerender_2
```

Opens napari in 3D mode.  The cumulative HASA reconstruction is a 4-D layer
`(window × depth × y × z)` — use the slider to scrub through convergence.

### 3. Cinematic animation

```powershell
python visualisation/napari/show_convergence.py `
  --run-dir=visualisation/outputs/20260509_170538_window_convergence/rerender_2 `
  --animate --fps 24 --orbit-frames 72
```

Saves `napari_cinematic.mp4` in the run folder.

Two phases:
- **Phase 1** — skull orbit: camera rotates 360° over `--orbit-frames` frames
- **Phase 2** — convergence build-up: camera locks, steps through each window

## Options

| Flag | Default | Description |
|---|---|---|
| `--run-dir` | *(required)* | Path to the window convergence run folder |
| `--animate` | off | Generate MP4 instead of opening interactive viewer |
| `--fps` | 24 | Animation frame rate |
| `--orbit-frames` | 72 | Frames for the skull-orbit intro (72 = 3 s at 24 fps) |
