# PAM Window Convergence Visualisation

This folder contains a self-contained 3D PAM animation pipeline. It generates a
synthetic vascular-network source distribution, simulates RF data, reconstructs
the case window by window with HASA, chooses the final source-F1-optimal
threshold, and renders a fixed-threshold convergence movie.

## Run

```powershell
julia --project=. visualisation/make_window_convergence.jl
```

Default case:

- 3D network source at `45:0:0 mm`
- CT skull aberrator, `slice-index=250`, skull distance `20 mm`
- `dx=0.2 mm`, `dy=0.2 mm`, `dz=0.2 mm`
- `t_max=300 us`
- `40 us` windows, `20 us` hop
- `40 kHz` reconstruction bandwidth
- random source phases resampled per reconstruction window
- GPU k-Wave and GPU reconstruction

The default run is intentionally heavy. A quick water smoke test is:

```powershell
julia --project=. visualisation/make_window_convergence.jl --aberrator=none --max-windows=2
```

Check the configuration without running simulation/reconstruction:

```powershell
julia --project=. visualisation/make_window_convergence.jl --dry-run=true
```

## Outputs

Outputs are written to:

```text
visualisation/outputs/<timestamp>_window_convergence/
```

The folder contains:

- `data.jld2`: RF data, medium arrays, final cumulative volume, truth mask,
  threshold search results, selected windows, and per-frame metrics
- `summary.json`: human-readable run summary
- `frames/frame_0001.png`, etc.
- `pam_window_convergence.mp4`

## Threshold Story

The script first reconstructs all selected windows and computes the final
cumulative HASA volume. It searches thresholds from `0.10:0.01:0.95`, chooses
the threshold with the best final source F1, and freezes the corresponding
absolute intensity cutoff for every frame.

Each frame therefore uses the same detection boundary. The orange region shows
the cumulative prediction above that fixed cutoff, white contours show the truth
mask, blue contours show the current-window contribution, and source points are
colored by detected versus not-yet-detected.

## ffmpeg

Install `ffmpeg` and make sure it is available on PATH to create the MP4. If it
is missing, the script still writes all PNG frames and prints the exact encoding
command.

Useful options:

```powershell
julia --project=. visualisation/make_window_convergence.jl `
  --out-dir=visualisation/outputs/test_run `
  --t-max-us=300 `
  --dy-mm=0.2 `
  --dz-mm=0.2 `
  --max-windows=0 `
  --fps=12 `
  --frames-only=false
```

To reuse an existing k-Wave simulation and medium cache:

```powershell
julia --project=. visualisation/make_window_convergence.jl `
  --from-data=visualisation/outputs/<timestamp>_window_convergence/data.jld2 `
  --out-dir=visualisation/outputs/rerender_attempt
```

`--from-data` skips medium construction and k-Wave RF simulation. It still
reruns the window reconstructions so you can adjust rendering and threshold
logic without paying the forward-simulation cost again.
