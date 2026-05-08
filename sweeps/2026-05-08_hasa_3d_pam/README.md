# HASA 3D PAM Tight-Band Vessel Sweep

Date: 2026-05-08

This sweep followed the earlier bandwidth/jitter result for 3D skull-aberrated squiggle vascular sources. The fixed baseline was:

- `recon-bandwidth-khz=40`
- `frequency-jitter-percent=1`
- `recon-window-us=40`
- `recon-hop-us=20`
- skull aberrator, `slice-index=250`, `skull-transducer-distance-mm=20`
- squiggle anchor baseline `42:0:0`, length `12 mm`, y/x squiggle amplitudes `0.3/0.2 mm`

Raw run directories:

- Hop continuation: `outputs/20260508_150710_pam_3d_jitter10_bandwidth_sweep`
- Vessel geometry/location sweep: `outputs/20260508_vessel_geometry_location_sweep`
- Weak-location rescue sweep: `outputs/20260508_location_rescue_sweep`

Bulky `result.jld2` and napari `.npz` arrays from the newly generated vessel/rescue runs were removed after summaries were written because the disk filled during the rescue sweep. Per-run `summary.json`, status files, logs, CSVs, and figures remain.

## Hop-Size Result

Shorter hop did **not** recover the `60 kHz / 2%` F1 gap.

| case | F1 | precision | recall | threshold | elapsed min |
|---|---:|---:|---:|---:|---:|
| `fjitter2_bw60` | 0.678 | 0.536 | 0.923 | 0.66 | 3.17 |
| `fjitter1_bw40` | 0.660 | 0.526 | 0.885 | 0.66 | 3.00 |
| `fjitter1_bw40_hop40` | 0.664 | 0.564 | 0.808 | 0.67 | 3.01 |
| `fjitter1_bw40_hop10` | 0.558 | 0.400 | 0.923 | 0.30 | 3.09 |
| `fjitter1_bw40_hop5` | 0.625 | 0.472 | 0.923 | 0.46 | 3.42 |

`hop=40 us` gives a tiny F1 bump over `hop=20 us`, but loses recall. The shorter-overlap runs are worse. I would keep the README/default setup at `40 kHz / 1% / hop 20 us`.

## Vessel Size And Shape

The current setup is robust to moderate centerline changes, but the exact scoring radius matters a lot:

- Baseline reproduces the prior tight-band result: F1 `0.660`.
- Denser sampling improves slightly: spacing `0.3 mm` gives F1 `0.680`.
- Straight vessels are easier: F1 `0.724`.
- Wider/high-curvature/long-wave squiggles stay in the same ballpark: F1 `0.645` to `0.674`.
- The `vascular-radius-mm` cases are metric sensitivity checks, not a physical RF change. Scoring at `0.5 mm` drops F1 to `0.279`; scoring at `1.5 mm` raises it to `0.835`.

## Location Sweep

Location dominates the failure modes:

- Shallow centered depth `32 mm` is excellent: F1 `0.828`.
- Baseline depth `42 mm` is decent: F1 `0.660`.
- Deeper centered vessels degrade hard: depth `52 mm` F1 `0.481`, depth `62 mm` F1 `0.376`.
- Lateral behavior is asymmetric. Negative-y and negative-z offsets are mostly acceptable, while positive-y and positive-y diagonal offsets are weak:
  - `42:-9:0`: F1 `0.716`
  - `42:9:0`: F1 `0.450`
  - `42:12:12`: F1 `0.429`

This looks more like skull/aperture/location geometry than a globally wrong bandwidth/jitter setting.

## Rescue Check

I reran weak locations with:

- `60 kHz / 2%`
- `80 kHz / 4%`

Those settings help some cases but do not fix the weak region globally:

| location | current 40/1 F1 | best rescue | best rescue F1 | conclusion |
|---|---:|---|---:|---|
| `52:0:0` | 0.481 | `80 kHz / 4%` | 0.538 | modest improvement |
| `62:0:0` | 0.376 | `80 kHz / 4%` | 0.403 | still poor |
| `42:9:0` | 0.450 | `80 kHz / 4%` | 0.494 | modest improvement |
| `42:18:0` | 0.491 | `60 kHz / 2%` | 0.545 | modest improvement |
| `42:12:-12` | 0.488 | `80 kHz / 4%` | 0.524 | modest improvement |
| `42:12:12` | 0.429 | `80 kHz / 4%` | 0.496 | modest improvement |

I would not replace the global default based on this. For difficult positive-y/deep targets, a local rescue setting around `80 kHz / 4%` is worth trying, but the bigger next question is why the skull/location geometry is so asymmetric.

## Decision

Keep `40 kHz / 1% jitter / 20 us hop` as the global sweet spot.

Use `60 kHz / 2%` as the high-F1 reference and `80 kHz / 4%` as a local rescue candidate for weak locations, not as the default.

Next useful work would be to inspect the CT skull slice/aperture geometry and test whether the positive-y failures track skull thickness, receiver aperture placement, or a coordinate/sign asymmetry.
