# 2D PAM Localization Accuracy — Schoen & Arvanitis (2020) Reproduction

**Setup:** 54 point sources (6 axial × 9 lateral, 30–80 mm depth, ±20 mm lateral),
simulated through a human skull CT slice (slice 250, skull at 20 mm) with k-Wave.
Frequency: 1 MHz, 10 cycles. Grid: 200 µm pitch, 40 ns time step.
Reconstructed with homogeneous ASA (geometric) and heterogeneous ASA (HASA, Eq. 6).

| Aperture | Geo (uncorrected)  | HASA (corrected)  | Paper HASA (Table II) |
|----------|--------------------|-------------------|-----------------------|
| 50 mm    | 10.1 ± 8.6 mm      | 1.1 ± 1.1 mm      | 1.2 ± 0.7 mm          |
| 75 mm    |  6.9 ± 6.0 mm      | 0.7 ± 0.5 mm      | 0.9 ± 0.5 mm          |
| 100 mm   |  5.4 ± 4.8 mm      | 0.6 ± 0.4 mm      | 0.8 ± 0.4 mm          |

HASA errors are within ~0.2 mm of the paper across all apertures.
Geometric errors are larger than the paper (3.7/2.5/3.5 mm), consistent with using a
single CT slice rather than an average over many skull positions.
