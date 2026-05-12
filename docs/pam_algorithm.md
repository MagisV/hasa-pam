# PAM Algorithm — High-Level Loop Structure

This document describes the loop structure of the PAM (Photoacoustic Microscopy)
reconstruction algorithm, for both the CPU and GPU execution paths, and explains
how the two implementations correspond to each other.

---

## What PAM does

Starting from RF pressure data recorded at a receiver plane, PAM back-propagates
the acoustic field one axial row at a time and accumulates the intensity `|p|²`
at each row to form an image.  The heterogeneous sound-speed field is handled by
the HASA correction; without it the method reduces to geometric ASA (a purely
linear phase shift).

---

## CPU path — `_reconstruct_pam_cpu_3d` / `reconstruct_pam` (CPU branch)

```
for each frequency bin f                         # outer: spectral
    build propagator, correction, eta operators

    current ← FFT(p₀)                           # initial condition from RF

    for each axial row (rr+1 → row_stop)        # outer spatial: depth march
        for each sub-step (1 → axial_substeps)  # inner: stability substeps

            p_space ← IFFT(current)             # back to spatial domain
            ┌ HASA (corrected=true):
            │   conv_term ← FFT(η[row] · p_space)
            │   next ← current · prop + corr · conv_term
            └ ASA  (corrected=false):
                next ← current · prop

            current ← next

        apply Tukey weighting once per row

        p_row ← IFFT(current)
        intensity[row] += |p_row|²              # accumulate

    (3D only) crop lateral window after row loop
```

**Loop nesting (2D):** `freq → row → substep`  
**Loop nesting (3D):** `freq → row → substep → (iy, iz) for crop`

Key points:
- Everything is serial; FFTW plans are measured once per call and reused.
- `eta = 1 − (c₀/c)²` — the speed-contrast field driving the HASA term.
- The Tukey weighting is applied once *per row* (not per substep) to suppress
  long-range numerical growth without coupling damping to substep count.
- Evanescent modes (imaginary `k_axial`) are zeroed before accumulation.

---

## GPU path — `_reconstruct_pam_cuda`

The GPU version restructures the same computation to maximise memory throughput
by **batching all frequency bins and all time-windows together** into a single
2-D array swept through the axial march.

```
build p0_d (padded_ny, nfreq×W)                # pack ALL freq×window ICs at once

current_d ← FFT(p0_d, lateral_dim)            # batched lateral FFT

for each axial row (rr+1 → row_stop)           # same depth march as CPU
    ┌ HASA (corrected=true):
    │   for each sub-step (1 → axial_substeps-1)
    │       p_space_d ← IFFT(current_d)                     # HASA
    │       tmp_d ← k0²_d · η[row] · p_space_d             # HASA
    │       FFT(tmp_d)                                       # HASA
    │       next_d ← current_d · prop_d + corr_d · tmp_d   # HASA
    │       swap current_d ↔ next_d
    │   final sub-step (uses prop_weight_d, corr_weight_d)
    └ ASA (corrected=false):
        current_d .*= prop_n_weight_d           # single element-wise multiply

    p_row_d ← IFFT(current_d)
    _accum_abs2_sum_batched! kernel:            # custom CUDA kernel
        for each lateral index i (one GPU thread):
            for each window w:
                for each freq f within window:
                    intensity[i, w, row] += |p_row_d[i, f]|²
```

**Loop nesting:** `row → substep` (on CPU dispatch) → GPU threads handle all
lateral positions `i` in parallel; frequency and window loops inside the kernel.

Key points:
- `(padded_ny, nfreq×W)` layout means every element-wise op hits the full
  batch with a single cuBLAS/broadcast call — no per-frequency dispatch overhead.
- `eta_yx_d` (2D) / `eta_yznx_d` (3D) is sliced per row as `[:, row:row]`;
  broadcasting makes this zero-copy.
- The final substep uses pre-fused `prop_weight_d = prop_d .* weight_d` to save
  one kernel launch.
- The `_accum_abs2_sum_batched!` kernel accumulates across the frequency
  dimension *on the GPU*, so only the finished `intensity_yWx_d` is downloaded.

---

## CPU → GPU correspondence

| CPU concept | GPU equivalent | Notes |
|---|---|---|
| `for (freq, bin) in ...` outer loop | Packed into dim-2 of `current_d` | All freqs live in one array; no dispatch loop |
| `for w in windows` (windowed API) | Packed into same dim-2 as `W` blocks | `nfreq_W = nfreq × W` columns |
| `ifft(current)` | `plan_bwd * p_space_d` | In-place cuFFT on the whole batch |
| `fft(η[row] .* p_space)` | `tmp_d .= k0²_d .* η[row:row] .* p_space_d; plan_fwd * tmp_d` | Row slice broadcasts over all freq×window columns |
| `current .* prop .+ corr .* conv_term` | `next_d .= current_d .* prop_d .+ corr_d .* tmp_d` | Element-wise; `prop_d` / `corr_d` are tiled `W` times at setup |
| `current .*= weighting` once per row | Fused into `prop_weight_d`, `corr_weight_d`, `prop_n_weight_d` | Saves a separate broadcast pass |
| `out[row, :] .+= abs2.(p_row)` | `_accum_abs2_sum_batched!` CUDA kernel | Kernel reduces over freq inside GPU; result goes to `intensity_yWx_d[:, :, row]` |
| Inner `for _ in 1:axial_substeps` | Same loop on CPU dispatch thread | Substep loop stays on host; each iteration launches batch GPU ops |
| CPU downloads result each row | Single download after all rows | `Array(intensity_yWx_d)` + permutedims at the end |

---

## Windowed reconstruction

`reconstruct_pam_windowed` wraps both paths:

1. Partition the full RF time axis into overlapping windows.
2. Skip windows below an energy threshold.
3. For GPU + `window_batch > 1`: build the CUDA setup once, then submit batches
   of `window_batch` windows at a time through `_reconstruct_pam_cuda`.
4. Accumulate each batch's intensity into the running `intensity` sum.

The inner reconstruction is identical to the single-window case — the windowing
layer only controls *which* slices of RF are fed in and how results are summed.

---

## Wave propagation

Each call to `simulate_point_sources` / `simulate_point_sources_3d` runs a
k-Wave acoustic simulation (**HASA**) to produce the synthetic RF data that the
reconstruction consumes.  The PAM loop structure above is the *reconstruction*
side; it never calls k-Wave directly.
