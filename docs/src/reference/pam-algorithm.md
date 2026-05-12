# PAM Algorithm

This note describes the high-level loop structure of the PAM reconstruction algorithm, for both CPU and GPU execution paths, and explains how the two implementations correspond to each other.

## What PAM Does

Starting from RF pressure data recorded at a receiver plane, PAM back-propagates the acoustic field one axial row at a time and accumulates the intensity `|p|^2` at each row to form an image. The heterogeneous sound-speed field is handled by the HASA correction. Without it, the method reduces to geometric ASA.

## CPU Path

The CPU reconstruction follows a direct frequency-by-frequency march:

```text
for each frequency bin f
    build propagator, correction, eta operators
    current <- FFT(p0)

    for each axial row
        for each axial sub-step
            p_space <- IFFT(current)
            if corrected
                conv_term <- FFT(eta[row] * p_space)
                next <- current * prop + corr * conv_term
            else
                next <- current * prop
            end
            current <- next
        end

        apply row weighting
        p_row <- IFFT(current)
        intensity[row] += |p_row|^2
    end
end
```

Key points:

- Loop nesting is `frequency -> row -> substep`.
- `eta = 1 - (c0 / c)^2` is the speed-contrast field driving the HASA term.
- Tukey weighting is applied once per row to suppress long-range numerical growth.
- Evanescent modes are zeroed before accumulation.

## GPU Path

The GPU version restructures the same computation by batching all selected frequency bins and time windows together into one array swept through the axial march.

```text
build p0_d with all frequency/window initial conditions
current_d <- FFT(p0_d)

for each axial row
    if corrected
        for each axial sub-step
            p_space_d <- IFFT(current_d)
            tmp_d <- k0^2 * eta[row] * p_space_d
            FFT(tmp_d)
            next_d <- current_d * prop_d + corr_d * tmp_d
            swap current_d and next_d
        end
    else
        current_d .*= prop_weight_d
    end

    p_row_d <- IFFT(current_d)
    accumulate |p_row_d|^2 across frequency/window batches
end
```

Key points:

- Frequency and window dimensions are packed into the second dimension of the GPU work array.
- Element-wise operations hit the full batch with one broadcast or kernel call.
- The final row weighting is fused into precomputed propagator/correction weights.
- The accumulated intensity is downloaded after the march instead of row by row.

## Windowed Reconstruction

`reconstruct_pam_windowed` wraps the single-window reconstruction:

1. Partition RF data into overlapping windows.
2. Skip windows below the configured energy threshold.
3. Reconstruct each active window.
4. Accumulate window intensity into the output image.

The inner ASA/HASA march is the same as a full reconstruction; windowing only controls which RF slices are fed into it and how their intensities are accumulated.

## Forward Simulation

`simulate_point_sources` and `simulate_point_sources_3d` produce synthetic RF data, usually through k-Wave. The PAM reconstruction loop consumes that RF data but does not call k-Wave directly.
