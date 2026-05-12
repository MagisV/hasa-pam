# Cluster Emission Models

Source implementations are in `src/pam/2d/sources.jl` and `src/pam/3d/sources3d.jl`.

## Bubble Cluster

For squiggle and network activity, generated bubble clusters emit a tapered harmonic signal:

```math
s(t) = A \cdot w_\mathrm{Tukey}(t; T_\mathrm{gate}, r) \cdot \sum_{n \in H} \alpha_n \cos(2\pi n f_0 t + \phi_n)
```

The signal is active during the configured gate and zero outside it.

## Parameters

| Symbol | Field | 2D default | 3D default |
|---|---|---:|---:|
| `f0` | `fundamental` | 500 kHz | 500 kHz |
| `A` | `amplitude` | 1.0 | 1.0 |
| `H` | `harmonics` | `{2, 3}` | `{2, 3, 4}` |
| `alpha_n` | `harmonic_amplitudes` | `[1.0, 0.6]` | `[1.0, 0.6, 0.3]` |
| `phi_n` | `harmonic_phases` | `[0.0, 0.0]` | `[0.0, 0.0, 0.0]` |
| `T_gate` | `gate_duration` | 50 us | 50 us |
| `r` | `taper_ratio` | 0.25 | 0.25 |

## Term Glossary

| Term | Meaning |
|---|---|
| `t` | Time measured from source onset after subtracting source delay. |
| `f0` | Fundamental emission frequency; harmonics are integer multiples of it. |
| `n` | Harmonic order. |
| `A` | Cluster pressure amplitude on a linear scale. |
| `alpha_n` | Relative amplitude of harmonic `n`. |
| `phi_n` | Phase of harmonic `n`, set by the selected phase mode. |
| `T_gate` | Total emission duration. |
| `w_Tukey` | Tapered rectangular window. `r=0` is rectangular; `r=1` is Hann-like. |
