# Cluster Emission Models

Sources: [`src/pam/2d/sources.jl`](../src/pam/2d/sources.jl), [`src/pam/3d/sources3d.jl`](../src/pam/3d/sources3d.jl)

## BubbleCluster (2D and 3D)

$$s(t) = A \;\cdot\; w_\text{Tukey}(t;\, T_\text{gate},\, r) \;\cdot\; \sum_{n \,\in\, \mathcal{H}} \alpha_n \cos\!\bigl(2\pi n f_0\, t + \phi_n\bigr), \qquad t \in [0,\, T_\text{gate}]$$

| Symbol | Field | 2D default | 3D default |
|---|---|---|---|
| $f_0$ | `fundamental` | 500 kHz | 500 kHz |
| $A$ | `amplitude` | 1.0 | 1.0 |
| $\mathcal{H}$ | `harmonics` | {2, 3} | {2, 3, 4} |
| $\alpha_n$ | `harmonic_amplitudes` | [1.0, 0.6] | [1.0, 0.6, 0.3] |
| $\phi_n$ | `harmonic_phases` | [0.0, 0.0] | [0.0, 0.0, 0.0] |
| $T_\text{gate}$ | `gate_duration` | 50 μs | 50 μs |
| $r$ | `taper_ratio` | 0.25 | 0.25 |

## Term glossary

| Symbol | Meaning |
|---|---|
| $t$ | Time, measured from source onset (after subtracting `delay`) |
| $f_0$ | Fundamental emission frequency. Harmonics are integer multiples $n f_0$ |
| $n$ | Harmonic order. Bubbles driven at $f_\text{drive}$ emit at $n f_\text{drive}$ |
| $A$ | Cluster pressure amplitude (linear scale) |
| $\alpha_n$ | Relative amplitude of harmonic $n$, normalised so $\alpha_2 = 1$ by convention |
| $\phi_n$ | Phase of harmonic $n$. Set by `phase_mode`: zero (coherent), geometric travel-time delay, or random |
| $T_\text{gate}$ | Total emission duration. Signal is zero outside $[0, T_\text{gate}]$ |
| $w_\text{Tukey}$ | Tukey (cosine-tapered rectangular) window. Taper fraction $r$ sets what fraction of the gate is the raised-cosine roll-on/off; $r=0$ is a rectangular gate, $r=1$ is a Hann window |

