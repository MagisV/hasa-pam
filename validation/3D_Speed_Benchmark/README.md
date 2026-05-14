# 3D PAM Speed Benchmark

Single centre source, 50 mm depth, 50 mm × 50 mm aperture, 0.2 mm grid,
CT skull at 20 mm.  Median over 5 timed runs after 1 warm-up.
RTX 3080 / AMD Ryzen 9 5900X.

## Wall-clock time

|          | CPU          | GPU         |
|----------|--------------|-------------|
| **ASA**  | 12.25 s      | 3.78 s      |
| **HASA** | 19.85 s      | 3.94 s      |

GPU/CPU wall speedup: **5.0× (HASA)**, 3.2× (ASA)

## Phase breakdown

| Mode | Wall | Setup | March | March share |
|------|------|-------|-------|-------------|
| CPU ASA  | 12.25 s | 5.41 s | 5.49 s | 45 % |
| CPU HASA | 19.85 s | 5.35 s | 13.28 s | 67 % |
| GPU ASA  |  3.78 s | 1.72 s |  0.02 s |  1 % |
| GPU HASA |  3.94 s | 1.66 s |  0.23 s |  6 % |

**GPU/CPU march speedup (HASA): 58.9×**

## Key observations

- **Wall-clock speedup understates GPU performance.** On GPU, the march
  (the part that repeats for every new RF recording) takes only 0.23 s —
  virtually free compared to setup. The 3.94 s total is dominated by the
  1.66 s operator setup, which is a one-time cost per session.

- **In live use (setup once, new RF each acquisition), GPU HASA runs at
  ~4 Hz** (0.23 s per reconstruction).

- **GPU ASA ≈ GPU HASA in total wall time** (3.78 s vs 3.94 s): adding skull
  correction costs only 0.21 s extra march time on GPU. On CPU, HASA march
  is 2.4× slower than ASA march (13.28 s vs 5.49 s).

- **GPU setup is also 3× faster than CPU setup** (1.7 s vs 5.4 s), though
  both are one-time costs.
