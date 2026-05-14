# 3D HASA GPU Profiling & Scaling Results

## Key takeaways

**1. GPU march time follows a Ny²-linear law in the well-utilised aperture range (50–75 mm):**

```
t_march ≈ Ny² × (9.1×10⁻⁷ + 2.3×10⁻⁹ × n_rows)   [seconds]
```

where Ny = aperture / 0.2 mm and n_rows = depth / 0.05 mm.
Both the fixed overhead (α·Ny²) and the per-row cost (β·Ny²·n_rows) scale with
the number of voxels per plane because the GPU is memory-bandwidth saturated at
~200 GB/s (~25% of RTX 3080 peak) in this regime.
Predictions match measurements to within 1–7% for 64–75 mm; ~11% for 50 mm.

**2. The practical aperture limit on an RTX 3080 is 75 mm.**
Below ~50 mm the GPU is under-occupied (bandwidth 10–57 GB/s, irregular timing).
Above 75 mm the setup pre-allocates a propagation operator stack that exceeds
10 GB VRAM; CUDA falls back to PCIe-speed unified memory (~3 GB/s) and the march
becomes 30–70× slower than the model predicts. 125 mm throws a hard OOM.
The fix — streaming operators in chunks rather than pre-allocating all rows —
would raise this limit substantially.

---

## Setup

All timing uses GPU HASA (`corrected=true`, `benchmark=true`).
RF data is synthetic (uninitialised Float32, correct shape) and the medium is
homogeneous (c = 1500 m/s), so no k-Wave simulation or DICOM loading is needed.
Apertures swept: 25, 50, 64, 75, 100, 125 mm.  Depths: 20–100 mm below skull
(skull at 20 mm → total axial range 40–120 mm from transducer).
N = 3 timed runs per point; stored value is the median `march_gpu_s`.
GPU: RTX 3080 (10 GB VRAM).

---

## Phase breakdown (depth = 70 mm, from `profile_cases.jl`)

| Phase | 25 mm (Ny = 125) | 75 mm (Ny = 375) |
|-------|-----------------|-----------------|
| **setup** | 0.37 s | 8.86 s |
| **march — GPU measured** | 0.20 s | 0.60 s |
| **march — wall clock** | 0.60 s | 1.17 s |
| **FFT share of march** | ~46 % | ~29 % |
| **element-wise share** | ~54 % | ~71 % |
| **download (GPU→CPU)** | 0.01 s | 0.09 s |
| **GPU memory allocated** | +160 MB | +6 112 MB |
| **bandwidth** | ~97 GB/s | ~198 GB/s |

Setup pre-computes propagation operators for all march rows at once, allocating
~6 GB for 75 mm. In practice setup runs once per session; all subsequent
reconstructions on new RF data pay only the march cost.

---

## Scalability sweep results — GPU march time [s]

| Depth \ Apt | 25 mm | 50 mm | 64 mm | 75 mm | 100 mm | 125 mm |
|-------------|-------|-------|-------|-------|--------|--------|
| 20 mm | 0.102 | 0.128 | 0.185 | 0.263 | 29.0 | OOM |
| 40 mm | 0.843 | 0.191 | 0.277 | 0.394 | 54.6 | OOM |
| 60 mm | 0.899 | 0.257 | 0.371 | 0.526 | 47.2 | OOM |
| 80 mm | 0.313 | 0.321 | 0.461 | 0.666 | 74.0 | OOM |
| 100 mm | 0.572 | 0.382 | 0.554 | 0.800 | 94.3 | OOM |

Memory bandwidth [GB/s] (RTX 3080 theoretical peak: 760 GB/s):

| Depth \ Apt | 25 mm | 50 mm | 64 mm | 75 mm | 100 mm |
|-------------|-------|-------|-------|-------|--------|
| 20 mm | 57 | 181 | 204 | 197 | 3 |
| 100 mm | 30 | 182 | 205 | 195 | 3 |

---

## Three operating regimes

### 25 mm — GPU under-occupied

Ny = 125 gives only 15 k voxels/plane, too few to saturate the RTX 3080.
Bandwidth is 10–57 GB/s (vs ~200 GB/s at larger apertures) and march times
are non-monotonic across depths due to variable kernel scheduling overhead.
The element-wise kernels are most affected: EW can be 0.06 s or 0.85 s for the
same aperture depending on GPU state.

### 50–75 mm — GPU well-utilised, clean scaling

Bandwidth is stable at 180–205 GB/s (~25% of peak, typical for mixed FFT +
element-wise workloads). March times increase monotonically with both depth and
aperture, allowing a reliable scaling model (see below).

### 100 mm — VRAM overflow to unified memory

Setup allocates > 10 GB, exceeding VRAM. CUDA silently falls back to unified
memory paging over PCIe rather than throwing OOM. Bandwidth collapses to ~3 GB/s
(PCIe speed), making march 30–70× slower than the scaling model predicts.
125 mm throws a hard `OutOfGPUMemoryError`.

**Practical limit on RTX 3080: 75 mm aperture.**
The root cause is setup pre-allocating the full operator stack at once
(one 2-D operator per march row). Streaming operators in chunks would push
this limit significantly higher.

---

## GPU march scaling law (50–75 mm regime)

In the well-utilised regime the march time follows:

```
t_march(Ny, n_rows) ≈ Ny² × (α + β × n_rows)
```

Fitting to the 64 mm and 75 mm columns (most reliable, highest GPU utilisation):

| Parameter | Value | Units |
|-----------|-------|-------|
| α | 9.1 × 10⁻⁷ | s / voxel |
| β | 2.3 × 10⁻⁹ | s / (voxel · row) |

Both terms scale as Ny² because each march row processes the full 2-D plane
(FFT + element-wise), and GPU occupancy is saturated so per-voxel cost is
constant.

**Depth (n_rows) is linear** — the slope per 400 additional rows is ~0.063 s
for 50 mm and ~0.134 s for 75 mm, both extremely consistent across the sweep.

**Aperture scales as Ny²** in the saturated regime (64–75 mm); the 50 mm column
fits within ~10% using the same coefficients.

### Predictions vs measurements

| Aperture | Depth | Predicted | Measured | Error |
|----------|-------|-----------|----------|-------|
| 64 mm | 100 mm | 0.55 s | 0.554 s | < 1 % |
| 75 mm | 60 mm | 0.52 s | 0.526 s | 1 % |
| 75 mm | 100 mm | 0.86 s | 0.800 s | 7 % |
| 50 mm | 100 mm | 0.34 s | 0.382 s | 11 % |

The larger error at 50 mm reflects that it sits at the edge of the well-utilised
regime (bandwidth 181 vs 204 GB/s), slightly underestimating per-voxel cost.
