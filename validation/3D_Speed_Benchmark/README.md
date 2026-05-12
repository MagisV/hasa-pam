# 3D PAM Speed Benchmark

Wall-clock reconstruction times for a single centre source at 50 mm depth,
64 mm × 64 mm aperture, 0.2 mm grid, CT skull at 20 mm.

|        | CPU     | GPU    |
|--------|---------|--------|
| **ASA**  | 37.1 s  | 34.4 s |
| **HASA** | 46.7 s  | 9.7 s  |

HASA massively benefits from GPU speedup. But not in a real time regime yet.