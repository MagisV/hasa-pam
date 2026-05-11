✔ The default environment has been installed.
Aperture sweep: [25, 50, 75] mm → Ny=Nz = [125, 250, 375]
  voxels/plane  : [15625, 62500, 140625]
Depth sweep    : [20, 40, 60, 80, 100] mm below skull
  march steps  : [400, 800, 1200, 1600, 2000] (×4 substeps for HASA)

k-Wave grid : 350×320×320, Nt=3000

Building skull medium for k-Wave…
┌ Info: reading DICOM series
└   dicom_dir = "C:\\Users\\AU-FUS-Valentin\\Desktop\\OBJ_0001"
┌ Info: cropping ROI
│   index_xyz = (170, 190, 400)
└   size_xyz = (705, 360, 450)
┌ Info: resampling x/y only
│   spacing_x_mm = 0.201171875
│   spacing_y_mm = 0.201171875
└   new_spacing_xy_mm = 0.2
Running k-Wave (GPU)…
WARNING:root:Highest prime factors in each dimension are [13  5  5]
WARNING:root:Use dimension sizes with lower prime factors to improve speed
C:\Users\AU-FUS-Valentin\Desktop\hasa-pam\.CondaPkg\.pixi\envs\default\Lib\site-packages\kwave\options\simulation_execution_options.py:111: UserWarning: Custom binary name set. Ignoring `is_gpu_simulation` state.
  warnings.warn("Custom binary name set. Ignoring `is_gpu_simulation` state.")
+---------------------------------------------------------------+
|                  kspaceFirstOrder-CUDA v1.3                   |
+---------------------------------------------------------------+
| Reading simulation configuration:                        Done |
| Selected GPU device id:                                     0 |
| GPU device name:                      NVIDIA GeForce RTX 3080 |
| Number of CPU threads:                                      1 |
| Processor name: AMD Ryzen 9 5900X 12-Core Processor           |
+---------------------------------------------------------------+
|                      Simulation details                       |
+---------------------------------------------------------------+
| Domain dimensions:                            390 x 360 x 360 |
| Medium type:                                               3D |
| Simulation time steps:                                   3000 |
+---------------------------------------------------------------+
|                        Initialization                         |
+---------------------------------------------------------------+
| Memory allocation:                                       Done |
| Data loading:                                            Done |
| Elapsed time:                                           0.40s |
+---------------------------------------------------------------+
| FFT plans creation:                                      Done |
| Pre-processing phase:                                    Done |
| Elapsed time:                                           2.77s |
+---------------------------------------------------------------+
|                    Computational resources                    |
+---------------------------------------------------------------+
| Current host memory in use:                            4500MB |
| Current device memory in use:                          6176MB |
| Expected output file size:                             1171MB |
+---------------------------------------------------------------+
|                          Simulation                           |
+----------+----------------+--------------+--------------------+
| Progress |  Elapsed time  |  Time to go  |  Est. finish time  |
+----------+----------------+--------------+--------------------+
|     0%   |        0.049s  |     73.451s  |  11/05/26 14:41:14 |
|     5%   |        7.586s  |    142.138s  |  11/05/26 14:42:31 |
|    10%   |       14.975s  |    133.783s  |  11/05/26 14:42:29 |
|    15%   |       22.374s  |    126.126s  |  11/05/26 14:42:30 |
|    20%   |       29.742s  |    118.474s  |  11/05/26 14:42:29 |
|    25%   |       37.087s  |    110.866s  |  11/05/26 14:42:28 |
|    30%   |       44.405s  |    103.283s  |  11/05/26 14:42:29 |
|    35%   |       51.838s  |     95.989s  |  11/05/26 14:42:28 |
|    40%   |       59.407s  |     88.863s  |  11/05/26 14:42:29 |
|    45%   |       66.740s  |     81.352s  |  11/05/26 14:42:29 |
|    50%   |       74.308s  |     74.110s  |  11/05/26 14:42:30 |
|    55%   |       81.615s  |     66.596s  |  11/05/26 14:42:29 |
|    60%   |       88.944s  |     59.131s  |  11/05/26 14:42:29 |
|    65%   |       96.281s  |     51.692s  |  11/05/26 14:42:28 |
|    70%   |      103.807s  |     44.348s  |  11/05/26 14:42:29 |
|    75%   |      111.348s  |     36.984s  |  11/05/26 14:42:29 |
|    80%   |      118.663s  |     29.542s  |  11/05/26 14:42:29 |
|    85%   |      126.291s  |     22.170s  |  11/05/26 14:42:30 |
|    90%   |      133.754s  |     14.752s  |  11/05/26 14:42:29 |
|    95%   |      141.251s  |      7.330s  |  11/05/26 14:42:29 |
+----------+----------------+--------------+--------------------+
| Elapsed time:                                         148.88s |
+---------------------------------------------------------------+
| Sampled data post-processing:                            Done |
| Elapsed time:                                           0.00s |
+---------------------------------------------------------------+
|                            Summary                            |
+---------------------------------------------------------------+
| Peak host memory in use:                               4500MB |
| Peak device memory in use:                             6176MB |
+---------------------------------------------------------------+
| Total execution time:                                 153.15s |
+---------------------------------------------------------------+
|                       End of computation                      |
+---------------------------------------------------------------+
  k-Wave wall-clock: 210.3 s

GPU warm-up (one call per aperture size)…
  aperture = 25 mm … ┌ Info: reading DICOM series
└   dicom_dir = "C:\\Users\\AU-FUS-Valentin\\Desktop\\OBJ_0001"
┌ Info: cropping ROI
│   index_xyz = (170, 190, 400)
└   size_xyz = (705, 360, 450)
┌ Info: resampling x/y only
│   spacing_x_mm = 0.201171875
│   spacing_y_mm = 0.201171875
└   new_spacing_xy_mm = 0.2
[ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
done.
  aperture = 50 mm … ┌ Info: reading DICOM series
└   dicom_dir = "C:\\Users\\AU-FUS-Valentin\\Desktop\\OBJ_0001"
┌ Info: cropping ROI
│   index_xyz = (170, 190, 400)
└   size_xyz = (705, 360, 450)
┌ Info: resampling x/y only
│   spacing_x_mm = 0.201171875
│   spacing_y_mm = 0.201171875
└   new_spacing_xy_mm = 0.2
[ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
done.
  aperture = 75 mm … ┌ Info: reading DICOM series
└   dicom_dir = "C:\\Users\\AU-FUS-Valentin\\Desktop\\OBJ_0001"
┌ Info: cropping ROI
│   index_xyz = (170, 190, 400)
└   size_xyz = (705, 360, 450)
┌ Info: resampling x/y only
│   spacing_x_mm = 0.201171875
│   spacing_y_mm = 0.201171875
└   new_spacing_xy_mm = 0.2
[ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
done.

┌ Info: reading DICOM series
└   dicom_dir = "C:\\Users\\AU-FUS-Valentin\\Desktop\\OBJ_0001"
┌ Info: cropping ROI
│   index_xyz = (170, 190, 400)
└   size_xyz = (705, 360, 450)
┌ Info: resampling x/y only
│   spacing_x_mm = 0.201171875
│   spacing_y_mm = 0.201171875
└   new_spacing_xy_mm = 0.2
[ 1/15] depth= 20 mm, aperture= 25 mm (Ny=Nz=125, 400 march rows, ETA ~? min)
  run 1/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
3.34 s  (march 0.10 s)
  run 2/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
3.19 s  (march 0.15 s)
  run 3/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
1.38 s  (march 0.18 s)
  → median: 3.19 s  (march 0.15 s)
    [progress saved — 1/15 done]
[ 2/15] depth= 20 mm, aperture= 50 mm (Ny=Nz=250, 400 march rows, ETA ~5 min)
  run 1/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
3.61 s  (march 0.13 s)
  run 2/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
3.31 s  (march 0.13 s)
  run 3/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
3.30 s  (march 0.13 s)
  → median: 3.31 s  (march 0.13 s)
    [progress saved — 2/15 done]
[ 3/15] depth= 20 mm, aperture= 75 mm (Ny=Nz=375, 400 march rows, ETA ~4 min)
  run 1/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
6.64 s  (march 0.27 s)
  run 2/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
48.00 s  (march 6.77 s)
  run 3/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
42.14 s  (march 0.27 s)
  → median: 42.14 s  (march 0.27 s)
    [progress saved — 3/15 done]
┌ Info: reading DICOM series
└   dicom_dir = "C:\\Users\\AU-FUS-Valentin\\Desktop\\OBJ_0001"
┌ Info: cropping ROI
│   index_xyz = (170, 190, 400)
└   size_xyz = (705, 360, 450)
┌ Info: resampling x/y only
│   spacing_x_mm = 0.201171875
│   spacing_y_mm = 0.201171875
└   new_spacing_xy_mm = 0.2
[ 4/15] depth= 40 mm, aperture= 25 mm (Ny=Nz=125, 800 march rows, ETA ~9 min)
  run 1/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
1.59 s  (march 0.09 s)
  run 2/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
1.34 s  (march 0.08 s)
  run 3/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
1.37 s  (march 0.09 s)
  → median: 1.37 s  (march 0.09 s)
    [progress saved — 4/15 done]
[ 5/15] depth= 40 mm, aperture= 50 mm (Ny=Nz=250, 800 march rows, ETA ~7 min)
  run 1/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
4.11 s  (march 0.19 s)
  run 2/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
3.98 s  (march 0.20 s)
  run 3/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
3.68 s  (march 0.20 s)
  → median: 3.98 s  (march 0.20 s)
    [progress saved — 5/15 done]
[ 6/15] depth= 40 mm, aperture= 75 mm (Ny=Nz=375, 800 march rows, ETA ~5 min)
  run 1/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
18.41 s  (march 0.40 s)
  run 2/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
29.63 s  (march 0.40 s)
  run 3/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
29.70 s  (march 0.52 s)
  → median: 29.63 s  (march 0.40 s)
    [progress saved — 6/15 done]
┌ Info: reading DICOM series
└   dicom_dir = "C:\\Users\\AU-FUS-Valentin\\Desktop\\OBJ_0001"
┌ Info: cropping ROI
│   index_xyz = (170, 190, 400)
└   size_xyz = (705, 360, 450)
┌ Info: resampling x/y only
│   spacing_x_mm = 0.201171875
│   spacing_y_mm = 0.201171875
└   new_spacing_xy_mm = 0.2
[ 7/15] depth= 60 mm, aperture= 25 mm (Ny=Nz=125, 1200 march rows, ETA ~6 min)
  run 1/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
1.50 s  (march 0.12 s)
  run 2/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
1.43 s  (march 0.11 s)
  run 3/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
1.46 s  (march 0.11 s)
  → median: 1.46 s  (march 0.11 s)
    [progress saved — 7/15 done]
[ 8/15] depth= 60 mm, aperture= 50 mm (Ny=Nz=250, 1200 march rows, ETA ~5 min)
  run 1/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
4.36 s  (march 0.26 s)
  run 2/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
4.54 s  (march 0.39 s)
  run 3/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
4.95 s  (march 0.26 s)
  → median: 4.54 s  (march 0.26 s)
    [progress saved — 8/15 done]
[ 9/15] depth= 60 mm, aperture= 75 mm (Ny=Nz=375, 1200 march rows, ETA ~4 min)
  run 1/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
17.01 s  (march 0.53 s)
  run 2/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
37.65 s  (march 0.56 s)
  run 3/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
34.52 s  (march 1.07 s)
  → median: 34.52 s  (march 0.56 s)
    [progress saved — 9/15 done]
┌ Info: reading DICOM series
└   dicom_dir = "C:\\Users\\AU-FUS-Valentin\\Desktop\\OBJ_0001"
┌ Info: cropping ROI
│   index_xyz = (170, 190, 400)
└   size_xyz = (705, 360, 450)
┌ Info: resampling x/y only
│   spacing_x_mm = 0.201171875
│   spacing_y_mm = 0.201171875
└   new_spacing_xy_mm = 0.2
[10/15] depth= 80 mm, aperture= 25 mm (Ny=Nz=125, 1600 march rows, ETA ~4 min)
  run 1/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
2.29 s  (march 0.15 s)
  run 2/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
1.53 s  (march 0.15 s)
  run 3/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
1.59 s  (march 0.14 s)
  → median: 1.59 s  (march 0.15 s)
    [progress saved — 10/15 done]
[11/15] depth= 80 mm, aperture= 50 mm (Ny=Nz=250, 1600 march rows, ETA ~3 min)
  run 1/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
4.59 s  (march 0.32 s)
  run 2/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
4.98 s  (march 0.32 s)
  run 3/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
4.71 s  (march 0.32 s)
  → median: 4.71 s  (march 0.32 s)
    [progress saved — 11/15 done]
[12/15] depth= 80 mm, aperture= 75 mm (Ny=Nz=375, 1600 march rows, ETA ~2 min)
  run 1/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
28.28 s  (march 0.86 s)
  run 2/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
33.95 s  (march 0.75 s)
  run 3/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
48.27 s  (march 5.44 s)
  → median: 33.95 s  (march 0.86 s)
    [progress saved — 12/15 done]
┌ Info: reading DICOM series
└   dicom_dir = "C:\\Users\\AU-FUS-Valentin\\Desktop\\OBJ_0001"
┌ Info: cropping ROI
│   index_xyz = (170, 190, 400)
└   size_xyz = (705, 360, 450)
┌ Info: resampling x/y only
│   spacing_x_mm = 0.201171875
│   spacing_y_mm = 0.201171875
└   new_spacing_xy_mm = 0.2
[13/15] depth=100 mm, aperture= 25 mm (Ny=Nz=125, 2000 march rows, ETA ~2 min)
  run 1/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
2.47 s  (march 0.18 s)
  run 2/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
1.58 s  (march 0.17 s)
  run 3/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
1.10 s  (march 0.17 s)
  → median: 1.58 s  (march 0.17 s)
    [progress saved — 13/15 done]
[14/15] depth=100 mm, aperture= 50 mm (Ny=Nz=250, 2000 march rows, ETA ~1 min)
  run 1/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
4.63 s  (march 0.38 s)
  run 2/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
5.35 s  (march 0.39 s)
  run 3/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
4.98 s  (march 0.39 s)
  → median: 4.98 s  (march 0.39 s)
    [progress saved — 14/15 done]
[15/15] depth=100 mm, aperture= 75 mm (Ny=Nz=375, 2000 march rows, ETA ~0 min)
  run 1/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
70.73 s  (march 1.86 s)
  run 2/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
85.08 s  (march 0.85 s)
  run 3/3 … [ PAM 3D ] HASA: GPU NVIDIA GeForce RTX 3080 (device 0), Float32 arithmetic, 1 freq bins × 1 windows batched
55.92 s  (march 14.26 s)
  → median: 70.73 s  (march 1.86 s)
    [progress saved — 15/15 done]

Sweep complete in 13.5 minutes.

Wall-clock GPU HASA [s]  (rows = depth from skull, cols = aperture)
  Depth \ Apt │     25 mm     50 mm     75 mm
  ───────────┼───────────────────────────
      20 mm   │     3.19 s     3.31 s    42.14 s
      40 mm   │     1.37 s     3.98 s    29.63 s
      60 mm   │     1.46 s     4.54 s    34.52 s
      80 mm   │     1.59 s     4.71 s    33.95 s
     100 mm   │     1.58 s     4.98 s    70.73 s