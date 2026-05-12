#!/usr/bin/env python3
"""
wavefront_aberration.py
=======================
Single Ricker impulse through water vs real skull CT slice — animated using
k-Wave's PML solver (kspaceFirstOrder-OMP) for clean, reflection-free boundaries.

Run:
  "/Users/vm/INI_code/Julia II/.CondaPkg/.pixi/envs/default/bin/python3" \
      wavefront_aberration.py
"""

import os, shutil, time, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["animation.ffmpeg_path"] = "/opt/homebrew/bin/ffmpeg"
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom
import pydicom

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions

# ── Physical constants ────────────────────────────────────────────────────────
C0       = 1500.0
C_BONE   = 4500#2500.0
RHO0     = 1000.0
RHO_BONE = 2800.0#2100
HU_THR   = 200

# ── Grid ─────────────────────────────────────────────────────────────────────
LX, LY = 0.100, 0.056          # 100 mm × 76 mm
DX      = 3e-4                  # 0.3 mm → PPW_water=6.25 at 800 kHz
NX      = int(round(LX / DX))  # 333
NY      = int(round(LY / DX))  # 253

x_m = np.arange(NX) * DX
y_m = np.arange(NY) * DX

SRC_IX = NX // 2                         # lateral centre (166)
SRC_IY = int(round(0.04 / DX))          # 40 mm depth   (200)

N_SEN  = 16
SEN_IX = np.linspace(NX // 6, 5 * NX // 6, N_SEN, dtype=int)

F0      = 500_000.0                       # 800 kHz — ~1.9 λ of differential delay
PML_SIZE = 20
N_SKIP  = 4
THRESH  = 3.0

# ── HU → (c, rho) — port of src/focus/medium.jl ──────────────────────────────
def hu_to_rho_c(hu: np.ndarray):
    hu   = np.clip(hu, -1000, 3000).astype(np.float32)
    mask = hu >= HU_THR
    c    = np.full_like(hu, C0)
    rho  = np.full_like(hu, RHO0)
    if mask.any():
        bone  = hu[mask]
        h_max = float(np.percentile(bone, 99.5))
        psi   = np.clip((h_max - bone) / max(h_max, 1.0), 0.0, 1.0)
        c[mask]   = C0       + (C_BONE   - C0)       * (1.0 - psi)
        rho[mask] = RHO0     + (RHO_BONE - RHO0)     * (1.0 - psi)
    return c, rho

# ── CT loading — mirrors make_pam_medium slice extraction ────────────────────
CT_PATH = (
    "/Users/vm/INI_code/Ultrasound/"
    "DIRU_20240404_human_skull_phase_correction_1_2_(skull_Normal)/"
    "DICOM/PAT_0000/STD_0000/SER_0002/OBJ_0001"
)
SLICE_IDX = 250

def load_skull_maps():
    print("Sorting DICOM headers …", flush=True)
    t0  = time.time()
    raw = [f for f in os.listdir(CT_PATH) if not f.startswith(".")]
    fz  = []
    for fname in raw:
        ds = pydicom.dcmread(os.path.join(CT_PATH, fname), stop_before_pixels=True)
        fz.append((float(ds.ImagePositionPatient[2]), fname))
    fz.sort()
    print(f"  {len(fz)} slices sorted  ({time.time()-t0:.1f} s)", flush=True)

    target = fz[SLICE_IDX][1]
    print(f"  Loading slice {SLICE_IDX}  z={fz[SLICE_IDX][0]:.0f} mm", flush=True)
    ds    = pydicom.dcmread(os.path.join(CT_PATH, target))
    hu    = ds.pixel_array.astype(np.float32) + float(getattr(ds, "RescaleIntercept", -1024))
    dx_ct = float(ds.PixelSpacing[1]) * 1e-3
    n_rows, n_cols = hu.shape
    print(f"  Shape {hu.shape}  dx={dx_ct*1e3:.2f} mm  HU [{hu.min():.0f},{hu.max():.0f}]",
          flush=True)

    # Orient: skull outer surface at low row index (near transducer)
    bone_per_row = (hu > HU_THR).sum(axis=1)
    significant  = np.where(bone_per_row > 30)[0]
    if len(significant) == 0:
        raise RuntimeError("No skull found in this slice.")
    y_top = int(significant[0])
    if y_top > n_rows // 2:
        hu        = hu[::-1, :]
        significant = n_rows - 1 - significant
        y_top     = int(significant.min())

    x_ctr      = n_cols // 2
    n_above_ct = int(round(0.020 / dx_ct))   # 30 mm water pad above skull
    ny_ct      = int(round(LY / dx_ct))
    nx_ct      = int(round(LX / dx_ct))
    y0 = y_top - n_above_ct;  y1 = y0 + ny_ct
    x0 = x_ctr - nx_ct // 2;  x1 = x0 + nx_ct

    pad = [(max(0, -y0), max(0, y1 - n_rows)),
           (max(0, -x0), max(0, x1 - n_cols))]
    y0 = max(0, y0);  y1 = min(n_rows, y1)
    x0 = max(0, x0);  x1 = min(n_cols, x1)
    region = hu[y0:y1, x0:x1]
    if any(v for pp in pad for v in pp):
        region = np.pad(region, pad, constant_values=0.0)

    # Resample to (NX, NY) FDTD grid
    hu_2d = zoom(region.astype(np.float32),
                 (NY / region.shape[0], NX / region.shape[1]),
                 order=1).T   # → (NX, NY): c[ix, iy]

    # hu_2d = _flatten_skull_slab(hu_2d)
    c_map, rho_map = hu_to_rho_c(hu_2d)
    print(f"  Bone fraction in domain: {(c_map > C0+20).mean()*100:.1f}%", flush=True)
    return c_map, rho_map, hu_2d


def _flatten_skull_slab(hu_2d: np.ndarray) -> np.ndarray:
    """
    Remap the curved skull arch to a flat horizontal slab.

    For each lateral column (ix), the bone pixels (HU ≥ HU_THR) are extracted
    and resampled into a uniform depth band whose position and thickness are the
    median across all columns.  This ensures every wavefront path — centre or
    edge — travels through the same bone depth, making aberration visible across
    the full aperture.  Real HU heterogeneity (diploë / cortical layers) is
    preserved within each column.
    """
    nx = hu_2d.shape[0]
    bone_mask = hu_2d >= HU_THR

    col_top = np.full(nx, -1, dtype=int)
    col_bot = np.full(nx, -1, dtype=int)
    for ix in range(nx):
        rows = np.where(bone_mask[ix, :])[0]
        if len(rows) >= 2:
            col_top[ix] = rows[0]
            col_bot[ix] = rows[-1]

    valid = col_top >= 0
    if valid.sum() < nx // 4:
        return hu_2d          # not enough skull columns — leave untouched

    slab_top   = int(np.median(col_top[valid]))
    slab_thick = int(round(0.020 / DX))   # force 20 mm slab (~1.9 λ of differential delay)

    hu_flat = np.zeros_like(hu_2d)          # 0 HU = water everywhere
    for ix in range(nx):
        if not valid[ix]:
            continue
        col = hu_2d[ix, col_top[ix]:col_bot[ix] + 1]
        if len(col) != slab_thick:
            col = zoom(col.astype(np.float32), slab_thick / len(col), order=1)
        hu_flat[ix, slab_top:slab_top + slab_thick] = col[:slab_thick]

    print(f"  Skull slab: top={slab_top} ({slab_top*DX*1e3:.1f} mm)  "
          f"thick={slab_thick} ({slab_thick*DX*1e3:.1f} mm)", flush=True)
    return hu_flat

# ── k-Wave simulation ─────────────────────────────────────────────────────────
# Use C_BONE to fix DT across both simulations so snapshots align in time.
T_END = 1.5 * SRC_IY * DX / C0   # ≈ 60 µs

def simulate_kwave(c_map: np.ndarray, rho_map: np.ndarray, label: str):
    print(f"\nRunning k-Wave ({label}) …", flush=True)
    t0 = time.time()

    kgrid = kWaveGrid([NX, NY], [DX, DX])
    kgrid.makeTime(C_BONE, cfl=0.3, t_end=T_END)   # fixes DT for both sims
    NT = int(kgrid.Nt)
    DT = float(kgrid.dt)
    print(f"  Grid {NX}×{NY}  NT={NT}  DT={DT*1e9:.1f} ns  "
          f"T_end={NT*DT*1e6:.0f} µs  λ_water={C0/F0*1e3:.2f} mm  "
          f"PPW_water={C0/F0/DX:.1f}", flush=True)

    medium = kWaveMedium(
        sound_speed=c_map.astype(np.float32),
        density=rho_map.astype(np.float32),
    )

    # Ricker wavelet: broadband single impulse
    t_v   = np.arange(NT) * DT
    xi    = np.pi * F0 * (t_v - 1.5 / F0)
    pulse = (1.0 - 2.0 * xi**2) * np.exp(-xi**2) * 3500.0

    source = kSource()
    p_mask = np.zeros((NX, NY), dtype=bool)
    p_mask[SRC_IX, SRC_IY] = True
    source.p_mask          = p_mask
    source.p               = pulse.reshape(1, -1).astype(np.float64)
    source.p_frequency_ref = F0
    source.medium          = medium

    # Full-field sensor (all NX×NY cells at every timestep)
    sensor = kSensor()
    sensor.mask   = np.ones((NX, NY), dtype=bool)
    sensor.record = ["p"]

    sim_dir = tempfile.mkdtemp()
    try:
        data = kspaceFirstOrder2D(
            kgrid=kgrid,
            medium=medium,
            source=source,
            sensor=sensor,
            simulation_options=SimulationOptions(
                pml_inside=False,
                pml_size=PML_SIZE,
                data_cast="single",     # float32 throughout
                data_recast=False,
                save_to_disk=True,
                data_path=sim_dir,
            ),
            execution_options=SimulationExecutionOptions(
                is_gpu_simulation=False,
                delete_data=True,
            ),
        )
    finally:
        shutil.rmtree(sim_dir, ignore_errors=True)

    # data["p"] shape: (NT, NX*NY), Fortran-ordered spatially (x varies fastest)
    p_raw = np.array(data["p"])  # float32, (NT, NX*NY)
    p_3d  = p_raw.T.reshape((NX, NY, NT), order="F")  # (NX, NY, NT)
    del p_raw

    # Full RF time series at each sensor (transducer row iy=0)
    rf_data = p_3d[SEN_IX, 0, :].copy()   # (N_SEN, NT)

    # Subsample snapshots for animation
    snap_idx = np.arange(0, NT, N_SKIP)
    snaps    = [p_3d[:, :, i].copy() for i in snap_idx]
    del p_3d

    print(f"  Done in {time.time()-t0:.1f} s", flush=True)
    return snaps, rf_data, NT, DT

# ── Load CT and run both simulations ─────────────────────────────────────────
print("Loading CT skull …")
c_skull, rho_skull, hu_map = load_skull_maps()
c_water   = np.full((NX, NY), C0,   dtype=np.float32)
rho_water = np.full((NX, NY), RHO0, dtype=np.float32)

snaps_w, rf_w, NT, DT = simulate_kwave(c_water,  rho_water,  "water")
snaps_s, rf_s, _,  _  = simulate_kwave(c_skull,  rho_skull,  "skull")

N_FR   = min(len(snaps_w), len(snaps_s))
T_US   = np.arange(NT) * DT * 1e6          # time axis in µs  (length NT)
print(f"\nFrames: {N_FR}  ({N_FR/25:.1f} s @ 25 fps)")

# ── Figure setup ─────────────────────────────────────────────────────────────
BG = "#070a0f";  FG = "#a8bbc8"

fig = plt.figure(figsize=(16, 10), facecolor=BG)
gs  = GridSpec(2, 2,
               height_ratios=[3.2, 1.0],
               hspace=0.08, wspace=0.09,
               left=0.06, right=0.96, top=0.92, bottom=0.06)
ax_w  = fig.add_subplot(gs[0, 0])
ax_s  = fig.add_subplot(gs[0, 1])
ax_bw = fig.add_subplot(gs[1, 0])
ax_bs = fig.add_subplot(gs[1, 1])

for ax in (ax_w, ax_s, ax_bw, ax_bs):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#18263a")
    ax.tick_params(colors=FG, labelsize=8)

# Pressure colormap: black at 0, blue for rarefaction, red for compression
_cd = {
    "red":   [(0, .04, .04), (0.5, 0., 0.), (1, .95, .95)],
    "green": [(0, .14, .14), (0.5, 0., 0.), (1, .10, .10)],
    "blue":  [(0, .90, .90), (0.5, 0., 0.), (1, .06, .06)],
}
PCMAP = LinearSegmentedColormap("glow", _cd)
VMAX  = 70.0

X_MM = x_m * 1e3;  Y_MM = y_m * 1e3
EXT  = [X_MM[0], X_MM[-1], Y_MM[-1], Y_MM[0]]

# ── Skull overlay (real c-map coloured by sound speed) ───────────────────────
skull_mask = c_skull > C0 + 20.0
sm_T       = skull_mask.T
c_skull_T  = c_skull.T
c_in   = c_skull_T[sm_T]
c_min  = float(c_in.min()) if c_in.size else C0
c_max  = float(c_in.max()) if c_in.size else C_BONE
c_norm_T = np.zeros_like(c_skull_T)
if c_in.size:
    c_norm_T[sm_T] = (c_skull_T[sm_T] - c_min) / max(c_max - c_min, 1.0)

bone_cm    = plt.cm.YlOrBr
skull_rgba = np.zeros((NY, NX, 4), dtype=np.float32)
if sm_T.any():
    rgba = bone_cm(c_norm_T[sm_T])
    skull_rgba[sm_T, :3] = rgba[:, :3]
    skull_rgba[sm_T,  3] = 0.65
water_rgba = np.zeros((NY, NX, 4), dtype=np.float32)

def arch_boundaries():
    outer = np.full(NX, np.nan);  inner = np.full(NX, np.nan)
    for ix in range(NX):
        idx = np.where(skull_mask[ix])[0]
        if len(idx):
            outer[ix] = Y_MM[idx[0]];  inner[ix] = Y_MM[idx[-1]]
    return outer, inner

outer_mm, inner_mm = arch_boundaries()

def setup_wave_ax(ax, title, tcol, overlay, show_skull_outline):
    im = ax.imshow(
        np.zeros((NY, NX)), origin="upper",
        extent=EXT, cmap=PCMAP, vmin=-VMAX, vmax=VMAX,
        aspect="equal", interpolation="bilinear", zorder=1,
    )
    ax.imshow(overlay, origin="upper", extent=EXT,
              aspect="equal", interpolation="nearest", zorder=2)
    if show_skull_outline:
        ax.plot(X_MM, outer_mm, color="#886600", lw=0.9, ls="--", zorder=3, alpha=0.75)
        ax.plot(X_MM, inner_mm, color="#886600", lw=0.9, ls="--", zorder=3, alpha=0.75)
    ax.scatter(X_MM[SEN_IX], np.ones(N_SEN) * Y_MM[1],
               s=30, c="#22ff88", marker="v", zorder=6, lw=0)
    ax.scatter([X_MM[SRC_IX]], [Y_MM[SRC_IY]],
               s=100, c="white", marker="*", zorder=6, lw=0)
    ax.set_xlim(X_MM[0], X_MM[-1]);  ax.set_ylim(Y_MM[-1], Y_MM[0])
    ax.set_xlabel("Lateral  [mm]", color=FG, fontsize=9)
    ax.set_ylabel("Depth  [mm]",   color=FG, fontsize=9)
    ax.set_title(title, color=tcol, fontsize=14, fontweight="bold", pad=7)
    return im

im_w = setup_wave_ax(ax_w, "Water only",      "#5599ff", water_rgba, show_skull_outline=False)
im_s = setup_wave_ax(ax_s, "With skull", "#ff8844", skull_rgba, show_skull_outline=True)

# Pressure legend (blue = rarefaction, red = compression)
_leg = ax_w.legend(
    handles=[
        mpatches.Patch(color="#0a22e6", label="Rarefaction  (−p)"),
        mpatches.Patch(color="#e61a0a", label="Compression  (+p)"),
    ],
    loc="lower left", fontsize=7.5, framealpha=0.45,
    facecolor=BG, edgecolor="#18263a", labelcolor=FG,
)
ax_w.add_artist(_leg)

ax_w.text(X_MM[-1]-1, Y_MM[1]+0.5, "Sensors",
          color="#22ff88", fontsize=7.5, ha="right", va="bottom", zorder=7)
mid_mm = float(np.nanmean([outer_mm[NX//2], inner_mm[NX//2]])) \
         if not np.isnan(outer_mm[NX//2]) else 35

from mpl_toolkits.axes_grid1 import make_axes_locatable
_div = make_axes_locatable(ax_s)
_cax = _div.append_axes("right", size="2.5%", pad=0.04)
_sm  = plt.cm.ScalarMappable(cmap=bone_cm, norm=plt.Normalize(c_min, c_max))
_cb  = fig.colorbar(_sm, cax=_cax)
_cb.set_label("c  [m/s]", color=FG, fontsize=8)
_cb.ax.tick_params(colors=FG, labelsize=7);  _cb.outline.set_edgecolor("#18263a")

# ── RF wiggle traces ──────────────────────────────────────────────────────────
# Normalise both panels identically so water/skull amplitudes are comparable.
rf_scale = max(np.abs(rf_w).max(), np.abs(rf_s).max(), 1.0) / 0.65

def setup_rf_ax(ax, color, label):
    ax.set_facecolor("#0b1520")
    for sp in ax.spines.values(): sp.set_edgecolor("#18263a")
    ax.tick_params(colors=FG, labelsize=7)

    # Faint zero-baseline rule for each sensor
    for s in range(N_SEN):
        ax.axhline(s, color="#18263a", lw=0.5, zorder=1)

    # One line object per sensor (data filled in during animation)
    lines = [ax.plot([], [], color=color, lw=0.75, alpha=0.9, zorder=3)[0]
             for _ in range(N_SEN)]

    # Vertical time cursor
    cursor, = ax.plot([], [], color=FG, lw=0.8, ls="--", alpha=0.35, zorder=5)

    ax.set_xlim(0, T_US[-1])
    ax.set_ylim(-0.6, N_SEN - 0.4)
    ax.set_xlabel("Time  [µs]", color=FG, fontsize=9)

    tick_s = [0, 4, 8, 12, N_SEN - 1]
    ax.set_yticks(tick_s)
    ax.set_yticklabels([f"{X_MM[SEN_IX[s]]:.0f} mm" for s in tick_s],
                       fontsize=6.5, color=FG)
    ax.set_ylabel("Sensor lateral pos.", color=FG, fontsize=8)
    ax.set_title(f"RF traces — {label}", color=color, fontsize=9.5, pad=3)
    return lines, cursor

lines_w, cursor_w = setup_rf_ax(ax_bw, "#4488ee", "water only")
lines_s, cursor_s = setup_rf_ax(ax_bs, "#ff7744", "skull in water")

fig.suptitle(
    f"Transcranial Ultrasound — Single Ricker Impulse ({F0/1e3:.0f} kHz) "
    f"Through a Real Skull  [CT slice {SLICE_IDX}]",
    color="white", fontsize=12.5, fontweight="bold", y=0.97,
)
t_label = fig.text(0.5, 0.935, "t = 0.0 µs",
                   ha="center", color=FG, fontsize=11)

# ── Animation ─────────────────────────────────────────────────────────────────
def update(fr):
    n    = min(fr * N_SKIP, NT - 1)
    t_us = T_US[n]
    t_label.set_text(f"t = {t_us:.1f} µs")
    im_w.set_data(snaps_w[fr].T)
    im_s.set_data(snaps_s[fr].T)

    t_slice = T_US[:n + 1]
    for s in range(N_SEN):
        lines_w[s].set_data(t_slice, rf_w[s, :n + 1] / rf_scale + s)
        lines_s[s].set_data(t_slice, rf_s[s, :n + 1] / rf_scale + s)
    cursor_w.set_data([t_us, t_us], [-0.6, N_SEN - 0.4])
    cursor_s.set_data([t_us, t_us], [-0.6, N_SEN - 0.4])

    return [im_w, im_s, *lines_w, *lines_s, cursor_w, cursor_s, t_label]

ani = animation.FuncAnimation(fig, update, frames=N_FR, interval=40, blit=True)

out_dir = os.path.dirname(os.path.abspath(__file__))
out_mp4 = os.path.join(out_dir, "wavefront_aberration.mp4")
out_gif = os.path.join(out_dir, "wavefront_aberration.gif")

try:
    print(f"\nSaving {out_mp4} …")
    ani.save(out_mp4,
             writer=animation.FFMpegWriter(fps=75, bitrate=4000,
                                           extra_args=["-pix_fmt", "yuv420p"]))
    print(f"Saved: {out_mp4}")
except Exception as e:
    print(f"FFMpeg failed ({e}), saving GIF …")
    ani.save(out_gif, writer=animation.PillowWriter(fps=20))
    print(f"Saved: {out_gif}")

plt.close(fig)
