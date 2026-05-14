#!/usr/bin/env python3
"""
time_reversed_impulses.py
=========================
Animate transducer-array time reversal through the same skull slice used by
wavefront_aberration.py.

The script first records the RF traces from a point source at the target, then
uses those traces in reverse as source signals at the array. The left panel uses
water-only receive delays and emits them through the skull (uncorrected). The
right panel uses skull receive traces and emits them back through the same skull
(time-reversal corrected), so the wavefront converges at the original source.

Run:
  "/Users/vm/INI_code/Julia II/.CondaPkg/.pixi/envs/default/bin/python3" \
      visualisation/time_reversed_impulses.py
"""

import os
import shutil
import tempfile
import time

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["animation.ffmpeg_path"] = "/opt/homebrew/bin/ffmpeg"

import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.ndimage import zoom

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions


# Physical constants
C0 = 1500.0
C_BONE = 4500.0
RHO0 = 1000.0
RHO_BONE = 2800.0
HU_THR = 200

# Grid and geometry. These match wavefront_aberration.py.
LX, LY = 0.100, 0.056
DX = 3e-4
NX = int(round(LX / DX))
NY = int(round(LY / DX))

x_m = np.arange(NX) * DX
y_m = np.arange(NY) * DX

SRC_IX = NX // 2
SRC_IY = int(round(0.04 / DX))

N_SEN = 16
SEN_IX = np.linspace(NX // 6, 5 * NX // 6, N_SEN, dtype=int)

F0 = 500_000.0
PML_SIZE = 20
N_SKIP = 4
T_END = 1.5 * SRC_IY * DX / C0

CT_PATH = (
    "/Users/vm/INI_code/Ultrasound/"
    "DIRU_20240404_human_skull_phase_correction_1_2_(skull_Normal)/"
    "DICOM/PAT_0000/STD_0000/SER_0002/OBJ_0001"
)
SLICE_IDX = 250


def hu_to_rho_c(hu: np.ndarray):
    hu = np.clip(hu, -1000, 3000).astype(np.float32)
    mask = hu >= HU_THR
    c = np.full_like(hu, C0)
    rho = np.full_like(hu, RHO0)
    if mask.any():
        bone = hu[mask]
        h_max = float(np.percentile(bone, 99.5))
        psi = np.clip((h_max - bone) / max(h_max, 1.0), 0.0, 1.0)
        c[mask] = C0 + (C_BONE - C0) * (1.0 - psi)
        rho[mask] = RHO0 + (RHO_BONE - RHO0) * (1.0 - psi)
    return c, rho


def load_skull_maps():
    print("Loading CT skull slice ...", flush=True)
    t0 = time.time()
    raw = [f for f in os.listdir(CT_PATH) if not f.startswith(".")]
    fz = []
    for fname in raw:
        ds = pydicom.dcmread(os.path.join(CT_PATH, fname), stop_before_pixels=True)
        fz.append((float(ds.ImagePositionPatient[2]), fname))
    fz.sort()
    print(f"  {len(fz)} slices sorted ({time.time() - t0:.1f} s)", flush=True)

    target = fz[SLICE_IDX][1]
    ds = pydicom.dcmread(os.path.join(CT_PATH, target))
    hu = ds.pixel_array.astype(np.float32) + float(getattr(ds, "RescaleIntercept", -1024))
    dx_ct = float(ds.PixelSpacing[1]) * 1e-3
    n_rows, n_cols = hu.shape
    print(
        f"  Slice {SLICE_IDX}: shape={hu.shape}, dx={dx_ct * 1e3:.2f} mm, "
        f"HU=[{hu.min():.0f},{hu.max():.0f}]",
        flush=True,
    )

    bone_per_row = (hu > HU_THR).sum(axis=1)
    significant = np.where(bone_per_row > 30)[0]
    if len(significant) == 0:
        raise RuntimeError("No skull found in this CT slice.")
    y_top = int(significant[0])
    if y_top > n_rows // 2:
        hu = hu[::-1, :]
        significant = n_rows - 1 - significant
        y_top = int(significant.min())

    x_ctr = n_cols // 2
    n_above_ct = int(round(0.020 / dx_ct))
    ny_ct = int(round(LY / dx_ct))
    nx_ct = int(round(LX / dx_ct))
    y0 = y_top - n_above_ct
    y1 = y0 + ny_ct
    x0 = x_ctr - nx_ct // 2
    x1 = x0 + nx_ct

    pad = [
        (max(0, -y0), max(0, y1 - n_rows)),
        (max(0, -x0), max(0, x1 - n_cols)),
    ]
    y0 = max(0, y0)
    y1 = min(n_rows, y1)
    x0 = max(0, x0)
    x1 = min(n_cols, x1)
    region = hu[y0:y1, x0:x1]
    if any(v for pp in pad for v in pp):
        region = np.pad(region, pad, constant_values=0.0)

    hu_2d = zoom(
        region.astype(np.float32),
        (NY / region.shape[0], NX / region.shape[1]),
        order=1,
    ).T

    c_map, rho_map = hu_to_rho_c(hu_2d)
    print(f"  Bone fraction in domain: {(c_map > C0 + 20).mean() * 100:.1f}%", flush=True)
    return c_map, rho_map, hu_2d


def make_grid():
    kgrid = kWaveGrid([NX, NY], [DX, DX])
    kgrid.makeTime(C_BONE, cfl=0.3, t_end=T_END)
    return kgrid


def ricker_pulse(nt, dt):
    t_v = np.arange(nt) * dt
    xi = np.pi * F0 * (t_v - 1.5 / F0)
    return ((1.0 - 2.0 * xi**2) * np.exp(-xi**2) * 3500.0).astype(np.float64)


def run_kwave(kgrid, medium, source, sensor, label):
    sim_dir = tempfile.mkdtemp()
    try:
        print(f"\nRunning k-Wave ({label}) ...", flush=True)
        t0 = time.time()
        data = kspaceFirstOrder2D(
            kgrid=kgrid,
            medium=medium,
            source=source,
            sensor=sensor,
            simulation_options=SimulationOptions(
                pml_inside=False,
                pml_size=PML_SIZE,
                data_cast="single",
                data_recast=False,
                save_to_disk=True,
                data_path=sim_dir,
            ),
            execution_options=SimulationExecutionOptions(
                is_gpu_simulation=False,
                delete_data=True,
            ),
        )
        print(f"  Done in {time.time() - t0:.1f} s", flush=True)
        return data
    finally:
        shutil.rmtree(sim_dir, ignore_errors=True)


def simulate_receive_rf(c_map: np.ndarray, rho_map: np.ndarray, label: str):
    kgrid = make_grid()
    nt = int(kgrid.Nt)
    dt = float(kgrid.dt)
    print(
        f"  Grid {NX}x{NY}, NT={nt}, DT={dt * 1e9:.1f} ns, "
        f"T_end={nt * dt * 1e6:.1f} us",
        flush=True,
    )

    medium = kWaveMedium(sound_speed=c_map.astype(np.float32), density=rho_map.astype(np.float32))

    src_mask = np.zeros((NX, NY), dtype=bool)
    src_mask[SRC_IX, SRC_IY] = True
    source = kSource()
    source.p_mask = src_mask
    source.p = ricker_pulse(nt, dt).reshape(1, -1)
    source.p_frequency_ref = F0
    source.medium = medium

    sen_mask = np.zeros((NX, NY), dtype=bool)
    sen_mask[SEN_IX, 0] = True
    sensor = kSensor()
    sensor.mask = sen_mask
    sensor.record = ["p"]

    data = run_kwave(kgrid, medium, source, sensor, f"receive RF: {label}")
    p = np.asarray(data["p"])
    rf = p.T if p.shape[0] == nt else p
    if rf.shape != (N_SEN, nt):
        raise RuntimeError(f"Unexpected RF shape {rf.shape}; expected {(N_SEN, nt)}")
    return rf.astype(np.float32), nt, dt


def simulate_array_emission(c_map: np.ndarray, rho_map: np.ndarray, signals: np.ndarray, label: str):
    kgrid = make_grid()
    nt = int(kgrid.Nt)
    dt = float(kgrid.dt)
    if signals.shape != (N_SEN, nt):
        raise ValueError(f"signals shape {signals.shape}; expected {(N_SEN, nt)}")

    medium = kWaveMedium(sound_speed=c_map.astype(np.float32), density=rho_map.astype(np.float32))

    src_mask = np.zeros((NX, NY), dtype=bool)
    src_mask[SEN_IX, 0] = True
    source = kSource()
    source.p_mask = src_mask
    source.p = signals.astype(np.float64)
    source.p_frequency_ref = F0
    source.medium = medium

    sensor = kSensor()
    sensor.mask = np.ones((NX, NY), dtype=bool)
    sensor.record = ["p"]

    data = run_kwave(kgrid, medium, source, sensor, f"array emission: {label}")
    p_raw = np.asarray(data["p"])
    if p_raw.shape[0] != nt:
        p_raw = p_raw.T
    p_3d = p_raw.T.reshape((NX, NY, nt), order="F")
    snap_idx = np.arange(0, nt, N_SKIP)
    snaps = [p_3d[:, :, i].copy() for i in snap_idx]
    del p_raw, p_3d
    return snaps


def time_reverse_drive(rf_water, rf_skull):
    rf_water = rf_water - rf_water[:, :20].mean(axis=1, keepdims=True)
    rf_skull = rf_skull - rf_skull[:, :20].mean(axis=1, keepdims=True)
    uncorr = rf_water[:, ::-1].copy()
    corr = rf_skull[:, ::-1].copy()

    uncorr *= 3500.0 / max(float(np.abs(uncorr).max()), 1.0)
    corr *= 3500.0 / max(float(np.abs(corr).max()), 1.0)
    return uncorr.astype(np.float32), corr.astype(np.float32)


print("Preparing media ...", flush=True)
c_skull, rho_skull, hu_map = load_skull_maps()
c_water = np.full((NX, NY), C0, dtype=np.float32)
rho_water = np.full((NX, NY), RHO0, dtype=np.float32)

rf_w, NT, DT = simulate_receive_rf(c_water, rho_water, "water only")
rf_s, _, _ = simulate_receive_rf(c_skull, rho_skull, "through skull")
drive_uncorr, drive_corr = time_reverse_drive(rf_w, rf_s)

snaps_uncorr = simulate_array_emission(
    c_skull,
    rho_skull,
    drive_uncorr,
    "uncorrected water-delay drive through skull",
)
snaps_corr = simulate_array_emission(
    c_skull,
    rho_skull,
    drive_corr,
    "skull time-reversal corrected drive",
)

N_FR = min(len(snaps_uncorr), len(snaps_corr))
T_US = np.arange(NT) * DT * 1e6
print(f"\nFrames: {N_FR} ({N_FR / 25:.1f} s @ 25 fps)", flush=True)


# Figure setup
BG = "#070a0f"
FG = "#a8bbc8"
BLUE = "#5599ff"
ORANGE = "#ff8844"
GREEN = "#22ff88"

fig = plt.figure(figsize=(16, 10), facecolor=BG)
gs = GridSpec(
    2,
    2,
    height_ratios=[3.25, 1.0],
    hspace=0.08,
    wspace=0.09,
    left=0.06,
    right=0.96,
    top=0.91,
    bottom=0.06,
)
ax_u = fig.add_subplot(gs[0, 0])
ax_c = fig.add_subplot(gs[0, 1])
ax_du = fig.add_subplot(gs[1, 0])
ax_dc = fig.add_subplot(gs[1, 1])

for ax in (ax_u, ax_c, ax_du, ax_dc):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#18263a")
    ax.tick_params(colors=FG, labelsize=8)

_cd = {
    "red": [(0, 0.04, 0.04), (0.5, 0.0, 0.0), (1, 0.95, 0.95)],
    "green": [(0, 0.14, 0.14), (0.5, 0.0, 0.0), (1, 0.10, 0.10)],
    "blue": [(0, 0.90, 0.90), (0.5, 0.0, 0.0), (1, 0.06, 0.06)],
}
PCMAP = LinearSegmentedColormap("glow", _cd)

sampled_uncorr = snaps_uncorr[:: max(1, len(snaps_uncorr) // 24)]
sampled_corr = snaps_corr[:: max(1, len(snaps_corr) // 24)]
VMAX_UNCORR = max(float(np.percentile(np.abs(s), 99.85)) for s in sampled_uncorr)
VMAX_CORR = max(float(np.percentile(np.abs(s), 99.85)) for s in sampled_corr)
VMAX_UNCORR = max(VMAX_UNCORR, 1.0)
VMAX_CORR = max(VMAX_CORR, 1.0)

X_MM = x_m * 1e3
Y_MM = y_m * 1e3
EXT = [X_MM[0], X_MM[-1], Y_MM[-1], Y_MM[0]]

skull_mask = c_skull > C0 + 20.0
sm_T = skull_mask.T
c_skull_T = c_skull.T
c_in = c_skull_T[sm_T]
c_min = float(c_in.min()) if c_in.size else C0
c_max = float(c_in.max()) if c_in.size else C_BONE
c_norm_T = np.zeros_like(c_skull_T)
if c_in.size:
    c_norm_T[sm_T] = (c_skull_T[sm_T] - c_min) / max(c_max - c_min, 1.0)

bone_cm = plt.cm.YlOrBr
skull_rgba = np.zeros((NY, NX, 4), dtype=np.float32)
if sm_T.any():
    rgba = bone_cm(c_norm_T[sm_T])
    skull_rgba[sm_T, :3] = rgba[:, :3]
    skull_rgba[sm_T, 3] = 0.62

outer_mm = np.full(NX, np.nan)
inner_mm = np.full(NX, np.nan)
for ix in range(NX):
    idx = np.where(skull_mask[ix])[0]
    if len(idx):
        outer_mm[ix] = Y_MM[idx[0]]
        inner_mm[ix] = Y_MM[idx[-1]]


def setup_wave_ax(ax, title, color, vmax):
    im = ax.imshow(
        np.zeros((NY, NX)),
        origin="upper",
        extent=EXT,
        cmap=PCMAP,
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
        interpolation="bilinear",
        zorder=1,
    )
    ax.imshow(skull_rgba, origin="upper", extent=EXT, aspect="equal", interpolation="nearest", zorder=2)
    ax.plot(X_MM, outer_mm, color="#886600", lw=0.9, ls="--", zorder=3, alpha=0.75)
    ax.plot(X_MM, inner_mm, color="#886600", lw=0.9, ls="--", zorder=3, alpha=0.75)
    emit = ax.scatter(
        X_MM[SEN_IX],
        np.ones(N_SEN) * Y_MM[1],
        s=np.ones(N_SEN) * 30,
        c=np.zeros(N_SEN),
        cmap=PCMAP,
        vmin=-1,
        vmax=1,
        marker="v",
        zorder=7,
        linewidths=0.25,
        edgecolors=GREEN,
    )
    ax.scatter([X_MM[SRC_IX]], [Y_MM[SRC_IY]], s=110, c="white", marker="*", zorder=8, lw=0)
    ax.text(
        X_MM[SRC_IX] + 2.0,
        Y_MM[SRC_IY] + 1.1,
        "source",
        color="white",
        fontsize=8,
        ha="left",
        va="center",
        zorder=8,
    )
    ax.text(X_MM[-1] - 1, Y_MM[1] + 0.5, "array", color=GREEN, fontsize=8, ha="right", va="bottom")
    ax.set_xlim(X_MM[0], X_MM[-1])
    ax.set_ylim(Y_MM[-1], Y_MM[0])
    ax.set_xlabel("Lateral [mm]", color=FG, fontsize=9)
    ax.set_ylabel("Depth [mm]", color=FG, fontsize=9)
    ax.set_title(title, color=color, fontsize=13.5, fontweight="bold", pad=7)
    return im, emit


im_u, emit_u = setup_wave_ax(
    ax_u,
    "Uncorrected: water delays emitted through skull",
    BLUE,
    VMAX_UNCORR,
)
im_c, emit_c = setup_wave_ax(
    ax_c,
    "Corrected: time-reversed skull RF emitted back",
    ORANGE,
    VMAX_CORR,
)

leg = ax_u.legend(
    handles=[
        mpatches.Patch(color="#0a22e6", label="Rarefaction (-p)"),
        mpatches.Patch(color="#e61a0a", label="Compression (+p)"),
        mpatches.Patch(color=GREEN, label="Emitting array elements"),
    ],
    loc="lower left",
    fontsize=7.5,
    framealpha=0.45,
    facecolor=BG,
    edgecolor="#18263a",
    labelcolor=FG,
)
ax_u.add_artist(leg)

from mpl_toolkits.axes_grid1 import make_axes_locatable

div = make_axes_locatable(ax_c)
cax = div.append_axes("right", size="2.5%", pad=0.04)
sm = plt.cm.ScalarMappable(cmap=bone_cm, norm=plt.Normalize(c_min, c_max))
cb = fig.colorbar(sm, cax=cax)
cb.set_label("c [m/s]", color=FG, fontsize=8)
cb.ax.tick_params(colors=FG, labelsize=7)
cb.outline.set_edgecolor("#18263a")

drive_scale = max(float(np.abs(drive_uncorr).max()), float(np.abs(drive_corr).max()), 1.0) / 0.65


def setup_drive_ax(ax, color, label):
    ax.set_facecolor("#0b1520")
    for sp in ax.spines.values():
        sp.set_edgecolor("#18263a")
    ax.tick_params(colors=FG, labelsize=7)
    for s in range(N_SEN):
        ax.axhline(s, color="#18263a", lw=0.5, zorder=1)
    lines = [ax.plot([], [], color=color, lw=0.8, alpha=0.9, zorder=3)[0] for _ in range(N_SEN)]
    cursor, = ax.plot([], [], color=FG, lw=0.8, ls="--", alpha=0.35, zorder=5)
    ax.set_xlim(0, T_US[-1])
    ax.set_ylim(-0.6, N_SEN - 0.4)
    ax.set_xlabel("Emission time [us]", color=FG, fontsize=9)
    tick_s = [0, 4, 8, 12, N_SEN - 1]
    ax.set_yticks(tick_s)
    ax.set_yticklabels([f"{X_MM[SEN_IX[s]]:.0f} mm" for s in tick_s], fontsize=6.5, color=FG)
    ax.set_ylabel("Element lateral pos.", color=FG, fontsize=8)
    ax.set_title(f"Time-reversed source signals - {label}", color=color, fontsize=9.5, pad=3)
    return lines, cursor


lines_u, cursor_u = setup_drive_ax(ax_du, BLUE, "water-delay")
lines_c, cursor_c = setup_drive_ax(ax_dc, ORANGE, "skull-corrected")

fig.suptitle(
    f"Array Time Reversal Through Skull - {F0 / 1e3:.0f} kHz Ricker Impulse "
    f"[CT slice {SLICE_IDX}]",
    color="white",
    fontsize=12.5,
    fontweight="bold",
    y=0.965,
)
t_label = fig.text(0.5, 0.927, "t = 0.0 us", ha="center", color=FG, fontsize=11)


def update(fr):
    n = min(fr * N_SKIP, NT - 1)
    t_us = T_US[n]
    t_label.set_text(f"t = {t_us:.1f} us")

    im_u.set_data(snaps_uncorr[fr].T)
    im_c.set_data(snaps_corr[fr].T)

    vals_u = drive_uncorr[:, n] / max(float(np.abs(drive_uncorr).max()), 1.0)
    vals_c = drive_corr[:, n] / max(float(np.abs(drive_corr).max()), 1.0)
    emit_u.set_array(vals_u)
    emit_c.set_array(vals_c)
    emit_u.set_sizes(35 + 300 * np.abs(vals_u))
    emit_c.set_sizes(35 + 300 * np.abs(vals_c))

    t_slice = T_US[: n + 1]
    for s in range(N_SEN):
        lines_u[s].set_data(t_slice, drive_uncorr[s, : n + 1] / drive_scale + s)
        lines_c[s].set_data(t_slice, drive_corr[s, : n + 1] / drive_scale + s)
    cursor_u.set_data([t_us, t_us], [-0.6, N_SEN - 0.4])
    cursor_c.set_data([t_us, t_us], [-0.6, N_SEN - 0.4])

    return [
        im_u,
        im_c,
        emit_u,
        emit_c,
        *lines_u,
        *lines_c,
        cursor_u,
        cursor_c,
        t_label,
    ]


ani = animation.FuncAnimation(fig, update, frames=N_FR, interval=40, blit=True)

out_dir = os.path.dirname(os.path.abspath(__file__))
out_mp4 = os.path.join(out_dir, "time_reversed_impulses.mp4")
out_gif = os.path.join(out_dir, "time_reversed_impulses.gif")

try:
    print(f"\nSaving {out_mp4} ...", flush=True)
    ani.save(
        out_mp4,
        writer=animation.FFMpegWriter(
            fps=75,
            bitrate=4500,
            extra_args=["-pix_fmt", "yuv420p"],
        ),
    )
    print(f"Saved: {out_mp4}", flush=True)
except Exception as exc:
    print(f"FFmpeg failed ({exc}), saving GIF ...", flush=True)
    ani.save(out_gif, writer=animation.PillowWriter(fps=20))
    print(f"Saved: {out_gif}", flush=True)

plt.close(fig)
