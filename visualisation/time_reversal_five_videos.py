#!/usr/bin/env python3
"""
time_reversal_five_videos.py
============================
Render a five-part time-reversal sequence using the same geometry and CT slice
as wavefront_aberration.py:

1. Water medium point source propagating outward.
2. Water-medium time reversal, leaving a cumulative intensity map.
3. The same water delays sent through the skull, leaving a skewed intensity map.
4. Skull-medium point source propagating outward while recording RF traces.
5. Skull RF traces time-reversed and sent back in, leaving a focused intensity map.

Run:
  "/Users/vm/INI_code/Julia II/.CondaPkg/.pixi/envs/default/bin/python3" -B \
      visualisation/time_reversal_five_videos.py
"""

import os
import shutil
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

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


C0 = 1500.0
C_BONE = 4500.0
RHO0 = 1000.0
RHO_BONE = 2800.0
HU_THR = 200

LX, LY = 0.100, 0.056
DX = 3e-4
NX = int(round(LX / DX))
NY = int(round(LY / DX))

x_m = np.arange(NX) * DX
y_m = np.arange(NY) * DX
X_MM = x_m * 1e3
Y_MM = y_m * 1e3
EXT = [X_MM[0], X_MM[-1], Y_MM[-1], Y_MM[0]]

SRC_IX = NX // 2
SRC_IY = int(round(0.04 / DX))

N_SEN = 16
SEN_IX = np.linspace(NX // 6, 5 * NX // 6, N_SEN, dtype=int)

F0 = 500_000.0
PML_SIZE = 20
N_SKIP = 4
T_END = 1.5 * SRC_IY * DX / C0
DT_FIXED = 0.3 * DX / C_BONE
ARRAY_NT_TARGET = 2500
ARRAY_T_END = (ARRAY_NT_TARGET - 1) * DT_FIXED
FPS = 75
INTENSITY_HOLD_SECONDS = 3
ARRAY_PRESSURE_VMAX = 3.0

CT_PATH = (
    "/Users/vm/INI_code/Ultrasound/"
    "DIRU_20240404_human_skull_phase_correction_1_2_(skull_Normal)/"
    "DICOM/PAT_0000/STD_0000/SER_0002/OBJ_0001"
)
SLICE_IDX = 250

BG = "#070a0f"
PANEL_BG = "#0b1520"
FG = "#a8bbc8"
BLUE = "#5599ff"
ORANGE = "#ff8844"
GREEN = "#22ff88"

PRESSURE_CMAP = LinearSegmentedColormap(
    "pressure_glow",
    {
        "red": [(0, 0.04, 0.04), (0.5, 0.0, 0.0), (1, 0.95, 0.95)],
        "green": [(0, 0.14, 0.14), (0.5, 0.0, 0.0), (1, 0.10, 0.10)],
        "blue": [(0, 0.90, 0.90), (0.5, 0.0, 0.0), (1, 0.06, 0.06)],
    },
)
INTENSITY_CMAP = plt.cm.magma
BONE_CMAP = plt.cm.YlOrBr


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


def make_grid(t_end=T_END):
    kgrid = kWaveGrid([NX, NY], [DX, DX])
    kgrid.makeTime(C_BONE, cfl=0.3, t_end=t_end)
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


def unpack_full_field(data, nt):
    p_raw = np.asarray(data["p"])
    if p_raw.shape[0] != nt:
        p_raw = p_raw.T
    p_3d = p_raw.T.reshape((NX, NY, nt), order="F")
    snap_idx = np.arange(0, nt, N_SKIP)
    snaps = [p_3d[:, :, i].copy() for i in snap_idx]
    rf = p_3d[SEN_IX, 0, :].copy().astype(np.float32)
    del p_raw, p_3d
    return snaps, rf


def simulate_point_source(c_map: np.ndarray, rho_map: np.ndarray, label: str):
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

    sensor = kSensor()
    sensor.mask = np.ones((NX, NY), dtype=bool)
    sensor.record = ["p"]

    data = run_kwave(kgrid, medium, source, sensor, f"point source: {label}")
    snaps, rf = unpack_full_field(data, nt)
    return snaps, rf, nt, dt


def simulate_array_emission(c_map: np.ndarray, rho_map: np.ndarray, signals: np.ndarray, label: str):
    kgrid = make_grid(ARRAY_T_END)
    nt = int(kgrid.Nt)
    dt = float(kgrid.dt)
    if signals.shape[0] != N_SEN or signals.shape[1] > nt:
        raise ValueError(f"signals shape {signals.shape}; expected ({N_SEN}, <= {nt})")
    padded_signals = np.zeros((N_SEN, nt), dtype=np.float32)
    padded_signals[:, : signals.shape[1]] = signals

    medium = kWaveMedium(sound_speed=c_map.astype(np.float32), density=rho_map.astype(np.float32))
    src_mask = np.zeros((NX, NY), dtype=bool)
    src_mask[SEN_IX, 0] = True

    source = kSource()
    source.p_mask = src_mask
    source.p = padded_signals.astype(np.float64)
    source.p_frequency_ref = F0
    source.medium = medium

    sensor = kSensor()
    sensor.mask = np.ones((NX, NY), dtype=bool)
    sensor.record = ["p"]

    data = run_kwave(kgrid, medium, source, sensor, f"array emission: {label}")
    snaps, _ = unpack_full_field(data, nt)
    return snaps, padded_signals, nt, dt


def time_reverse_drive(rf: np.ndarray):
    centered = rf - rf[:, :20].mean(axis=1, keepdims=True)
    drive = centered[:, ::-1].copy()
    drive *= 3500.0 / max(float(np.abs(drive).max()), 1.0)
    return drive.astype(np.float32)


def pressure_vmax(snaps, top_crop=0):
    sampled = snaps[:: max(1, len(snaps) // 24)]
    values = []
    for snap in sampled:
        field = snap[:, top_crop:] if top_crop else snap
        values.append(float(np.percentile(np.abs(field), 97.5)))
    return max(1.0, max(values))


def intensity_vmax(snaps):
    accum = np.zeros_like(snaps[0], dtype=np.float32)
    for snap in snaps:
        np.maximum(accum, np.abs(snap), out=accum)
    return max(1.0, float(np.percentile(accum, 99.75)))


def make_skull_overlay(c_skull):
    skull_mask = c_skull > C0 + 20.0
    sm_t = skull_mask.T
    c_skull_t = c_skull.T
    c_in = c_skull_t[sm_t]
    c_min = float(c_in.min()) if c_in.size else C0
    c_max = float(c_in.max()) if c_in.size else C_BONE

    c_norm_t = np.zeros_like(c_skull_t)
    if c_in.size:
        c_norm_t[sm_t] = (c_skull_t[sm_t] - c_min) / max(c_max - c_min, 1.0)

    skull_rgba = np.zeros((NY, NX, 4), dtype=np.float32)
    if sm_t.any():
        rgba = BONE_CMAP(c_norm_t[sm_t])
        skull_rgba[sm_t, :3] = rgba[:, :3]
        skull_rgba[sm_t, 3] = 0.58

    outer_mm = np.full(NX, np.nan)
    inner_mm = np.full(NX, np.nan)
    for ix in range(NX):
        idx = np.where(skull_mask[ix])[0]
        if len(idx):
            outer_mm[ix] = Y_MM[idx[0]]
            inner_mm[ix] = Y_MM[idx[-1]]
    return skull_rgba, outer_mm, inner_mm, c_min, c_max


def style_axis(ax):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#18263a")
    ax.tick_params(colors=FG, labelsize=8)


def setup_trace_axis(ax, line_color, label, t_us, data):
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#18263a")
    ax.tick_params(colors=FG, labelsize=7)
    for s in range(N_SEN):
        ax.axhline(s, color="#18263a", lw=0.5, zorder=1)
    lines = [ax.plot([], [], color=line_color, lw=0.78, alpha=0.92, zorder=3)[0] for _ in range(N_SEN)]
    cursor, = ax.plot([], [], color=FG, lw=0.8, ls="--", alpha=0.35, zorder=5)
    ax.set_xlim(0, t_us[-1])
    ax.set_ylim(-0.6, N_SEN - 0.4)
    ax.set_xlabel("Time [us]", color=FG, fontsize=9)
    tick_s = [0, 4, 8, 12, N_SEN - 1]
    ax.set_yticks(tick_s)
    ax.set_yticklabels([f"{X_MM[SEN_IX[s]]:.0f} mm" for s in tick_s], fontsize=6.5, color=FG)
    ax.set_ylabel("Array lateral pos.", color=FG, fontsize=8)
    ax.set_title(label, color=line_color, fontsize=9.5, pad=3)
    scale = max(float(np.abs(data).max()), 1.0) / 0.65
    return lines, cursor, scale


def render_video(
    out_path,
    title,
    snaps,
    nt,
    dt,
    trace_data,
    trace_label,
    trace_color,
    skull_overlay=None,
    show_intensity=False,
    emitter_data=None,
):
    print(f"\nRendering {os.path.basename(out_path)} ...", flush=True)
    n_frames = len(snaps)
    hold_frames = FPS * INTENSITY_HOLD_SECONDS if show_intensity else 0
    total_frames = n_frames + hold_frames
    t_us = np.arange(nt) * dt * 1e6
    wave_top_crop = 8 if emitter_data is not None else 0
    p_vmax = pressure_vmax(snaps, top_crop=wave_top_crop)
    if emitter_data is not None:
        p_vmax = ARRAY_PRESSURE_VMAX
    i_vmax = intensity_vmax(snaps) if show_intensity else 1.0
    final_intensity = None
    if show_intensity:
        final_intensity = np.zeros_like(snaps[0], dtype=np.float32)
        for snap in snaps:
            np.maximum(final_intensity, np.abs(snap), out=final_intensity)
        final_intensity = np.clip(final_intensity / i_vmax, 0.0, 1.0)

    fig = plt.figure(figsize=(12, 9), facecolor=BG)
    gs = GridSpec(
        2,
        1,
        height_ratios=[3.4, 1.0],
        hspace=0.17,
        left=0.08,
        right=0.96,
        top=0.90,
        bottom=0.08,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_tr = fig.add_subplot(gs[1, 0])
    style_axis(ax)

    pressure_im = ax.imshow(
        np.zeros((NY, NX)),
        origin="upper",
        extent=EXT,
        cmap=PRESSURE_CMAP,
        vmin=-p_vmax,
        vmax=p_vmax,
        aspect="equal",
        interpolation="bilinear",
        zorder=1,
    )
    intensity_im = None
    if show_intensity:
        intensity_im = ax.imshow(
            np.zeros((NY, NX)),
            origin="upper",
            extent=EXT,
            cmap=INTENSITY_CMAP,
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
            interpolation="bilinear",
            zorder=2,
        )
        intensity_im.set_alpha(np.zeros((NY, NX), dtype=np.float32))
        intensity_im.set_visible(False)

    if skull_overlay is not None:
        skull_rgba, outer_mm, inner_mm, c_min, c_max = skull_overlay
        ax.imshow(skull_rgba, origin="upper", extent=EXT, aspect="equal", interpolation="nearest", zorder=3)
        ax.plot(X_MM, outer_mm, color="#886600", lw=0.9, ls="--", zorder=4, alpha=0.78)
        ax.plot(X_MM, inner_mm, color="#886600", lw=0.9, ls="--", zorder=4, alpha=0.78)

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="2.2%", pad=0.04)
        sm = plt.cm.ScalarMappable(cmap=BONE_CMAP, norm=plt.Normalize(c_min, c_max))
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label("c [m/s]", color=FG, fontsize=8)
        cb.ax.tick_params(colors=FG, labelsize=7)
        cb.outline.set_edgecolor("#18263a")

    emit = ax.scatter(
        X_MM[SEN_IX],
        np.ones(N_SEN) * Y_MM[1],
        s=np.ones(N_SEN) * 34,
        c=np.zeros(N_SEN),
        cmap=PRESSURE_CMAP,
        vmin=-1,
        vmax=1,
        marker="v",
        zorder=7,
        linewidths=0.25,
        edgecolors=GREEN,
    )
    ax.scatter([X_MM[SRC_IX]], [Y_MM[SRC_IY]], s=105, c="white", marker="*", zorder=8, lw=0)
    ax.text(X_MM[-1] - 1, Y_MM[1] + 0.5, "array", color=GREEN, fontsize=8, ha="right", va="bottom")
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
    ax.set_xlim(X_MM[0], X_MM[-1])
    ax.set_ylim(Y_MM[-1], Y_MM[0])
    ax.set_xlabel("")
    ax.set_ylabel("Depth [mm]", color=FG, fontsize=9)

    legend_handles = [
        mpatches.Patch(color="#0a22e6", label="Rarefaction (-p)"),
        mpatches.Patch(color="#e61a0a", label="Compression (+p)"),
        mpatches.Patch(color=GREEN, label="Array elements"),
    ]
    if show_intensity:
        legend_handles.append(mpatches.Patch(color="#ffb000", label="Final intensity map"))
    leg = ax.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=7.5,
        framealpha=0.45,
        facecolor=BG,
        edgecolor="#18263a",
        labelcolor=FG,
    )
    ax.add_artist(leg)

    lines, cursor, trace_scale = setup_trace_axis(ax_tr, trace_color, trace_label, t_us, trace_data)
    fig.suptitle(title, color="white", fontsize=12.5, fontweight="bold", y=0.965)
    time_label = fig.text(0.5, 0.92, "t = 0.0 us", ha="center", color=FG, fontsize=11)

    accum = np.zeros_like(snaps[0], dtype=np.float32)

    def update(fr):
        if fr == 0:
            accum.fill(0.0)
        wave_fr = min(fr, n_frames - 1)
        n = min(wave_fr * N_SKIP, nt - 1)
        is_hold = show_intensity and fr >= n_frames
        time_label.set_text(
            f"final intensity map ({INTENSITY_HOLD_SECONDS} s hold)"
            if is_hold
            else f"t = {t_us[n]:.1f} us"
        )
        if is_hold:
            pressure_im.set_data(np.zeros((NY, NX), dtype=np.float32))
        else:
            pressure_im.set_data(snaps[wave_fr].T)

        artists = [pressure_im, emit, *lines, cursor, time_label]
        if show_intensity:
            if is_hold:
                intensity_im.set_visible(True)
                intensity_im.set_data(final_intensity.T)
                intensity_im.set_alpha((0.08 + 0.82 * np.power(final_intensity, 0.42)).T)
            else:
                intensity_im.set_visible(False)
            artists.append(intensity_im)

        if emitter_data is not None and not is_hold:
            vals = emitter_data[:, n] / max(float(np.abs(emitter_data).max()), 1.0)
            emit.set_array(vals)
            emit.set_sizes(35 + 300 * np.abs(vals))
        else:
            emit.set_array(np.zeros(N_SEN))
            emit.set_sizes(np.ones(N_SEN) * 34)

        trace_n = nt - 1 if is_hold else n
        t_slice = t_us[: trace_n + 1]
        for s in range(N_SEN):
            lines[s].set_data(t_slice, trace_data[s, : trace_n + 1] / trace_scale + s)
        cursor.set_data([t_us[trace_n], t_us[trace_n]], [-0.6, N_SEN - 0.4])
        return artists

    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=40, blit=True)
    ani.save(
        out_path,
        writer=animation.FFMpegWriter(fps=FPS, bitrate=4500, extra_args=["-pix_fmt", "yuv420p"]),
    )
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "time_reversal_five_videos")
    os.makedirs(out_dir, exist_ok=True)

    print("Preparing media ...", flush=True)
    c_skull, rho_skull, _ = load_skull_maps()
    c_water = np.full((NX, NY), C0, dtype=np.float32)
    rho_water = np.full((NX, NY), RHO0, dtype=np.float32)
    skull_overlay = make_skull_overlay(c_skull)

    snaps_w_src, rf_w, nt, dt = simulate_point_source(c_water, rho_water, "water")
    render_video(
        os.path.join(out_dir, "01_water_point_source.mp4"),
        f"1. Water Medium Point Source - {F0 / 1e3:.0f} kHz Ricker Impulse",
        snaps_w_src,
        nt,
        dt,
        rf_w,
        "RF traces recorded at the water array",
        BLUE,
    )

    drive_w = time_reverse_drive(rf_w)
    snaps_w_tr, drive_w_emit, nt_w_emit, dt_w_emit = simulate_array_emission(
        c_water,
        rho_water,
        drive_w,
        "water time reversal",
    )
    render_video(
        os.path.join(out_dir, "02_water_time_reversal_intensity.mp4"),
        "2. Water Time Reversal - Waves, Then Final Intensity Map",
        snaps_w_tr,
        nt_w_emit,
        dt_w_emit,
        drive_w_emit,
        "Time-reversed impulses emitted by the array",
        BLUE,
        show_intensity=True,
        emitter_data=drive_w_emit,
    )
    del snaps_w_tr

    snaps_skull_uncorr, drive_w_skull_emit, nt_w_skull_emit, dt_w_skull_emit = simulate_array_emission(
        c_skull,
        rho_skull,
        drive_w,
        "water delays through skull",
    )
    render_video(
        os.path.join(out_dir, "03_water_delays_through_skull_skewed_intensity.mp4"),
        "3. Water Delays Through Skull - Waves, Then Skewed Intensity Map",
        snaps_skull_uncorr,
        nt_w_skull_emit,
        dt_w_skull_emit,
        drive_w_skull_emit,
        "Same water-derived delays emitted through the skull",
        BLUE,
        skull_overlay=skull_overlay,
        show_intensity=True,
        emitter_data=drive_w_skull_emit,
    )
    del snaps_skull_uncorr, snaps_w_src

    snaps_s_src, rf_s, _, _ = simulate_point_source(c_skull, rho_skull, "skull")
    render_video(
        os.path.join(out_dir, "04_skull_point_source_record_rf.mp4"),
        "4. Skull Medium Point Source - Outward Propagation and Recorded RF",
        snaps_s_src,
        nt,
        dt,
        rf_s,
        "RF traces recorded after propagation through the skull",
        ORANGE,
        skull_overlay=skull_overlay,
    )
    del snaps_s_src

    drive_s = time_reverse_drive(rf_s)
    snaps_s_tr, drive_s_emit, nt_s_emit, dt_s_emit = simulate_array_emission(
        c_skull,
        rho_skull,
        drive_s,
        "skull time reversal",
    )
    render_video(
        os.path.join(out_dir, "05_skull_time_reversal_focused_intensity.mp4"),
        "5. Skull-Corrected Time Reversal - Waves, Then Focused Intensity Map",
        snaps_s_tr,
        nt_s_emit,
        dt_s_emit,
        drive_s_emit,
        "Skull RF traces time-reversed and emitted back",
        ORANGE,
        skull_overlay=skull_overlay,
        show_intensity=True,
        emitter_data=drive_s_emit,
    )

    print("\nAll five videos written to:", out_dir, flush=True)


if __name__ == "__main__":
    main()
