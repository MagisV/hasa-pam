"""
Napari PAM window convergence viewer.

Loads a napari_data.h5 export and displays the SOS field, vascular network,
truth sources, and cumulative HASA reconstruction.

Interactive mode (default):
    The cumulative HASA stack is a 4-D layer (window × depth × y × z) with a
    slider so you can scrub through convergence manually.

Animation mode (--animate):
    Two-phase cinematic MP4:
      Phase 1 — camera orbits 360° around the skull + vascular network
      Phase 2 — camera locks at a fixed view; cumulative HASA builds up
                window by window

Usage:
    python show_convergence.py --run-dir path/to/run_folder
    python show_convergence.py --run-dir path/to/run_folder --animate
    python show_convergence.py --run-dir path/to/run_folder --animate --fps 24 --orbit-frames 72

Requirements:
    pip install napari[all] h5py imageio imageio-ffmpeg
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


# ── Data loading ──────────────────────────────────────────────────────────────

def load_run(run_dir: Path) -> dict:
    hdf = run_dir / "napari_data.h5"
    if not hdf.exists():
        raise FileNotFoundError(
            f"{hdf} not found.\n"
            "Export first:\n"
            "  julia --project=. visualisation/napari/export_for_napari.jl "
            f"--run-dir={run_dir}"
        )

    data = {}
    with h5py.File(hdf, "r") as f:
        for key in ("sos", "x_mm", "y_mm", "z_mm",
                    "source_depth_mm", "source_y_mm", "source_z_mm",
                    "truth_mask"):
            data[key] = f[key][:]

        data["receiver_row"] = int(f["receiver_row"][()])

        if "centerline_ids" in f:
            data["cl_depth"] = f["centerline_depth_mm"][:]
            data["cl_y"]     = f["centerline_y_mm"][:]
            data["cl_z"]     = f["centerline_z_mm"][:]
            data["cl_ids"]   = f["centerline_ids"][:]

        keys = sorted(k for k in f if k.startswith("window_"))
        data["windows"] = [f[k][:] for k in keys]

    return data


# ── Cumulative stack ──────────────────────────────────────────────────────────

def precompute_cumulatives(windows: list) -> np.ndarray:
    """Return shape (n_windows, nx, ny, nz) of normalized cumulative means."""
    n = len(windows)
    if n == 0:
        raise ValueError("No windows found in the data file.")
    shape  = windows[0].shape
    result = np.zeros((n, *shape), dtype=np.float32)
    cumul  = np.zeros(shape, dtype=np.float64)
    for i, w in enumerate(windows):
        cumul += w.astype(np.float64)
        mean   = cumul / (i + 1)
        vmax   = float(mean.max())
        result[i] = (mean / vmax if vmax > 0 else mean).astype(np.float32)
    return result


# ── Napari layer setup ────────────────────────────────────────────────────────

def setup_viewer(data: dict, cumulatives: np.ndarray):
    """
    Build the napari viewer.

    cumulatives: (n_windows, nx, ny, nz) — added as a 4-D HASA layer so
    napari shows a window-index slider in interactive mode.
    """
    import napari

    x_mm = data["x_mm"]
    y_mm = data["y_mm"]
    z_mm = data["z_mm"]

    dx = float(np.diff(x_mm).mean())
    dy = float(np.diff(y_mm).mean())
    dz = float(np.diff(z_mm).mean())
    scale     = (dx, dy, dz)
    translate = (x_mm[0], y_mm[0], z_mm[0])

    viewer = napari.Viewer(ndisplay=3, title="PAM window convergence")

    # ── SOS / skull ──────────────────────────────────────────────────────────
    viewer.add_image(
        data["sos"],
        name="skull (SOS)",
        scale=scale,
        translate=translate,
        colormap="gray",
        contrast_limits=[1480.0, 2600.0],
        gamma=0.7,
        opacity=0.35,
        blending="additive",
    )

    # ── Cumulative HASA — 4-D: (window, depth, y, z) ─────────────────────────
    # Axis-0 scale=1.0 puts the window-index slider in screen units.
    # The spatial axes (1,2,3) use the same mm scale as every other layer.
    hasa_layer = viewer.add_image(
        cumulatives,
        name="cumulative HASA",
        scale=(1.0, dx, dy, dz),
        translate=(0.0, *translate),
        colormap="inferno",
        rendering="translucent",
        contrast_limits=[0.0, 1.0],
        opacity=0.9,
        blending="additive",
    )

    # ── Truth mask ───────────────────────────────────────────────────────────
    viewer.add_image(
        data["truth_mask"].astype(np.float32),
        name="truth mask",
        scale=scale,
        translate=translate,
        colormap="cyan",
        rendering="iso",
        iso_threshold=0.5,
        opacity=0.25,
        blending="additive",
    )

    # ── Source ground-truth points ────────────────────────────────────────────
    rr  = data["receiver_row"]
    x0  = x_mm[rr]
    src_pts = np.column_stack([
        x0 + data["source_depth_mm"],
        data["source_y_mm"],
        data["source_z_mm"],
    ])
    viewer.add_points(
        src_pts,
        name="truth sources",
        size=1.5,
        face_color="red",
        symbol="cross",
        blending="translucent",
    )

    # ── Vascular network centerlines ─────────────────────────────────────────
    if "cl_ids" in data:
        ids    = data["cl_ids"]
        coords = np.column_stack([
            x0 + data["cl_depth"],
            data["cl_y"],
            data["cl_z"],
        ])
        tracks = np.column_stack([ids, coords])
        viewer.add_tracks(
            tracks,
            name="vascular network",
            tail_width=2,
            tail_length=len(tracks),
            colormap="gray",
            blending="additive",
        )

    # ── Receiver plane ───────────────────────────────────────────────────────
    y_ext = [y_mm[0], y_mm[-1]]
    z_ext = [z_mm[0], z_mm[-1]]
    plane_corners = np.array([
        [x0, y_ext[0], z_ext[0]],
        [x0, y_ext[0], z_ext[1]],
        [x0, y_ext[1], z_ext[1]],
        [x0, y_ext[1], z_ext[0]],
    ])
    viewer.add_shapes(
        [plane_corners],
        shape_type="polygon",
        name="receiver array",
        face_color=[1.0, 0.0, 1.0, 0.15],
        edge_color=[1.0, 0.0, 1.0, 0.8],
        edge_width=0.5,
    )

    viewer.camera.angles = (0.0, -35.0, 45.0)
    viewer.camera.zoom   = 1.8

    return viewer, hasa_layer


# ── Animation ─────────────────────────────────────────────────────────────────

def animate(viewer, hasa_layer, run_dir: Path,
            fps: int = 24, orbit_frames: int = 72):
    """
    Two-phase cinematic animation saved to napari_cinematic.mp4.

    Phase 1 — skull orbit:
        HASA layer hidden; camera rotates 360° so the viewer can appreciate the
        3-D skull geometry and vascular network before any reconstruction is shown.

    Phase 2 — convergence build-up:
        Camera locks at a fixed view angle; the window-index slider steps through
        each cumulative HASA volume one frame at a time.
    """
    import imageio

    frames_dir = run_dir / "napari_frames"
    frames_dir.mkdir(exist_ok=True)

    n_windows = hasa_layer.data.shape[0]
    paths     = []
    fnum      = 0

    def snap(title: str):
        nonlocal fnum
        viewer.title = title
        img = viewer.screenshot(canvas_only=True, flash=False)
        p   = frames_dir / f"frame_{fnum:04d}.png"
        imageio.imwrite(p, img)
        paths.append(p)
        fnum += 1

    # ── Phase 1: orbit ────────────────────────────────────────────────────────
    print(f"  phase 1: skull orbit ({orbit_frames} frames)")
    hasa_layer.visible = False
    elev = -25.0
    for i in range(orbit_frames):
        azimuth = (i / orbit_frames) * 360.0 - 90.0
        viewer.camera.angles = (elev, azimuth, 0.0)
        snap(f"skull overview — {i + 1}/{orbit_frames}")

    # ── Phase 2: HASA convergence ─────────────────────────────────────────────
    print(f"  phase 2: HASA convergence ({n_windows} windows)")
    hasa_layer.visible = True
    viewer.camera.angles = (elev, 45.0, 0.0)
    viewer.camera.zoom   = 2.0

    for i in range(n_windows):
        step       = list(viewer.dims.current_step)
        step[0]    = i
        viewer.dims.current_step = tuple(step)
        snap(f"window convergence — {i + 1}/{n_windows}")

    # ── Encode ────────────────────────────────────────────────────────────────
    mp4 = run_dir / "napari_cinematic.mp4"
    print(f"  encoding {len(paths)} frames → {mp4}")
    with imageio.get_writer(str(mp4), fps=fps) as writer:
        for p in paths:
            writer.append_data(imageio.imread(str(p)))

    print(f"Saved: {mp4}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="napari PAM convergence viewer")
    parser.add_argument("--run-dir", required=True,
                        help="Path to window convergence run folder")
    parser.add_argument("--animate", action="store_true",
                        help="Generate two-phase cinematic animation instead of interactive view")
    parser.add_argument("--fps", type=int, default=24,
                        help="Animation frame rate (default: 24)")
    parser.add_argument("--orbit-frames", type=int, default=72,
                        help="Number of frames for the skull-orbit intro (default: 72 = 3 s at 24 fps)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    data    = load_run(run_dir)

    print(f"Precomputing {len(data['windows'])} cumulative windows...")
    cumulatives = precompute_cumulatives(data["windows"])
    print(f"  stack shape: {cumulatives.shape}  ({cumulatives.nbytes / 1e9:.2f} GB)")

    viewer, hasa_layer = setup_viewer(data, cumulatives)

    if args.animate:
        animate(viewer, hasa_layer, run_dir,
                fps=args.fps, orbit_frames=args.orbit_frames)
    else:
        import napari
        napari.run()


if __name__ == "__main__":
    main()
