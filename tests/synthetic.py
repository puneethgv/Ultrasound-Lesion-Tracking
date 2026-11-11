"""Build a small fake dataset on disk for end-to-end tests (random numbers only, no patient data).

Layout mirrors the real one::

    <root>/gui/gui_<class>[_halle]/<case>.xlsx                    GUI boxes (Large_ROI / Small_ROI)
    <root>/gui/raw_data_<class>[_halle]/<case>/<scan>_extracted/  C3_large_rf.raw + .yml
    <root>/FLL_ROI/<case>_roi.pkl                                 QuantUS ROI export
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from usloc.labels.quantus import QuantusScGeometry

N_LINES, N_SAMPLES = 64, 800
DEPTH_MM, RADIUS_MM, PITCH_UM = 98.0, 45.0, 900.0

Box = tuple[float, float, float, float]  # (line0, sample0, line1, sample1)


def moving_tissue(n_frames: int, lesion: Box, velocity=(0, 0), seed: int = 0) -> np.ndarray:
    """RF cine ``(frames, lines, samples)``: speckle with a bright block at ``lesion`` in frame 0.
    Everything moves rigidly by ``velocity`` = (lines, samples) per frame, like tissue under the probe."""
    vl, vs = int(velocity[0]), int(velocity[1])
    pl, ps = abs(vl) * n_frames + 1, abs(vs) * n_frames + 1
    rng = np.random.default_rng(seed)
    field = rng.standard_normal((N_LINES + 2 * pl, N_SAMPLES + 2 * ps)) * 20
    l0, s0, l1, s1 = (int(v) for v in lesion)
    field[pl + l0: pl + l1 + 1, ps + s0: ps + s1 + 1] *= 40
    frames = [
        field[pl - k * vl: pl - k * vl + N_LINES, ps - k * vs: ps - k * vs + N_SAMPLES]
        for k in range(n_frames)
    ]
    return np.stack(frames).astype(np.int16)


def write_recording(folder: Path, rf: np.ndarray, preset: str = "C3_large") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    n_frames, n_lines, n_samples = rf.shape
    (folder / f"{preset}_rf.yml").write_text(
        f"type: RF\nsize:\n  samples per line: {n_samples}\n  number of lines: {n_lines}\n"
        f"  sample size: 2 bytes\nframes: {n_frames}\nimaging depth: {DEPTH_MM:g} mm\n"
        f"probe:\n  radius: {RADIUS_MM:g} mm\n  pitch: {PITCH_UM:g} um\n"
    )
    with open(folder / f"{preset}_rf.raw", "wb") as f:
        np.array([2, n_frames, n_lines, n_samples, 2], dtype="<i4").tofile(f)
        for k in range(n_frames):
            np.array([k], dtype="<i8").tofile(f)
            rf[k].astype("<i2").tofile(f)


def write_roi(path: Path, box: Box, *, scan: str = "raw_0_0", frame: int = 0,
              start_depth_factor: float = 0.25) -> None:
    """Save the outline of ``box`` the way QuantUS exports an ROI drawn on its own picture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    qg = QuantusScGeometry.for_clarius(n_lines=N_LINES, imaging_depth=DEPTH_MM,
                                       probe_radius_mm=RADIUS_MM, start_depth_factor=start_depth_factor)
    l0, s0, l1, s1 = box
    t = np.linspace(0, 1, 40)
    lines = np.r_[l0 + (l1 - l0) * t, np.full(40, l1), l1 - (l1 - l0) * t, np.full(40, l0)]
    samples = np.r_[np.full(40, s0), s0 + (s1 - s0) * t, np.full(40, s1), s1 - (s1 - s0) * t]
    x, y = qg.to_sc(lines, samples)
    with open(path, "wb") as f:
        pickle.dump({"Spline X": tuple(x), "Spline Y": tuple(y), "Scan Name": scan,
                     "Phantom Name": "phantom", "Frame": frame}, f)


def write_boxes(path: Path, large: Box, small: Box | None = None, *, frame: int = 0,
                sample_name: str | None = None, device: str = "C3") -> None:
    """GUI box file. Boxes are (line0, sample0, line1, sample1); the GUI stores h = sample, v = line."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def sheet(b: Box) -> pd.DataFrame:
        return pd.DataFrame([{"sample_name": sample_name or path.stem, "device": device, "frame": frame,
                              "h1": b[1], "h2": b[3], "v1": b[0], "v2": b[2]}])

    if small is None:
        ml, ms = (large[2] - large[0]) / 4, (large[3] - large[1]) / 4
        small = (large[0] + ml, large[1] + ms, large[2] - ml, large[3] - ms)
    with pd.ExcelWriter(path) as xl:
        sheet(large).to_excel(xl, sheet_name="Large_ROI", index=False)
        sheet(small).to_excel(xl, sheet_name="Small_ROI", index=False)


def shifted(box: Box, velocity, frames: int) -> Box:
    dl, ds = velocity[0] * frames, velocity[1] * frames
    return (box[0] + dl, box[1] + ds, box[2] + dl, box[3] + ds)


def add_case(
    root: Path,
    case: str,
    *,
    cls: str = "metastasis",
    lesion: Box = (24, 300, 39, 459),
    n_frames: int = 3,
    velocity=(0, 0),
    box: Box | None = None,
    box_frame: int = 0,
    roi_frame: int = 1,
    scan: str = "raw_0_0",
    roi_start_depth_factor: float = 0.25,
    with_rf: bool = True,
    with_roi: bool = True,
    seed: int = 0,
) -> None:
    """Add one case. The lesion sits at ``lesion`` in frame 0 and moves by ``velocity`` per frame.
    The ROI outlines it on ``roi_frame``; the GUI box (default: the lesion on ``box_frame``) is on ``box_frame``."""
    suffix = "_halle" if case.upper().startswith("UKH") else ""
    gui = root / "gui"
    if with_rf:
        rf = moving_tissue(n_frames, lesion, velocity, seed)
        write_recording(gui / f"raw_data_{cls}{suffix}" / case / f"{scan}_extracted", rf)
    else:
        (gui / f"raw_data_{cls}{suffix}" / case).mkdir(parents=True, exist_ok=True)
    write_boxes(gui / f"gui_{cls}{suffix}" / f"{case}.xlsx",
                box if box is not None else shifted(lesion, velocity, box_frame), frame=box_frame)
    if with_roi:
        write_roi(root / "FLL_ROI" / f"{case}_roi.pkl", shifted(lesion, velocity, roi_frame),
                  scan=scan, frame=roi_frame, start_depth_factor=roi_start_depth_factor)


def use(root: Path, monkeypatch) -> None:
    """Point the package at the fake dataset under ``root``."""
    from usloc import config as C

    monkeypatch.setattr(C, "GUI_ROOT", root / "gui")
    monkeypatch.setattr(C, "FLL_ROI_DIR", root / "FLL_ROI")
    monkeypatch.setattr(C, "DERIVED", root / "derived")
    monkeypatch.setattr(C, "MASTER_XLSX", root / "missing.xlsx")
    (root / "FLL_ROI").mkdir(parents=True, exist_ok=True)
