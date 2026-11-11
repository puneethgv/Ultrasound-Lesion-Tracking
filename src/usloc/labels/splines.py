"""Parse FLL_ROI spline-contour masks (``FLL_ROI/<case>_roi.pkl``).

Each pickle is a dict: {'Spline X': (...), 'Spline Y': (...), 'Scan Name': str, 'Phantom Name': str,
'Frame': int}. Spline (X, Y) is a closed lesion contour, ~171 points.

A redrawn ROI is saved next to the original as ``<case>_roi_<n>.pkl`` (``n`` = 2, 3, ...).
:func:`find_spline_file` returns the highest ``n``, so a redrawn ROI replaces the original without the
original being overwritten.
"""

from __future__ import annotations

import importlib
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import config as C

_REVISION = re.compile(r"_roi(?:_(\d+))?\.pkl$")


@dataclass
class SplineLabel:
    case: str
    x: np.ndarray  # (N,) contour x
    y: np.ndarray  # (N,) contour y
    scan_name: str
    frame: int
    source: str = ""  # file name the ROI was read from


def spline_files(case: str, roi_dir: str | Path | None = None) -> list[Path]:
    """All ROI files of ``case``, oldest first: ``<case>_roi.pkl``, then ``<case>_roi_<n>.pkl`` by ``n``."""
    roi_dir = Path(roi_dir or C.FLL_ROI_DIR)
    found: list[tuple[int, Path]] = []
    for p in roi_dir.glob(f"{case}_roi*.pkl"):
        m = _REVISION.search(p.name)
        if m and p.name[: m.start()] == case:
            found.append((int(m.group(1) or 1), p))
    return [p for _, p in sorted(found, key=lambda t: (t[0], t[1].name))]


def find_spline_file(case: str, roi_dir: str | Path | None = None) -> Path | None:
    """The ROI file to use for ``case`` (the newest revision), or ``None`` if there is none."""
    files = spline_files(case, roi_dir)
    return files[-1] if files else None


class _Unpickler(pickle.Unpickler):
    """``pickle`` with one addition: NumPy 2 names its internals ``numpy._core`` and NumPy 1 names
    them ``numpy.core``. A file written under one is read under the other by trying both names."""

    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            if module.startswith("numpy._core"):
                other = "numpy.core" + module[len("numpy._core"):]
            elif module.startswith("numpy.core"):
                other = "numpy._core" + module[len("numpy.core"):]
            else:
                raise
            return getattr(importlib.import_module(other), name)


def load_spline(pkl_path: str | Path) -> SplineLabel:
    with open(pkl_path, "rb") as f:
        d = _Unpickler(f).load()
    name = Path(pkl_path).name
    m = _REVISION.search(name)
    case = name[: m.start()] if m else Path(pkl_path).stem.replace("_roi", "")
    return SplineLabel(
        case=case,
        x=np.asarray(d.get("Spline X", ()), dtype=np.float64),
        y=np.asarray(d.get("Spline Y", ()), dtype=np.float64),
        scan_name=str(d.get("Scan Name", "")),
        frame=int(d.get("Frame", 0)),
        source=name,
    )


def rasterize_spline(x: np.ndarray, y: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize a polygon contour to a binary uint8 mask of ``shape`` = (H, W)."""
    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.stack([np.asarray(x), np.asarray(y)], axis=1).round().astype(np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 1)
    return mask
