"""Parse FLL_ROI spline-contour masks (``FLL_ROI/<case>_roi.pkl``).

Each pickle is a dict: {'Spline X': (...), 'Spline Y': (...), 'Scan Name': str, 'Phantom Name': str,
'Frame': int}. Spline (X, Y) is a closed lesion contour, ~171 points.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SplineLabel:
    case: str
    x: np.ndarray  # (N,) contour x
    y: np.ndarray  # (N,) contour y
    scan_name: str
    frame: int


def load_spline(pkl_path: str | Path) -> SplineLabel:
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    case = Path(pkl_path).stem.replace("_roi", "")
    return SplineLabel(
        case=case,
        x=np.asarray(d.get("Spline X", ()), dtype=np.float64),
        y=np.asarray(d.get("Spline Y", ()), dtype=np.float64),
        scan_name=str(d.get("Scan Name", "")),
        frame=int(d.get("Frame", 0)),
    )


def rasterize_spline(x: np.ndarray, y: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize a polygon contour to a binary uint8 mask of ``shape`` = (H, W)."""
    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.stack([np.asarray(x), np.asarray(y)], axis=1).round().astype(np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 1)
    return mask
