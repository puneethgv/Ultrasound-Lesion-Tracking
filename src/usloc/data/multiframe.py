"""Multi-frame expansion — propagate a case's single spline annotation across its ``0_0.dcm`` cine.

The lesion persists across the grayscale B-mode cine, but by different amounts per case (breathing /
probe motion). We therefore keep only frames whose **annotated-box region still correlates** with the
annotated frame (default τ=0.85), so the propagated mask/box stays approximately valid. This turns the
~104 single-frame training images into a few thousand, without a tracker (that is a later upgrade).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config as C
from ..deid import deidentify_frames, ultrasound_region
from ..io import read_dicom
from ..labels.register import _find_0_0_dicom, spline_offset
from ..labels.splines import load_spline, rasterize_spline
from .discover import find_case_dir


@dataclass
class FrameSample:
    case: str
    frame: int
    image: np.ndarray                       # (H, W) de-identified B-mode
    mask: np.ndarray                        # (H, W) uint8 (propagated from the annotated frame)
    bbox: tuple[int, int, int, int]         # (x0, y0, x1, y1)
    polygon: np.ndarray                     # (N, 2) in image coords
    is_annotated: bool                      # True for the originally-labelled frame


def _box_corr(proc: np.ndarray, ref: np.ndarray, box) -> float:
    x0, y0, x1, y1 = box
    v = proc[y0:y1 + 1, x0:x1 + 1].ravel().astype(np.float32)
    if v.std() < 1e-6 or ref.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(ref, v)[0, 1])


def iter_case_frames(
    case: str,
    *,
    corr_thresh: float = 0.85,
    max_frames: int = 40,
    space: str = "crop",
) -> list[FrameSample]:
    """Return correlation-gated frames for a case, each carrying the propagated mask/box.

    The annotated frame is always included. Other frames are kept when their box-region correlation
    with the annotated frame is >= ``corr_thresh``; if more than ``max_frames`` qualify, the
    highest-correlation ones are kept.
    """
    case_dir = find_case_dir(case)
    if case_dir is None:
        return []
    dcm = _find_0_0_dicom(case_dir)
    if dcm is None:
        return []
    spline_path = C.FLL_ROI_DIR / f"{case}_roi.pkl"
    if not spline_path.exists():
        return []

    import pydicom

    region = ultrasound_region(pydicom.dcmread(str(dcm)))
    sp = load_spline(spline_path)
    _, frames = read_dicom(dcm, deidentify=False)
    n = frames.shape[0]
    af = int(np.clip(sp.frame, 0, n - 1))

    mode = "crop" if (space == "crop" and region is not None) else "mask"
    proc = deidentify_frames(frames, region=region, mode=mode)  # (N, H, W)
    H, W = proc.shape[-2], proc.shape[-1]

    ox, oy = spline_offset(region, space)
    px = np.clip(np.asarray(sp.x) + ox, 0, W - 1)
    py = np.clip(np.asarray(sp.y) + oy, 0, H - 1)
    if len(px) < 3:
        return []
    polygon = np.stack([px, py], axis=1)
    mask = rasterize_spline(px, py, (H, W))
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    ref = proc[af, box[1]:box[3] + 1, box[0]:box[2] + 1].ravel().astype(np.float32)
    scored = {f: (1.0 if f == af else _box_corr(proc[f], ref, box)) for f in range(n)}
    keep = [f for f in range(n) if f == af or scored[f] >= corr_thresh]
    if len(keep) > max_frames:
        keep = sorted(keep, key=lambda f: -scored[f])[:max_frames]
    keep = sorted(set(keep))

    return [
        FrameSample(
            case=case, frame=f, image=proc[f].astype(np.float32),
            mask=mask, bbox=box, polygon=polygon, is_annotated=(f == af),
        )
        for f in keep
    ]
