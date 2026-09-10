"""Build lesion-crop datasets for the stage-2 classifier.

Crops come from the multi-frame GT boxes (padded), organized as ImageFolder:
``<out>/<split>/<label>/<case>_f<idx>.png``. Same patient-level Dresden/Halle split as the detector.
``task='binary'`` → benign vs malignant (robust to fnh=18); ``task='multiclass'`` → 3 classes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import config as C
from ..data.cohort import build_cohort
from ..data.multiframe import iter_case_frames
from ..datasets.export_yolo import _assign_splits

BINARY_MAP = {"fnh": "benign", "hemangioma": "benign", "metastasis": "malignant"}


def export_crops(
    out_dir: str | Path | None = None,
    *,
    task: str = "binary",
    pad: float = 0.15,
    val_frac: float = 0.2,
    seed: int = 0,
    corr_thresh: float = 0.85,
    max_frames: int = 40,
    min_size: int = 16,
) -> dict:
    """Write padded lesion crops as an ImageFolder tree and return per-(split,label) counts."""
    import cv2

    out_dir = Path(out_dir or (C.DERIVED / "crops" / task))
    cohort = build_cohort()
    df = cohort[cohort.has_spline & (~cohort.excluded.astype(bool))]
    splits = _assign_splits(df, val_frac, seed)

    counts: dict[str, dict[str, int]] = {s: {} for s in ("train", "val", "test")}
    for _, row in df.iterrows():
        case = row["case"]
        split = splits.get(case)
        if split is None:
            continue
        label = BINARY_MAP[row["class"]] if task == "binary" else row["class"]
        dest = out_dir / split / label
        dest.mkdir(parents=True, exist_ok=True)
        for s in iter_case_frames(case, corr_thresh=corr_thresh, max_frames=max_frames):
            x0, y0, x1, y1 = s.bbox
            bw, bh = x1 - x0, y1 - y0
            dx, dy = int(bw * pad), int(bh * pad)
            H, W = s.image.shape
            cx0, cy0 = max(0, x0 - dx), max(0, y0 - dy)
            cx1, cy1 = min(W, x1 + dx), min(H, y1 + dy)
            crop = s.image[cy0:cy1, cx0:cx1]
            if crop.shape[0] < min_size or crop.shape[1] < min_size:
                continue
            img = (np.clip(crop, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(str(dest / f"{case}_f{s.frame:03d}.png"), img)
            counts[split][label] = counts[split].get(label, 0) + 1
    return {"out_dir": str(out_dir), "task": task, "counts": counts}
