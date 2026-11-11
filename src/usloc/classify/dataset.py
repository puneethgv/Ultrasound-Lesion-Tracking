"""Build lesion-crop datasets for the stage-2 classifier.

Crops come from the multi-frame GT boxes (padded), organized as ImageFolder:
``<out>/<split>/<label>/<case>_f<idx>.png``. Same patient-level Dresden/Halle split as the detector.
``task='binary'`` → benign vs malignant (robust to fnh=18); ``task='multiclass'`` → 3 classes.

``region='beside'`` writes a control set instead: for every frame, a box of the same size at the same
depth *next to* the lesion (liver, no lesion). A classifier trained on those crops cannot be reading
the lesion, so its score shows how much of the "lesion" classifier's score comes from everything else
(patient, scanner settings, liver background). If the two scores are close, the lesion is not what
the classifier uses.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .. import config as C
from ..data.cohort import build_cohort, drop_training_cases
from ..data.multiframe import iter_case_frames
from ..datasets.export_yolo import _assign_splits

BINARY_MAP = {"fnh": "benign", "hemangioma": "benign", "metastasis": "malignant"}
REGIONS = ("lesion", "beside")


def beside_box(box, image: np.ndarray, gap: float = 0.25, max_blank: float = 0.05):
    """A box of the same size at the same depth beside ``box`` (right, else left), or ``None`` if
    neither side lies inside the fan. ``gap`` is the free space between the two, in box widths."""
    x0, y0, x1, y1 = box
    shift = int(round((x1 - x0) * (1 + gap))) + 1
    width = image.shape[1]
    for dx in (shift, -shift):
        nx0, nx1 = x0 + dx, x1 + dx
        if nx0 < 0 or nx1 >= width:
            continue
        crop = image[y0: y1 + 1, nx0: nx1 + 1]
        if crop.size and float((crop == 0).mean()) <= max_blank:  # outside the fan the image is 0
            return (nx0, y0, nx1, y1)
    return None


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
    source: str = "rf",
    region: str = "lesion",
    exclude: Iterable[str] = (),
) -> dict:
    """Write padded crops as an ImageFolder tree and return per-(split,label) counts.

    ``region``: ``'lesion'`` (default) or ``'beside'`` (control crops, see the module docstring).
    ``exclude``: cases to leave out of train/val (the Halle test set is never reduced).
    """
    import cv2

    if region not in REGIONS:
        raise ValueError(f"unknown region {region!r} (expected one of {REGIONS})")
    name = task if region == "lesion" else f"{task}_{region}"
    out_dir = Path(out_dir or (C.DERIVED / "crops" / name))
    cohort = build_cohort()
    df = cohort[cohort.has_spline & (~cohort.excluded.astype(bool))]
    splits = _assign_splits(df, val_frac, seed)  # split first: leaving cases out must not reshuffle it
    df = drop_training_cases(df, exclude)

    counts: dict[str, dict[str, int]] = {s: {} for s in ("train", "val", "test")}
    for _, row in df.iterrows():
        case = row["case"]
        split = splits.get(case)
        if split is None:
            continue
        label = BINARY_MAP[row["class"]] if task == "binary" else row["class"]
        dest = out_dir / split / label
        dest.mkdir(parents=True, exist_ok=True)
        for s in iter_case_frames(case, corr_thresh=corr_thresh, max_frames=max_frames, source=source):
            box = s.bbox if region == "lesion" else beside_box(s.bbox, s.image)
            if box is None:
                continue
            x0, y0, x1, y1 = box
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
    return {"out_dir": str(out_dir), "task": task, "region": region, "counts": counts}
