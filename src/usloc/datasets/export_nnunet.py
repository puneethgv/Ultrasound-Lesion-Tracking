"""Export the registered spline cases into **nnU-Net v2** raw format for 2D lesion segmentation.

Layout produced under ``$nnUNet_raw/Dataset<ID>_<name>/``:
    imagesTr/<case>_0000.png   labelsTr/<case>.png     (Dresden — nnU-Net does its own 5-fold CV)
    imagesTs/<case>_0000.png   labelsTs/<case>.png     (Halle — locked external test)
    dataset.json

One frame per patient (the annotated frame) so nnU-Net's internal case-level CV stays patient-clean.
Binary task: background=0, lesion=1 (localization). Reuses ``register_case`` (de-identified crop).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .. import config as C
from ..data.cohort import build_cohort
from ..labels import register_case


def export_nnunet(
    dataset_id: int = 1,
    name: str = "Lesion",
    *,
    space: str = "crop",
    include_test: bool = True,
    raw_root: str | Path | None = None,
) -> dict:
    import cv2

    raw_root = Path(raw_root or os.environ["nnUNet_raw"])
    root = raw_root / f"Dataset{dataset_id:03d}_{name}"
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    cohort = build_cohort()
    df = cohort[cohort.has_spline & (~cohort.excluded.astype(bool))]
    n_tr, n_ts, skipped = 0, 0, []
    for _, row in df.iterrows():
        s = register_case(row["case"], space=space)
        if s is None or s.bbox is None:
            skipped.append(row["case"])
            continue
        img = (np.clip(s.image, 0, 1) * 255).astype(np.uint8)
        mask = (s.mask > 0).astype(np.uint8)
        case = row["case"]
        if row["site"] == C.SITE_TRAIN:
            cv2.imwrite(str(root / "imagesTr" / f"{case}_0000.png"), img)
            cv2.imwrite(str(root / "labelsTr" / f"{case}.png"), mask)
            n_tr += 1
        elif include_test:
            cv2.imwrite(str(root / "imagesTs" / f"{case}_0000.png"), img)
            cv2.imwrite(str(root / "labelsTs" / f"{case}.png"), mask)
            n_ts += 1

    dataset_json = {
        "channel_names": {"0": "grayscale"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": n_tr,
        "file_ending": ".png",
    }
    (root / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    return {"root": str(root), "numTraining": n_tr, "numTest": n_ts, "skipped": skipped}
