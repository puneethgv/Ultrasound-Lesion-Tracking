"""Label ↔ image registration — resolved coordinate mapping between annotations and a canonical image.

Findings (validated visually across both sites, all 3 classes, and varying region offsets):

* **Spline masks** (``FLL_ROI/*.pkl``) are drawn in the **scan-converted "sector-crop" space** — the
  DICOM display cropped to its ``SequenceOfUltrasoundRegions`` rectangle. So:
      DICOM(x, y) = (spline_x + region.x0, spline_y + region.y0)
  and ``spline.Frame`` indexes the ``0_0.dcm`` cine (which corresponds to acquisition ``raw_0_0``).
  In the cropped canonical image the spline coordinates are **native** (no offset).
* **GUI boxes** (``gui_*/*.xlsx``) live in the *pre-scan RF grid* of ``raw_0_0`` (h→axial sample,
  v→scan line); mapping those into the display requires forward scan-conversion (future work). For the
  142 spline cases the bounding box is derived directly from the mask.

Canonical training image = the **de-identified sector crop** of ``0_0.dcm`` at the annotated frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import config as C
from ..data.discover import find_case_dir, list_dicoms
from ..deid import Region, anonymize_header, deidentify_frames, ultrasound_region
from ..io import read_dicom
from .splines import load_spline, rasterize_spline


@dataclass
class RegisteredSample:
    case: str
    image: np.ndarray          # (H, W) de-identified B-mode, canonical space
    mask: np.ndarray           # (H, W) uint8 lesion mask (0/1), or empty if no spline
    bbox: tuple[int, int, int, int] | None  # (x0, y0, x1, y1) in image coords
    polygon: np.ndarray | None  # (N, 2) lesion contour in image coords, or None
    frame: int
    region: Region
    space: str                 # "crop" or "full"

    @property
    def has_lesion(self) -> bool:
        return self.bbox is not None


def _find_0_0_dicom(case_dir: Path):
    dcms = list_dicoms(case_dir)
    if not dcms:
        return None
    for p in dcms:
        if p.name.startswith("0_0"):
            return p
    return dcms[0]


def register_case(case: str, *, space: str = "crop", frame: int | None = None) -> RegisteredSample | None:
    """Build the registered (image, mask, bbox) sample for a case from its spline annotation.

    ``space='crop'`` (default) returns the de-identified sector crop with mask/bbox in crop coords;
    ``space='full'`` returns the full de-identified frame (mask/bbox in full-image coords).
    Returns ``None`` if the case has no DICOM.
    """
    case_dir = find_case_dir(case)
    if case_dir is None:
        return None
    dcm = _find_0_0_dicom(case_dir)
    if dcm is None:
        return None

    import pydicom

    ds_raw = pydicom.dcmread(str(dcm))
    region = ultrasound_region(ds_raw)
    _, frames = read_dicom(dcm, deidentify=False)  # raw pixels; we de-identify explicitly below

    # spline (optional)
    spline_path = C.FLL_ROI_DIR / f"{case}_roi.pkl"
    sp = load_spline(spline_path) if spline_path.exists() else None
    fr = frame if frame is not None else (sp.frame if sp is not None else frames.shape[0] // 2)
    fr = int(np.clip(fr, 0, frames.shape[0] - 1))

    if space == "crop" and region is not None:
        image = deidentify_frames(frames[fr], region=region, mode="crop")
        ox, oy = 0, 0  # spline coords are native in crop space
    else:
        image = deidentify_frames(frames[fr], region=region, mode="mask")
        ox, oy = (region.x0, region.y0) if region is not None else (0, 0)
        space = "full"

    H, W = image.shape[-2], image.shape[-1]
    mask = np.zeros((H, W), dtype=np.uint8)
    bbox = None
    polygon = None
    if sp is not None and len(sp.x) >= 3:
        px = np.clip(np.asarray(sp.x) + ox, 0, W - 1)
        py = np.clip(np.asarray(sp.y) + oy, 0, H - 1)
        polygon = np.stack([px, py], axis=1)
        mask = rasterize_spline(px, py, (H, W))
        ys, xs = np.where(mask)
        if len(xs):
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    return RegisteredSample(
        case=case, image=np.asarray(image, dtype=np.float32), mask=mask,
        bbox=bbox, polygon=polygon, frame=fr, region=region, space=space,
    )
