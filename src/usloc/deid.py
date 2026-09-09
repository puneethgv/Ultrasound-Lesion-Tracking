"""De-identification pipeline for the ultrasound cines.

Two layers of PHI exist:

1. **Burned-in pixel text** — case ID, institution, acquisition date and the *operator* name are
   rendered into the banner rows *outside* the ultrasound image. The DICOM declares the image
   rectangle in ``SequenceOfUltrasoundRegions`` (here y in [182, 617], full width); keeping only
   pixels inside that rectangle removes every burned-in identifier **without touching the ultrasound
   sector itself**. A geometric fallback (blank top/bottom banners) is used if the metadata is absent.

2. **Header tags** — ``OperatorsName``, ``InstitutionName``, ``StudyDate/Time``, ``StationName``,
   ``DeviceSerialNumber`` … scrubbed by :func:`anonymize_header`, keeping the pseudonymized
   ``PatientID`` (our case key).

Design goal: the ultrasound image is never altered — only non-image chrome is removed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Header elements that may carry identifying information (staff, site, device, timing).
PHI_TAGS = (
    "PatientName", "PatientBirthDate", "PatientAddress", "PatientTelephoneNumbers",
    "OtherPatientIDs", "OtherPatientNames", "ReferringPhysicianName", "PerformingPhysicianName",
    "OperatorsName", "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName",
    "StationName", "DeviceSerialNumber", "StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate",
    "StudyTime", "SeriesTime", "AcquisitionTime", "ContentTime", "AcquisitionDateTime",
    "AccessionNumber", "StudyID",
)


@dataclass(frozen=True)
class Region:
    x0: int
    y0: int
    x1: int
    y1: int  # inclusive pixel bounds

    @property
    def slices(self) -> tuple[slice, slice]:
        return slice(self.y0, self.y1 + 1), slice(self.x0, self.x1 + 1)


def ultrasound_region(ds) -> Region | None:
    """Union of ``SequenceOfUltrasoundRegions`` rectangles, or ``None`` if not present."""
    seq = getattr(ds, "SequenceOfUltrasoundRegions", None)
    if not seq:
        return None
    x0 = min(int(r.RegionLocationMinX0) for r in seq)
    y0 = min(int(r.RegionLocationMinY0) for r in seq)
    x1 = max(int(r.RegionLocationMaxX1) for r in seq)
    y1 = max(int(r.RegionLocationMaxY1) for r in seq)
    return Region(x0, y0, x1, y1)


def _apply(frames: np.ndarray, region: Region, mode: str) -> np.ndarray:
    ys, xs = region.slices
    if mode == "crop":
        return frames[..., ys, xs] if frames.ndim == 3 else frames[ys, xs]
    out = np.zeros_like(frames)  # "mask": keep size, zero everything outside the region
    if frames.ndim == 3:
        out[:, ys, xs] = frames[:, ys, xs]
    else:
        out[ys, xs] = frames[ys, xs]
    return out


def deidentify_frames(
    frames: np.ndarray,
    ds=None,
    *,
    region: Region | None = None,
    mode: str = "mask",
    fallback_top: float = 0.24,
    fallback_bottom: float = 0.24,
) -> np.ndarray:
    """Remove burned-in PHI from a frame ``(H, W)`` or cine ``(N, H, W)``.

    Uses the DICOM ultrasound region when available (``region`` overrides ``ds``); otherwise blanks
    the top/bottom banner fractions. ``mode='mask'`` keeps the image size (zeros outside the region);
    ``mode='crop'`` returns just the ultrasound rectangle.
    """
    frames = np.asarray(frames)
    region = region or (ultrasound_region(ds) if ds is not None else None)
    if region is not None:
        return _apply(frames, region, mode)

    # geometric fallback: blank banner rows, keep the middle band
    h = frames.shape[-2]
    y0 = int(round(h * fallback_top))
    y1 = h - int(round(h * fallback_bottom)) - 1
    return _apply(frames, Region(0, y0, frames.shape[-1] - 1, y1), mode)


def anonymize_header(ds, keep: tuple[str, ...] = ("PatientID",)):
    """Return ``ds`` with PHI tags blanked/removed in place (keeps ``keep`` tags, e.g. case id)."""
    for tag in PHI_TAGS:
        if tag in keep:
            continue
        if tag in ds:
            try:
                ds.data_element(tag).value = ""
            except Exception:
                del ds[tag]
    # scrub private tags that may echo identifiers
    ds.remove_private_tags()
    return ds
