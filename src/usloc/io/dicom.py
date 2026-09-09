"""Minimal DICOM cine reader for the scan-converted display clips (e.g. 800x800, N frames)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_dicom(path: str | Path, deidentify: bool = True):
    """Return ``(dataset, frames)`` where ``frames`` is ``(N, H, W)`` grayscale float in [0, 1].

    Ultrasound cines are often RGB; we collapse to luminance for B-mode display.

    ``deidentify=True`` (default) strips burned-in PHI by keeping only the declared ultrasound
    region and blanks PHI header tags — the ultrasound image itself is never modified. Pass
    ``deidentify=False`` only for internal calibration where raw display chrome is needed.
    """
    import pydicom

    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array  # (N,H,W) or (N,H,W,3) or (H,W[,3])
    if arr.ndim == 2:  # single grayscale frame
        arr = arr[None]
    elif arr.ndim == 3 and arr.shape[-1] == 3:  # single RGB frame
        arr = arr[None]
    if arr.ndim == 4 and arr.shape[-1] == 3:  # (N,H,W,3) RGB -> luminance
        arr = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])
    arr = arr.astype(np.float32)
    mx = float(arr.max()) or 1.0
    arr = arr / mx
    if deidentify:
        from ..deid import anonymize_header, deidentify_frames

        arr = deidentify_frames(arr, ds, mode="mask")
        ds = anonymize_header(ds)
    return ds, arr


def dicom_meta(ds) -> dict:
    """A few useful header fields for sanity checks."""
    g = lambda tag, default=None: getattr(ds, tag, default)  # noqa: E731
    return {
        "Rows": g("Rows"),
        "Columns": g("Columns"),
        "NumberOfFrames": g("NumberOfFrames"),
        "PhotometricInterpretation": g("PhotometricInterpretation"),
        "Manufacturer": g("Manufacturer"),
        "TransducerData": g("TransducerData"),
        "PhysicalDeltaX": g("PhysicalDeltaX"),
        "PhysicalDeltaY": g("PhysicalDeltaY"),
    }
