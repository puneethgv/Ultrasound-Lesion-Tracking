"""Label ↔ image registration — resolved coordinate mapping between annotations and a canonical image.

Findings:

* **Spline masks** (``FLL_ROI/*.pkl``) are **QuantUS ROI exports**. They were drawn on QuantUS's own
  scan-conversion of the Clarius RF acquisition named by ``Scan Name`` (e.g. ``raw_0_0``), and
  ``Frame`` indexes that acquisition's **RF frames**. See :mod:`usloc.labels.quantus` for the exact
  (non-physical) geometry. They are *not* DICOM pixels: the earlier assumption
  ``DICOM(x, y) = (spline_x + region.x0, spline_y + region.y0)`` with ``Frame`` indexing ``0_0.dcm``
  misplaces the lesion (visible in ``notebooks/03_mask_alignment_check.ipynb``, panel B vs C).
* **GUI boxes** (``gui_*/*.xlsx``) live in the *pre-scan RF grid* of ``raw_0_0`` (h→axial sample,
  v→scan line). Inverting the QuantUS warp puts the spline in this same grid, so the two independent
  annotations can be cross-checked (``scripts/validate_alignment.py``).

Canonical training image (``source='rf'``, default) = B-mode reconstructed from the spline's own RF
acquisition and frame, scan-converted with :func:`usloc.recon.fan_grid`. The lesion polygon is carried
through the same grid (QuantUS pixels → RF ``(line, sample)`` → fan pixels), so image and mask are
aligned by construction. RF-derived images carry no burned-in PHI.

``source='dicom'`` keeps the legacy DICOM-crop behaviour for comparison only.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import config as C
from ..data.discover import find_case_dir, list_dicoms, list_extracted
from ..deid import Region, deidentify_frames, ultrasound_region
from ..io import RawGeometry, parse_rawdata_yml, read_dicom, read_raw
from ..recon import FanGrid, fan_grid, linear_grid, rf_to_bmode, sector_angle_rad, warp_to_fan
from .quantus import spline_to_prescan
from .splines import SplineLabel, load_spline, rasterize_spline

DEFAULT_SCAN = "raw_0_0"
RF_OUT_H = 512  # height (px) of the canonical RF fan image


@dataclass
class RegisteredSample:
    case: str
    image: np.ndarray          # (H, W) B-mode in [0, 1], canonical space
    mask: np.ndarray           # (H, W) uint8 lesion mask (0/1), or empty if no spline
    bbox: tuple[int, int, int, int] | None  # (x0, y0, x1, y1) in image coords
    polygon: np.ndarray | None  # (N, 2) lesion contour in image coords, or None
    frame: int
    region: Region | None      # DICOM ultrasound region (dicom source only)
    space: str                 # "fan" (rf source), "crop" or "full" (dicom source)

    @property
    def has_lesion(self) -> bool:
        return self.bbox is not None


# ----------------------------------------------------------------------------- RF (aligned) path


@dataclass
class RfCase:
    """Everything needed to render a case's RF acquisition and its spline in one canonical space."""

    case: str
    spline: SplineLabel | None
    geom: RawGeometry
    rf: np.ndarray              # (frames, lines, samples) raw RF
    grid: FanGrid
    prescan_polygon: np.ndarray | None  # (N, 2) as (line, sample)
    polygon: np.ndarray | None  # (N, 2) fan-image (x, y)
    mask: np.ndarray            # (H, W) uint8
    bbox: tuple[int, int, int, int] | None

    @property
    def n_frames(self) -> int:
        return self.rf.shape[0]

    def frame_image(self, frame: int) -> np.ndarray:
        prescan = rf_to_bmode(self.rf[int(frame)].astype(np.float32), axis=1)  # (lines, samples)
        return warp_to_fan(prescan, self.grid)


def find_rf_acquisition(case: str, scan_name: str = DEFAULT_SCAN) -> dict[str, Path] | None:
    """Return the file roles (``rf_raw``, ``rf_yml`` …) of ``<scan_name>_extracted`` for ``case``.

    Mirrors QuantUS, which loads the ``rf.raw`` found inside the acquisition's extracted folder.
    """
    case_dir = find_case_dir(case)
    if case_dir is None:
        return None
    for e in list_extracted(case_dir):
        if e.directory.name != f"{scan_name}_extracted":
            continue
        cands = {p: r for p, r in e.presets.items() if "rf_raw" in r and "rf_yml" in r}
        if not cands:
            return None
        if len(cands) > 1:
            warnings.warn(f"{case}/{scan_name}: several RF presets {sorted(cands)}; using {sorted(cands)[0]}",
                          stacklevel=2)
        return cands[sorted(cands)[0]]
    return None


def canonical_grid(geom: RawGeometry, out_h: int = RF_OUT_H) -> FanGrid:
    """Physical-ish display grid for an RF acquisition (fan for convex probes, rectangle for linear)."""
    if geom.imaging_depth_mm is None or geom.probe_pitch_um is None:
        raise ValueError("RF yml lacks imaging depth / probe pitch; cannot build the display grid")
    n_lines, n_samples = geom.number_of_lines, geom.samples_per_line
    if geom.probe_radius_mm is None:
        return linear_grid(n_lines, n_samples, width_mm=n_lines * geom.probe_pitch_um / 1000.0,
                           depth_mm=geom.imaging_depth_mm, out_h=out_h)
    return fan_grid(
        n_lines, n_samples,
        radius_mm=geom.probe_radius_mm,
        depth_mm=geom.imaging_depth_mm,
        sector_rad=sector_angle_rad(n_lines, geom.probe_pitch_um, geom.probe_radius_mm),
        out_h=out_h,
    )


def load_rf_case(case: str, *, out_h: int = RF_OUT_H) -> RfCase | None:
    """Load the spline's RF acquisition and map the spline into the canonical fan image."""
    spline_path = C.FLL_ROI_DIR / f"{case}_roi.pkl"
    sp = load_spline(spline_path) if spline_path.exists() else None
    scan = (sp.scan_name or DEFAULT_SCAN) if sp is not None else DEFAULT_SCAN
    roles = find_rf_acquisition(case, scan)
    if roles is None:
        return None
    geom = parse_rawdata_yml(roles["rf_yml"])
    rf = read_raw(roles["rf_raw"], geom)
    grid = canonical_grid(geom, out_h=out_h)
    H, W = grid.shape

    prescan_poly = polygon = bbox = None
    mask = np.zeros((H, W), dtype=np.uint8)
    if sp is not None and len(sp.x) >= 3:
        line, sample = spline_to_prescan(sp.x, sp.y, geom)
        line = np.clip(line, 0, geom.number_of_lines - 1)
        sample = np.clip(sample, 0, geom.samples_per_line - 1)
        prescan_poly = np.stack([line, sample], axis=1)
        px, py = grid.forward(line, sample)
        polygon = np.stack([np.clip(px, 0, W - 1), np.clip(py, 0, H - 1)], axis=1)
        mask = rasterize_spline(polygon[:, 0], polygon[:, 1], (H, W))
        ys, xs = np.where(mask)
        if len(xs):
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    return RfCase(case=case, spline=sp, geom=geom, rf=rf, grid=grid, prescan_polygon=prescan_poly,
                  polygon=polygon, mask=mask, bbox=bbox)


def _register_case_rf(case: str, frame: int | None) -> RegisteredSample | None:
    rc = load_rf_case(case)
    if rc is None:
        return None
    fr = frame if frame is not None else (rc.spline.frame if rc.spline is not None else rc.n_frames // 2)
    fr = int(np.clip(fr, 0, rc.n_frames - 1))
    return RegisteredSample(
        case=case, image=rc.frame_image(fr), mask=rc.mask, bbox=rc.bbox, polygon=rc.polygon,
        frame=fr, region=None, space="fan",
    )


# ----------------------------------------------------------------------------- legacy DICOM path


def _find_0_0_dicom(case_dir: Path):
    dcms = list_dicoms(case_dir)
    if not dcms:
        return None
    for p in dcms:
        if p.name.startswith("0_0"):
            return p
    return dcms[0]


def spline_offset(region: Region | None, space: str) -> tuple[int, int]:
    """Legacy: offset added to native spline (x, y) under the (incorrect) DICOM-crop assumption."""
    if space == "crop" and region is not None:
        return 0, 0
    return (region.x0, region.y0) if region is not None else (0, 0)


def _register_case_dicom(case: str, space: str, frame: int | None) -> RegisteredSample | None:
    warnings.warn(
        "source='dicom' treats QuantUS spline pixels as DICOM pixels; masks are misaligned. "
        "Use source='rf'.", stacklevel=3,
    )
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

    spline_path = C.FLL_ROI_DIR / f"{case}_roi.pkl"
    sp = load_spline(spline_path) if spline_path.exists() else None
    fr = frame if frame is not None else (sp.frame if sp is not None else frames.shape[0] // 2)
    fr = int(np.clip(fr, 0, frames.shape[0] - 1))

    mode = "crop" if (space == "crop" and region is not None) else "mask"
    image = deidentify_frames(frames[fr], region=region, mode=mode)
    ox, oy = spline_offset(region, space)
    if mode == "mask":
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


# ----------------------------------------------------------------------------- public entry point


def register_case(
    case: str, *, source: str = "rf", space: str = "crop", frame: int | None = None,
) -> RegisteredSample | None:
    """Build the registered (image, mask, bbox) sample for a case from its spline annotation.

    ``source='rf'`` (default): image reconstructed from the spline's RF acquisition/frame, mask mapped
    through the QuantUS geometry — aligned. ``space`` is ignored (always ``"fan"``).
    ``source='dicom'``: legacy DICOM crop (``space='crop'|'full'``), kept for comparison; misaligned.
    Returns ``None`` if the needed files are missing.
    """
    if source == "rf":
        return _register_case_rf(case, frame)
    if source == "dicom":
        return _register_case_dicom(case, space, frame)
    raise ValueError(f"unknown source {source!r} (expected 'rf' or 'dicom')")
