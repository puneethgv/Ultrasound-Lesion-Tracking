"""Coordinate model of the QuantUS ROI-drawing image — the space the ``FLL_ROI/*.pkl`` splines live in.

The spline pickles (``Spline X/Y``, ``Scan Name``, ``Phantom Name``, ``Frame``) are QuantUS ROI exports.
QuantUS draws ROIs on *its own* scan-conversion of the Clarius RF (``raw_<i>_<j>`` = ``Scan Name``),
**not** on the scanner DICOM. ``Frame`` indexes the RF frames of that acquisition. Its Clarius loader
builds that image with fixed, non-physical settings, reproduced here (geometry only):

  * RF lines zero-padded at the end to ``2928`` samples (C3/L15), then envelope + log-compression;
  * ``scanConvert(width = 2 * probe_radius_mm [used as degrees], tilt = 0,
    startDepth = imaging_depth / 4, endDepth = imaging_depth, desiredHeight = 500)``.

Because start depth is a fixed fraction of end depth, the pixel geometry is scale-invariant: the
depth's unit (mm vs m) does not change the mapping. A QuantUS sample index equals the raw ``_rf.raw``
sample index (QuantUS appends its delay/zero padding after the signal).

:meth:`QuantusScGeometry.to_prescan` inverts that warp, taking a spline from QuantUS pixels to the RF
pre-scan grid ``(line, sample)``, which is the same grid the GUI boxes use. From there it can be
carried into any image built from the RF.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

QUANTUS_SAMPLES = 2928   # QuantUS pads C3 and L15 lines to this many samples
QUANTUS_HEIGHT = 500     # desiredHeight of the QuantUS scan conversion


@dataclass(frozen=True)
class QuantusScGeometry:
    n_lines: int
    sector_deg: float
    start_depth: float
    end_depth: float
    n_samples: int = QUANTUS_SAMPLES
    height: int = QUANTUS_HEIGHT
    # derived (mirrors QuantUS scanConvert)
    start_angle: float = field(init=False)
    angle_inc: float = field(init=False)
    depth_inc: float = field(init=False)
    xmin: float = field(init=False)
    xmax: float = field(init=False)
    ymin: float = field(init=False)
    ymax: float = field(init=False)
    width: int = field(init=False)

    def __post_init__(self) -> None:
        sa = np.deg2rad(270 - self.sector_deg / 2) % np.pi
        ea = sa + np.deg2rad(self.sector_deg)
        ai = np.deg2rad(self.sector_deg) / (self.n_lines - 1)
        angles = np.arange(sa, ea + ai, ai)
        depths = np.array([self.start_depth, self.end_depth])
        xmin = -1 * max(np.cos(sa) * depths)
        xmax = -1 * min(np.cos(ea) * depths)
        ymin = min(np.sin(angles) * self.start_depth)
        ymax = max(np.sin(angles) * self.end_depth)
        width = int(np.ceil(self.height * abs((xmax - xmin) / (ymax - ymin))))
        set_ = object.__setattr__
        set_(self, "start_angle", float(sa))
        set_(self, "angle_inc", float(ai))
        set_(self, "depth_inc", float((self.end_depth - self.start_depth) / (self.n_samples - 1)))
        set_(self, "xmin", float(xmin))
        set_(self, "xmax", float(xmax))
        set_(self, "ymin", float(ymin))
        set_(self, "ymax", float(ymax))
        set_(self, "width", width)

    @classmethod
    def for_clarius(cls, *, n_lines: int, imaging_depth: float, probe_radius_mm: float) -> QuantusScGeometry:
        """Geometry QuantUS uses for a curvilinear Clarius acquisition (values from the ``_rf.yml``)."""
        return cls(
            n_lines=int(n_lines),
            sector_deg=2.0 * float(probe_radius_mm),
            start_depth=float(imaging_depth) / 4.0,
            end_depth=float(imaging_depth),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def to_prescan(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """QuantUS image pixels ``(x, y)`` -> RF pre-scan ``(line, sample)`` (continuous indices)."""
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        X = x / (self.width - 1) * (self.xmax - self.xmin) + self.xmin
        Y = y / (self.height - 1) * (self.ymax - self.ymin) + self.ymin
        angle = np.arctan2(Y, -X)
        r = np.hypot(X, Y)
        # QuantUS picks beam ceil(t) - 1, so bin k spans t in (k, k+1] and its centre is t = k + 0.5
        line = (angle - self.start_angle) / self.angle_inc - 0.5
        sample = (r - self.start_depth) / self.depth_inc - 0.5
        return line, sample

    def to_sc(self, line, sample) -> tuple[np.ndarray, np.ndarray]:
        """RF pre-scan ``(line, sample)`` -> QuantUS image pixels ``(x, y)`` (inverse of :meth:`to_prescan`)."""
        line, sample = np.asarray(line, dtype=np.float64), np.asarray(sample, dtype=np.float64)
        angle = self.start_angle + (line + 0.5) * self.angle_inc
        r = self.start_depth + (sample + 0.5) * self.depth_inc
        X, Y = -r * np.cos(angle), r * np.sin(angle)
        x = (X - self.xmin) / (self.xmax - self.xmin) * (self.width - 1)
        y = (Y - self.ymin) / (self.ymax - self.ymin) * (self.height - 1)
        return x, y


def quantus_geometry(geom) -> QuantusScGeometry | None:
    """QuantUS geometry for an RF :class:`~usloc.io.RawGeometry`, or ``None`` for a linear probe
    (no probe radius -> QuantUS shows the un-converted pre-scan, so spline ``(x, y) == (line, sample)``)."""
    if geom.probe_radius_mm is None:
        return None
    if geom.imaging_depth_mm is None:
        raise ValueError("RF yml has no imaging depth; cannot reproduce the QuantUS geometry")
    return QuantusScGeometry.for_clarius(
        n_lines=geom.number_of_lines,
        imaging_depth=geom.imaging_depth_mm,
        probe_radius_mm=geom.probe_radius_mm,
    )


def spline_to_prescan(x, y, geom) -> tuple[np.ndarray, np.ndarray]:
    """Map a QuantUS spline to the RF pre-scan grid ``(line, sample)`` for the given RF geometry."""
    qg = quantus_geometry(geom)
    if qg is None:
        return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return qg.to_prescan(x, y)
