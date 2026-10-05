"""Curvilinear scan-conversion: RF/envelope pre-scan (lines × samples) → Cartesian fan image.

Approximate geometry (visual verification, not vendor-calibrated): each scan line fans out from the
convex-probe apex; radial distance runs from the probe radius R to R + imaging-depth. The image and any
annotation are warped with the *same* grid (:func:`fan_grid`), so label alignment does not depend on
the geometry being exact — only realism does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def sector_angle_rad(n_lines: int, pitch_um: float, radius_mm: float) -> float:
    """Angular span of the sector ≈ n_lines · element_pitch / radius (one line per element)."""
    return n_lines * (pitch_um / 1000.0) / radius_mm


@dataclass(frozen=True)
class FanGrid:
    """Sampling grid of a fan image: for each output pixel, the pre-scan (sample, line) it reads."""

    map_x: np.ndarray  # (H, W) float32 sample index
    map_y: np.ndarray  # (H, W) float32 line index
    valid: np.ndarray  # (H, W) bool, inside the sector
    forward: Callable  # (line_idx, sample_idx) -> (x_px, y_px); accepts arrays

    @property
    def shape(self) -> tuple[int, int]:
        return self.valid.shape


def fan_grid(
    n_lines: int,
    n_samples: int,
    *,
    radius_mm: float,
    depth_mm: float,
    sector_rad: float,
    out_h: int = 700,
) -> FanGrid:
    """Build the fan sampling grid for a ``(n_lines, n_samples)`` pre-scan."""
    r0, r1 = float(radius_mm), float(radius_mm + depth_mm)
    half = sector_rad / 2.0

    x_max = r1 * np.sin(half)
    x_min = -x_max
    y_top = r0 * np.cos(half)
    y_bot = r1
    w_mm, h_mm = (x_max - x_min), (y_bot - y_top)
    ppm = out_h / h_mm
    H, W = int(round(h_mm * ppm)), int(round(w_mm * ppm))

    xs = np.linspace(x_min, x_max, W)
    ys = np.linspace(y_top, y_bot, H)
    X, Y = np.meshgrid(xs, ys)
    r = np.sqrt(X * X + Y * Y)
    phi = np.arctan2(X, Y)  # angle from the vertical axis
    valid = (r >= r0) & (r <= r1) & (np.abs(phi) <= half)

    map_x = np.clip((r - r0) / (r1 - r0) * (n_samples - 1), 0, n_samples - 1).astype(np.float32)
    map_y = np.clip((phi + half) / sector_rad * (n_lines - 1), 0, n_lines - 1).astype(np.float32)

    def forward(line_idx, sample_idx):
        ph = (np.asarray(line_idx, dtype=np.float64) / (n_lines - 1)) * sector_rad - half
        rr = r0 + (np.asarray(sample_idx, dtype=np.float64) / (n_samples - 1)) * (r1 - r0)
        x, y = rr * np.sin(ph), rr * np.cos(ph)
        return (x - x_min) / w_mm * (W - 1), (y - y_top) / h_mm * (H - 1)

    return FanGrid(map_x=map_x, map_y=map_y, valid=valid, forward=forward)


def linear_grid(
    n_lines: int, n_samples: int, *, width_mm: float, depth_mm: float, out_h: int = 700,
) -> FanGrid:
    """Rectangular grid for a linear probe (lines span ``width_mm``, samples span ``depth_mm``)."""
    H = int(out_h)
    W = max(2, int(round(width_mm / depth_mm * H)))
    map_x = np.repeat(np.linspace(0, n_samples - 1, H, dtype=np.float32)[:, None], W, axis=1)
    map_y = np.repeat(np.linspace(0, n_lines - 1, W, dtype=np.float32)[None, :], H, axis=0)

    def forward(line_idx, sample_idx):
        x = np.asarray(line_idx, dtype=np.float64) / (n_lines - 1) * (W - 1)
        y = np.asarray(sample_idx, dtype=np.float64) / (n_samples - 1) * (H - 1)
        return x, y

    return FanGrid(map_x=map_x, map_y=map_y, valid=np.ones((H, W), dtype=bool), forward=forward)


def warp_to_fan(prescan: np.ndarray, grid: FanGrid, *, nearest: bool = False) -> np.ndarray:
    """Warp a ``(n_lines, n_samples)`` array onto ``grid`` (``nearest=True`` for label maps)."""
    import cv2

    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    out = cv2.remap(np.asarray(prescan, dtype=np.float32), grid.map_x, grid.map_y,
                    interpolation=interp, borderValue=0.0)
    out[~grid.valid] = 0.0
    return out


def scan_convert(
    prescan: np.ndarray,
    *,
    radius_mm: float,
    depth_mm: float,
    sector_rad: float,
    out_h: int = 700,
) -> tuple[np.ndarray, Callable[[float, float], tuple[float, float]]]:
    """Warp a ``(n_lines, n_samples)`` pre-scan B-mode into a fan image.

    Returns ``(image, forward_map)`` where ``forward_map(line_idx, sample_idx) -> (x_px, y_px)`` maps a
    point in the pre-scan grid to the output fan-image pixel (for overlaying the RF-grid box).
    """
    n_lines, n_samples = prescan.shape
    grid = fan_grid(n_lines, n_samples, radius_mm=radius_mm, depth_mm=depth_mm,
                    sector_rad=sector_rad, out_h=out_h)
    return warp_to_fan(prescan, grid), grid.forward
