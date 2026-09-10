"""Curvilinear scan-conversion: RF/envelope pre-scan (lines × samples) → Cartesian fan image.

Approximate geometry (visual verification, not vendor-calibrated): each scan line fans out from the
convex-probe apex; radial distance runs from the probe radius R to R + imaging-depth. Used to check
that a lesion annotated in the RF grid lands where the DICOM/mask says it does.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def sector_angle_rad(n_lines: int, pitch_um: float, radius_mm: float) -> float:
    """Angular span of the sector ≈ n_lines · element_pitch / radius (one line per element)."""
    return n_lines * (pitch_um / 1000.0) / radius_mm


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
    import cv2

    n_lines, n_samples = prescan.shape
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
    out = cv2.remap(prescan.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderValue=0.0)
    out[~valid] = 0.0

    def forward_map(line_idx: float, sample_idx: float) -> tuple[float, float]:
        ph = (line_idx / (n_lines - 1)) * sector_rad - half
        rr = r0 + (sample_idx / (n_samples - 1)) * (r1 - r0)
        x, y = rr * np.sin(ph), rr * np.cos(ph)
        return (x - x_min) / w_mm * (W - 1), (y - y_top) / h_mm * (H - 1)

    return out, forward_map
