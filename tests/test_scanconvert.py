"""Unit tests for curvilinear scan-conversion (dataset-free)."""

import numpy as np

from usloc.recon import scan_convert, sector_angle_rad


def test_sector_angle():
    # 192 lines, 300 um pitch, 45 mm radius -> ~1.28 rad
    a = sector_angle_rad(192, 300.0, 45.0)
    assert 1.2 < a < 1.35


def test_scan_convert_shapes_and_map():
    prescan = np.random.rand(64, 400).astype(np.float32)
    fan, fwd = scan_convert(prescan, radius_mm=45.0, depth_mm=98.0, sector_rad=1.28, out_h=300)
    assert fan.ndim == 2 and fan.shape[0] == 300
    H, W = fan.shape
    # center line (middle index) maps near the horizontal center; deeper sample -> larger y
    x_mid, _ = fwd((prescan.shape[0] - 1) / 2, prescan.shape[1] // 2)
    assert abs(x_mid - W / 2) < W * 0.05
    _, y_shallow = fwd(0, 0)
    _, y_deep = fwd(0, prescan.shape[1] - 1)
    assert y_deep > y_shallow
