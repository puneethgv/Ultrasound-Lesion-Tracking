"""Unit tests for curvilinear scan-conversion (dataset-free)."""

import numpy as np

from usloc.recon import fan_grid, linear_grid, scan_convert, sector_angle_rad, warp_to_fan


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


def test_mask_warp_lands_on_forward_mapped_point():
    # a small blob in the pre-scan grid must land where forward() says, so image and label agree
    n_lines, n_samples = 64, 400
    grid = fan_grid(n_lines, n_samples, radius_mm=45.0, depth_mm=98.0, sector_rad=1.28, out_h=300)
    blob = np.zeros((n_lines, n_samples), dtype=np.float32)
    blob[20:25, 240:260] = 1.0
    fan = warp_to_fan(blob, grid, nearest=True)
    ys, xs = np.nonzero(fan)
    assert set(np.unique(fan)) <= {0.0, 1.0}
    fx, fy = grid.forward(22, 250)
    assert abs(xs.mean() - fx) < 3 and abs(ys.mean() - fy) < 3


def test_linear_grid_forward():
    grid = linear_grid(128, 1000, width_mm=38.4, depth_mm=40.0, out_h=200)
    H, W = grid.shape
    assert H == 200 and abs(W - 192) <= 1
    x, y = grid.forward(127, 999)
    assert abs(x - (W - 1)) < 1e-6 and abs(y - (H - 1)) < 1e-6
