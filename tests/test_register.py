"""Registration tests — gated on the mounted dataset (skipped in CI without data)."""

import pytest

from usloc import config as C

pytestmark = pytest.mark.skipif(
    not C.FLL_ROI_DIR.exists(), reason="ultrasound dataset not mounted"
)


def test_crop_vs_full_offset_consistency():
    from usloc.labels import register_case

    crop = register_case("CEUS014", space="crop")
    full = register_case("CEUS014", space="full")
    assert crop is not None and full is not None
    assert crop.bbox is not None and full.bbox is not None
    # same mask area regardless of space
    assert int(crop.mask.sum()) == int(full.mask.sum())
    # full-space bbox is the crop bbox shifted down by region.y0 (x0 == 0 here)
    y0 = crop.region.y0
    cx0, cy0, cx1, cy1 = crop.bbox
    fx0, fy0, fx1, fy1 = full.bbox
    assert (fx0, fx1) == (cx0, cx1)
    assert fy0 == cy0 + y0 and fy1 == cy1 + y0


def test_crop_image_height_matches_region():
    from usloc.labels import register_case

    s = register_case("CEUS014", space="crop")
    assert s.image.shape[0] == s.region.y1 - s.region.y0 + 1
