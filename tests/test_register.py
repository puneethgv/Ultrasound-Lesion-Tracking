"""Registration tests — gated on the mounted dataset (skipped in CI without data)."""

import glob

import pytest

from usloc import config as C

pytestmark = pytest.mark.skipif(
    not C.FLL_ROI_DIR.exists(), reason="ultrasound dataset not mounted"
)


def test_rf_registration_has_mask_on_spline_frame():
    from usloc.labels import load_spline, register_case

    s = register_case("CEUS014")
    sp = load_spline(str(C.FLL_ROI_DIR / "CEUS014_roi.pkl"))
    assert s is not None and s.space == "fan" and s.bbox is not None
    assert s.frame == sp.frame
    assert s.image.shape == s.mask.shape and s.mask.sum() > 100


def test_rf_spline_overlaps_gui_box():
    # spline (QuantUS-inverted) and GUI box are independent annotations in the same RF grid
    from usloc.labels import load_boxes, load_rf_case

    rc = load_rf_case("CEUS014")
    ln, sm = rc.prescan_polygon[:, 0], rc.prescan_polygon[:, 1]
    box = next(b for b in load_boxes(glob.glob(str(C.GUI_ROOT / "gui_*" / "CEUS014.xlsx"))[0])
               if b.roi_kind == "large")
    h0, v0, h1, v1 = box.xyxy  # h = sample, v = line
    assert ln.min() < v1 and ln.max() > v0 and sm.min() < h1 and sm.max() > h0
    assert abs(ln.min() - v0) < 10 and abs(ln.max() - v1) < 10


def test_legacy_dicom_crop_vs_full_offset_consistency():
    from usloc.labels import register_case

    with pytest.warns(UserWarning):
        crop = register_case("CEUS014", source="dicom", space="crop")
        full = register_case("CEUS014", source="dicom", space="full")
    assert crop is not None and full is not None
    assert crop.bbox is not None and full.bbox is not None
    assert int(crop.mask.sum()) == int(full.mask.sum())
    y0 = crop.region.y0
    cx0, cy0, cx1, cy1 = crop.bbox
    fx0, fy0, fx1, fy1 = full.bbox
    assert (fx0, fx1) == (cx0, cx1)
    assert fy0 == cy0 + y0 and fy1 == cy1 + y0
