"""Unit tests for the de-identification pipeline (dataset-free)."""

import numpy as np

from usloc.deid import Region, anonymize_header, deidentify_frames, ultrasound_region


def _frame():
    # 20x10 with a bright "banner" in the top 5 rows and content in the middle band
    f = np.zeros((20, 10), dtype=np.float32)
    f[:5, :] = 1.0      # banner (PHI)
    f[8:15, :] = 0.5    # ultrasound content
    return f


def test_mask_zeros_outside_region_keeps_inside():
    f = _frame()
    reg = Region(0, 8, 9, 14)
    out = deidentify_frames(f, region=reg, mode="mask")
    assert out.shape == f.shape
    assert out[:8].max() == 0.0 and out[15:].max() == 0.0   # banner + below removed
    assert np.array_equal(out[8:15], f[8:15])               # content untouched


def test_crop_returns_region_only():
    f = _frame()
    reg = Region(0, 8, 9, 14)
    out = deidentify_frames(f, region=reg, mode="crop")
    assert out.shape == (7, 10)
    assert np.array_equal(out, f[8:15])


def test_cine_masking_batched():
    cine = np.stack([_frame(), _frame() * 0.7])
    out = deidentify_frames(cine, region=Region(0, 8, 9, 14), mode="mask")
    assert out.shape == cine.shape
    assert out[:, :8].max() == 0.0


def test_fallback_when_no_region():
    f = _frame()
    out = deidentify_frames(f, ds=None, mode="mask", fallback_top=0.25, fallback_bottom=0.25)
    assert out[:5].max() == 0.0  # top banner blanked by geometric fallback


def test_anonymize_header_blanks_phi_keeps_case_id():
    pydicom = __import__("pydicom")
    ds = pydicom.dataset.Dataset()
    ds.PatientID = "CEUS014"
    ds.OperatorsName = "Doe^Jane"
    ds.InstitutionName = "Some Hospital"
    ds.StudyDate = "20230608"
    anonymize_header(ds, keep=("PatientID",))
    assert str(ds.PatientID) == "CEUS014"
    assert str(ds.OperatorsName) == ""
    assert str(ds.InstitutionName) == ""
    assert str(ds.StudyDate) == ""


def test_ultrasound_region_union():
    pydicom = __import__("pydicom")
    ds = pydicom.dataset.Dataset()
    r1 = pydicom.dataset.Dataset()
    r1.RegionLocationMinX0, r1.RegionLocationMinY0 = 0, 182
    r1.RegionLocationMaxX1, r1.RegionLocationMaxY1 = 799, 617
    ds.SequenceOfUltrasoundRegions = [r1]
    reg = ultrasound_region(ds)
    assert (reg.x0, reg.y0, reg.x1, reg.y1) == (0, 182, 799, 617)
