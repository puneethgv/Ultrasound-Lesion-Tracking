"""Fast unit tests that do not require the mounted dataset."""

import numpy as np

from usloc import config as C
from usloc.labels.boxes import BoxLabel
from usloc.recon import envelope_to_bmode
from usloc.viz import montage


def test_site_of():
    assert C.site_of("UKHCEUS013") == "Halle"
    assert C.site_of("CEUS014") == "Dresden"
    assert C.site_of("UKDCEUS034") == "Dresden"


def test_box_xyxy_orders_corners():
    b = BoxLabel(case="x", roi_kind="small", device="C3", frame=0, h1=983, h2=322, v1=134, v2=67)
    assert b.xyxy == (322, 67, 983, 134)


def test_envelope_to_bmode_range():
    env = np.abs(np.random.randn(64, 128)).astype(np.float32)
    img = envelope_to_bmode(env, dynamic_range_db=60)
    assert img.shape == (64, 128)
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0


def test_montage_shape():
    frames = np.zeros((5, 10, 8), dtype=np.float32)
    m = montage(frames, ncols=3, pad=2)
    # 5 frames -> 2 rows x 3 cols
    assert m.shape == (2 * 10 + 2, 3 * 8 + 2 * 2)


def test_classes_consistent():
    assert set(C.BENIGN) | set(C.MALIGNANT) == set(C.CLASSES)
    assert C.CLASS_TO_ID["fnh"] == 0
