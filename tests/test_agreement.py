"""Box-vs-mask agreement and tissue tracking (dataset-free)."""

import cv2
import numpy as np
import pytest

from usloc.labels.agreement import box_iou, compare, frac_inside, shift_box
from usloc.labels.tracking import track_box

BOX = (20.0, 300.0, 60.0, 700.0)  # (line0, sample0, line1, sample1)


def test_iou_and_fractions():
    assert box_iou(BOX, BOX) == 1.0
    assert box_iou(BOX, shift_box(BOX, 100, 0)) == 0.0
    half = (20.0, 300.0, 40.0, 700.0)
    assert box_iou(half, BOX) == pytest.approx(0.5)
    assert frac_inside(half, BOX) == pytest.approx(1.0)
    assert frac_inside(BOX, half) == pytest.approx(0.5)
    assert box_iou((0, 0, 0, 0), BOX) == 0.0  # empty box


def test_relations():
    assert compare(BOX, BOX).relation == "match"
    # a small outline inside a generous box: same place, but a low IoU
    small = (35.0, 450.0, 45.0, 550.0)
    a = compare(small, BOX)
    assert a.iou < 0.1 and a.relation == "inside_box" and a.spline_in_box == pytest.approx(1.0)
    # the other way round
    assert compare(BOX, small).relation == "covers_box"
    # overlapping, neither contains the other
    assert compare(shift_box(BOX, 30, 300), BOX).relation == "shifted"
    assert compare(shift_box(BOX, 60, 0), BOX).relation == "apart"


def test_centre_offsets_are_spline_minus_box():
    a = compare(shift_box(BOX, 5, -40), BOX)
    assert a.d_line == pytest.approx(5) and a.d_sample == pytest.approx(-40)


def test_track_box_follows_moving_speckle():
    rng = np.random.default_rng(1)
    field = cv2.GaussianBlur(rng.standard_normal((400, 2400)).astype(np.float32), (0, 0),
                             sigmaX=6, sigmaY=1.5)
    v_line, v_sample = 0.7, -9.0  # tissue motion per frame

    def frame_at(k):
        m = np.float32([[1, 0, k * v_sample], [0, 1, k * v_line]])
        moved = cv2.warpAffine(field, m, (field.shape[1], field.shape[0]), flags=cv2.INTER_LINEAR)
        return moved[100:292, 200:2120]

    box = (60.0, 500.0, 110.0, 1100.0)
    d_line, d_sample, conf = track_box(frame_at, box, 0, 12)
    assert d_line == pytest.approx(12 * v_line, abs=1.5)
    assert d_sample == pytest.approx(12 * v_sample, abs=3.0)
    assert conf > 0.5
    back = track_box(frame_at, box, 12, 0)
    assert back[0] == pytest.approx(-12 * v_line, abs=1.5) and back[1] == pytest.approx(-12 * v_sample, abs=3.0)
    assert track_box(frame_at, box, 4, 4) == (0.0, 0.0, 1.0)
