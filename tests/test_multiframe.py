"""Multi-frame expansion tests — gated on the mounted dataset."""

import pytest

from usloc import config as C

pytestmark = pytest.mark.skipif(
    not C.FLL_ROI_DIR.exists(), reason="ultrasound dataset not mounted"
)


def test_annotated_frame_always_included():
    from usloc.data.multiframe import iter_case_frames
    from usloc.labels import load_spline

    frames = iter_case_frames("CEUS014", corr_thresh=0.85, max_frames=40)
    assert len(frames) >= 1
    assert sum(f.is_annotated for f in frames) == 1
    sp = load_spline(str(C.FLL_ROI_DIR / "CEUS014_roi.pkl"))
    assert any(f.frame == sp.frame and f.is_annotated for f in frames)


def test_corr_gate_and_cap_monotonic():
    from usloc.data.multiframe import iter_case_frames

    strict = iter_case_frames("CEUS014", corr_thresh=0.99, max_frames=40)
    loose = iter_case_frames("CEUS014", corr_thresh=0.5, max_frames=40)
    assert len(strict) <= len(loose)          # stricter gate keeps fewer
    assert len(loose) <= 40                    # cap respected


def test_all_frames_share_propagated_box():
    from usloc.data.multiframe import iter_case_frames

    frames = iter_case_frames("CEUS014", corr_thresh=0.85, max_frames=10)
    boxes = {f.bbox for f in frames}
    assert len(boxes) == 1                     # mask/box propagated from the annotated frame
