"""QuantUS ROI-image geometry (dataset-free)."""

import numpy as np

from usloc.io import RawGeometry
from usloc.labels.quantus import QuantusScGeometry, quantus_geometry, spline_to_prescan

# CEUS014 C3_large: 192 lines, 98 mm imaging depth, 45 mm probe radius (from its _rf.yml)
CEUS014 = QuantusScGeometry.for_clarius(n_lines=192, imaging_depth=98.0, probe_radius_mm=45.0)


def test_quantus_image_size():
    # QuantUS: desiredHeight=500, width from the (90-degree) sector aspect
    assert CEUS014.shape == (500, 859)


def test_round_trip():
    rng = np.random.default_rng(0)
    line = rng.uniform(5, 185, 200)
    sample = rng.uniform(50, 1900, 200)
    x, y = CEUS014.to_sc(line, sample)
    l2, s2 = CEUS014.to_prescan(x, y)
    assert np.allclose(l2, line, atol=1e-6) and np.allclose(s2, sample, atol=1e-6)


def test_centre_line_is_image_centre():
    # within ~half a beam: QuantUS bins beams with ceil(t) - 1, shifting its fan by half a beam
    x, _ = CEUS014.to_sc((192 - 1) / 2, 1000)
    assert abs(x - (CEUS014.width - 1) / 2) < 2.0


def test_depth_unit_does_not_matter():
    g_m = QuantusScGeometry.for_clarius(n_lines=192, imaging_depth=0.98, probe_radius_mm=45.0)
    pts = (np.array([352.0, 522.0, 437.0]), np.array([190.0, 190.0, 94.0]))
    assert np.allclose(CEUS014.to_prescan(*pts), g_m.to_prescan(*pts))


def test_ceus014_spline_matches_independent_gui_box():
    # Extremes of the CEUS014 spline in QuantUS pixels (x in [352, 522], y in [94, 247]) vs its GUI
    # "large" box, annotated independently in the RF grid: lines 67-134, samples 322-983.
    line, sample = CEUS014.to_prescan(np.array([352.0, 522.0, 437.0]), np.array([190.0, 190.0, 94.0]))
    left, right, top = line[0], line[1], sample[2]
    assert abs(left - 67) <= 5 and abs(right - 134) <= 5 and abs(top - 322) <= 5
    # the legacy reading (spline == DICOM-crop pixels) has no such correspondence


def test_linear_probe_is_identity():
    g = RawGeometry(samples_per_line=1920, number_of_lines=128, sample_bytes=2, frames=10,
                    kind="RF", imaging_depth_mm=40.0, probe_radius_mm=None, probe_pitch_um=300.0)
    assert quantus_geometry(g) is None
    line, sample = spline_to_prescan([10.0, 20.0], [100.0, 200.0], g)
    assert list(line) == [10.0, 20.0] and list(sample) == [100.0, 200.0]
