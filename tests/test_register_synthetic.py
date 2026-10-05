"""End-to-end alignment on a synthetic case (dataset-free).

A fake Clarius acquisition has a bright "lesion" at a known RF position; its spline is written in
QuantUS pixels exactly as QuantUS would export it. ``register_case`` must put the mask on the lesion.
"""

import pickle

import numpy as np

from usloc import config as C
from usloc.labels import register_case
from usloc.labels.quantus import QuantusScGeometry
from usloc.recon import warp_to_fan

N_LINES, N_SAMPLES, N_FRAMES = 64, 800, 3
LESION = (slice(24, 40), slice(300, 460))  # (lines, samples)


def _make_case(root):
    gui_root = root / "gui"
    roi_dir = root / "FLL_ROI"
    ext = gui_root / "raw_data_metastasis" / "SYN001" / "raw_0_0_extracted"
    ext.mkdir(parents=True)
    roi_dir.mkdir()

    (ext / "C3_large_rf.yml").write_text(
        "type: RF\nsize:\n  samples per line: 800\n  number of lines: 64\n  sample size: 2 bytes\n"
        "frames: 3\nimaging depth: 98 mm\nprobe:\n  radius: 45 mm\n  pitch: 900 um\n"
    )
    rng = np.random.default_rng(0)
    rf = (rng.standard_normal((N_FRAMES, N_LINES, N_SAMPLES)) * 20).astype(np.int16)
    rf[:, LESION[0], LESION[1]] *= 40
    with open(ext / "C3_large_rf.raw", "wb") as f:
        np.array([2, N_FRAMES, N_LINES, N_SAMPLES, 2], dtype="<i4").tofile(f)
        for k in range(N_FRAMES):
            np.array([k], dtype="<i8").tofile(f)
            rf[k].astype("<i2").tofile(f)

    # spline = lesion outline in QuantUS pixels (QuantUS pads to 2928 samples; geometry from the yml)
    qg = QuantusScGeometry.for_clarius(n_lines=N_LINES, imaging_depth=98.0, probe_radius_mm=45.0)
    l0, l1 = LESION[0].start, LESION[0].stop - 1
    s0, s1 = LESION[1].start, LESION[1].stop - 1
    t = np.linspace(0, 1, 40)
    lines = np.r_[l0 + (l1 - l0) * t, np.full(40, l1), l1 - (l1 - l0) * t, np.full(40, l0)]
    samples = np.r_[np.full(40, s0), s0 + (s1 - s0) * t, np.full(40, s1), s1 - (s1 - s0) * t]
    x, y = qg.to_sc(lines, samples)
    with open(roi_dir / "SYN001_roi.pkl", "wb") as f:
        pickle.dump({"Spline X": tuple(x), "Spline Y": tuple(y), "Scan Name": "raw_0_0",
                     "Phantom Name": "phantom", "Frame": 1}, f)
    return gui_root, roi_dir


def test_rf_registration_puts_mask_on_lesion(tmp_path, monkeypatch):
    gui_root, roi_dir = _make_case(tmp_path)
    monkeypatch.setattr(C, "GUI_ROOT", gui_root)
    monkeypatch.setattr(C, "FLL_ROI_DIR", roi_dir)

    s = register_case("SYN001")
    assert s is not None and s.space == "fan" and s.frame == 1 and s.bbox is not None

    from usloc.labels import load_rf_case

    truth = np.zeros((N_LINES, N_SAMPLES), dtype=np.float32)
    truth[LESION] = 1.0
    truth_fan = warp_to_fan(truth, load_rf_case("SYN001").grid, nearest=True) > 0
    m = s.mask.astype(bool)
    iou = (m & truth_fan).sum() / (m | truth_fan).sum()
    assert iou > 0.85, iou
    assert s.image[m].mean() > 2 * s.image[~m & (s.image > 0)].mean()
