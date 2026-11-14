"""ROI files: redrawn revisions and reading (dataset-free)."""

import io

import synthetic as syn
from usloc.labels.splines import _Unpickler, find_spline_file, load_spline, spline_files

BOX = (24, 300, 39, 459)


def test_newest_revision_wins(tmp_path):
    for name in ("CEUS014_roi.pkl", "CEUS014_roi_2.pkl", "CEUS014_roi_10.pkl",
                 "UKDCEUS014_roi.pkl", "CEUS01_roi_3.pkl", "CEUS014_roi_backup.pkl"):
        syn.write_roi(tmp_path / name, BOX)
    assert [p.name for p in spline_files("CEUS014", tmp_path)] == [
        "CEUS014_roi.pkl", "CEUS014_roi_2.pkl", "CEUS014_roi_10.pkl"]
    assert find_spline_file("CEUS014", tmp_path).name == "CEUS014_roi_10.pkl"
    assert find_spline_file("UKDCEUS014", tmp_path).name == "UKDCEUS014_roi.pkl"
    assert find_spline_file("CEUS01", tmp_path).name == "CEUS01_roi_3.pkl"
    assert find_spline_file("CEUS999", tmp_path) is None


def test_load_spline_reads_case_and_source(tmp_path):
    syn.write_roi(tmp_path / "UKDCEUS057_roi_2.pkl", BOX, scan="raw_0_1", frame=18)
    sp = load_spline(tmp_path / "UKDCEUS057_roi_2.pkl")
    assert sp.case == "UKDCEUS057" and sp.source == "UKDCEUS057_roi_2.pkl"
    assert sp.scan_name == "raw_0_1" and sp.frame == 18 and len(sp.x) == len(sp.y) == 160


def test_numpy_internal_module_names_are_interchangeable():
    # NumPy 2 writes "numpy._core...", NumPy 1 writes "numpy.core..."; either must load under both.
    for module in ("numpy._core.numeric", "numpy.core.numeric"):
        obj = _Unpickler(io.BytesIO(f"c{module}\n_frombuffer\n.".encode())).load()
        assert callable(obj)


def test_register_uses_the_redrawn_roi(tmp_path, monkeypatch):
    syn.use(tmp_path, monkeypatch)
    syn.add_case(tmp_path, "SYN001")
    moved = (34, 500, 49, 659)
    syn.write_roi(tmp_path / "FLL_ROI" / "SYN001_roi_2.pkl", moved, frame=2)

    from usloc.labels import load_rf_case

    rc = load_rf_case("SYN001")
    assert rc.spline.source == "SYN001_roi_2.pkl" and rc.spline.frame == 2
    line, sample = rc.prescan_polygon[:, 0], rc.prescan_polygon[:, 1]
    assert abs(line.min() - moved[0]) < 1 and abs(sample.max() - moved[3]) < 2
