"""Mask-vs-box check end to end, on a fake dataset (random numbers, no patient data)."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

import synthetic as syn
from usloc.labels.check import gui_boxes, score_case
from usloc.labels.quantus import padded_start_depth_factor
from usloc.labels.register import explain_missing_rf

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_agreeing_case_is_not_flagged(tmp_path, monkeypatch):
    syn.use(tmp_path, monkeypatch)
    syn.add_case(tmp_path, "SYN001")
    rec, rc = score_case("SYN001")
    assert rec["status"] == "ok" and rc is not None
    assert rec["relation"] == "match" and rec["iou_large"] > 0.85 and not rec["flag"]
    assert (rec["frame"], rec["box_frame"], rec["frame_gap"]) == (1, 0, 1)
    assert rec["frame_in_range"] and rec["spline_on_picture"]
    assert rec["roi_file"] == "SYN001_roi.pkl" and rec["box_sample_name"] == "SYN001"
    assert "iou_tracked" not in rec  # tracking only runs for low-IoU cases


def test_mask_inside_a_larger_box_is_the_same_place(tmp_path, monkeypatch):
    syn.use(tmp_path, monkeypatch)
    syn.add_case(tmp_path, "SYN002", box=(8, 100, 56, 700))
    rec, _ = score_case("SYN002")
    assert rec["iou_large"] < 0.3 and rec["relation"] == "inside_box"
    assert rec["spline_inside_large"] > 0.95
    assert rec["flag"] and "inside a larger box" in rec["flag_reason"]


def test_lesion_that_moved_between_the_two_frames(tmp_path, monkeypatch):
    syn.use(tmp_path, monkeypatch)
    # box drawn on frame 0, ROI on frame 10; the tissue moves 1 line and 15 samples per frame
    syn.add_case(tmp_path, "SYN003", n_frames=12, velocity=(1, 15), box_frame=0, roi_frame=10)
    rec, _ = score_case("SYN003")
    assert rec["iou_large"] < 0.05 and rec["relation"] in ("shifted", "apart")
    assert rec["frame_gap"] == 10
    assert abs(rec["track_d_line"] - 10) <= 1.5 and abs(rec["track_d_sample"] - 150) <= 5
    assert rec["iou_tracked"] > 0.6 and rec["relation_tracked"] == "match"
    assert rec["flag"] and "following the tissue" in rec["flag_reason"]
    # without tracking the case is still flagged, just without that hint
    rec2, _ = score_case("SYN003", track=False)
    assert rec2["flag"] and "iou_tracked" not in rec2 and "following" not in rec2["flag_reason"]


def test_mask_on_another_recording_is_still_compared(tmp_path, monkeypatch):
    syn.use(tmp_path, monkeypatch)
    syn.add_case(tmp_path, "SYN004", scan="raw_0_1")
    rec, _ = score_case("SYN004")
    assert rec["scan"] == "raw_0_1" and rec["relation"] == "match" and not rec["flag"]


def test_roi_drawn_on_a_newer_quantus_picture(tmp_path, monkeypatch):
    syn.use(tmp_path, monkeypatch)
    syn.add_case(tmp_path, "SYN005", roi_start_depth_factor=padded_start_depth_factor(syn.N_SAMPLES))
    rec, _ = score_case("SYN005")
    assert rec["iou_large"] < 0.3 and rec["iou_new_quantus"] > 0.85
    assert rec["flag"] and "newer QuantUS" in rec["flag_reason"]
    # read with the matching setting, the same ROI lands on the lesion
    from usloc.labels import load_rf_case

    rc = load_rf_case("SYN005", start_depth_factor=padded_start_depth_factor(syn.N_SAMPLES))
    assert abs(rc.prescan_polygon[:, 0].min() - 24) < 1 and abs(rc.prescan_polygon[:, 1].min() - 300) < 2


def test_frame_outside_the_recording_is_flagged(tmp_path, monkeypatch):
    syn.use(tmp_path, monkeypatch)
    syn.add_case(tmp_path, "SYN006", n_frames=3, roi_frame=7)
    rec, _ = score_case("SYN006")
    assert not rec["frame_in_range"] and rec["flag"] and "outside the recording" in rec["flag_reason"]


def test_missing_rf_says_why(tmp_path, monkeypatch):
    syn.use(tmp_path, monkeypatch)
    syn.add_case(tmp_path, "SYN007", with_rf=False)
    rec, rc = score_case("SYN007")
    assert rc is None and rec["status"].startswith("no RF: no folder raw_0_0_extracted")

    case_dir = tmp_path / "gui" / "raw_data_metastasis" / "SYN007"
    (case_dir / "raw_0_1_extracted").mkdir()
    assert "extracted folders: raw_0_1_extracted" in explain_missing_rf("SYN007", "raw_0_0")
    (case_dir / "raw_0_0_extracted").mkdir()
    (case_dir / "raw_0_0_extracted" / "C3_large_rf.raw.lzo").write_bytes(b"")
    why = explain_missing_rf("SYN007", "raw_0_0")
    assert "no *_rf.raw + *_rf.yml pair" in why and "C3_large_rf.raw.lzo" in why
    assert explain_missing_rf("NOPE001") == "no raw-data folder for this case"


def test_box_row_for_the_masks_recording_is_preferred(tmp_path, monkeypatch):
    syn.use(tmp_path, monkeypatch)
    path = tmp_path / "gui" / "gui_metastasis" / "SYN008.xlsx"
    path.parent.mkdir(parents=True)
    rows = pd.DataFrame([
        {"sample_name": "raw_0_0", "device": "C3", "frame": 0, "h1": 100, "h2": 200, "v1": 10, "v2": 20},
        {"sample_name": "raw_0_1", "device": "C3", "frame": 4, "h1": 500, "h2": 600, "v1": 30, "v2": 40},
    ])
    with pd.ExcelWriter(path) as xl:
        rows.to_excel(xl, sheet_name="Large_ROI", index=False)
    boxes, n = gui_boxes("SYN008", "raw_0_1")
    assert n == 2 and boxes["large"].frame == 4 and boxes["large"].xyxy == (500, 30, 600, 40)
    assert gui_boxes("SYN008", "raw_0")[0]["large"].frame == 0     # "raw_0" is not "raw_0_0"
    assert gui_boxes("SYN008")[0]["large"].frame == 0
    assert gui_boxes("NOPE001") == ({}, 0)


def test_cohort_counts_a_case_with_only_a_redrawn_roi(tmp_path, monkeypatch):
    syn.use(tmp_path, monkeypatch)
    syn.add_case(tmp_path, "SYN009", with_roi=False)
    syn.write_roi(tmp_path / "FLL_ROI" / "SYN009_roi_2.pkl", (24, 300, 39, 459), frame=1)

    from usloc.data.cohort import build_cohort

    cohort = build_cohort()
    assert bool(cohort.set_index("case").loc["SYN009", "has_spline"])


def test_script_writes_report_and_review_pictures(tmp_path, monkeypatch, capsys):
    syn.use(tmp_path, monkeypatch)
    syn.add_case(tmp_path, "SYN001")
    syn.add_case(tmp_path, "SYN002", box=(8, 100, 56, 700))
    syn.add_case(tmp_path, "SYN003", n_frames=12, velocity=(1, 15), box_frame=0, roi_frame=10)
    syn.add_case(tmp_path, "SYN007", with_rf=False)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["validate_alignment.py", "--out", str(out), "--review", "--gallery", "2"])
    _load_script("validate_alignment").main()

    df = pd.read_csv(out / "alignment_report.csv").set_index("case")
    assert list(df.index) == ["SYN001", "SYN002", "SYN003", "SYN007"]
    assert df.loc["SYN001", "relation"] == "match" and not df.loc["SYN001", "flag"]
    assert df.loc["SYN002", "relation"] == "inside_box"
    assert df.loc["SYN003", "iou_tracked"] > 0.6
    assert df.loc["SYN007", "status"].startswith("no RF")
    assert {p.name for p in (out / "review").iterdir()} == {"SYN002.png", "SYN003.png"}
    assert (out / "alignment_gallery.png").exists()
    printed = capsys.readouterr().out
    assert "flagged for review: 2" in printed and "carried to the mask's frame: 1" in printed
