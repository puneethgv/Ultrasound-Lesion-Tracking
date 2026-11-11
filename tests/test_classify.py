"""Classifier scores per patient, control crops and case exclusion (dataset-free)."""

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

import synthetic as syn
from usloc.classify.dataset import beside_box, export_crops
from usloc.classify.metrics import auc, balanced_accuracy, case_of, confusion, patient_level
from usloc.data.cohort import build_cohort, drop_training_cases, read_case_list

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_case_of():
    assert case_of("UKDCEUS057_f018.png") == "UKDCEUS057"
    assert case_of("some/dir/UKHBIOPSIE013_f000.png") == "UKHBIOPSIE013"
    assert case_of("CEUS014.png") == "CEUS014"


def test_basic_scores():
    y, p = [0, 0, 0, 1], [0, 0, 1, 1]
    assert confusion(y, p, 2).tolist() == [[2, 1], [0, 1]]
    assert balanced_accuracy(y, p, 2) == pytest.approx((2 / 3 + 1) / 2)
    assert balanced_accuracy([0, 0], [0, 1], 2) == 0.5      # a class that never occurs is ignored
    assert auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auc([0, 1], [0.5, 0.5]) == 0.5
    assert np.isnan(auc([1, 1], [0.2, 0.3]))


def test_patient_level_counts_each_patient_once():
    classes = ["benign", "malignant"]
    # patient A (malignant): 40 crops, all right. Patients B, C (benign): 1 crop each, both wrong.
    names = [f"A_f{i:03d}.png" for i in range(40)] + ["B_f000.png", "C_f000.png"]
    y = [1] * 40 + [0, 0]
    probs = np.array([[0.1, 0.9]] * 40 + [[0.3, 0.7], [0.4, 0.6]])
    per_crop = balanced_accuracy(y, probs.argmax(1), 2)
    pat = patient_level(names, y, probs, classes)
    assert per_crop == 0.5 and pat["patient_balanced_acc"] == 0.5
    assert pat["n_patients"] == 3 and pat["patient_confusion"] == [[0, 2], [0, 1]]
    assert (np.asarray(y) == probs.argmax(1)).mean() > 0.95      # plain accuracy looks great per crop
    assert pat["patient_auc"] == 1.0
    # averaging: one patient whose crops disagree gets the label of the mean probability
    pat2 = patient_level(["D_f000.png", "D_f001.png", "D_f002.png"], [0, 0, 0],
                         np.array([[0.9, 0.1], [0.45, 0.55], [0.45, 0.55]]), classes)
    assert pat2["patients"][0]["pred"] == 0 and pat2["patients"][0]["n_crops"] == 3
    with pytest.raises(ValueError):
        patient_level(["E_f000.png", "E_f001.png"], [0, 1], np.array([[1, 0], [0, 1]]), classes)
    assert patient_level([], [], np.zeros((0, 2)), classes)["n_patients"] == 0


def test_beside_box():
    img = np.full((100, 300), 0.5, dtype=np.float32)
    assert beside_box((100, 40, 139, 59), img) == (150, 40, 189, 59)       # right side first
    assert beside_box((250, 40, 289, 59), img) == (200, 40, 239, 59)       # no room right -> left
    assert beside_box((100, 40, 199, 59), img[:, :260]) is None            # no room on either side
    img[:, 150:] = 0.0                                                     # right side is outside the fan
    assert beside_box((100, 40, 139, 59), img) == (50, 40, 89, 59)


def _dataset(root, monkeypatch):
    syn.use(root, monkeypatch)
    for i, cls in enumerate(["metastasis", "metastasis", "metastasis", "hemangioma", "hemangioma", "fnh"]):
        syn.add_case(root, f"SYN{i:03d}", cls=cls, seed=i)
    syn.add_case(root, "UKHSYN100", cls="metastasis", seed=10)
    syn.add_case(root, "UKHSYN101", cls="hemangioma", seed=11)


def _pngs(folder: Path) -> list[Path]:
    return sorted(folder.rglob("*.png"))


def test_control_crops_are_the_same_size_but_miss_the_lesion(tmp_path, monkeypatch):
    _dataset(tmp_path, monkeypatch)
    lesion = export_crops(tmp_path / "crops", task="binary")
    beside = export_crops(tmp_path / "crops_beside", task="binary", region="beside")
    a, b = _pngs(tmp_path / "crops"), _pngs(tmp_path / "crops_beside")
    assert len(a) == len(b) == 8 * 3 and lesion["region"] == "lesion" and beside["region"] == "beside"
    assert [p.relative_to(tmp_path / "crops") for p in a] == [p.relative_to(tmp_path / "crops_beside") for p in b]
    for pa, pb in zip(a, b, strict=True):
        ia, ib = cv2.imread(str(pa), cv2.IMREAD_GRAYSCALE), cv2.imread(str(pb), cv2.IMREAD_GRAYSCALE)
        assert ia.shape == ib.shape
        assert ia.mean() > 1.5 * ib.mean()      # the fake lesion is bright; the tissue beside it is not
    assert set(lesion["counts"]["test"]) == {"benign", "malignant"}
    with pytest.raises(ValueError):
        export_crops(tmp_path / "x", region="nowhere")


def test_case_lists(tmp_path):
    txt = tmp_path / "leave_out.txt"
    txt.write_text("# cases to leave out\nSYN001\n\nSYN003  # unsure\nSYN001\n", encoding="utf-8")
    assert read_case_list(txt) == ["SYN001", "SYN003"]
    report = tmp_path / "alignment_report.csv"
    pd.DataFrame({"case": ["SYN000", "SYN001", "UKHSYN100"], "flag": [False, True, True]}).to_csv(report, index=False)
    assert read_case_list(report) == ["SYN001", "UKHSYN100"]
    plain = tmp_path / "cases.csv"
    pd.DataFrame({"case": ["SYN004", "SYN005"]}).to_csv(plain, index=False)
    assert read_case_list(plain) == ["SYN004", "SYN005"]
    pd.DataFrame({"id": [1]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError):
        read_case_list(tmp_path / "bad.csv")


def test_excluded_cases_leave_training_but_not_the_test_set(tmp_path, monkeypatch):
    _dataset(tmp_path, monkeypatch)
    cohort = build_cohort()
    kept = drop_training_cases(cohort, ["SYN001", "UKHSYN100"])
    assert "SYN001" not in set(kept["case"]) and "UKHSYN100" in set(kept["case"])
    assert "UKHSYN100" not in set(drop_training_cases(cohort, ["UKHSYN100"], also_test=True)["case"])
    assert len(drop_training_cases(cohort, [])) == len(cohort)

    from usloc.datasets.export_nnunet import export_nnunet

    full = export_nnunet(1, "Full", raw_root=tmp_path / "raw")
    less = export_nnunet(2, "Less", raw_root=tmp_path / "raw", exclude=["SYN001", "SYN004", "UKHSYN100"])
    assert (full["numTraining"], full["numTest"]) == (6, 2)
    assert (less["numTraining"], less["numTest"], less["excluded"]) == (4, 2, 2)
    names = {p.name for p in (tmp_path / "raw" / "Dataset002_Less" / "labelsTr").iterdir()}
    assert "SYN001.png" not in names and "SYN000.png" in names
    assert json.loads((tmp_path / "raw" / "Dataset002_Less" / "dataset.json").read_text())["numTraining"] == 4

    from usloc.datasets.export_yolo import export_yolo_seg

    y_full = export_yolo_seg(tmp_path / "yolo_full")
    y_less = export_yolo_seg(tmp_path / "yolo_less", exclude=["SYN001", "UKHSYN100"])
    assert sum(y_full["cases"].values()) == 8 and sum(y_less["cases"].values()) == 7
    assert y_less["cases"]["test"] == y_full["cases"]["test"] == 2

    crops = export_crops(tmp_path / "crops_less", exclude=["SYN001"])
    assert not list((tmp_path / "crops_less").rglob("SYN001_f*.png"))
    assert sum(crops["counts"]["test"].values()) == 2 * 3


def test_export_nnunet_script(tmp_path, monkeypatch, capsys):
    _dataset(tmp_path, monkeypatch)
    listed = tmp_path / "leave_out.txt"
    listed.write_text("SYN002\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["export_nnunet.py", "--dataset-id", "5", "--name", "Cli",
                                      "--raw-root", str(tmp_path / "raw"), "--exclude-file", str(listed)])
    spec = importlib.util.spec_from_file_location("export_nnunet_cli", SCRIPTS / "export_nnunet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()
    out = capsys.readouterr().out
    assert '"numTraining": 5' in out and "1 listed, 1 removed" in out
    assert not (tmp_path / "raw" / "Dataset005_Cli" / "imagesTr" / "SYN002_0000.png").exists()
