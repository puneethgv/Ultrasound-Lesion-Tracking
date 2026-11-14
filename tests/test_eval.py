"""Per-case scores and no-model baselines (dataset-free)."""

import importlib.util
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from usloc.eval import (
    bootstrap_ci,
    box_fill_dice,
    case_scores,
    dice,
    format_summary,
    location_prior,
    median_area_fraction,
    prior_mask,
    score_folder,
    summarize,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SHAPE = (200, 300)


def disc(cx: float, cy: float, r: float, shape=SHAPE) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r).astype(np.uint8)


def test_dice():
    a = disc(150, 100, 30)
    assert dice(a, a) == 1.0 and dice(a, np.zeros_like(a)) == 0.0
    assert dice(np.zeros_like(a), np.zeros_like(a)) == 1.0
    assert dice(a, disc(150, 100, 30) * 255) == 1.0  # any non-zero value counts


def test_filling_the_true_box_already_scores_088_on_a_round_lesion():
    # this is why "0.88 Dice when given the box" says little about the model that was given the box
    assert box_fill_dice(disc(150, 100, 40)) == pytest.approx(2 * math.pi / (4 + math.pi), abs=0.02)
    assert box_fill_dice(np.zeros(SHAPE, np.uint8)) == 1.0


def test_case_scores_separate_the_outlined_lesion_from_blobs_elsewhere():
    gt = disc(100, 100, 30)
    on = disc(105, 100, 30)
    s = case_scores(on | disc(240, 60, 20), gt)
    assert s["hit"] and s["n_blobs"] == 2 and s["n_blobs_off_lesion"] == 1
    assert s["dice_on_lesion"] == pytest.approx(dice(on, gt))
    assert s["dice"] < s["dice_on_lesion"]
    miss = case_scores(disc(240, 60, 20), gt)
    assert not miss["hit"] and miss["dice"] == 0.0 and miss["dice_on_lesion"] == 0.0
    assert miss["n_blobs_off_lesion"] == 1
    empty = case_scores(np.zeros_like(gt), gt)
    assert not empty["hit"] and empty["n_blobs"] == 0 and empty["dice"] == 0.0


def test_location_prior_is_the_best_fixed_guess():
    rng = np.random.default_rng(0)
    train = [disc(150 + rng.normal(0, 6), 100 + rng.normal(0, 6), 28) for _ in range(40)]
    prior = location_prior(train)
    assert prior.shape == (128, 128) and 0.0 <= prior.min() and prior.max() <= 1.0
    area = median_area_fraction(train)
    guess = prior_mask(prior, area, SHAPE)
    assert guess.sum() == pytest.approx(area * guess.size, rel=0.02)
    assert dice(guess, disc(150, 100, 28)) > 0.8        # a centred lesion: the guess is good
    assert dice(guess, disc(40, 40, 28)) == 0.0         # an off-centre one: it is useless
    assert prior_mask(prior, area, (100, 150)).shape == (100, 150)
    with pytest.raises(ValueError):
        location_prior([])


def test_bootstrap_interval():
    v = np.r_[np.zeros(20), np.ones(16)]  # e.g. 16 of 36 cases found
    lo, hi = bootstrap_ci(v)
    assert lo < v.mean() < hi and 0.2 < hi - lo < 0.4
    assert bootstrap_ci(v) == (lo, hi)  # reproducible
    wide = bootstrap_ci(v[::4])
    assert wide[1] - wide[0] > hi - lo
    s = summarize(v)
    assert s["n"] == 36 and s["over_0.5"] == pytest.approx(16 / 36)
    assert summarize([])["n"] == 0


def _write(folder: Path, name: str, mask: np.ndarray) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", mask.astype(np.uint8))[1].tofile(str(folder / f"{name}.png"))


def _fake_run(root: Path) -> Path:
    raw = root / "raw" / "Dataset007_Fake"
    rng = np.random.default_rng(1)
    for i in range(30):
        _write(raw / "labelsTr", f"TRAIN{i:03d}", disc(150 + rng.normal(0, 8), 100 + rng.normal(0, 8), 26))
    truths = {"UKHA": disc(150, 100, 26), "UKHB": disc(60, 50, 22), "UKHC": disc(230, 150, 24)}
    for case, gt in truths.items():
        _write(raw / "labelsTs", case, gt)
    pred = root / "pred"
    _write(pred, "UKHA", disc(152, 101, 26))                      # found
    _write(pred, "UKHB", disc(60, 50, 22) | disc(240, 150, 15))   # found, plus a blob elsewhere
    _write(pred, "UKHC", np.zeros(SHAPE, np.uint8))               # missed
    return raw


def test_score_folder_and_summary(tmp_path):
    raw = _fake_run(tmp_path)
    df = score_folder(raw / "labelsTs", tmp_path / "pred", raw / "labelsTr").set_index("case")
    assert list(df.index) == ["UKHA", "UKHB", "UKHC"]
    assert df.loc["UKHA", "dice"] > 0.9 and df.loc["UKHC", "dice"] == 0.0
    assert df.loc["UKHB", "n_blobs_off_lesion"] == 1 and df.loc["UKHB", "dice_on_lesion"] == 1.0
    assert (df["box_fill_dice"] > 0.85).all()
    assert df.loc["UKHA", "prior_dice"] > 0.7 and df.loc["UKHB", "prior_dice"] == 0.0

    text = format_summary(df.reset_index().assign(site="Halle"), title="Fake", model="nnU-Net", by=("site",))
    assert "nnU-Net" in text and "location prior" in text and "fill the box" in text
    assert "blobs away from the outlined lesion in 1 of 3" in text and "By site:" in text

    # labels only: the baselines still work, and a missing prediction leaves the model columns empty
    only = score_folder(raw / "labelsTs")
    assert "dice" not in only and "box_fill_dice" in only
    (tmp_path / "pred" / "UKHC.png").unlink()
    part = score_folder(raw / "labelsTs", tmp_path / "pred")
    assert pd.isna(part.set_index("case").loc["UKHC", "dice"])
    assert "no prediction for 1 labelled cases" in format_summary(part)


def test_eval_script_scores_existing_predictions(tmp_path, monkeypatch, capsys):
    raw = _fake_run(tmp_path)
    from usloc import config as C

    monkeypatch.setattr(C, "DERIVED", tmp_path / "derived")
    monkeypatch.setattr(C, "GUI_ROOT", tmp_path / "no_gui")
    monkeypatch.setenv("nnUNet_raw", str(raw.parent))
    monkeypatch.setattr(sys, "argv", ["eval_nnunet.py", "--dataset-id", "7", "--name", "Fake",
                                      "--pred", str(tmp_path / "pred"), "--skip-predict"])
    spec = importlib.util.spec_from_file_location("eval_nnunet", SCRIPTS / "eval_nnunet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()
    out = capsys.readouterr().out
    assert "=== Halle external (n=3 labelled cases) ===" in out and "fill the box" in out
    cases = pd.read_csv(tmp_path / "pred_cases.csv")
    assert len(cases) == 3 and {"dice", "dice_on_lesion", "box_fill_dice", "prior_dice"} <= set(cases.columns)
