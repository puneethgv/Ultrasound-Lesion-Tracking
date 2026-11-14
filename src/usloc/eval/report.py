"""Score a folder of predicted masks against the label masks, case by case, next to the baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import (
    box_fill_dice,
    case_scores,
    dice,
    location_prior,
    median_area_fraction,
    prior_mask,
    summarize,
)


def load_masks(folder: str | Path) -> dict[str, np.ndarray]:
    """``<case>.png`` masks of ``folder`` by case (empty if the folder does not exist)."""
    import cv2

    folder = Path(folder)
    out: dict[str, np.ndarray] = {}
    if not folder.is_dir():
        return out
    for p in sorted(folder.glob("*.png")):
        if p.name.startswith("._"):
            continue
        m = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        out[p.stem] = (m[..., 0] if m.ndim == 3 else m) > 0
    return out


def score_folder(
    label_dir: str | Path,
    pred_dir: str | Path | None = None,
    train_label_dir: str | Path | None = None,
) -> pd.DataFrame:
    """One row per labelled case.

    Always: ``gt_px`` and ``box_fill_dice`` (what filling the true bounding box scores).
    With ``train_label_dir``: ``prior_dice`` (the best fixed guess learnt from the training masks).
    With ``pred_dir``: the model's scores from :func:`usloc.eval.metrics.case_scores` for every case
    that has a prediction; cases without one keep empty model columns.
    """
    labels = load_masks(label_dir)
    preds = load_masks(pred_dir) if pred_dir is not None else {}
    train = list(load_masks(train_label_dir).values()) if train_label_dir is not None else []
    prior = location_prior(train) if train else None
    area = median_area_fraction(train) if train else None

    rows = []
    for case, gt in labels.items():
        row: dict = {"case": case}
        if case in preds:
            if preds[case].shape != gt.shape:
                raise ValueError(f"{case}: prediction is {preds[case].shape}, label is {gt.shape}")
            row.update(case_scores(preds[case], gt))
        row["gt_px"] = int(gt.sum())
        row["box_fill_dice"] = box_fill_dice(gt)
        if prior is not None:
            row["prior_dice"] = dice(prior_mask(prior, area, gt.shape), gt)
        rows.append(row)
    return pd.DataFrame(rows)


def _line(name: str, values) -> str:
    s = summarize(pd.Series(values).dropna().to_numpy(dtype=float))
    if not s["n"]:
        return f"  {name:18s} no cases"
    return (f"  {name:18s} mean Dice {s['mean']:.3f} (95% CI {s['ci_low']:.2f}-{s['ci_high']:.2f})  "
            f"median {s['median']:.3f}  >0.1: {s['over_0.1']:.0%}  >0.3: {s['over_0.3']:.0%}  "
            f">0.5: {s['over_0.5']:.0%}  n={s['n']}")


def format_summary(df: pd.DataFrame, title: str = "Test set", model: str = "model",
                   by: tuple[str, ...] = ()) -> str:
    """Readable summary of :func:`score_folder` output; ``by`` adds a breakdown per value of a column."""
    out = [f"=== {title} (n={len(df)} labelled cases) ==="]
    has_model = "dice" in df and df["dice"].notna().any()
    if has_model:
        scored = df[df["dice"].notna()]
        out.append(_line(model, scored["dice"]))
        out.append(_line("  on the lesion", scored["dice_on_lesion"]))
        extra = int((scored["n_blobs_off_lesion"] > 0).sum())
        out.append(f"  lesion touched in {scored['hit'].mean():.0%} of cases; "
                   f"blobs away from the outlined lesion in {extra} of {len(scored)}")
        if len(scored) < len(df):
            out.append(f"  no prediction for {len(df) - len(scored)} labelled cases (left out above)")
    out.append("Baselines (no model):")
    if "prior_dice" in df:
        out.append(_line("location prior", df["prior_dice"]) + "   <- best fixed guess")
    out.append(_line("fill the box", df["box_fill_dice"]) + "   <- free with the true box")
    if has_model:
        for col in by:
            if col not in df or df[col].isna().all():
                continue
            out.append(f"By {col}:")
            for value, part in df[df["dice"].notna()].groupby(col):
                out.append(_line(f"{value}", part["dice"]))
    return "\n".join(out)
