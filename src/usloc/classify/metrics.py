"""Patient-level scores for the stage-2 classifier (plain NumPy, no torch).

The crops are frames: one patient contributes up to 40 nearly identical crops, so a per-crop score
counts some patients 40 times and others once. :func:`patient_level` first averages the crop
probabilities of each patient and then scores patients, which is the number to report.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def case_of(name: str) -> str:
    """``<case>_f<idx>.png`` -> ``<case>`` (the crop exporter's naming)."""
    return re.sub(r"_f\d+$", "", Path(str(name)).stem)


def confusion(y_true, y_pred, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int), strict=True):
        cm[t, p] += 1
    return cm


def balanced_accuracy(y_true, y_pred, n_classes: int) -> float:
    """Mean per-class recall over the classes that occur in ``y_true``."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    accs = [float((y_pred[y_true == c] == c).mean()) for c in range(n_classes) if (y_true == c).any()]
    return float(np.mean(accs)) if accs else 0.0


def auc(is_positive, score) -> float:
    """Area under the ROC curve (rank formula, ties count half); ``nan`` if only one class is present."""
    pos = np.asarray(score, dtype=float)[np.asarray(is_positive, dtype=bool)]
    neg = np.asarray(score, dtype=float)[~np.asarray(is_positive, dtype=bool)]
    if not len(pos) or not len(neg):
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


def patient_level(names: Sequence[str], y_true, probs, classes: Sequence[str]) -> dict:
    """Average each patient's crop probabilities, then score patients.

    ``names`` are the crop file names (``<case>_f<idx>.png``), ``probs`` is ``(n_crops, n_classes)``.
    Returns the patient count, balanced accuracy, confusion matrix, for two classes the AUC of the
    second class (``malignant`` if present), and the per-patient table.
    """
    y_true, probs = np.asarray(y_true, dtype=int), np.asarray(probs, dtype=float)
    n = len(classes)
    if not len(y_true):
        return {"n_patients": 0, "patient_balanced_acc": 0.0, "patient_confusion": [], "patients": []}
    cases = np.array([case_of(nm) for nm in names])
    rows = []
    for case in sorted(set(cases)):
        m = cases == case
        labels = set(y_true[m].tolist())
        if len(labels) != 1:
            raise ValueError(f"{case}: crops carry different labels {sorted(labels)}")
        mean = probs[m].mean(axis=0)
        rows.append({"case": case, "label": labels.pop(), "pred": int(mean.argmax()),
                     "n_crops": int(m.sum()), "probs": mean.tolist()})
    yt = np.array([r["label"] for r in rows])
    yp = np.array([r["pred"] for r in rows])
    out = {
        "n_patients": len(rows),
        "patient_balanced_acc": balanced_accuracy(yt, yp, n),
        "patient_confusion": confusion(yt, yp, n).tolist(),
        "patients": rows,
    }
    if n == 2:
        pos = list(classes).index("malignant") if "malignant" in classes else 1
        out["patient_auc"] = auc(yt == pos, np.array([r["probs"][pos] for r in rows]))
    return out
