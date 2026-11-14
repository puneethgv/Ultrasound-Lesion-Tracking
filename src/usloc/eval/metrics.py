"""Segmentation scores per case, and the trivial baselines every result should be read against.

A mean Dice says little on its own in this project:

* **Fill the box.** A method that is *given* the lesion's bounding box and simply fills it already
  scores about 0.88 Dice on a round lesion (``2*pi / (4 + pi)``). A box-prompted model has to beat that
  number, not zero.
* **Location prior.** Sonographers centre the lesion, so the average training mask is itself a decent
  guess. A model that has learnt to *find* lesions has to beat the best fixed guess.
* **Only one lesion per patient is outlined.** A second, real lesion that the model finds counts as
  a mistake. ``dice_on_lesion`` ignores predicted blobs that do not touch the outlined lesion, and
  ``n_blobs_off_lesion`` counts them, so that effect can be seen.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = np.asarray(pred) > 0, np.asarray(gt) > 0
    denom = int(p.sum()) + int(g.sum())
    return 1.0 if denom == 0 else float(2 * int((p & g).sum()) / denom)


def box_fill(gt: np.ndarray) -> np.ndarray:
    """The mask's tight bounding box, filled."""
    g = np.asarray(gt) > 0
    out = np.zeros(g.shape, dtype=np.uint8)
    ys, xs = np.where(g)
    if len(xs):
        out[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1] = 1
    return out


def box_fill_dice(gt: np.ndarray) -> float:
    """Dice a method gets for free when it is handed the lesion's bounding box."""
    return dice(box_fill(gt), gt)


def location_prior(train_masks: Sequence[np.ndarray], grid: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Average training mask on a common ``grid`` (H, W): how often each image position is lesion."""
    import cv2

    if not len(train_masks):
        raise ValueError("no training masks")
    acc = np.zeros(grid, dtype=np.float64)
    for m in train_masks:
        acc += cv2.resize((np.asarray(m) > 0).astype(np.float32), (grid[1], grid[0]),
                          interpolation=cv2.INTER_AREA)
    return (acc / len(train_masks)).astype(np.float32)


def median_area_fraction(masks: Sequence[np.ndarray]) -> float:
    return float(np.median([(np.asarray(m) > 0).mean() for m in masks]))


def prior_mask(prior: np.ndarray, area_fraction: float, shape: tuple[int, int]) -> np.ndarray:
    """The fixed guess: the ``area_fraction`` most lesion-prone pixels of ``prior``, at image ``shape``."""
    import cv2

    p = cv2.resize(prior, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    k = int(np.clip(round(area_fraction * p.size), 1, p.size))
    out = np.zeros(p.size, dtype=np.uint8)
    out[np.argpartition(p.ravel(), -k)[-k:]] = 1
    return out.reshape(shape)


def case_scores(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Scores of one prediction against the one outlined lesion."""
    import cv2

    p, g = (np.asarray(pred) > 0), (np.asarray(gt) > 0)
    n, labels = cv2.connectedComponents(p.astype(np.uint8), connectivity=8)
    touching = [k for k in range(1, n) if (g & (labels == k)).any()]
    on_lesion = np.isin(labels, touching) if touching else np.zeros_like(p)
    return {
        "dice": dice(p, g),
        "dice_on_lesion": dice(on_lesion, g),
        "hit": bool(touching),
        "n_blobs": int(n - 1),
        "n_blobs_off_lesion": int(n - 1 - len(touching)),
        "gt_px": int(g.sum()),
        "pred_px": int(p.sum()),
    }


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of ``values`` (cases resampled with replacement)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summarize(values: Sequence[float]) -> dict:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    lo, hi = bootstrap_ci(v)
    return {
        "n": int(v.size), "mean": float(v.mean()), "ci_low": lo, "ci_high": hi,
        "median": float(np.median(v)),
        "over_0.1": float((v > 0.1).mean()), "over_0.3": float((v > 0.3).mean()),
        "over_0.5": float((v > 0.5).mean()),
    }
