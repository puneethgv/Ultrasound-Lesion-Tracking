"""Evaluation helpers: per-case scores, confidence intervals and trivial baselines."""

from .metrics import (
    bootstrap_ci,
    box_fill,
    box_fill_dice,
    case_scores,
    dice,
    location_prior,
    median_area_fraction,
    prior_mask,
    summarize,
)
from .report import format_summary, load_masks, score_folder

__all__ = [
    "dice", "case_scores", "bootstrap_ci", "summarize",
    "box_fill", "box_fill_dice", "location_prior", "median_area_fraction", "prior_mask",
    "score_folder", "load_masks", "format_summary",
]
