"""Predict on the Halle external test set with a trained nnU-Net fold and report Dice, case by case,
next to two baselines that need no model (``usloc.eval``):

    location prior   the best fixed guess learnt from the training masks (lesions are usually centred)
    fill the box     what any method scores when it is handed the true bounding box (~0.88 if round)

Requires the nnUNet_* env vars to be set (nnUNet_raw / nnUNet_preprocessed / nnUNet_results).

Usage:
    python scripts/eval_nnunet.py --trainer nnUNetTrainer_100epochs --fold 0
    python scripts/eval_nnunet.py --dataset-id 2 --name LesionRF --pred <folder> --skip-predict
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd

from usloc import config as C
from usloc.eval import format_summary, score_folder


def _add_case_info(df: pd.DataFrame) -> pd.DataFrame:
    """Add class (cohort) and the mask-vs-box check result, when those are available on this machine."""
    try:
        from usloc.data.cohort import build_cohort

        df = df.merge(build_cohort()[["case", "class"]], on="case", how="left")
    except Exception:  # noqa: BLE001 - the dataset may not be mounted
        pass
    report = C.DERIVED / "alignment" / "alignment_report.csv"
    if report.exists():
        cols = [c for c in ("case", "relation", "flag") if c in pd.read_csv(report, nrows=0).columns]
        df = df.merge(pd.read_csv(report, usecols=cols), on="case", how="left")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", type=int, default=1)
    ap.add_argument("--name", default="Lesion")
    ap.add_argument("--trainer", default="nnUNetTrainer_100epochs")
    ap.add_argument("--fold", default="0")
    ap.add_argument("--checkpoint", default="checkpoint_best.pth",
                    help="checkpoint_best.pth (peak, avoids overfit epochs) or checkpoint_final.pth")
    ap.add_argument("--pred", default=None, help="prediction folder (default: $USLOC_DERIVED/nnunet/pred_halle)")
    ap.add_argument("--skip-predict", action="store_true", help="score the predictions already in --pred")
    args = ap.parse_args()

    raw = Path(os.environ["nnUNet_raw"]) / f"Dataset{args.dataset_id:03d}_{args.name}"
    out = Path(args.pred) if args.pred else C.DERIVED / "nnunet" / "pred_halle"
    if not args.skip_predict:
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["nnUNetv2_predict", "-i", str(raw / "imagesTs"), "-o", str(out),
             "-d", str(args.dataset_id), "-c", "2d", "-f", args.fold, "-tr", args.trainer,
             "-chk", args.checkpoint],
            check=True,
        )

    df = _add_case_info(score_folder(raw / "labelsTs", out, raw / "labelsTr"))
    print()
    print(format_summary(df, title="Halle external", model="nnU-Net", by=("class", "relation", "flag")))
    out.mkdir(parents=True, exist_ok=True)
    csv = out.parent / f"{out.name}_cases.csv"
    df.to_csv(csv, index=False)
    print("per-case scores ->", csv)


if __name__ == "__main__":
    main()
