"""Predict on the Halle external test set with a trained nnU-Net fold and report Dice.

Requires the nnUNet_* env vars to be set (nnUNet_raw / nnUNet_preprocessed / nnUNet_results).

Usage:
    python scripts/eval_nnunet.py --trainer nnUNetTrainer_100epochs --fold 0
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
from pathlib import Path

import numpy as np

from usloc import config as C


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred > 0, gt > 0
    denom = p.sum() + g.sum()
    return 1.0 if denom == 0 else float(2 * (p & g).sum() / denom)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", type=int, default=1)
    ap.add_argument("--name", default="Lesion")
    ap.add_argument("--trainer", default="nnUNetTrainer_100epochs")
    ap.add_argument("--fold", default="0")
    ap.add_argument("--checkpoint", default="checkpoint_best.pth",
                    help="checkpoint_best.pth (peak, avoids overfit epochs) or checkpoint_final.pth")
    args = ap.parse_args()

    import cv2

    raw = Path(os.environ["nnUNet_raw"]) / f"Dataset{args.dataset_id:03d}_{args.name}"
    out = C.DERIVED / "nnunet" / "pred_halle"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["nnUNetv2_predict", "-i", str(raw / "imagesTs"), "-o", str(out),
         "-d", str(args.dataset_id), "-c", "2d", "-f", args.fold, "-tr", args.trainer,
         "-chk", args.checkpoint],
        check=True,
    )

    dices = []
    for lp in sorted(glob.glob(str(raw / "labelsTs" / "*.png"))):
        pred = out / os.path.basename(lp)
        if not pred.exists():
            continue
        gt = cv2.imread(lp, cv2.IMREAD_UNCHANGED)
        pr = cv2.imread(str(pred), cv2.IMREAD_UNCHANGED)
        dices.append(dice(pr, gt))
    dices = np.array(dices)
    print(f"\n=== Halle external (n={len(dices)}) ===")
    print(f"mean Dice   : {dices.mean():.3f}")
    print(f"median Dice : {np.median(dices):.3f}")
    print(f"detected (Dice>0.10): {(dices > 0.10).mean():.2f}   (Dice>0.30): {(dices > 0.30).mean():.2f}")


if __name__ == "__main__":
    main()
