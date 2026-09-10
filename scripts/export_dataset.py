"""CLI to export the YOLO-seg dataset from the registered spline cases.

Usage:
    python scripts/export_dataset.py                 # -> $USLOC_DERIVED/yolo_seg
    python scripts/export_dataset.py --val-frac 0.2 --seed 0
"""

from __future__ import annotations

import argparse
import json

from usloc.datasets import export_yolo_seg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--space", default="crop", choices=["crop", "full"])
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-mask-area", type=int, default=30)
    ap.add_argument("--multiframe", action="store_true", help="expand each case across its cine")
    ap.add_argument("--class-agnostic", action="store_true", help="single 'lesion' class")
    ap.add_argument("--corr-thresh", type=float, default=0.85)
    ap.add_argument("--max-frames", type=int, default=40)
    args = ap.parse_args()

    summary = export_yolo_seg(
        args.out, space=args.space, val_frac=args.val_frac,
        seed=args.seed, min_mask_area=args.min_mask_area,
        multiframe=args.multiframe, class_agnostic=args.class_agnostic,
        corr_thresh=args.corr_thresh, max_frames=args.max_frames,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "skipped"}, indent=2))
    print("skipped:", len(summary["skipped"]))


if __name__ == "__main__":
    main()
