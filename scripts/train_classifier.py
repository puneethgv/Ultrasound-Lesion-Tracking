"""Export lesion crops and train the stage-2 classifier.

Usage:
    python scripts/train_classifier.py --task binary       # benign vs malignant
    python scripts/train_classifier.py --task multiclass   # fnh / hemangioma / metastasis
"""

from __future__ import annotations

import argparse
import json

from usloc import config as C
from usloc.classify import export_crops
from usloc.classify.train import train_classifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="binary", choices=["binary", "multiclass"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    out = C.DERIVED / "crops" / args.task
    if not args.skip_export:
        summary = export_crops(out, task=args.task, max_frames=args.max_frames)
        print(json.dumps(summary, indent=2))

    _, results = train_classifier(out, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
    print("\n=== RESULTS ===")
    print(json.dumps({k: v for k, v in results.items() if k != "confusion"}, indent=2, default=str))


if __name__ == "__main__":
    main()
