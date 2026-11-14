"""Export lesion crops and train the stage-2 classifier.

Usage:
    python scripts/train_classifier.py --task binary       # benign vs malignant
    python scripts/train_classifier.py --task multiclass   # fnh / hemangioma / metastasis
    # control: the same classifier on crops next to the lesion. If it scores about as well, the
    # classifier is not reading the lesion.
    python scripts/train_classifier.py --task binary --region beside
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from usloc import config as C
from usloc.classify import export_crops
from usloc.classify.dataset import REGIONS
from usloc.classify.train import train_classifier
from usloc.data.cohort import read_case_list


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="binary", choices=["binary", "multiclass"])
    ap.add_argument("--region", default="lesion", choices=list(REGIONS),
                    help="lesion = crops of the lesion (default); beside = control crops next to it")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--source", default="rf", choices=["rf", "dicom"])
    ap.add_argument("--workers", type=int, default=4, help="data-loader workers (0 on Windows)")
    ap.add_argument("--exclude-file", default=None,
                    help="cases to leave out of train/val (text file, or CSV with a 'case' column)")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    out = C.DERIVED / "crops" / (args.task if args.region == "lesion" else f"{args.task}_{args.region}")
    if not args.skip_export:
        exclude = read_case_list(args.exclude_file) if args.exclude_file else []
        summary = export_crops(out, task=args.task, max_frames=args.max_frames, source=args.source,
                               region=args.region, exclude=exclude)
        print(json.dumps(summary, indent=2))

    _, results = train_classifier(out, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                                  workers=args.workers)
    for split in ("val", "test"):
        if split in results:  # per-patient predictions, one row per patient
            patients = pd.DataFrame(results[split].pop("patients"))
            patients.to_csv(out / f"patients_{split}.csv", index=False)
            print("per-patient predictions ->", out / f"patients_{split}.csv")
    print("\n=== RESULTS ===")
    print(json.dumps({k: v for k, v in results.items() if k != "confusion"}, indent=2, default=str))


if __name__ == "__main__":
    main()
