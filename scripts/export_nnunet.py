"""CLI to export the nnU-Net v2 dataset (Dresden -> imagesTr/labelsTr, Halle -> imagesTs/labelsTs).

Requires the nnUNet_raw env var (or --raw-root).

Usage:
    python scripts/export_nnunet.py --dataset-id 2 --name LesionRF
    # the same without the cases the mask-vs-box check flagged, to measure what they change:
    python scripts/export_nnunet.py --dataset-id 3 --name LesionRFclean \
        --exclude-file $USLOC_DERIVED/alignment/alignment_report.csv
"""

from __future__ import annotations

import argparse
import json

from usloc.data.cohort import read_case_list
from usloc.datasets.export_nnunet import export_nnunet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", type=int, default=1)
    ap.add_argument("--name", default="Lesion")
    ap.add_argument("--source", default="rf", choices=["rf", "dicom"],
                    help="rf = aligned RF fan images (default); dicom = legacy, misaligned masks")
    ap.add_argument("--raw-root", default=None, help="default: $nnUNet_raw")
    ap.add_argument("--no-test", action="store_true", help="do not write the Halle test images")
    ap.add_argument("--exclude-file", default=None,
                    help="cases to leave out of training: a text file (one case per line) or a CSV with a "
                         "'case' column; of alignment_report.csv only the flagged rows are used")
    args = ap.parse_args()

    exclude = read_case_list(args.exclude_file) if args.exclude_file else []
    summary = export_nnunet(args.dataset_id, args.name, source=args.source, raw_root=args.raw_root,
                            include_test=not args.no_test, exclude=exclude)
    print(json.dumps({k: v for k, v in summary.items() if k != "skipped"}, indent=2))
    print("left out of training on request:", len(exclude), "listed,", summary["excluded"], "removed")
    print("skipped (no image or mask):", len(summary["skipped"]), summary["skipped"])


if __name__ == "__main__":
    main()
