"""Train a YOLO11-seg baseline (box + mask + 3-class) and evaluate on the locked Halle test split.

Usage:
    python scripts/train_baseline.py --epochs 120
    python scripts/train_baseline.py --epochs 2 --name smoke   # quick pipeline check
"""

from __future__ import annotations

import argparse
from pathlib import Path

from usloc import config as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s-seg.pt")
    ap.add_argument("--data", default=str(C.DERIVED / "yolo_seg" / "dataset.yaml"))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="yolo11s_seg_baseline")
    ap.add_argument("--device", default="0")
    ap.add_argument("--patience", type=int, default=40)
    args = ap.parse_args()

    from ultralytics import YOLO

    project = str(C.DERIVED / "runs")
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=project,
        name=args.name,
        seed=0,
        patience=args.patience,
        # ultrasound-appropriate augmentation: no vertical flip (depth is meaningful),
        # muted color jitter (grayscale), mild geometric jitter.
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        mosaic=0.5,
        plots=True,
        verbose=True,
    )

    print("\n=== External test (Halle) evaluation ===")
    metrics = model.val(data=args.data, split="test", project=project, name=f"{args.name}_test", plots=True)
    box = metrics.box
    seg = metrics.seg
    print(f"Box  mAP50: {box.map50:.3f}  mAP50-95: {box.map:.3f}")
    print(f"Seg  mAP50: {seg.map50:.3f}  mAP50-95: {seg.map:.3f}")
    for i, name in C.ID_TO_CLASS.items():
        try:
            print(f"  {name:12} box AP50={box.ap50[i]:.3f}  seg AP50={seg.ap50[i]:.3f}")
        except Exception:
            pass
    print("weights:", Path(project) / args.name / "weights" / "best.pt")


if __name__ == "__main__":
    main()
