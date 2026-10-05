"""Validate spline-mask alignment against the independent GUI boxes, for every spline case.

The spline (QuantUS pixels) is mapped to the RF pre-scan grid with the inverse QuantUS warp
(``usloc.labels.quantus``); the GUI boxes were annotated directly in that grid (h = sample, v = line).
Two independent annotations of the same lesion should overlap there. Cases that do not are flagged
for manual review.

Usage:
    python scripts/validate_alignment.py                 # report -> $USLOC_DERIVED/alignment/
    python scripts/validate_alignment.py --gallery 24    # + overlay gallery of 24 cases
"""

from __future__ import annotations

import argparse
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from usloc import config as C
from usloc.data.cohort import build_cohort
from usloc.labels import load_boxes, load_rf_case
from usloc.labels.register import DEFAULT_SCAN

warnings.simplefilter("ignore")


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def _frac_inside(box, ref) -> float:
    """Fraction of ``box`` area lying inside ``ref``."""
    ix = max(0.0, min(box[2], ref[2]) - max(box[0], ref[0]))
    iy = max(0.0, min(box[3], ref[3]) - max(box[1], ref[1]))
    area = (box[2] - box[0]) * (box[3] - box[1])
    return float(ix * iy / area) if area > 0 else 0.0


def _gui_boxes(case: str) -> dict:
    paths = glob.glob(str(C.GUI_ROOT / "gui_*" / f"{case}.xlsx"))
    if not paths:
        return {}
    out = {}
    for b in load_boxes(paths[0]):
        if None in (b.h1, b.h2, b.v1, b.v2):
            continue
        h0, v0, h1, v1 = b.xyxy               # h = sample, v = line
        out.setdefault(b.roi_kind, (v0, h0, v1, h1))  # -> (line0, sample0, line1, sample1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-iou", type=float, default=0.3, help="flag if IoU(spline, large box) below this")
    ap.add_argument("--gallery", type=int, default=0, help="number of cases to plot (0 = none)")
    args = ap.parse_args()

    out_dir = C.DERIVED / "alignment"
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort = build_cohort()
    cases = cohort[cohort.has_spline]

    rows, keep = [], {}
    for _, row in cases.iterrows():
        case = row["case"]
        rec = {"case": case, "class": row["class"], "site": row["site"]}
        try:
            rc = load_rf_case(case)
        except Exception as e:  # noqa: BLE001 — report and continue
            rows.append({**rec, "status": f"error: {type(e).__name__}: {e}"})
            continue
        if rc is None or rc.prescan_polygon is None:
            rows.append({**rec, "status": "no RF acquisition / spline"})
            continue
        ln, sm = rc.prescan_polygon[:, 0], rc.prescan_polygon[:, 1]
        sbox = (ln.min(), sm.min(), ln.max(), sm.max())
        rec.update(status="ok", scan=rc.spline.scan_name, frame=rc.spline.frame, n_frames=rc.n_frames,
                   spline_lines=f"{sbox[0]:.0f}-{sbox[2]:.0f}", spline_samples=f"{sbox[1]:.0f}-{sbox[3]:.0f}")
        boxes = _gui_boxes(case)
        for kind, b in boxes.items():
            rec[f"{kind}_box"] = f"L{b[0]:.0f}-{b[2]:.0f} S{b[1]:.0f}-{b[3]:.0f}"
            rec[f"iou_{kind}"] = round(_iou(sbox, b), 3)
            rec[f"{kind}_inside_spline"] = round(_frac_inside(b, sbox), 3)
        # GUI boxes were annotated on raw_0_0; a spline on another acquisition is not comparable
        if not boxes:
            rec["flag"], rec["flag_reason"] = True, "no GUI box"
        elif Path(rc.spline.scan_name).stem != DEFAULT_SCAN:
            rec["flag"], rec["flag_reason"] = True, f"spline on {rc.spline.scan_name}, boxes on {DEFAULT_SCAN}"
        elif rec.get("iou_large", rec.get("iou_small", 0.0)) < args.min_iou:
            rec["flag"], rec["flag_reason"] = True, f"IoU < {args.min_iou}"
        else:
            rec["flag"], rec["flag_reason"] = False, ""
        rows.append(rec)
        if len(keep) < args.gallery:  # keep only what the gallery draws (full RF cines are large)
            fr = int(np.clip(rc.spline.frame, 0, rc.n_frames - 1))
            keep[case] = (rc.frame_image(fr), rc.mask, rc.grid.forward, boxes, fr)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "alignment_report.csv", index=False)
    ok = df[df.status == "ok"]
    print(f"spline cases: {len(df)}   mapped: {len(ok)}   failed: {len(df) - len(ok)}")
    if len(ok):
        for col in ("iou_large", "large_inside_spline", "iou_small", "small_inside_spline"):
            if col in ok:
                print(f"  {col:22s} median={ok[col].median():.3f}  min={ok[col].min():.3f}")
        flagged = ok[ok["flag"].astype(bool)]
        print(f"  flagged for review: {len(flagged)}")
        if len(flagged):
            cols = ["case", "flag_reason", "spline_lines", "spline_samples"]
            print(flagged[cols + [c for c in ("large_box", "iou_large") if c in ok]].to_string(index=False))
    print("report ->", out_dir / "alignment_report.csv")

    if keep:
        _gallery(keep, out_dir / "alignment_gallery.png")


def _gallery(keep: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from usloc.viz import overlay_mask

    cases = list(keep)
    ncol = 4
    nrow = int(np.ceil(len(cases) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)
    for a in ax.ravel():
        a.axis("off")
    for a, case in zip(ax.ravel(), cases, strict=False):
        image, mask, forward, boxes, fr = keep[case]
        a.imshow(overlay_mask(image, mask, (255, 0, 200), 0.4))
        for kind, color in (("large", "#ff3d00"), ("small", "#00e5ff")):
            if kind in boxes:
                l0, s0, l1, s1 = boxes[kind]
                ls = np.r_[np.linspace(l0, l1, 20), np.full(20, l1), np.linspace(l1, l0, 20), np.full(20, l0)]
                ss = np.r_[np.full(20, s0), np.linspace(s0, s1, 20), np.full(20, s1), np.linspace(s1, s0, 20)]
                x, y = forward(ls, ss)
                a.plot(x, y, "-", color=color, lw=1.0)
        a.set_title(f"{case}  frame {fr}", fontsize=8)
    fig.suptitle("Spline mask (magenta, QuantUS-inverted) vs GUI boxes (red large / cyan small) on RF fan",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    print("gallery ->", path)


if __name__ == "__main__":
    main()
