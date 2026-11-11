"""Validate spline-mask alignment against the independent GUI boxes, for every spline case.

The spline (QuantUS pixels) is mapped to the RF pre-scan grid with the inverse QuantUS warp
(``usloc.labels.quantus``); the GUI boxes were annotated directly in that grid (h = sample, v = line).
Two independent annotations of the same lesion should overlap there. Cases that do not are flagged
for manual review, together with what is known about why (``usloc.labels.check``):

    relation         mask inside a larger box / covers a smaller box / shifted / apart
    frame_gap        mask frame minus box frame (the liver moves between frames)
    iou_tracked      IoU after carrying the box to the mask's frame with the tissue motion
    iou_new_quantus  IoU if the ROI had been drawn on the picture of a newer QuantUS

Usage:
    python scripts/validate_alignment.py                 # report -> $USLOC_DERIVED/alignment/
    python scripts/validate_alignment.py --gallery 24    # + overlay gallery of 24 cases
    python scripts/validate_alignment.py --review        # + one two-frame picture per flagged case
    python scripts/validate_alignment.py --cases UKDCEUS056 UKDCEUS057
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from usloc import config as C
from usloc.data.cohort import build_cohort
from usloc.labels.agreement import RELATION_TEXT, RELATIONS, shift_box
from usloc.labels.check import as_box, gui_boxes, score_case

warnings.simplefilter("ignore")

COLUMNS = [
    "case", "class", "site", "status", "flag", "flag_reason", "relation",
    "iou_large", "spline_inside_large", "large_inside_spline", "iou_small", "spline_inside_small",
    "small_inside_spline", "d_line", "d_sample",
    "frame", "box_frame", "frame_gap", "n_frames", "frame_in_range",
    "iou_tracked", "relation_tracked", "track_d_line", "track_d_sample", "track_conf",
    "iou_new_quantus", "spline_on_picture",
    "roi_file", "scan", "box_kind", "box_sample_name", "box_device", "n_box_rows",
    "spline_lines", "spline_samples", "large_box", "small_box",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-iou", type=float, default=0.3, help="flag if IoU(spline, large box) below this")
    ap.add_argument("--gallery", type=int, default=0, help="number of cases to plot (0 = none)")
    ap.add_argument("--review", action="store_true",
                    help="write one two-frame picture per flagged case to <out>/review/")
    ap.add_argument("--no-track", action="store_true", help="skip the frame-to-frame motion estimate")
    ap.add_argument("--cases", nargs="*", default=None, help="only these cases (default: all)")
    ap.add_argument("--out", default=None, help="output folder (default: $USLOC_DERIVED/alignment)")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else C.DERIVED / "alignment"
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort = build_cohort()
    cases = cohort[cohort.has_spline]
    if args.cases:
        cases = cases[cases["case"].isin(args.cases)]

    rows, keep, n_review = [], {}, 0
    for _, row in cases.iterrows():
        case = row["case"]
        rec, rc = score_case(case, min_iou=args.min_iou, track=not args.no_track)
        rows.append({"case": case, "class": row["class"], "site": row["site"], **rec})
        if rc is None or rec.get("status") != "ok":
            continue
        want_gallery = len(keep) < args.gallery
        want_review = args.review and rec.get("flag") and "box_frame" in rec
        if not (want_gallery or want_review):
            continue
        boxes = {k: as_box(b) for k, b in gui_boxes(case, Path(rc.spline.scan_name).stem)[0].items()}
        if want_gallery:  # keep only what the gallery draws (full RF cines are large)
            fr = int(np.clip(rc.spline.frame, 0, rc.n_frames - 1))
            keep[case] = (rc.frame_image(fr), rc.mask, rc.grid.forward, boxes, fr)
        if want_review:
            (out_dir / "review").mkdir(exist_ok=True)
            _review(case, rc, rec, boxes, out_dir / "review" / f"{case}.png")
            n_review += 1

    df = pd.DataFrame(rows)
    df = df[[c for c in COLUMNS if c in df] + [c for c in df if c not in COLUMNS]]
    for col in ("frame", "box_frame", "frame_gap", "n_frames", "n_box_rows"):
        if col in df:  # whole numbers, also when some cases have none
            df[col] = df[col].astype("Int64")
    df.to_csv(out_dir / "alignment_report.csv", index=False)
    _summary(df, args.min_iou)
    print("report ->", out_dir / "alignment_report.csv")
    if n_review:
        print(f"review pictures ({n_review}) ->", out_dir / "review")
    if keep:
        _gallery(keep, out_dir / "alignment_gallery.png")


def _summary(df: pd.DataFrame, min_iou: float) -> None:
    ok = df[df.status == "ok"] if "status" in df else df.iloc[0:0]
    print(f"spline cases: {len(df)}   scored: {len(ok)}   not scored: {len(df) - len(ok)}")
    if "status" in df:
        for _, r in df[df.status != "ok"].iterrows():
            print(f"  {r['case']}: {r['status']}")
    if not len(ok):
        return
    for col in ("iou_large", "large_inside_spline", "iou_small", "small_inside_spline"):
        if col in ok:
            print(f"  {col:22s} median={ok[col].median():.3f}  min={ok[col].min():.3f}")
    if "relation" in ok:
        print("  mask vs box:")
        for rel in RELATIONS:
            n = int((ok["relation"] == rel).sum())
            if n:
                print(f"    {rel:11s}{n:5d}   {RELATION_TEXT[rel]}")
    flagged = ok[ok["flag"].astype(bool)]
    print(f"  flagged for review: {len(flagged)}")
    if not len(flagged):
        return
    if "iou_tracked" in flagged:
        n = int((flagged["iou_tracked"] >= min_iou).sum())
        print(f"    of these, the box agrees once it is carried to the mask's frame: {n}")
    iou_col = "iou_large" if "iou_large" in flagged else "iou_small"
    order = flagged.sort_values(iou_col) if iou_col in flagged else flagged
    for _, r in order.iterrows():
        iou = f"IoU {r[iou_col]:.2f}" if iou_col in r and pd.notna(r[iou_col]) else "IoU  n/a"
        print(f"    {r['case']:16s}{iou}   {r['flag_reason']}")


def _outline(forward, box, n: int = 20):
    """Fan-image outline of a pre-scan box ``(line0, sample0, line1, sample1)``."""
    l0, s0, l1, s1 = box
    ls = np.r_[np.linspace(l0, l1, n), np.full(n, l1), np.linspace(l1, l0, n), np.full(n, l0)]
    ss = np.r_[np.full(n, s0), np.linspace(s0, s1, n), np.full(n, s1), np.linspace(s1, s0, n)]
    return forward(ls, ss)


def _review(case: str, rc, rec: dict, boxes: dict, path) -> None:
    """Left: the box on the frame it was drawn on. Right: the mask on its frame, with the box as drawn
    (dashed) and, if the two frames differ, the box carried along with the tissue (yellow, dotted)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from usloc.viz import overlay_mask

    f_mask = int(np.clip(rc.spline.frame, 0, rc.n_frames - 1))
    f_box = int(np.clip(rec["box_frame"], 0, rc.n_frames - 1))
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax[0].imshow(rc.frame_image(f_box), cmap="gray", vmin=0, vmax=1)
    ax[1].imshow(overlay_mask(rc.frame_image(f_mask), rc.mask, (255, 0, 200), 0.4))
    for kind, color in (("large", "#ff3d00"), ("small", "#00e5ff")):
        if kind in boxes:
            x, y = _outline(rc.grid.forward, boxes[kind])
            ax[0].plot(x, y, "-", color=color, lw=1.2)
            ax[1].plot(x, y, "--", color=color, lw=0.9)
    if rc.polygon is not None:
        ax[0].plot(np.r_[rc.polygon[:, 0], rc.polygon[0, 0]], np.r_[rc.polygon[:, 1], rc.polygon[0, 1]],
                   "--", color="#ff00c8", lw=0.9)
    if "track_d_line" in rec and rec.get("box_kind") in boxes:
        moved = shift_box(boxes[rec["box_kind"]], rec["track_d_line"], rec["track_d_sample"])
        x, y = _outline(rc.grid.forward, moved)
        ax[1].plot(x, y, ":", color="#ffee00", lw=1.5)
    ax[0].set_title(f"GUI box on its frame ({f_box}); mask outline dashed", fontsize=9)
    ax[1].set_title(f"Mask on its frame ({f_mask}); box dashed, box moved with the tissue dotted", fontsize=9)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    head = f"{case}   IoU {rec.get('iou_large', rec.get('iou_small', float('nan'))):.2f}"
    if "iou_tracked" in rec:
        head += f"   after following the tissue {rec['iou_tracked']:.2f}"
    fig.suptitle(f"{head}\n{rec.get('flag_reason', '')}", fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


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
                x, y = _outline(forward, boxes[kind])
                a.plot(x, y, "-", color=color, lw=1.0)
        a.set_title(f"{case}  frame {fr}", fontsize=8)
    fig.suptitle("Spline mask (magenta, QuantUS-inverted) vs GUI boxes (red large / cyan small) on RF fan",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    print("gallery ->", path)


if __name__ == "__main__":
    main()
