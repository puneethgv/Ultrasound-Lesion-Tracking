"""Compare a case's spline mask with its GUI boxes: the logic behind ``scripts/validate_alignment.py``.

The spline (QuantUS pixels) is mapped to the RF pre-scan grid with the inverse QuantUS warp; the GUI
boxes were annotated directly in that grid (h = sample, v = line). Two independent annotations of the
same lesion should agree there. When they do not, :func:`score_case` records *how* they differ:

* ``relation``: the mask lies inside a larger box, covers a smaller box, overlaps it shifted, or is apart;
* ``frame_gap``: the box and the mask are usually drawn on different frames of the recording;
* ``iou_tracked``: the IoU after carrying the box to the mask's frame with the tissue motion;
* ``iou_new_quantus``: the IoU if the ROI had been drawn on the picture of a newer QuantUS.

Nothing here assumes which recording a box was drawn on: the comparison is made on the mask's own
recording and the box's ``sample_name`` is reported, so a mismatch shows up as a low IoU, not as a
case that is skipped.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np

from .. import config as C
from ..recon import rf_to_bmode
from .agreement import RELATION_TEXT, Agreement, Box, compare, shift_box
from .boxes import BoxLabel, load_boxes
from .quantus import padded_start_depth_factor, quantus_geometry, spline_to_prescan
from .register import DEFAULT_SCAN, RfCase, explain_missing_rf, load_rf_case
from .splines import find_spline_file, load_spline
from .tracking import track_box


def as_box(b: BoxLabel) -> Box:
    """GUI box as ``(line0, sample0, line1, sample1)`` (the GUI stores h = sample, v = line)."""
    h0, v0, h1, v1 = b.xyxy
    return (float(v0), float(h0), float(v1), float(h1))


def _names_scan(name: str, scan: str) -> bool:
    """True if ``name`` mentions the recording ``scan`` (``raw_0`` does not match ``raw_0_0``)."""
    return re.search(rf"(^|[^0-9A-Za-z]){re.escape(scan)}($|[^0-9_])", name) is not None


def gui_boxes(case: str, scan: str | None = None) -> tuple[dict[str, BoxLabel], int]:
    """The GUI boxes of ``case`` by kind (``large`` / ``small``), and how many rows the file has.

    A file normally has one row per kind. If it has several and their ``sample_name`` names a
    recording, the row for ``scan`` is preferred; otherwise the first complete row is used.
    """
    paths = sorted(glob.glob(str(C.GUI_ROOT / "gui_*" / f"{case}.xlsx")))
    if not paths:
        return {}, 0
    rows = [b for b in load_boxes(paths[0]) if None not in (b.h1, b.h2, b.v1, b.v2)]
    out: dict[str, BoxLabel] = {}
    for b in rows:
        cur = out.get(b.roi_kind)
        if cur is None or (scan and _names_scan(b.case, scan) and not _names_scan(cur.case, scan)):
            out[b.roi_kind] = b
    return out, len(rows)


def _bounds(line: np.ndarray, sample: np.ndarray) -> Box:
    return (float(line.min()), float(sample.min()), float(line.max()), float(sample.max()))


def score_case(case: str, *, min_iou: float = 0.3, track: bool = True) -> tuple[dict, RfCase | None]:
    """Score one case. Returns ``(record, rf_case)``; the record's ``status`` is ``ok`` if it was scored."""
    path = find_spline_file(case)
    if path is None:
        return {"status": "no ROI file"}, None
    rec: dict = {"roi_file": path.name}
    try:
        sp = load_spline(path)
        scan = Path(sp.scan_name).stem if sp.scan_name else DEFAULT_SCAN
        rc = load_rf_case(case)
    except Exception as e:  # noqa: BLE001 - report and continue
        return {**rec, "status": f"error: {type(e).__name__}: {e}"}, None
    rec.update(scan=sp.scan_name, frame=sp.frame)
    if rc is None:
        return {**rec, "status": f"no RF: {explain_missing_rf(case, scan)}"}, None
    if rc.prescan_polygon is None:
        return {**rec, "status": "ROI has fewer than 3 points"}, rc

    geom, n = rc.geom, rc.n_frames
    sbox = _bounds(rc.prescan_polygon[:, 0], rc.prescan_polygon[:, 1])
    qg = quantus_geometry(geom)
    rec.update(
        status="ok", n_frames=n, frame_in_range=bool(0 <= sp.frame < n),
        spline_lines=f"{sbox[0]:.0f}-{sbox[2]:.0f}", spline_samples=f"{sbox[1]:.0f}-{sbox[3]:.0f}",
        spline_on_picture=True if qg is None else qg.contains(sp.x, sp.y),
    )

    reasons: list[str] = []
    try:
        boxes, n_rows = gui_boxes(case, scan)
    except Exception as e:  # noqa: BLE001 - one unreadable box file must not stop the run
        boxes, n_rows = {}, 0
        reasons.append(f"GUI box file not readable ({type(e).__name__}: {e})")
    rec["n_box_rows"] = n_rows
    scored: dict[str, tuple[Box, Agreement]] = {}
    for kind, b in boxes.items():
        gb = as_box(b)
        a = compare(sbox, gb, min_iou=min_iou)
        scored[kind] = (gb, a)
        rec[f"{kind}_box"] = f"L{gb[0]:.0f}-{gb[2]:.0f} S{gb[1]:.0f}-{gb[3]:.0f}"
        rec[f"iou_{kind}"] = round(a.iou, 3)
        rec[f"{kind}_inside_spline"] = round(a.box_in_spline, 3)
        rec[f"spline_inside_{kind}"] = round(a.spline_in_box, 3)

    kind = "large" if "large" in scored else next(iter(scored), None)
    if kind is None:
        if not reasons:
            reasons.append("no GUI box")
    else:
        gb, a = scored[kind]
        b = boxes[kind]
        rec.update(box_kind=kind, box_frame=b.frame, box_sample_name=b.case, box_device=b.device,
                   frame_gap=int(sp.frame - b.frame), d_line=round(a.d_line, 1),
                   d_sample=round(a.d_sample, 1), relation=a.relation)

        alt = None
        if qg is not None and geom.samples_per_line <= qg.n_samples:
            factor = padded_start_depth_factor(geom.samples_per_line)
            line, sample = spline_to_prescan(sp.x, sp.y, geom, factor)
            line = np.clip(line, 0, geom.number_of_lines - 1)
            sample = np.clip(sample, 0, geom.samples_per_line - 1)
            alt = compare(_bounds(line, sample), gb, min_iou=min_iou)
            rec["iou_new_quantus"] = round(alt.iou, 3)

        low = a.iou < min_iou
        tracked = None
        f_box, f_mask = (int(np.clip(f, 0, n - 1)) for f in (b.frame, sp.frame))
        if track and low and f_box != f_mask:
            cache: dict[int, np.ndarray] = {}

            def frame_at(k: int) -> np.ndarray:
                if k not in cache:
                    cache.clear()  # only neighbouring frames are needed; keep memory flat
                    cache[k] = rf_to_bmode(rc.rf[k].astype(np.float32), axis=1)
                return cache[k]

            try:
                dl, ds, conf = track_box(frame_at, gb, f_box, f_mask)
            except Exception:  # noqa: BLE001 - the motion estimate is a hint; go on without it
                pass
            else:
                tracked = compare(sbox, shift_box(gb, dl, ds), min_iou=min_iou)
                rec.update(track_d_line=round(dl, 1), track_d_sample=round(ds, 1),
                           track_conf=round(conf, 2), iou_tracked=round(tracked.iou, 3),
                           relation_tracked=tracked.relation)

        if low:
            reasons.append(f"IoU < {min_iou:g}: {RELATION_TEXT[a.relation]}")
            if rec["frame_gap"]:
                reasons.append(f"box on frame {b.frame}, mask on frame {sp.frame}")
            if tracked is not None and tracked.iou >= min_iou:
                reasons.append(f"agrees after following the tissue between the frames (IoU {tracked.iou:.2f})")
            if alt is not None and alt.iou >= 0.5 and alt.iou >= a.iou + 0.3:
                reasons.append(f"fits the picture of a newer QuantUS (IoU {alt.iou:.2f})")
            if scan != DEFAULT_SCAN:
                reasons.append(f"mask on {sp.scan_name}")

    if not rec["frame_in_range"]:
        reasons.append(f"Frame {sp.frame} is outside the recording ({n} frames)")
    if not rec["spline_on_picture"]:
        reasons.append("ROI leaves the QuantUS picture (other QuantUS settings?)")
    rec["flag"] = bool(reasons)
    rec["flag_reason"] = "; ".join(reasons)
    return rec, rc
