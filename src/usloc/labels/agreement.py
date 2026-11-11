"""Agreement between a spline mask and a GUI box in the RF pre-scan grid ``(line, sample)``.

Boxes are ``(line0, sample0, line1, sample1)``. IoU alone mixes *where* two annotations are with *how
big* they are: a small lesion outlined inside a generous box scores a low IoU although both mark the
same place. :func:`compare` keeps the IoU and adds what is needed to tell the situations apart.
"""

from __future__ import annotations

from dataclasses import dataclass

Box = tuple[float, float, float, float]

#: How a spline's bounding rectangle relates to a GUI box, most to least reassuring.
RELATIONS = ("match", "inside_box", "covers_box", "shifted", "apart")

RELATION_TEXT = {
    "match": "mask and box agree",
    "inside_box": "mask lies inside a larger box (same place)",
    "covers_box": "mask covers a smaller box (same place)",
    "shifted": "mask and box overlap but are shifted",
    "apart": "mask and box do not overlap",
}


def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def box_iou(a: Box, b: Box) -> float:
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def frac_inside(box: Box, ref: Box) -> float:
    """Fraction of ``box`` area lying inside ``ref``."""
    area = _area(box)
    return float(_intersection(box, ref) / area) if area > 0 else 0.0


def shift_box(box: Box, d_line: float, d_sample: float) -> Box:
    return (box[0] + d_line, box[1] + d_sample, box[2] + d_line, box[3] + d_sample)


@dataclass(frozen=True)
class Agreement:
    iou: float
    spline_in_box: float  # fraction of the spline's rectangle inside the box
    box_in_spline: float  # fraction of the box inside the spline's rectangle
    d_line: float         # spline centre minus box centre, in scan lines
    d_sample: float       # the same in samples (positive = deeper)
    relation: str         # one of RELATIONS


def compare(spline_box: Box, gui_box: Box, *, min_iou: float = 0.3, inside: float = 0.8) -> Agreement:
    """Describe how the spline's bounding rectangle sits relative to a GUI box.

    ``match`` if the IoU reaches ``min_iou``. Otherwise ``inside_box`` / ``covers_box`` when at least
    ``inside`` of one rectangle lies within the other (same place, different size), ``shifted`` when
    they merely overlap, and ``apart`` when they do not overlap at all.
    """
    iou = box_iou(spline_box, gui_box)
    s_in_b = frac_inside(spline_box, gui_box)
    b_in_s = frac_inside(gui_box, spline_box)
    if iou >= min_iou:
        relation = "match"
    elif s_in_b >= inside:
        relation = "inside_box"
    elif b_in_s >= inside:
        relation = "covers_box"
    elif _intersection(spline_box, gui_box) > 0:
        relation = "shifted"
    else:
        relation = "apart"
    return Agreement(
        iou=iou,
        spline_in_box=s_in_b,
        box_in_spline=b_in_s,
        d_line=(spline_box[0] + spline_box[2]) / 2 - (gui_box[0] + gui_box[2]) / 2,
        d_sample=(spline_box[1] + spline_box[3]) / 2 - (gui_box[1] + gui_box[3]) / 2,
        relation=relation,
    )
