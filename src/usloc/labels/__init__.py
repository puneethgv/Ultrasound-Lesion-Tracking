"""Ground-truth parsing: GUI bounding boxes, FLL_ROI spline masks, and label↔image registration."""

from .boxes import BoxLabel, load_boxes
from .register import RegisteredSample, register_case
from .splines import SplineLabel, load_spline, rasterize_spline

__all__ = [
    "BoxLabel", "load_boxes",
    "SplineLabel", "load_spline", "rasterize_spline",
    "RegisteredSample", "register_case",
]
