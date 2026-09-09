"""Ground-truth parsing: GUI bounding boxes and FLL_ROI spline masks."""

from .boxes import BoxLabel, load_boxes
from .splines import SplineLabel, load_spline, rasterize_spline

__all__ = ["BoxLabel", "load_boxes", "SplineLabel", "load_spline", "rasterize_spline"]
