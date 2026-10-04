"""Ground-truth parsing: GUI bounding boxes, FLL_ROI spline masks, and label↔image registration."""

from .boxes import BoxLabel, load_boxes
from .quantus import QuantusScGeometry, quantus_geometry, spline_to_prescan
from .register import RegisteredSample, RfCase, load_rf_case, register_case
from .splines import SplineLabel, load_spline, rasterize_spline

__all__ = [
    "BoxLabel", "load_boxes",
    "SplineLabel", "load_spline", "rasterize_spline",
    "QuantusScGeometry", "quantus_geometry", "spline_to_prescan",
    "RegisteredSample", "RfCase", "load_rf_case", "register_case",
]
