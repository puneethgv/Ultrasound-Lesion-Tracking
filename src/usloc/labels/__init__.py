"""Ground-truth parsing: GUI bounding boxes, FLL_ROI spline masks, and label↔image registration."""

from .agreement import Agreement, box_iou, compare
from .boxes import BoxLabel, load_boxes
from .quantus import (
    QuantusScGeometry,
    padded_start_depth_factor,
    quantus_geometry,
    spline_to_prescan,
)
from .register import RegisteredSample, RfCase, explain_missing_rf, load_rf_case, register_case
from .splines import SplineLabel, find_spline_file, load_spline, rasterize_spline, spline_files
from .tracking import track_box

__all__ = [
    "BoxLabel", "load_boxes",
    "SplineLabel", "load_spline", "rasterize_spline", "find_spline_file", "spline_files",
    "QuantusScGeometry", "quantus_geometry", "spline_to_prescan", "padded_start_depth_factor",
    "RegisteredSample", "RfCase", "load_rf_case", "register_case", "explain_missing_rf",
    "Agreement", "box_iou", "compare", "track_box",
]
