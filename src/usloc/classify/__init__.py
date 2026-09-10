"""Stage-2 lesion-type classifier (operates on lesion crops from the detector / GT boxes)."""

from .dataset import BINARY_MAP, export_crops

__all__ = ["export_crops", "BINARY_MAP"]
