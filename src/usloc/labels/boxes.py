"""Parse GUI per-case bounding-box labels (``gui_<class>[_halle]/<case>.xlsx``).

Each file has sheets ``Large_ROI`` and ``Small_ROI`` with columns:
    sample_name, device, frame, h1, h2, v1, v2, h1_new, h2_new, v1_new, v2_new
where (h1,h2) is the horizontal extent and (v1,v2) the vertical extent of the lesion box, in the
raw-acquisition coordinate grid (registration to a canonical image is resolved separately).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class BoxLabel:
    case: str
    roi_kind: str  # "large" or "small"
    device: str
    frame: int
    h1: float
    h2: float
    v1: float
    v2: float
    # refined ("_new") coordinates when present
    h1_new: float | None = None
    h2_new: float | None = None
    v1_new: float | None = None
    v2_new: float | None = None

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """Axis-ordered (x_min, y_min, x_max, y_max) from h=horizontal, v=vertical."""
        return (min(self.h1, self.h2), min(self.v1, self.v2),
                max(self.h1, self.h2), max(self.v1, self.v2))


def load_boxes(xlsx_path: str | Path) -> list[BoxLabel]:
    """Return all box labels in a per-case GUI file (usually one Large and one Small)."""
    case = Path(xlsx_path).stem
    out: list[BoxLabel] = []
    xl = pd.ExcelFile(xlsx_path)
    for sheet in xl.sheet_names:
        kind = "large" if "large" in sheet.lower() else "small" if "small" in sheet.lower() else sheet
        df = xl.parse(sheet)
        for _, r in df.iterrows():
            def g(col):
                return None if col not in df.columns or pd.isna(r[col]) else float(r[col])

            out.append(
                BoxLabel(
                    case=str(r.get("sample_name", case)),
                    roi_kind=kind,
                    device=str(r.get("device", "")),
                    frame=int(r["frame"]) if "frame" in df.columns and not pd.isna(r["frame"]) else 0,
                    h1=g("h1"), h2=g("h2"), v1=g("v1"), v2=g("v2"),
                    h1_new=g("h1_new"), h2_new=g("h2_new"),
                    v1_new=g("v1_new"), v2_new=g("v2_new"),
                )
            )
    return out
