"""Build the case-level cohort table by scanning the GUI labels and cross-referencing raw data,
spline masks, and the clinical / QC spreadsheets.

Output columns: case, class, class_id, site, benign_malignant, n_tar (raw acquisitions),
has_raw_dir, has_box, has_spline, ground_truth, qc_note, excluded.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import pandas as pd

from .. import config as C


def _clean_cases(paths: list[str]) -> list[str]:
    return sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in paths
        if not os.path.basename(p).startswith("._")
    )


def load_master_clinical(path: str | Path | None = None) -> pd.DataFrame:
    """Return the ``focal_lesions`` sheet keyed by case id (patients_id)."""
    path = Path(path or C.MASTER_XLSX)
    df = pd.read_excel(path, sheet_name="focal_lesions")
    df = df.rename(columns={"patients_id": "case"})
    df["case"] = df["case"].astype(str).str.strip()
    return df


def load_qc_flags(path: str | Path | None = None) -> pd.DataFrame:
    """Parse the 'Tabelle1' DATENANALYSE TOOLBOX sheet into per-case QC notes / exclusion flags."""
    path = Path(path or C.MASTER_XLSX)
    raw = pd.read_excel(path, sheet_name="Tabelle1", header=None)
    # locate the header row that contains 'patients_id'
    id_col = None
    header_row = 0
    for i in range(min(5, len(raw))):
        for j, v in enumerate(raw.iloc[i]):
            if isinstance(v, str) and "patients_id" in v.lower():
                header_row, id_col = i, j
                break
        if id_col is not None:
            break
    if id_col is None:
        return pd.DataFrame(columns=["case", "qc_note", "excluded"])

    rows = []
    for i in range(header_row + 1, len(raw)):
        case = raw.iat[i, id_col]
        if not isinstance(case, str) or not case.strip():
            continue
        notes = [
            str(v).strip()
            for j, v in enumerate(raw.iloc[i])
            if j != id_col and isinstance(v, str) and v.strip()
        ]
        note = " | ".join(notes)
        excluded = bool(re.search(r"aussortiert", note, re.IGNORECASE))
        rows.append({"case": case.strip(), "qc_note": note, "excluded": excluded})
    return pd.DataFrame(rows)


def build_cohort(gui_root: str | Path | None = None) -> pd.DataFrame:
    """Scan ``gui_<class>[_halle]/*.xlsx`` and assemble the cohort table."""
    gui_root = Path(gui_root or C.GUI_ROOT)
    records = []
    for cls in C.CLASSES:
        for suffix in ("", "_halle"):
            gui_dir = gui_root / f"gui_{cls}{suffix}"
            raw_dir_base = gui_root / f"raw_data_{cls}{suffix}"
            if not gui_dir.is_dir():
                continue
            for case in _clean_cases(glob.glob(str(gui_dir / "*.xlsx"))):
                case_raw = raw_dir_base / case
                tars = (
                    [t for t in os.listdir(case_raw) if t.endswith(".tar") and not t.startswith("._")]
                    if case_raw.is_dir()
                    else []
                )
                spline = C.FLL_ROI_DIR / f"{case}_roi.pkl"
                records.append(
                    {
                        "case": case,
                        "class": cls,
                        "class_id": C.CLASS_TO_ID[cls],
                        "site": C.site_of(case),
                        "benign_malignant": "benign" if cls in C.BENIGN else "malignant",
                        "n_tar": len(tars),
                        "has_raw_dir": case_raw.is_dir(),
                        "has_box": True,
                        "has_spline": spline.exists(),
                    }
                )
    cohort = pd.DataFrame(records).drop_duplicates("case").reset_index(drop=True)

    # merge clinical ground_truth + QC (best-effort; dataset spreadsheets may be absent off-machine)
    try:
        clin = load_master_clinical()[["case", "ground_truth"]]
        cohort = cohort.merge(clin, on="case", how="left")
    except Exception:
        cohort["ground_truth"] = pd.NA
    try:
        qc = load_qc_flags()
        cohort = cohort.merge(qc, on="case", how="left")
    except Exception:
        cohort["qc_note"] = pd.NA
        cohort["excluded"] = False
    cohort["excluded"] = cohort.get("excluded", False).fillna(False)
    return cohort
