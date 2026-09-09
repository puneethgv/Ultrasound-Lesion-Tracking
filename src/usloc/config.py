"""Central paths and dataset constants.

Everything can be overridden with environment variables so the code is portable off this machine:
  USLOC_DATA_ROOT  -> root of the Dresden_new dataset
  USLOC_DERIVED    -> where reconstructed images / exported datasets are cached
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Dataset location -------------------------------------------------------
DATA_ROOT = Path(
    os.environ.get(
        "USLOC_DATA_ROOT",
        "/media/ahmed-el-kaffas/20TB-HDD/Puneeth/Dresden_new",
    )
)
FLL_DATA = DATA_ROOT / "FLL Data"
GUI_ROOT = FLL_DATA / "johanna_samples_gui"
FLL_ROI_DIR = FLL_DATA / "FLL_ROI"
MASTER_XLSX = DATA_ROOT / "samples_normalisierte_dresden_halle.xlsx"
REGISTRY_XLSX = DATA_ROOT / "Bericht Rohdaten FLL.xlsx"

# --- Label taxonomy ---------------------------------------------------------
# Only the three GUI-labeled localization classes (others exist clinically but have no ROI labels).
CLASSES = ("fnh", "hemangioma", "metastasis")
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
ID_TO_CLASS = {i: c for c, i in CLASS_TO_ID.items()}
# Coarse benign/malignant grouping (stabilises the tiny FNH class).
BENIGN = ("fnh", "hemangioma")
MALIGNANT = ("metastasis",)

# --- Sites ------------------------------------------------------------------
# Site is encoded in the case ID prefix: UKH... = Halle, everything else (CEUS.../UKD...) = Dresden.
SITE_TRAIN = "Dresden"
SITE_TEST = "Halle"

# --- Raw-file quirk ---------------------------------------------------------
# Every Clarius-style *.raw file on this dataset has filesize == frames*lines*samples*bytes + 564.
# Whether those 564 bytes are a leading header or a trailing footer is validated empirically in the
# exploration notebook (Phase 2/3); this constant documents the observation.
RAW_TAIL_BYTES = 564

# --- Derived outputs (gitignored) -------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DERIVED = Path(os.environ.get("USLOC_DERIVED", PROJECT_ROOT / "data" / "derived"))


def site_of(case_id: str) -> str:
    """Return 'Halle' for UKH* case IDs, else 'Dresden'."""
    return SITE_TEST if str(case_id).upper().startswith("UKH") else SITE_TRAIN
