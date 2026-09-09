"""Locate the on-disk files for a given case: raw-data directory, DICOM cines, extracted RF/env sets."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path

from .. import config as C


@dataclass
class ExtractedSet:
    """One ``raw_<i>_<j>_extracted`` acquisition with its RF/env file group."""

    directory: Path
    presets: dict[str, dict[str, Path]] = field(default_factory=dict)  # e.g. {"C3_small": {...}}


def _class_and_suffix(case: str, gui_root: Path) -> tuple[str, str] | None:
    for cls in C.CLASSES:
        for suffix in ("", "_halle"):
            if (gui_root / f"raw_data_{cls}{suffix}" / case).is_dir():
                return cls, suffix
    return None


def find_case_dir(case: str, gui_root: str | Path | None = None) -> Path | None:
    gui_root = Path(gui_root or C.GUI_ROOT)
    cs = _class_and_suffix(case, gui_root)
    if cs is None:
        return None
    cls, suffix = cs
    return gui_root / f"raw_data_{cls}{suffix}" / case


def list_dicoms(case_dir: Path) -> list[Path]:
    return sorted(
        Path(p)
        for p in glob.glob(str(case_dir / "*.dcm"))
        if not os.path.basename(p).startswith("._")
    )


def list_extracted(case_dir: Path) -> list[ExtractedSet]:
    """Group the ``*_extracted`` folders by acquisition and probe preset (C3_small, L15_large, ...)."""
    out: list[ExtractedSet] = []
    for d in sorted(glob.glob(str(case_dir / "*_extracted"))):
        d = Path(d)
        if d.name.startswith("._"):
            continue
        presets: dict[str, dict[str, Path]] = {}
        for f in os.listdir(d):
            if f.startswith("._"):
                continue
            m = _match_preset(f)
            if m:
                preset, role = m
                presets.setdefault(preset, {})[role] = d / f
        out.append(ExtractedSet(directory=d, presets=presets))
    return out


_ROLE_PATTERNS = {
    "rf_yml": "_rf.yml",
    "rf_raw": "_rf.raw",
    "rf_npy": "_rf_no_tgc.npy",
    "env_yml": "_env.yml",
    "env_raw": "_env.raw",
    "delay_xlsx": "_rf_delay_samples.xlsx",
}


def _match_preset(fname: str) -> tuple[str, str] | None:
    for role, suffix in _ROLE_PATTERNS.items():
        if fname.endswith(suffix) and not fname.endswith(".lzo"):
            preset = fname[: -len(suffix)]
            return preset, role
    return None
