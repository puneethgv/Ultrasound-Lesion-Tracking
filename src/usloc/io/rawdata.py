"""Readers for the Clarius-style raw acquisition files (``*_rf.raw`` / ``*_env.raw``) and their
YAML sidecars, plus the pre-extracted ``*_rf_no_tgc.npy`` arrays.

Geometry facts confirmed from the dataset (C3 curvilinear "hd3" probe):
  * RF  yml: samples/line=1040, lines=64,  bytes=2 (int16), frames=68, sampling 15 MHz
  * ENV yml: samples/line=480,  lines=192, bytes=1 (uint8), frames=68, type "B pre-scan"
  * filesize(raw) == frames*lines*samples*bytes + 564  (see config.RAW_TAIL_BYTES)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RawGeometry:
    """Geometry parsed from a ``*_rf.yml`` / ``*_env.yml`` sidecar."""

    samples_per_line: int
    number_of_lines: int
    sample_bytes: int
    frames: int
    kind: str  # "RF" or "B pre-scan"
    imaging_depth_mm: float | None = None
    sampling_rate_mhz: float | None = None
    transmit_freq_mhz: float | None = None
    delay_samples: int | None = None
    probe_radius_mm: float | None = None
    probe_pitch_um: float | None = None
    probe_elements: int | None = None

    @property
    def dtype(self) -> np.dtype:
        return np.dtype({1: np.uint8, 2: np.int16}[self.sample_bytes])

    @property
    def frame_len(self) -> int:
        return self.number_of_lines * self.samples_per_line


def _search_float(text: str, key: str) -> float | None:
    m = re.search(rf"{re.escape(key)}\s*:\s*([-+]?\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else None


def _search_int(text: str, key: str) -> int | None:
    v = _search_float(text, key)
    return int(v) if v is not None else None


def parse_rawdata_yml(path: str | Path) -> RawGeometry:
    """Parse the geometry we need with tolerant regexes (the files are YAML-ish but contain
    unit-suffixed scalars and degree symbols that trip strict parsers)."""
    text = Path(path).read_text(errors="ignore")

    samples = _search_int(text, "samples per line")
    lines = _search_int(text, "number of lines")
    sbytes = _search_int(text, "sample size")
    frames = _search_int(text, "frames")
    kind_m = re.search(r"^type:\s*(.+)$", text, flags=re.MULTILINE)
    kind = kind_m.group(1).strip() if kind_m else "unknown"

    if None in (samples, lines, sbytes, frames):
        raise ValueError(f"Could not parse core geometry from {path}")

    # probe block
    radius = _search_float(text, "radius")
    pitch = _search_float(text, "pitch")
    elements = _search_int(text, "elements")

    return RawGeometry(
        samples_per_line=samples,
        number_of_lines=lines,
        sample_bytes=sbytes,
        frames=frames,
        kind=kind,
        imaging_depth_mm=_search_float(text, "imaging depth"),
        sampling_rate_mhz=_search_float(text, "sampling rate"),
        transmit_freq_mhz=_search_float(text, "transmit frequency"),
        delay_samples=_search_int(text, "delay samples"),
        probe_radius_mm=radius,
        probe_pitch_um=pitch,
        probe_elements=elements,
    )


def read_raw(
    raw_path: str | Path,
    geom: RawGeometry,
    *,
    header_bytes: int = 0,
    order: str = "lines_first",
) -> np.ndarray:
    """Read a raw file into ``(frames, lines, samples)``.

    Parameters
    ----------
    header_bytes : bytes to skip at the start (use to test the 564-byte-header hypothesis).
    order : "lines_first" -> each frame stored line-by-line (default, matches "samples per line").
    """
    buf = np.fromfile(str(raw_path), dtype=np.uint8)
    if header_bytes:
        buf = buf[header_bytes:]
    arr = np.frombuffer(buf.tobytes(), dtype=geom.dtype)
    n_frames = min(geom.frames, arr.size // geom.frame_len)
    arr = arr[: n_frames * geom.frame_len]
    if order == "lines_first":
        arr = arr.reshape(n_frames, geom.number_of_lines, geom.samples_per_line)
    else:  # samples_first
        arr = arr.reshape(n_frames, geom.samples_per_line, geom.number_of_lines)
        arr = np.transpose(arr, (0, 2, 1))
    return arr


def load_npy_rf(npy_path: str | Path) -> np.ndarray:
    """Load a ``*_rf_no_tgc.npy`` array (memory-mapped). Observed shape (lines, samples, frames)."""
    return np.load(str(npy_path), mmap_mode="r")
