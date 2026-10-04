"""Readers for the Clarius-style raw acquisition files (``*_rf.raw`` / ``*_env.raw``) and their
YAML sidecars, plus the pre-extracted ``*_rf_no_tgc.npy`` arrays.

Geometry varies per preset (e.g. CEUS014 C3_large: RF 192 lines x 1920 samples int16, 42 frames;
ENV 192 x 480 uint8). Read it from the sidecar, never hard-code it.

Clarius ``*.raw`` layout (explains the old "+564 bytes" observation: 20 + 68*8 = 564, 20 + 42*8 = 356):
  * header: 5 x int32 = (id, frames, lines, samples, sample_size)          -> 20 bytes
  * then per frame: int64 timestamp, followed by lines*samples*sample_size data bytes
Reading the file as one flat block (ignoring the header and per-frame timestamps) shifts frame k by
(20 + 8k) bytes, i.e. a growing depth offset — :func:`read_raw` parses the layout properly.
"""

from __future__ import annotations

import re
import warnings
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


CLARIUS_HEADER_BYTES = 20  # 5 x int32
CLARIUS_TIMESTAMP_BYTES = 8  # int64 per frame


def read_clarius_raw(raw_path: str | Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """Parse a Clarius ``*.raw`` file -> ``(header, timestamps (frames,), data (frames, lines, samples))``.

    Raises ``ValueError`` if the file size does not match the Clarius layout.
    """
    buf = np.fromfile(str(raw_path), dtype=np.uint8)
    if buf.size < CLARIUS_HEADER_BYTES:
        raise ValueError(f"{raw_path}: too small for a Clarius header")
    hid, frames, lines, samples, ssize = (int(v) for v in buf[:CLARIUS_HEADER_BYTES].view("<i4"))
    if ssize not in (1, 2) or min(frames, lines, samples) <= 0:
        raise ValueError(f"{raw_path}: implausible Clarius header {(hid, frames, lines, samples, ssize)}")
    frame_bytes = lines * samples * ssize
    expected = CLARIUS_HEADER_BYTES + frames * (CLARIUS_TIMESTAMP_BYTES + frame_bytes)
    if buf.size != expected:
        raise ValueError(f"{raw_path}: size {buf.size} != Clarius layout {expected}")
    body = buf[CLARIUS_HEADER_BYTES:].reshape(frames, CLARIUS_TIMESTAMP_BYTES + frame_bytes)
    timestamps = body[:, :CLARIUS_TIMESTAMP_BYTES].copy().view("<i8").ravel()
    dtype = np.dtype("<i2") if ssize == 2 else np.dtype(np.uint8)
    data = body[:, CLARIUS_TIMESTAMP_BYTES:].copy().view(dtype).reshape(frames, lines, samples)
    header = {"id": hid, "frames": frames, "lines": lines, "samples": samples, "sample_size": ssize}
    return header, timestamps, data


def read_raw(
    raw_path: str | Path,
    geom: RawGeometry,
    *,
    header_bytes: int | None = None,
    order: str = "lines_first",
) -> np.ndarray:
    """Read a raw file into ``(frames, lines, samples)``.

    Parameters
    ----------
    header_bytes : ``None`` (default) parses the Clarius layout (header + per-frame timestamps), falling
        back to a flat read if the file does not match it. An int forces the legacy flat read after
        skipping that many bytes (kept only to reproduce the old exploration plots).
    order : "lines_first" -> each frame stored line-by-line (default, matches "samples per line").
    """
    if header_bytes is None and order == "lines_first":
        try:
            _, _, data = read_clarius_raw(raw_path)
            if data.shape[1:] == (geom.number_of_lines, geom.samples_per_line):
                return data
            reason = f"header shape {data.shape[1:]} != yml geometry"
        except ValueError as e:
            reason = str(e)
        warnings.warn(f"{raw_path}: not parsed as Clarius layout ({reason}); flat read may be offset",
                      stacklevel=2)
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
    """Load a ``*_rf_no_tgc.npy`` array (memory-mapped). Observed shape (lines, samples, frames).

    These come from the QuantUS Clarius parser: RF lines zero-padded at the END to 2928 samples
    (C3/L15), so only the first ``samples_per_line`` samples carry signal.
    """
    return np.load(str(npy_path), mmap_mode="r")
