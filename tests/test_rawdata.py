"""Clarius raw-file parsing (dataset-free, synthetic file)."""

import numpy as np

from usloc.io import RawGeometry, read_clarius_raw, read_raw


def _write_clarius(path, data: np.ndarray, timestamps: np.ndarray) -> None:
    frames, lines, samples = data.shape
    with open(path, "wb") as f:
        np.array([2, frames, lines, samples, data.itemsize], dtype="<i4").tofile(f)
        for k in range(frames):
            np.array([timestamps[k]], dtype="<i8").tofile(f)
            data[k].astype(data.dtype.newbyteorder("<")).tofile(f)


def _geom(data):
    frames, lines, samples = data.shape
    return RawGeometry(samples_per_line=samples, number_of_lines=lines, sample_bytes=data.itemsize,
                       frames=frames, kind="RF")


def test_overhead_matches_dataset_observation(tmp_path):
    # the dataset showed +564 bytes for 68 frames and +356 for 42: 20-byte header + 8 bytes/frame
    for frames, overhead in ((68, 564), (42, 356)):
        data = np.zeros((frames, 2, 3), dtype=np.uint8)
        p = tmp_path / f"f{frames}.raw"
        _write_clarius(p, data, np.arange(frames))
        assert p.stat().st_size - data.nbytes == overhead


def test_read_clarius_raw_exact(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.integers(-2000, 2000, size=(5, 4, 16), dtype=np.int16)
    ts = np.arange(5, dtype=np.int64) * 71_000_000
    p = tmp_path / "x_rf.raw"
    _write_clarius(p, data, ts)
    hdr, t, d = read_clarius_raw(p)
    assert hdr["frames"] == 5 and hdr["lines"] == 4 and hdr["samples"] == 16
    assert np.array_equal(t, ts) and np.array_equal(d, data)
    assert np.array_equal(read_raw(p, _geom(data)), data)  # default path parses the layout


def test_legacy_flat_read_is_offset(tmp_path):
    data = np.arange(3 * 2 * 8, dtype=np.int16).reshape(3, 2, 8)
    p = tmp_path / "y_rf.raw"
    _write_clarius(p, data, np.zeros(3, dtype=np.int64))
    flat = read_raw(p, _geom(data), header_bytes=0)
    assert not np.array_equal(flat, data)
