"""Follow a patch of tissue through an RF cine (frame-to-frame phase correlation).

The GUI box and the spline are often drawn on different frames of the same recording. Between those
frames the liver moves with breathing, so a lesion can sit beside "its" box although both annotations
are right. :func:`track_box` estimates how far the tissue inside a box moved between two frames, so the
box can be carried to the spline's frame before the two are compared.

This is a rigid-shift estimate meant for triage, not a registration: check a case it "resolves" by eye.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .agreement import Box


def _window(box: Box, shape: tuple[int, int], margin: float, min_lines: int, min_samples: int):
    """Integer window ``(l0, l1, s0, s1)`` (half-open) around ``box``, padded and kept on the grid."""
    n_lines, n_samples = shape
    cl, cs = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hl = max((box[2] - box[0]) * (0.5 + margin), min_lines / 2)
    hs = max((box[3] - box[1]) * (0.5 + margin), min_samples / 2)
    hl, hs = min(hl, n_lines / 2), min(hs, n_samples / 2)
    l0 = int(round(min(max(cl - hl, 0), n_lines - 2 * hl)))
    s0 = int(round(min(max(cs - hs, 0), n_samples - 2 * hs)))
    return l0, l0 + int(round(2 * hl)), s0, s0 + int(round(2 * hs))


def track_box(
    frame_at: Callable[[int], np.ndarray],
    box: Box,
    f_from: int,
    f_to: int,
    *,
    margin: float = 0.5,
    min_lines: int = 16,
    min_samples: int = 64,
) -> tuple[float, float, float]:
    """Displacement ``(d_line, d_sample)`` of the tissue in ``box`` from frame ``f_from`` to ``f_to``.

    ``frame_at(k)`` returns pre-scan frame ``k`` as a ``(lines, samples)`` array (B-mode works well).
    ``box`` is ``(line0, sample0, line1, sample1)``. The window follows the tissue from frame to frame.
    Also returns the mean phase-correlation peak (0..1) as a rough confidence; it drops when the speckle
    decorrelates, e.g. when the lesion leaves the imaging plane.
    """
    import cv2

    f_from, f_to = int(f_from), int(f_to)
    if f_from == f_to:
        return 0.0, 0.0, 1.0
    step = 1 if f_to > f_from else -1
    d_line = d_sample = 0.0
    responses = []
    prev = np.asarray(frame_at(f_from), dtype=np.float32)
    for k in range(f_from, f_to, step):
        nxt = np.asarray(frame_at(k + step), dtype=np.float32)
        moved = (box[0] + d_line, box[1] + d_sample, box[2] + d_line, box[3] + d_sample)
        l0, l1, s0, s1 = _window(moved, prev.shape, margin, min_lines, min_samples)
        a, b = prev[l0:l1, s0:s1], nxt[l0:l1, s0:s1]
        if a.shape[0] < 4 or a.shape[1] < 4:
            break
        a = np.ascontiguousarray(a - a.mean())
        b = np.ascontiguousarray(b - b.mean())
        hann = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(a, b, hann)  # x = samples (columns), y = lines (rows)
        d_sample += float(dx)
        d_line += float(dy)
        responses.append(float(response))
        prev = nxt
    conf = float(np.mean(responses)) if responses else 0.0
    return d_line, d_sample, conf
