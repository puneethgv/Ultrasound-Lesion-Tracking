"""Classical B-mode reconstruction: (Hilbert) envelope -> log-compression -> normalized image.

Scan-conversion of the curvilinear geometry is a separate step (added when the label<->image
registration is fixed); these functions produce the *pre-scan* (rectangular, depth x line) B-mode
that the ROI annotations appear to live in.
"""

from __future__ import annotations

import numpy as np


def envelope_to_bmode(env: np.ndarray, dynamic_range_db: float = 60.0) -> np.ndarray:
    """Log-compress an amplitude/envelope map to a normalized [0, 1] B-mode image.

    ``env`` is a 2D array (any orientation); returns float32 in [0, 1].
    """
    x = np.asarray(env, dtype=np.float32)
    x = np.abs(x)
    peak = float(x.max()) or 1.0
    x = x / peak
    log = 20.0 * np.log10(x + 1e-4)
    log = np.clip(log, -dynamic_range_db, 0.0)
    return ((log + dynamic_range_db) / dynamic_range_db).astype(np.float32)


def rf_to_bmode(rf: np.ndarray, axis: int = -1, dynamic_range_db: float = 60.0) -> np.ndarray:
    """RF (2D) -> B-mode via analytic-signal envelope along the axial ``axis`` then log-compress."""
    from scipy.signal import hilbert

    rf = np.asarray(rf, dtype=np.float32)
    env = np.abs(hilbert(rf, axis=axis))
    return envelope_to_bmode(env, dynamic_range_db=dynamic_range_db)
