"""Error metrics used in the polynomial-order comparison workflow."""

from __future__ import annotations

import numpy as np


def relative_l2_error(values, reference) -> float:
    """Return the relative L2 error between `values` and `reference`."""
    values_arr = np.asarray(values, dtype=float)
    ref_arr = np.asarray(reference, dtype=float)
    denom = np.linalg.norm(ref_arr.ravel())
    if denom == 0.0:
        raise ValueError("reference norm is zero")
    return float(np.linalg.norm((values_arr - ref_arr).ravel()) / denom)


def linf_error(values, reference) -> float:
    """Return the L-infinity error between `values` and `reference`."""
    values_arr = np.asarray(values, dtype=float)
    ref_arr = np.asarray(reference, dtype=float)
    return float(np.max(np.abs(values_arr - ref_arr)))


def front_position(x, concentration, threshold):
    """Return the x-location where concentration first crosses a threshold."""
    raise NotImplementedError
