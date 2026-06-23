"""Error metrics used in the polynomial-order comparison workflow."""

from __future__ import annotations

import numpy as np


def _masked_arrays(values, reference, mask=None) -> tuple[np.ndarray, np.ndarray]:
    values_arr = np.asarray(values, dtype=float)
    ref_arr = np.asarray(reference, dtype=float)
    if mask is None:
        mask_arr = np.isfinite(values_arr) & np.isfinite(ref_arr)
    else:
        mask_arr = np.asarray(mask, dtype=bool) & np.isfinite(values_arr) & np.isfinite(ref_arr)

    if not np.any(mask_arr):
        raise ValueError("No valid points are available for metric calculation.")

    return values_arr[mask_arr], ref_arr[mask_arr]


def relative_l2_error(values, reference, mask=None) -> float:
    """Return the relative L2 error between `values` and `reference`."""
    values_arr, ref_arr = _masked_arrays(values, reference, mask)
    denom = np.sum(ref_arr**2)
    if denom == 0.0:
        raise ValueError("Reference L2 denominator is zero.")
    return float(np.sqrt(np.sum((values_arr - ref_arr) ** 2) / denom))


def absolute_linf_error(values, reference, mask=None) -> float:
    """Return the L-infinity error between `values` and `reference`."""
    values_arr, ref_arr = _masked_arrays(values, reference, mask)
    return float(np.max(np.abs(values_arr - ref_arr)))


def linf_error(values, reference) -> float:
    """Return the L-infinity error between `values` and `reference`."""
    return absolute_linf_error(values, reference)


def relative_linf_error(values, reference, mask=None) -> float:
    """Return the relative L-infinity error between `values` and `reference`."""
    values_arr, ref_arr = _masked_arrays(values, reference, mask)
    denom = np.max(np.abs(ref_arr))
    if denom == 0.0:
        raise ValueError("Reference L-infinity denominator is zero.")
    return float(np.max(np.abs(values_arr - ref_arr)) / denom)


def front_position(Xi, C_grid, threshold, mask=None) -> float:
    """Return the maximum x location where concentration exceeds `threshold`."""
    x_arr = np.asarray(Xi, dtype=float)
    c_arr = np.asarray(C_grid, dtype=float)
    if mask is None:
        mask_arr = np.isfinite(x_arr) & np.isfinite(c_arr)
    else:
        mask_arr = np.asarray(mask, dtype=bool) & np.isfinite(x_arr) & np.isfinite(c_arr)

    front_mask = mask_arr & (c_arr > threshold)
    if not np.any(front_mask):
        return float("nan")
    return float(np.max(x_arr[front_mask]))
