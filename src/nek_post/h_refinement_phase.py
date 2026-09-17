"""Saved-slice phase/morphology diagnostics, separate from convergence metrics.

No arrays are registered, smoothed, or substituted into the reported errors.
Integer x-shifts use a fixed interior support, no wrapping or extrapolation.
Distribution distance deliberately discards spatial arrangement; neither test
establishes physical equivalence or proves chaotic divergence.
"""
from __future__ import annotations

import numpy as np


def morphology_diagnostics(values, reference, mask) -> dict:
    """Compare spatial RMS with distribution-only RMS on the same finite mask."""
    a, b = np.asarray(values, float), np.asarray(reference, float)
    if a.shape != b.shape or np.shape(mask) != a.shape:
        raise ValueError("Fields and mask must have the same shape.")
    valid = np.asarray(mask, bool) & np.isfinite(a) & np.isfinite(b)
    if not valid.any():
        raise ValueError("No finite points for morphology diagnostics.")
    x, y = a[valid], b[valid]
    rms = float(np.sqrt(np.mean((x-y)**2)))
    distribution = float(np.sqrt(np.mean((np.sort(x)-np.sort(y))**2)))
    centered_x, centered_y = x-x.mean(), y-y.mean()
    denom = float(np.linalg.norm(centered_x)*np.linalg.norm(centered_y))
    return {
        "valid_point_count": int(valid.sum()), "spatial_rms": rms,
        "distribution_rms": distribution,
        "distribution_to_spatial_rms": distribution/rms if rms > 0 else None,
        "spatial_correlation": float(np.dot(centered_x, centered_y)/denom) if denom > 0 else None,
        "mean_difference": float(x.mean()-y.mean()),
        "standard_deviation_difference": float(x.std()-y.std()),
    }


def translation_diagnostics(values, reference, mask, x_coordinates, *, max_shift=.5) -> list[dict]:
    """Find bounded discrete x-shifts separately on x<0 and x>0 halves.

    At candidate lag k, compare values[:,j+k] with reference[:,j]. Positive
    reported shifts mean sampling values at larger x. This is not a velocity
    estimate or front reconstruction. All lags use identical interior support.
    """
    a, b = np.asarray(values, float), np.asarray(reference, float)
    x = np.asarray(x_coordinates, float)
    if a.ndim != 2 or a.shape != b.shape or np.shape(mask) != a.shape or x.shape != (a.shape[1],):
        raise ValueError("Matching 2D fields/mask and 1D x coordinates required.")
    if not np.isfinite(max_shift) or max_shift < 0 or not np.all(np.isfinite(x)) or x.size < 2:
        raise ValueError("Finite coordinates and nonnegative shift bound required.")
    dx = float(x[1]-x[0])
    if dx <= 0 or not np.allclose(np.diff(x), dx, rtol=1e-9, atol=1e-12):
        raise ValueError("Translation diagnostics require increasing uniform x coordinates.")
    max_lag = int(np.floor(max_shift/dx))
    rows = []
    for label, columns in (("x_negative", np.flatnonzero(x < 0)), ("x_positive", np.flatnonzero(x > 0))):
        if len(columns) <= 2*max_lag:
            raise ValueError("Insufficient half-domain width for requested fixed-support shift search.")
        targets = columns[max_lag:len(columns)-max_lag] if max_lag else columns
        support = np.asarray(mask, bool)[:, targets] & np.isfinite(b[:, targets])
        for lag in range(-max_lag, max_lag+1):
            support &= np.asarray(mask, bool)[:, targets+lag] & np.isfinite(a[:, targets+lag])
        if not support.any():
            rows.append({"half": label, "valid_point_count": 0, "raw_rms": None,
                         "best_rms": None, "best_shift_x": None, "rms_reduction_fraction": None,
                         "search_boundary_hit": None, "maximum_tested_shift": max_lag*dx})
            continue
        ref = b[:,targets][support]
        candidates = []
        for lag in range(-max_lag, max_lag+1):
            rms = float(np.sqrt(np.mean((a[:,targets+lag][support]-ref)**2)))
            candidates.append((rms, abs(lag), lag))
        best, _, best_lag = min(candidates)
        raw = next(rms for rms, _, lag in candidates if lag == 0)
        rows.append({"half": label, "valid_point_count": int(support.sum()), "raw_rms": raw,
                     "best_rms": best, "best_shift_x": best_lag*dx,
                     "rms_reduction_fraction": 1-best/raw if raw > 0 else None,
                     "search_boundary_hit": bool(max_lag and abs(best_lag)==max_lag),
                     "maximum_tested_shift": max_lag*dx})
    return rows
