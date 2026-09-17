"""Pure time alignment and spanwise diagnostics for sampled leading edges.

All spanwise averages use equally weighted, endpoint-excluded periodic samples.
Failed crossings are masked, never filled or bridged in time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def validate_front_history(time, front, success):
    time = np.asarray(time, dtype=float)
    front = np.asarray(front, dtype=float)
    success = np.asarray(success)
    if time.ndim != 1 or not time.size or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("Physical times must be finite, nonempty and strictly increasing.")
    if front.ndim != 2 or front.shape[0] != time.size or not front.shape[1]:
        raise ValueError("Front arrays must have shape (time, y), with nonempty y.")
    if success.dtype != np.bool_ or success.shape != front.shape:
        raise ValueError("Success masks must be boolean and match the front arrays.")
    if np.isinf(front).any() or np.any(success & ~np.isfinite(front)):
        raise ValueError("Successful crossings must be finite; infinities are invalid.")
    return time, front, success & np.isfinite(front)


@dataclass(frozen=True)
class AlignedFront:
    time: np.ndarray
    front: np.ndarray
    valid: np.ndarray
    left_position: np.ndarray
    right_position: np.ndarray
    right_weight: np.ndarray


def align_fronts(time, front, success, targets) -> AlignedFront:
    """Piecewise linear time interpolation, requiring both adjacent crossings.

    Exact stored times use only that frame's mask. No temporal extrapolation,
    tolerance snapping, smoothing, or interpolation across a missing frame.
    """
    time, front, valid = validate_front_history(time, front, success)
    targets = np.asarray(targets, dtype=float)
    if (targets.ndim != 1 or not targets.size or not np.isfinite(targets).all()
            or np.any(np.diff(targets) <= 0)):
        raise ValueError("Target times must be finite, nonempty and strictly increasing.")
    if targets[0] < time[0] or targets[-1] > time[-1]:
        raise ValueError("Temporal extrapolation is not allowed.")
    right = np.searchsorted(time, targets, side="left")
    exact = time[right] == targets
    left = np.where(exact, right, right - 1)
    weight = np.zeros(targets.size)
    between = ~exact
    weight[between] = ((targets[between] - time[left[between]]) /
                       (time[right[between]] - time[left[between]]))
    mask = valid[left] & valid[right]
    values = np.full((targets.size, front.shape[1]), np.nan)
    # Copy exact samples directly: 0 * NaN in a neighbor must not invalidate them.
    values[exact] = np.where(mask[exact], front[right[exact]], np.nan)
    values[between] = np.where(mask[between],
        (1 - weight[between, None]) * front[left[between]]
        + weight[between, None] * front[right[between]], np.nan)
    return AlignedFront(targets.copy(), values, mask, left, right, weight)


def common_reference_timeline(times_by_case, reference_case):
    """Reference's stored times restricted to the all-case range intersection."""
    if reference_case not in times_by_case or len(times_by_case) < 2:
        raise ValueError("A reference and at least two case time arrays are required.")
    times = {}
    for case, values in times_by_case.items():
        array = np.asarray(values, dtype=float)
        if (array.ndim != 1 or not array.size or not np.isfinite(array).all()
                or np.any(np.diff(array) <= 0)):
            raise ValueError(f"{case}: times must be finite and strictly increasing.")
        times[case] = array
    lower = max(t[0] for t in times.values())
    upper = min(t[-1] for t in times.values())
    ref = times[reference_case]
    selected = ref[(ref >= lower) & (ref <= upper)]
    if selected.size < 2:
        raise ValueError("Fewer than two reference times lie in the shared time range.")
    return selected.copy(), (float(lower), float(upper))


def front_statistics(front, valid):
    front = np.asarray(front, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if front.ndim != 1 or front.shape != valid.shape or not front.size:
        raise ValueError("Front and mask must be matching nonempty y vectors.")
    mask = valid & np.isfinite(front)
    values = front[mask]
    result = {"successful_y_count": int(mask.sum()),
              "failed_y_count": int(mask.size - mask.sum()),
              "valid_fraction": float(mask.mean())}
    result.update({name: float(function(values)) if values.size else float("nan")
                   for name, function in (("mean_front", np.mean), ("std_front", np.std),
                                          ("min_front", np.min), ("max_front", np.max),
                                          ("peak_to_peak", np.ptp))})
    return result


def front_difference(front, reference, common_valid):
    """Total, signed bulk, and mean-removed differences on identical support.

    Both means are computed on the supplied common mask. With this convention,
    total_rms**2 = bulk_difference**2 + shape_rms**2 (up to roundoff).
    Empty support produces NaN metrics rather than a fabricated zero.
    """
    front, reference = np.asarray(front, float), np.asarray(reference, float)
    valid = np.asarray(common_valid, bool)
    if front.ndim != 1 or front.shape != reference.shape or front.shape != valid.shape:
        raise ValueError("Difference inputs must be matching y vectors.")
    if np.any(valid & (~np.isfinite(front) | ~np.isfinite(reference))):
        raise ValueError("The common metric mask includes nonfinite crossings.")
    a, b = front[valid], reference[valid]
    result = {"valid_y_count": int(valid.sum()), "valid_fraction": float(valid.mean())}
    names = ("total_mae", "total_rms", "total_max_abs", "bulk_difference",
             "shape_mae", "shape_rms", "shape_max_abs")
    if not a.size:
        return dict(result, **dict.fromkeys(names, float("nan")))
    difference = a - b
    bulk = float(a.mean() - b.mean())
    shape = (a - a.mean()) - (b - b.mean())
    result.update(zip(names, (float(np.mean(abs(difference))),
        float(np.sqrt(np.mean(difference**2))), float(np.max(abs(difference))), bulk,
        float(np.mean(abs(shape))), float(np.sqrt(np.mean(shape**2))), float(np.max(abs(shape))))))
    return result


def compare_extraction_methods(first, first_success, second, second_success):
    """Compare two raw same-time curves; report every value/mask difference."""
    a, b = np.asarray(first, float), np.asarray(second, float)
    ma, mb = np.asarray(first_success, bool), np.asarray(second_success, bool)
    if a.ndim != 1 or a.shape != b.shape or ma.shape != a.shape or mb.shape != a.shape:
        raise ValueError("Method-comparison inputs must be matching y vectors.")
    valid = ma & mb & np.isfinite(a) & np.isfinite(b)
    same_values = (a == b) | (np.isnan(a) & np.isnan(b))
    differing = np.flatnonzero(~same_values | (ma != mb))
    difference = a[valid] - b[valid]
    return {"common_valid_y_count": int(valid.sum()),
            "common_valid_fraction": float(valid.mean()),
            "rms_difference": float(np.sqrt(np.mean(difference**2))) if difference.size else float("nan"),
            "max_abs_difference": float(np.max(abs(difference))) if difference.size else float("nan"),
            "exactly_identical": bool(not differing.size),
            "different_y_count": int(differing.size),
            "mask_mismatch_count": int(np.count_nonzero(ma != mb)),
            "different_y_positions": differing}
