"""Shared row-wise concentration-threshold intersection utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdIntersection:
    """One row intersection and the adjacent samples that support it.

    ``left_ix`` and ``right_ix`` identify adjacent physical-x samples when
    such a pair supports the intersection. For an exact-threshold plateau,
    ``x`` is always the plateau's rightmost coordinate, while the supporting
    pair is taken at the plateau's left edge when available. This preserves
    the validated plateau location and exposes an upstream heavy-side sample
    for downstream heavy-to-light filtering.
    """

    x: float
    left_ix: int | None
    right_ix: int | None


def row_threshold_intersections(
    x: object,
    concentration: object,
    threshold: object,
) -> tuple[ThresholdIntersection, ...]:
    """Return all validated rightmost-compatible intersections in one row.

    Strict sign changes use linear interpolation between adjacent finite
    samples. Every maximal exact-threshold plateau contributes once at its
    rightmost physical x. Non-finite samples neither form nor bridge strict
    crossings.
    """
    try:
        x_values = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("x must be a finite, strictly increasing vector.") from exc
    try:
        if np.iscomplexobj(concentration):
            raise ValueError("concentration must be a real numeric vector.")
        row = np.asarray(concentration, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("concentration must be a real numeric vector.") from exc
    if (
        x_values.ndim != 1
        or x_values.size == 0
        or not np.all(np.isfinite(x_values))
        or np.any(np.diff(x_values) <= 0.0)
    ):
        raise ValueError("x must be a finite, strictly increasing vector.")
    if row.ndim != 1 or row.shape != x_values.shape:
        raise ValueError("concentration must be a vector matching x.")
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be finite.") from exc
    if not np.isfinite(threshold_value):
        raise ValueError("threshold must be finite.")

    finite = np.isfinite(row)
    shifted = row - threshold_value
    exact = finite & (shifted == 0.0)
    intersections: list[ThresholdIntersection] = []

    # Every maximal exact run is represented once at its rightmost x. Prefer
    # the pair entering the plateau from the left: that pair exposes the
    # heavy-side support needed for a downstream heavy-to-light intersection.
    index = 0
    while index < x_values.size:
        if not exact[index]:
            index += 1
            continue
        plateau_start = index
        plateau_end = index
        while plateau_end + 1 < x_values.size and exact[plateau_end + 1]:
            plateau_end += 1
        if plateau_start > 0 and finite[plateau_start - 1]:
            left_ix = plateau_start - 1
            right_ix = plateau_start
        elif plateau_end + 1 < x_values.size and finite[plateau_end + 1]:
            left_ix = plateau_end
            right_ix = plateau_end + 1
        else:
            left_ix = None
            right_ix = None
        intersections.append(
            ThresholdIntersection(
                x=float(x_values[plateau_end]),
                left_ix=left_ix,
                right_ix=right_ix,
            )
        )
        index = plateau_end + 1

    # Exact endpoints are excluded, so plateaus cannot also create duplicate
    # adjacent-pair intersections.
    for index in range(x_values.size - 1):
        if not finite[index] or not finite[index + 1]:
            continue
        left_shifted = float(shifted[index])
        right_shifted = float(shifted[index + 1])
        strict_change = (
            left_shifted < 0.0 < right_shifted
            or right_shifted < 0.0 < left_shifted
        )
        if not strict_change:
            continue
        left_concentration = float(row[index])
        right_concentration = float(row[index + 1])
        crossing = float(
            x_values[index]
            + (threshold_value - left_concentration)
            * (x_values[index + 1] - x_values[index])
            / (right_concentration - left_concentration)
        )
        intersections.append(
            ThresholdIntersection(
                x=crossing,
                left_ix=index,
                right_ix=index + 1,
            )
        )

    return tuple(intersections)


__all__ = ("ThresholdIntersection", "row_threshold_intersections")
