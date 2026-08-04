"""Pure numerical extraction of spanwise gravity-current leading edges."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class LeadingEdgeCurve:
    """Rightmost concentration-threshold intersection for each supplied y row."""

    y: NDArray[np.float64]
    x_front: NDArray[np.float64]
    success_mask: NDArray[np.bool_]
    crossing_count: NDArray[np.int64]
    threshold: float


def _readonly_copy(
    array: object,
    dtype: np.dtype | type,
) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _coordinate_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional finite array.") from exc
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional finite array.")
    if result.size == 0:
        raise ValueError(f"{name} must contain at least one coordinate.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite coordinates.")
    if np.any(np.diff(result) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing and unique.")
    return result


def _threshold_value(threshold: float) -> float:
    try:
        result = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be finite.") from exc
    if not np.isfinite(result):
        raise ValueError("threshold must be finite.")
    return result


def _concentration_array(
    concentration: ArrayLike,
    expected_shape: tuple[int, int],
) -> NDArray[np.float64]:
    if np.iscomplexobj(concentration):
        raise ValueError("concentration must be a two-dimensional real numeric array.")
    try:
        result = np.asarray(concentration, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "concentration must be a two-dimensional real numeric array."
        ) from exc
    if result.ndim != 2:
        raise ValueError("concentration must be a two-dimensional numeric array.")
    if result.shape != expected_shape:
        raise ValueError(
            f"concentration shape mismatch: expected {expected_shape}, "
            f"got {result.shape}."
        )
    return result


def _row_intersections(
    x: NDArray[np.float64],
    concentration: NDArray[np.float64],
    threshold: float,
) -> list[float]:
    finite = np.isfinite(concentration)
    shifted = concentration - threshold
    exact = finite & (shifted == 0.0)
    intersections: list[float] = []

    # Every maximal run of exact-threshold samples is one intersection. Its
    # rightmost coordinate represents that plateau for leading-edge selection.
    index = 0
    while index < x.size:
        if not exact[index]:
            index += 1
            continue
        plateau_end = index
        while plateau_end + 1 < x.size and exact[plateau_end + 1]:
            plateau_end += 1
        intersections.append(float(x[plateau_end]))
        index = plateau_end + 1

    # Exact endpoints are deliberately excluded here, so a plateau cannot also
    # generate a duplicate adjacent-pair crossing.
    for index in range(x.size - 1):
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
        left_concentration = float(concentration[index])
        right_concentration = float(concentration[index + 1])
        crossing = float(
            x[index]
            + (threshold - left_concentration)
            * (x[index + 1] - x[index])
            / (right_concentration - left_concentration)
        )
        intersections.append(crossing)

    return intersections


def extract_spanwise_leading_edge(
    x: ArrayLike,
    y: ArrayLike,
    concentration: ArrayLike,
    *,
    threshold: float = 0.1,
) -> LeadingEdgeCurve:
    """Return the rightmost threshold intersection independently in each y row.

    ``concentration`` must have shape ``(len(y), len(x))``. Only adjacent finite
    samples within a row participate. Strict sign changes are located by linear
    interpolation in the supplied physical x coordinates. A contiguous run of
    exact-threshold samples counts once and is represented by its rightmost x.
    No interpolation or endpoint duplication is performed in periodic y.
    """
    x_array = _coordinate_array(x, "x")
    y_array = _coordinate_array(y, "y")
    threshold_value = _threshold_value(threshold)
    concentration_array = _concentration_array(
        concentration,
        (y_array.size, x_array.size),
    )

    x_front = np.full(y_array.size, np.nan, dtype=np.float64)
    crossing_count = np.zeros(y_array.size, dtype=np.int64)
    for row_index, row in enumerate(concentration_array):
        intersections = _row_intersections(x_array, row, threshold_value)
        crossing_count[row_index] = len(intersections)
        if intersections:
            x_front[row_index] = max(intersections)
    success_mask = crossing_count > 0

    return LeadingEdgeCurve(
        y=_readonly_copy(y_array, np.float64),
        x_front=_readonly_copy(x_front, np.float64),
        success_mask=_readonly_copy(success_mask, np.bool_),
        crossing_count=_readonly_copy(crossing_count, np.int64),
        threshold=threshold_value,
    )


__all__ = ("LeadingEdgeCurve", "extract_spanwise_leading_edge")
