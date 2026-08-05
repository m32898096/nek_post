"""Shared types and validation for leading-edge extraction methods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


DEFAULT_LEADING_EDGE_METHOD = "rightmost-crossing"
SUPPORTED_LEADING_EDGE_METHODS = (
    DEFAULT_LEADING_EDGE_METHOD,
    "moore-boundary",
)


def normalize_leading_edge_method(value: object) -> str:
    """Return a supported canonical extraction-method name."""
    if not isinstance(value, str):
        raise ValueError("extraction_method must be a supported method name.")
    method = value.strip().lower()
    if method not in SUPPORTED_LEADING_EDGE_METHODS:
        supported = ", ".join(SUPPORTED_LEADING_EDGE_METHODS)
        raise ValueError(
            f"Unsupported leading-edge extraction method {value!r}; "
            f"choose one of: {supported}."
        )
    return method


@dataclass(frozen=True)
class PeriodicYContext:
    """Explicit physical context for an endpoint-excluded periodic y grid."""

    periodic_y: bool
    y_period: float

    def __post_init__(self) -> None:
        if not isinstance(self.periodic_y, (bool, np.bool_)):
            raise ValueError("periodic_y must be a boolean.")
        if not bool(self.periodic_y):
            raise ValueError("periodic_y must be True for leading-edge extraction.")
        if isinstance(self.y_period, (bool, np.bool_)):
            raise ValueError("y_period must be finite and positive.")
        try:
            period = float(self.y_period)
        except (TypeError, ValueError) as exc:
            raise ValueError("y_period must be finite and positive.") from exc
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("y_period must be finite and positive.")
        object.__setattr__(self, "periodic_y", True)
        object.__setattr__(self, "y_period", period)


def validate_periodic_y_coordinates(
    y: object,
    context: PeriodicYContext,
) -> NDArray[np.float64]:
    """Validate that *y* is increasing and excludes its periodic endpoint."""
    values = np.asarray(y, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.all(np.isfinite(values))
        or np.any(np.diff(values) <= 0.0)
    ):
        raise ValueError("y must be a nonempty, finite, strictly increasing vector.")
    endpoint = float(values[0] + context.y_period)
    scale = max(1.0, abs(float(values[0])), abs(endpoint), context.y_period)
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    if values[-1] >= endpoint - tolerance:
        raise ValueError(
            "The computational y vector must exclude the periodic upper endpoint."
        )
    return values


def _readonly_copy(values: object, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class LeadingEdgeExtractionResult:
    """Immutable method-independent leading-edge extraction result."""

    method: str
    y: NDArray[np.float64]
    x_front: NDArray[np.float64]
    success_mask: NDArray[np.bool_]
    crossing_count: NDArray[np.int64]
    threshold: float

    def __post_init__(self) -> None:
        method = normalize_leading_edge_method(self.method)
        y = _readonly_copy(self.y, np.float64)
        x_front = _readonly_copy(self.x_front, np.float64)
        success_mask = _readonly_copy(self.success_mask, np.bool_)
        crossing_count = _readonly_copy(self.crossing_count, np.int64)
        if y.ndim != 1 or any(
            values.shape != y.shape
            for values in (x_front, success_mask, crossing_count)
        ):
            raise ValueError("Leading-edge result arrays must be matching vectors.")
        threshold = float(self.threshold)
        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite.")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "x_front", x_front)
        object.__setattr__(self, "success_mask", success_mask)
        object.__setattr__(self, "crossing_count", crossing_count)
        object.__setattr__(self, "threshold", threshold)


__all__ = (
    "DEFAULT_LEADING_EDGE_METHOD",
    "LeadingEdgeExtractionResult",
    "PeriodicYContext",
    "SUPPORTED_LEADING_EDGE_METHODS",
    "normalize_leading_edge_method",
    "validate_periodic_y_coordinates",
)
