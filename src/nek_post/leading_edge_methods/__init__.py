"""Dispatch leading-edge extraction to a supported numerical method."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nek_post.leading_edge_methods.common import (
    DEFAULT_LEADING_EDGE_METHOD,
    SUPPORTED_LEADING_EDGE_METHODS,
    LeadingEdgeExtractionResult,
    PeriodicYContext,
    normalize_leading_edge_method,
    validate_periodic_y_coordinates,
)
from nek_post.leading_edge_methods.rightmost_crossing import (
    extract_rightmost_crossing,
)


def extract_leading_edge(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    concentration: NDArray[np.float64],
    *,
    threshold: float,
    method: object = DEFAULT_LEADING_EDGE_METHOD,
    periodic_y: bool,
    y_period: float,
) -> LeadingEdgeExtractionResult:
    """Extract one leading-edge curve using explicit periodic-y context."""
    canonical_method = normalize_leading_edge_method(method)
    context = PeriodicYContext(periodic_y=periodic_y, y_period=y_period)
    validated_y = validate_periodic_y_coordinates(y, context)
    if canonical_method == DEFAULT_LEADING_EDGE_METHOD:
        return extract_rightmost_crossing(
            x,
            validated_y,
            concentration,
            threshold=threshold,
        )
    raise AssertionError(f"Unhandled leading-edge method {canonical_method!r}.")


__all__ = (
    "DEFAULT_LEADING_EDGE_METHOD",
    "LeadingEdgeExtractionResult",
    "PeriodicYContext",
    "SUPPORTED_LEADING_EDGE_METHODS",
    "extract_leading_edge",
    "normalize_leading_edge_method",
)
