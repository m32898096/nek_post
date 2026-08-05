"""Adapter for the validated legacy rightmost-crossing extractor."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nek_post.leading_edge_extraction import extract_spanwise_leading_edge
from nek_post.leading_edge_methods.common import (
    DEFAULT_LEADING_EDGE_METHOD,
    LeadingEdgeExtractionResult,
)


def extract_rightmost_crossing(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    concentration: NDArray[np.float64],
    *,
    threshold: float,
) -> LeadingEdgeExtractionResult:
    """Call the legacy extractor and adapt its result without changing math."""
    legacy = extract_spanwise_leading_edge(
        x,
        y,
        concentration,
        threshold=threshold,
    )
    return LeadingEdgeExtractionResult(
        method=DEFAULT_LEADING_EDGE_METHOD,
        y=legacy.y,
        x_front=legacy.x_front,
        success_mask=legacy.success_mask,
        crossing_count=legacy.crossing_count,
        threshold=legacy.threshold,
    )


__all__ = ("extract_rightmost_crossing",)
