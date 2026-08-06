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
from nek_post.leading_edge_methods.moore_boundary import (
    MOORE_BOUNDARY_METHOD,
    MooreBoundaryComponent,
    MooreBoundaryTrace,
    MooreTraceCandidateDiagnostic,
    MooreTraceSelection,
    extract_moore_boundary,
    label_periodic_boundary_components,
    moore_trace_winding_number,
    primary_moore_component_mask,
    select_primary_moore_component,
    select_periodic_moore_trace,
    trace_moore_boundary,
)


EXTRACTION_X_CONDITION = "strict-greater-than"


def normalize_extraction_x_min(value: object) -> float | None:
    """Return a finite extraction lower bound, or ``None`` for unrestricted."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("x_min must be finite or None.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("x_min must be finite or None.") from exc
    if not np.isfinite(result):
        raise ValueError("x_min must be finite or None.")
    return result


def restrict_leading_edge_x_domain(
    x: object,
    concentration: object,
    *,
    x_min: float | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return immutable extraction arrays restricted to strict ``x > x_min``.

    ``None`` preserves the complete physical x domain.  A finite bound is
    applied before any method-specific threshold candidate construction.
    """
    x_values = np.asarray(x, dtype=np.float64)
    field = np.asarray(concentration, dtype=np.float64)
    if x_values.ndim != 1:
        raise ValueError("x must be a one-dimensional vector.")
    if field.ndim != 2 or field.shape[1] != x_values.size:
        raise ValueError("concentration must have shape (y.size, x.size).")

    bound = normalize_extraction_x_min(x_min)
    if bound is None:
        keep = np.ones(x_values.size, dtype=np.bool_)
    else:
        keep = x_values > bound

    x_work = np.array(x_values[keep], dtype=np.float64, copy=True)
    concentration_work = np.array(field[:, keep], dtype=np.float64, copy=True)
    if x_work.size == 0:
        raise ValueError("The extraction x domain contains no retained points.")
    if x_work.size < 2:
        raise ValueError(
            "The extraction x domain must contain at least two retained points."
        )
    if not np.all(np.isfinite(x_work)) or np.any(np.diff(x_work) <= 0.0):
        raise ValueError(
            "The restricted extraction x vector must be finite and strictly "
            "increasing."
        )
    if concentration_work.shape != (field.shape[0], x_work.size):
        raise ValueError("Restricted concentration shape is inconsistent with x.")
    x_work.setflags(write=False)
    concentration_work.setflags(write=False)
    return x_work, concentration_work


def extract_leading_edge(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    concentration: NDArray[np.float64],
    *,
    threshold: float,
    method: object = DEFAULT_LEADING_EDGE_METHOD,
    periodic_y: bool,
    y_period: float,
    x_min: float | None = None,
) -> LeadingEdgeExtractionResult:
    """Extract one curve after optional strict physical-x domain restriction."""
    canonical_method = normalize_leading_edge_method(method)
    context = PeriodicYContext(periodic_y=periodic_y, y_period=y_period)
    validated_y = validate_periodic_y_coordinates(y, context)
    field = np.asarray(concentration, dtype=np.float64)
    x_values = np.asarray(x, dtype=np.float64)
    if field.shape != (validated_y.size, x_values.size):
        raise ValueError("concentration must have shape (y.size, x.size).")
    x_work, concentration_work = restrict_leading_edge_x_domain(
        x_values,
        field,
        x_min=x_min,
    )
    if canonical_method == DEFAULT_LEADING_EDGE_METHOD:
        return extract_rightmost_crossing(
            x_work,
            validated_y,
            concentration_work,
            threshold=threshold,
        )
    if canonical_method == MOORE_BOUNDARY_METHOD:
        return extract_moore_boundary(
            x_work,
            validated_y,
            concentration_work,
            threshold=threshold,
        )
    raise AssertionError(f"Unhandled leading-edge method {canonical_method!r}.")


__all__ = (
    "DEFAULT_LEADING_EDGE_METHOD",
    "EXTRACTION_X_CONDITION",
    "LeadingEdgeExtractionResult",
    "MOORE_BOUNDARY_METHOD",
    "MooreBoundaryComponent",
    "MooreBoundaryTrace",
    "MooreTraceCandidateDiagnostic",
    "MooreTraceSelection",
    "PeriodicYContext",
    "SUPPORTED_LEADING_EDGE_METHODS",
    "extract_leading_edge",
    "extract_moore_boundary",
    "label_periodic_boundary_components",
    "moore_trace_winding_number",
    "normalize_extraction_x_min",
    "normalize_leading_edge_method",
    "primary_moore_component_mask",
    "restrict_leading_edge_x_domain",
    "select_primary_moore_component",
    "select_periodic_moore_trace",
    "trace_moore_boundary",
)
