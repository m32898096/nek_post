"""Single-frame leading edges from the original solution on refined GLL nodes."""

from __future__ import annotations

from nek_post.leading_edge_methods import (
    LeadingEdgeExtractionResult,
    extract_leading_edge,
)
from nek_post.refined_gll_horizontal_slice import (
    RefinedGLLHorizontalSlicePlan,
    apply_refined_gll_horizontal_slice_plan,
)


def extract_refined_gll_leading_edge_frame(
    data: object,
    *,
    plan: RefinedGLLHorizontalSlicePlan,
    threshold: float,
    extraction_method: str,
    x_min: float | None = None,
) -> LeadingEdgeExtractionResult:
    """Extract one curve directly from a non-uniform GLL horizontal plane.

    An N7 solution evaluated on a 10-node-per-element GLL representation is
    still the original P7 solution. Denser sampling changes threshold-contour
    reconstruction; it adds no solution information.

    The existing public dispatcher receives the actual physical x/y vectors
    and concentration without remeshing. It owns method selection, validation,
    and strict ``x > x_min`` filtering before extraction. Its periodic-y
    contract requires an endpoint-excluded grid with a positive physical period.
    Neither the plan nor the plane is modified. No evolution workflow or output
    writing is performed.
    """
    plane = apply_refined_gll_horizontal_slice_plan(data, plan)
    return extract_leading_edge(
        plane.x,
        plane.y,
        plane.concentration,
        threshold=threshold,
        method=extraction_method,
        periodic_y=plane.periodic_y,
        y_period=plane.y_period,
        x_min=x_min,
    )
