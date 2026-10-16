"""Native-GLL streamwise curl followed by signed scalar mass-lumped recovery."""
from __future__ import annotations

import numpy as np

from nek_post.fields import get_coordinates, get_velocity
from nek_post.l2_projection import build_l2_projection_operator, project_scalar_field, _positive_integer
from nek_post.spectral_derivatives import (
    DEFAULT_SINGULAR_RTOL, element_jacobian, reference_derivatives,
)


def element_streamwise_vorticity(element, *, singular_rtol=DEFAULT_SINGULAR_RTOL):
    """Return dw/dy - dv/dz, computing only two physical derivatives.

    J has physical rows and reference r/s/t columns. Solving J*c=e_y/e_z
    gives only the two required columns of J^-1. Contract them with the
    existing local GLL derivatives of w and v. Geometry is validated once;
    no u derivatives, full gradient tensor, other curl components or eigensolve
    are evaluated. All temporary derivatives are bounded to one element.
    """
    jacobian, _ = element_jacobian(get_coordinates(element), singular_rtol=singular_rtol)
    _, v, w = get_velocity(element)
    if np.shape(v) != jacobian.shape[:-2] or np.shape(w) != jacobian.shape[:-2]:
        raise ValueError("Velocity field shape must match coordinate shape.")
    rhs = np.broadcast_to(np.eye(3)[:, 1:], jacobian.shape[:-2] + (3, 2))
    try:
        columns = np.linalg.solve(jacobian, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Physical curl transformation failed: singular Jacobian.") from exc
    with np.errstate(over="ignore", invalid="ignore"):
        omega = np.einsum('...r,...r->...', reference_derivatives(w), columns[..., 0])
        omega -= np.einsum('...r,...r->...', reference_derivatives(v), columns[..., 1])
    if not np.isfinite(omega).all():
        raise FloatingPointError("Streamwise vorticity contains non-finite values.")
    return omega


def compute_projected_omega_x(data, operator=None, *, periodic_axes=('x', 'y'),
                              expected_element_counts=None, chunk_size=256, progress=None):
    """Assemble one local scalar volume, then project with the shared kernel.

    An optional prepared operator must match the exact geometry/element order.
    Return element-local float64 scalars; no interpolation or smoothing.
    """
    chunk_size = _positive_integer(chunk_size, 'chunk_size')
    if operator is None:
        operator = build_l2_projection_operator(data, periodic_axes=periodic_axes,
            expected_element_counts=expected_element_counts, chunk_size=chunk_size)
    else:
        operator.node_map.validate_mesh(data)
    raw = np.empty((operator.node_map.element_count, *operator.node_map.element_shape), dtype=np.float64)
    for ei, element in enumerate(data.elem):
        raw[ei] = element_streamwise_vorticity(element)
        if progress and (ei + 1) % 4096 == 0:
            progress(f'omega_x derivatives: {ei + 1}/{len(data.elem)} elements')
    return project_scalar_field(operator, raw, chunk_size=chunk_size)
