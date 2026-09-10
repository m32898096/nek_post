"""Element-local post-processing resampling of existing GLL polynomials."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nek_post.gll import gll_interpolation_matrix


def refine_gll_tensor3(
    values: object,
    *,
    target_node_count: int,
) -> NDArray[np.float64]:
    """Evaluate one element's tensor polynomial on a target GLL nodal set.

    ``values`` must be a finite real 3-D tensor with at least two GLL nodes
    per axis, ordered as ascending reference nodes in each NumPy axis. Source
    node counts are inferred independently from its shape; axis order is
    preserved without assigning physical x/y/z directions. The returned float64
    array has shape ``(target_node_count,) * 3`` and does not modify the input.

    ``node_count`` means GLL node count; ``polynomial_order = node_count - 1``.
    In particular, (8, 8, 8) -> (10, 10, 10) evaluates the same P7 spectral-element
    polynomial on the order-9 GLL nodal set. This is post-processing resampling,
    not a true N9 simulation or new solution information. A smaller target count
    is allowed, but its nodal values need not retain the full source polynomial.
    """
    if np.iscomplexobj(values):
        raise ValueError("values must contain only real values.")
    values_arr = np.asarray(values, dtype=np.float64)
    if values_arr.ndim != 3:
        raise ValueError("values must be a three-dimensional tensor.")
    if any(count < 2 for count in values_arr.shape):
        raise ValueError("values must have at least 2 GLL nodes per axis.")
    if not np.all(np.isfinite(values_arr)):
        raise ValueError("values must contain only finite values.")

    matrix0, matrix1, matrix2 = (
        gll_interpolation_matrix(source_node_count, target_node_count)
        for source_node_count in values_arr.shape
    )
    return np.einsum(
        "ai,bj,ck,ijk->abc",
        matrix0,
        matrix1,
        matrix2,
        values_arr,
        optimize=True,
    )
