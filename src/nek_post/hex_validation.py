"""Validate linear VTK hex topology and the full trilinear cell Jacobian.

This validates the VTU subcell map, not an original high-order SEM map. Curved
SEM faces are represented only by their eight-corner trilinear subcells here.
Nonuniform cuboids, skew affine cells and certifiably positive warped cells
are supported. No point movement or changes to scientific scalars occur.
"""
from __future__ import annotations

from itertools import product

import numpy as np

from nek_post.l2_projection import _positive_integer


_CORNERS = np.asarray([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                       (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)])
_SAMPLES = np.asarray(list(product((0., .5, 1.), repeat=3)))
_EDGE_PAIRS = (((0, 3, 4, 7), (1, 2, 5, 6)),
               ((0, 1, 4, 5), (3, 2, 7, 6)),
               ((0, 1, 2, 3), (4, 5, 6, 7)))
_BERNSTEIN_FROM_VALUES = np.asarray([[1., 0., 0.], [-.5, 2., -.5], [0., 0., 1.]])
_CORNER_INDICES = np.ix_((0, 2), (0, 2), (0, 2))
JACOBIAN_RTOL = 1e-12
MAX_SUBDIVISION_DEPTH = 8
MAX_SUBDIVISION_BOXES = 4096


def _edge_weights():
    weights = []
    for axis in range(3):
        other = [a for a in range(3) if a != axis]
        low, _ = _EDGE_PAIRS[axis]
        weights.append(np.prod(np.where(_CORNERS[list(low)][:, other][None],
                                        _SAMPLES[:, None, other], 1 - _SAMPLES[:, None, other]), axis=-1))
    return tuple(weights)


_EDGE_WEIGHTS = _edge_weights()


def _split_bernstein(coefficients, axis):
    """Degree-two de Casteljau subdivision at 1/2, along one tensor axis."""
    a, b, c = np.moveaxis(coefficients, axis, 0)
    ab, bc = .5 * (a + b), .5 * (b + c)
    mid = .5 * (ab + bc)
    return tuple(np.moveaxis(np.stack(v), 0, axis) for v in ((a, ab, mid), (mid, bc, c)))


def _certify_positive(coefficients):
    """Convex-hull bounds certify positivity over the entire reference cube.

    det(J) is degree <=(2,2,2) for a trilinear hex. Its tensor Bernstein
    coefficients bound every value, since basis functions are nonnegative and
    sum to one. Subdivision tightens inconclusive bounds without relying on a
    finite set of sample points. The normalized 1e-12 margin exceeds ordinary
    float64 conversion/determinant roundoff; this is not interval arithmetic.
    """
    stack = [(coefficients, 0)]
    visited = 0
    while stack:
        values, depth = stack.pop()
        visited += 1
        if np.min(values) > JACOBIAN_RTOL:
            continue
        if np.any(values[_CORNER_INDICES] <= JACOBIAN_RTOL):
            raise ValueError('Hexahedral Jacobian is nonpositive or numerically degenerate.')
        if depth >= MAX_SUBDIVISION_DEPTH or visited >= MAX_SUBDIVISION_BOXES:
            raise ValueError('Cannot certify a positive hexahedral Jacobian within subdivision limits.')
        children = [values]
        for axis in range(3):
            children = [child for box in children for child in _split_bernstein(box, axis)]
        stack.extend((child, depth + 1) for child in children)


def _validate_warped(points):
    # Each Jacobian column is a convex combination of four parallel-reference
    # edges. Normalize those edges before evaluation to avoid cancellation of
    # large translations or unrelated anisotropic coordinate components.
    jacobian = np.empty((len(points), 27, 3, 3))
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        for axis, (lo, hi) in enumerate(_EDGE_PAIRS):
            edges = points[:, hi] - points[:, lo]
            scale = np.max(np.abs(edges), axis=(1, 2))
            if not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError('Hexahedral geometry has invalid or zero-length reference directions.')
            jacobian[..., axis] = np.einsum('cni,sn->csi', edges / scale[:, None, None], _EDGE_WEIGHTS[axis])
        determinants = np.linalg.det(jacobian).reshape(-1, 3, 3, 3)
    if not np.isfinite(determinants).all() or np.any(determinants <= JACOBIAN_RTOL):
        raise ValueError('Hexahedral Jacobian is nonpositive or numerically degenerate.')
    coefficients = np.einsum('ai,bj,dk,cijk->cabd', *([_BERNSTEIN_FROM_VALUES] * 3), determinants)
    for values in coefficients:
        _certify_positive(values)


def validate_hexahedral_grid(grid, *, chunk_size=4096):
    """Reject malformed IDs/offsets, repeated vertices and nonpositive maps.

    Bounded chunks avoid reconstructing the full legacy packed-cell array.
    Exact physical x/y/z cuboids use an analytical positive-diagonal-J test;
    all other cells use the full-domain Bernstein certificate above. Extremely
    ill-conditioned or uncertifiable cells fail closed with an explicit error.
    """
    from vtkmodules.vtkCommonDataModel import VTK_HEXAHEDRON

    chunk_size = _positive_integer(chunk_size, 'chunk_size')
    points = np.asarray(grid.points)
    connectivity = np.asarray(grid.cell_connectivity)
    offsets = np.asarray(grid.offset)
    types = np.asarray(grid.celltypes)
    if (grid.n_cells < 1 or points.shape != (grid.n_points, 3) or grid.n_points < 8 or
            points.dtype != np.float64 or not np.isfinite(points).all()):
        raise ValueError('Hexahedral grid requires finite float64 3D points.')
    if (connectivity.dtype.kind not in 'iu' or connectivity.ndim != 1 or
            offsets.shape != (grid.n_cells + 1,) or offsets[0] != 0 or
            offsets[-1] != len(connectivity) or len(connectivity) != 8 * grid.n_cells or
            types.shape != (grid.n_cells,) or not np.all(types == VTK_HEXAHEDRON)):
        raise ValueError('Malformed VTK hexahedral connectivity, offsets or cell types.')
    for start in range(0, grid.n_cells, chunk_size):
        stop = min(start + chunk_size, grid.n_cells)
        if not np.all(np.diff(offsets[start:stop + 1]) == 8):
            raise ValueError('Each VTK hexahedron must contain exactly eight vertices.')
        ids = connectivity[8 * start:8 * stop].reshape(-1, 8)
        if np.any(ids < 0) or np.any(ids >= grid.n_points):
            raise ValueError('Hexahedral point IDs are outside the point array.')
        if np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
            raise ValueError('Each hexahedron must have eight unique vertex IDs.')
        p = points[ids]
        low, high = p[:, 0], p[:, 6]
        expected = np.where(_CORNERS[None, :, :], high[:, None, :], low[:, None, :])
        # The shortcut proves a positive diagonal J only in this orientation.
        # Other orientations, including two reversed axes with det(J)>0,
        # must use the unchanged general Jacobian/Bernstein validation.
        cuboids = np.all(p == expected, axis=(1, 2)) & np.all(high > low, axis=1)
        with np.errstate(over='ignore', invalid='ignore'):
            spans = high - low
        if np.any(cuboids & ~np.isfinite(spans).all(axis=1)):
            raise ValueError('Hexahedral Jacobian is nonpositive or degenerate.')
        if np.any(~cuboids):
            other = p[~cuboids]
            for vertex in range(7):
                if np.any(np.all(other[:, vertex, None] == other[:, vertex + 1:], axis=-1)):
                    raise ValueError('Hexahedral vertices must have distinct physical coordinates.')
            _validate_warped(other)
