"""GLL-quadrature lumped L2 recovery on continuous structured meshes.

The caller supplies confirmed ``periodic_axes``; periodicity is never inferred
from coordinates. Geometry must be a complete axis-aligned affine tensor
partition (within the existing coordinate-storage tolerance), with no
intentionally disconnected coincident interfaces. Curved/unstructured global
connectivity is outside this implementation. Coordinates are never changed.

Scalar storage is (nt, ns, nr); reference (r, s, t) follows axes (2, 1, 0).
Tensor rows are velocity components, columns are physical x/y/z. This is the
GLL quadrature-discrete diagonal-mass version of global gradient recovery,
not the consistent-mass Galerkin solve in Engsig-Karup et al. (2016),
Section 3.1.3, Eqs. (17)-(19), https://arxiv.org/abs/1512.02548.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from types import SimpleNamespace

import numpy as np
from numpy.typing import NDArray

from nek_post.fields import get_coordinates
from nek_post.gll import gll_quadrature_weights
from nek_post.gll_directional_integration import (
    _assemble_global_coordinate,
    _cluster_intervals,
    _detect_geometry_mapping,
    _elements,
    _physical_cell_profiles,
    _validate_complete_topology,
    _validated_geometry,
)
from nek_post.spectral_derivatives import DEFAULT_SINGULAR_RTOL, element_jacobian
from nek_post.spectral_interpolation import element_coordinate_signature


DEFAULT_CHUNK_SIZE = 256


def _positive_integer(value: object, name: str, minimum: int = 1) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _integer_triple(values: object, name: str, minimum: int) -> tuple[int, int, int]:
    try:
        entries = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{name} must contain three integers.") from exc
    if len(entries) != 3:
        raise ValueError(f"{name} must contain three integers.")
    return tuple(_positive_integer(v, name, minimum) for v in entries)


def _periodic_axes(values: object) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError("periodic_axes must be an explicit iterable of distinct x/y/z names.")
    try:
        axes = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("periodic_axes must be supplied explicitly, including () for none.") from exc
    if any(not isinstance(a, str) or a not in ("x", "y", "z") for a in axes):
        raise ValueError("periodic_axes may contain only x, y, z.")
    if len(set(axes)) != len(axes):
        raise ValueError("periodic_axes must not contain duplicates.")
    return tuple(a for a in "xyz" if a in axes)


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    result = array.view()
    result.setflags(write=False)
    return result


def structured_global_node_shape(
    element_counts: object, nodes_per_element_xyz: object, *, periodic_axes: object,
) -> tuple[int, int, int]:
    """Return global node counts in physical x/y/z order, without allocating IDs.

    Both triples use physical directions, not array-axis order. Nodes are
    counts, not polynomial degrees. A periodic direction omits its upper
    endpoint, so each dimension is ``elements * (nodes - 1) + not_periodic``.
    """
    counts = _integer_triple(element_counts, "element_counts", 1)
    nodes = _integer_triple(nodes_per_element_xyz, "nodes_per_element_xyz", 2)
    periodic = _periodic_axes(periodic_axes)
    shape = tuple(c * (n - 1) + (a not in periodic) for c, n, a in zip(counts, nodes, "xyz"))
    if shape[0] * shape[1] * shape[2] > np.iinfo(np.int64).max:
        raise ValueError("Global node count exceeds signed int64 indexing capacity.")
    return shape


@dataclass(frozen=True)
class StructuredGLLNodeMap:
    """Compact, element-order-specific connectivity; IDs are generated in chunks.

    Axis permutations must be consistent across elements. Each element's
    increasing/decreasing orientation is checked and retained independently.
    Validated physical axis coordinate vectors include both endpoints even
    when their IDs are periodically equivalent; agreeing replicas may be
    averaged within storage uncertainty. Inputs remain untouched. Reuse
    requires identical geometry/order.
    """

    element_shape: tuple[int, int, int]
    element_interval_counts: tuple[int, int, int]
    physical_to_array_axes: tuple[int, int, int]
    periodic_axes: tuple[str, ...]
    global_shape_xyz: tuple[int, int, int]
    element_cell_indices: NDArray[np.int64]
    element_axis_increasing: NDArray[np.bool_]
    element_coordinate_signatures: tuple[str, ...]
    physical_axis_coordinates: tuple[NDArray[np.float64], ...]

    @property
    def element_count(self) -> int:
        return len(self.element_coordinate_signatures)

    @property
    def global_node_count(self) -> int:
        nx, ny, nz = self.global_shape_xyz
        return nx * ny * nz

    @property
    def shared_node_count(self) -> int:
        unshared = 1
        for i, a in enumerate("xyz"):
            nodes = self.element_shape[self.physical_to_array_axes[i]]
            unshared *= self.element_interval_counts[i] * (nodes - 2) + (0 if a in self.periodic_axes else 2)
        return self.global_node_count - unshared

    def node_ids(self, start: int, stop: int) -> np.ndarray:
        """IDs for elements [start:stop], shape (chunk, nt, ns, nr).

        Flattening is x-fast: ``(Iz * Ny + Iy) * Nx + Ix``. Each physical
        index uses its corresponding local array axis, reversed when needed.
        int32 is used when the entire global numbering fits; otherwise int64.
        """
        start = _positive_integer(start, "start", 0)
        stop = _positive_integer(stop, "stop", 0)
        if not start <= stop <= self.element_count:
            raise ValueError("Element slice must satisfy 0 <= start <= stop <= element_count.")
        dtype = np.int32 if self.global_node_count <= np.iinfo(np.int32).max else np.int64
        indices = []
        for i, a in enumerate("xyz"):
            axis = self.physical_to_array_axes[i]
            count = self.element_shape[axis]
            shape = [1, 1, 1, 1]
            shape[axis + 1] = count
            local = np.arange(count, dtype=dtype).reshape(shape)
            increasing = self.element_axis_increasing[start:stop, i, None, None, None]
            local = np.where(increasing, local, count - 1 - local)
            index = (count - 1) * self.element_cell_indices[start:stop, i].astype(dtype)[:, None, None, None] + local
            if a in self.periodic_axes:
                index %= self.global_shape_xyz[i]
            indices.append(index)
        nx, ny, _ = self.global_shape_xyz
        return (indices[2] * ny + indices[1]) * nx + indices[0]

    def validate_mesh(self, data: object) -> None:
        """Reject changed geometry or reordered elements before operator reuse."""
        elements = _elements(data)
        if len(elements) != self.element_count:
            raise ValueError("Mesh element count differs from the prepared mapping.")
        for i, element in enumerate(elements):
            signature = element_coordinate_signature(*get_coordinates(element))
            if signature != self.element_coordinate_signatures[i]:
                raise ValueError(f"Mesh geometry or element ordering differs at element {i}.")


def build_structured_node_map(
    data: object, *, periodic_axes: object, chunk_size: int = DEFAULT_CHUNK_SIZE,
    expected_element_counts: object | None = None,
) -> StructuredGLLNodeMap:
    """Validate a continuous affine box mesh and build compact structured IDs.

    ``data.elem`` provides coordinate-bearing reader elements. The explicit
    periodic axes assert pure translation with no rotations or disconnected
    coincident interfaces. Complete tiling, all GLL profiles, orientation and
    transverse consistency (including opposite boundary planes) are checked.
    No node is merged by arbitrary spatial hashing or tolerance rounding.
    Supply expected_element_counts to also reject a removed complete outer
    layer, which would otherwise describe a different, complete smaller box.
    """
    periodic = _periodic_axes(periodic_axes)
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    expected = None if expected_element_counts is None else _integer_triple(expected_element_counts, "expected_element_counts", 1)
    elements = _elements(data)
    bounds = np.empty((len(elements), 3, 2), dtype=np.float64)
    tolerances = np.empty((len(elements), 3), dtype=np.float64)
    increasing = np.empty((len(elements), 3), dtype=np.bool_)
    all_profiles, signatures = [], []
    shape = mapping = None
    for start in range(0, len(elements), chunk_size):
        _, geometry, batch_shape, batch_signatures = _validated_geometry(
            SimpleNamespace(elem=elements[start:start + chunk_size]))
        batch_mapping, profiles, inc, batch_bounds, tol = _detect_geometry_mapping(geometry, batch_shape)
        if shape is not None and (shape != batch_shape or mapping != batch_mapping):
            raise ValueError("Element shapes or physical/array-axis mappings are inconsistent.")
        shape, mapping = batch_shape, batch_mapping
        stop = start + len(geometry)
        bounds[start:stop], tolerances[start:stop], increasing[start:stop] = batch_bounds, tol, inc
        all_profiles.extend(profiles)
        signatures.extend(batch_signatures)
        del geometry
    assert shape is not None and mapping is not None
    cells = np.empty((len(elements), 3), dtype=np.int64)
    intervals = []
    for i, a in enumerate("xyz"):
        axis_intervals, cells[:, i] = _cluster_intervals(bounds[:, i], float(tolerances[:, i].max()), a)
        intervals.append(axis_intervals)
    counts = tuple(len(v) for v in intervals)
    if expected is not None and counts != expected:
        raise ValueError(f"Expected element counts {expected}, found {counts}.")
    _validate_complete_topology(cells, counts)
    physical_profiles = _physical_cell_profiles(tuple(all_profiles), increasing, cells, counts, tolerances)
    axes = tuple(_readonly(_assemble_global_coordinate(p)[0]) for p in physical_profiles)
    global_shape = structured_global_node_shape(counts, tuple(shape[mapping[i]] for i in range(3)), periodic_axes=periodic)
    return StructuredGLLNodeMap(
        shape, counts, mapping, periodic, global_shape, _readonly(cells),
        _readonly(increasing), tuple(signatures), axes,
    )


def element_gll_weights(
    coordinates: object, *, singular_rtol: float = DEFAULT_SINGULAR_RTOL,
) -> NDArray[np.float64]:
    """Return wt*ws*wr*abs(detJ) in (nt, ns, nr) order using Stage 1.

    This local weight function also accepts valid curved elements. The
    structured global-map builder remains restricted to affine box geometry.
    Consistently negative determinants are permitted; folded/singular nodal
    Jacobians are rejected by Stage 1. No determinant is approximated by a
    bounding-box volume.
    """
    _, determinant = element_jacobian(coordinates, singular_rtol=singular_rtol)
    wt, ws, wr = (gll_quadrature_weights(n) for n in determinant.shape)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        weights = np.abs(determinant) * wt[:, None, None] * ws[None, :, None] * wr[None, None, :]
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Element integration weights must be finite and strictly positive.")
    return weights


@dataclass(frozen=True)
class L2ProjectionOperator:
    """Reusable mapping, float64 nodal weights and positive lumped mass.

    Storage is O(local_nodes + global_nodes); no full local-ID array or dense
    mass matrix is retained. Geometry weights come from the original nodal
    coordinates, including float32 storage variations, via Stage 1.
    """

    node_map: StructuredGLLNodeMap
    weights: NDArray[np.float64]
    mass: NDArray[np.float64]

    @property
    def storage_bytes(self) -> int:
        """Persistent NumPy payload bytes (excludes Python/signature overhead)."""
        mapping = self.node_map
        return sum(a.nbytes for a in (
            self.weights, self.mass, mapping.element_cell_indices,
            mapping.element_axis_increasing, *mapping.physical_axis_coordinates,
        ))


def build_l2_projection_operator(
    data: object, *, periodic_axes: object, chunk_size: int = DEFAULT_CHUNK_SIZE,
    singular_rtol: float = DEFAULT_SINGULAR_RTOL,
    expected_element_counts: object | None = None,
) -> L2ProjectionOperator:
    """Prepare one stationary mesh; validate geometry and assemble mass by scatter.

    Coordinate processing is bounded by chunk_size during mapping and one
    element during Stage 1 determinant evaluation. No velocity is accessed.
    All weights and global masses must be finite and strictly positive.
    For the verified N7_H mesh, use periodic_axes=("x", "y") and
    expected_element_counts=(272, 12, 8); these are not generic defaults.
    """
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    mapping = build_structured_node_map(
        data, periodic_axes=periodic_axes, chunk_size=chunk_size,
        expected_element_counts=expected_element_counts,
    )
    elements = _elements(data)
    weights = np.empty((mapping.element_count, *mapping.element_shape), dtype=np.float64)
    mass = np.zeros(mapping.global_node_count, dtype=np.float64)
    for start in range(0, mapping.element_count, chunk_size):
        stop = min(start + chunk_size, mapping.element_count)
        for i in range(start, stop):
            weights[i] = element_gll_weights(get_coordinates(elements[i]), singular_rtol=singular_rtol)
        with np.errstate(over="ignore", invalid="ignore"):
            np.add.at(mass, mapping.node_ids(start, stop).ravel(), weights[start:stop].ravel())
    with np.errstate(over="ignore", invalid="ignore"):
        total_mass = mass.sum()
    if not np.all(np.isfinite(mass)) or np.any(mass <= 0) or not np.isfinite(total_mass):
        raise ValueError("Assembled global mass must be finite and strictly positive at every node.")
    return L2ProjectionOperator(mapping, _readonly(weights), _readonly(mass))


def _gradient_array(
    node_map: StructuredGLLNodeMap, gradients: object, chunk_size: int,
) -> np.ndarray:
    try:
        array = np.asarray(gradients)
    except (ValueError, TypeError) as exc:
        raise ValueError("Gradients must contain real numeric values.") from exc
    shape = (node_map.element_count, *node_map.element_shape, 3, 3)
    if array.shape != shape:
        raise ValueError(f"Gradient tensor must have shape {shape}.")
    if array.dtype.kind not in "fiu":
        raise ValueError("Gradients must contain real numeric values.")
    for start in range(0, node_map.element_count, chunk_size):
        if not np.all(np.isfinite(array[start:start + chunk_size])):
            raise ValueError("Gradients must contain only finite values.")
    return array


def shared_node_discontinuity(
    node_map: StructuredGLLNodeMap, gradients: object, *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> NDArray[np.float64]:
    """Maximum contributor range per component, shape (3,3), including seams.

    Range = max(local values) - min(local values) at each global node; the
    largest range equals the maximum pairwise absolute jump. Unique nodes
    contribute zero. Two global vectors are reused across nine components;
    chunked ID/value arrays avoid full-size temporary local arrays.
    """
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    array = _gradient_array(node_map, gradients, chunk_size)
    low = np.empty(node_map.global_node_count, dtype=np.float64)
    high = np.empty_like(low)
    jumps = np.empty((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            low.fill(np.inf)
            high.fill(-np.inf)
            for start in range(0, node_map.element_count, chunk_size):
                stop = min(start + chunk_size, node_map.element_count)
                ids = node_map.node_ids(start, stop).ravel()
                values = array[start:stop, ..., i, j].reshape(-1)
                np.minimum.at(low, ids, values)
                np.maximum.at(high, ids, values)
            with np.errstate(over="ignore", invalid="ignore"):
                np.subtract(high, low, out=high)
            if not np.all(np.isfinite(high)):
                raise FloatingPointError("Shared-node discontinuity is non-finite or exceeds float64 range.")
            jumps[i, j] = np.max(high)
    return _readonly(jumps)


@dataclass(frozen=True)
class L2ProjectionDiagnostics:
    global_node_count: int
    shared_node_count: int
    mass_min: float
    mass_max: float
    mass_sum: float
    discontinuity_before: NDArray[np.float64]
    discontinuity_after: NDArray[np.float64]


@dataclass(frozen=True)
class L2ProjectionResult:
    values: NDArray[np.float64]
    diagnostics: L2ProjectionDiagnostics | None


def project_velocity_gradient(
    operator: L2ProjectionOperator, gradients: object, *, mesh: object | None = None,
    out: np.ndarray | None = None, diagnostics: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> L2ProjectionResult:
    """Recover all nine gradient components by GLL weighted global averaging.

    Input/output shape is (nelem, nt, ns, nr, 3, 3), A[i,j] = du_i/dx_j.
    The input is never converted wholesale to float64 and is not modified.
    Output is float64 in the original local layout. Optional out must be a
    writable C-contiguous float64 array of that shape without overlap with
    inputs/operator (contiguity also rules out internally overlapping strides).
    Only one global RHS vector is used at a time. Diagnostics measure jumps
    before/after independently and use two global workspace vectors.

    Input gradients must follow the prepared mesh's element and node order.
    Pass mesh to verify stationary geometry/order before reusing an operator
    with a later snapshot; anonymous gradient arrays alone cannot prove order.
    No eigenvalues, derived scalar or global coordinate averaging is performed.
    """
    if not isinstance(operator, L2ProjectionOperator):
        raise ValueError("operator must be a prepared L2ProjectionOperator.")
    if not isinstance(diagnostics, (bool, np.bool_)):
        raise ValueError("diagnostics must be boolean.")
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    mapping = operator.node_map
    if mesh is not None:
        mapping.validate_mesh(mesh)
    array = _gradient_array(mapping, gradients, chunk_size)
    if out is None:
        output = np.empty(array.shape, dtype=np.float64)
    else:
        if not isinstance(out, np.ndarray) or out.shape != array.shape or out.dtype != np.float64 or not out.flags.writeable or not out.flags.c_contiguous:
            raise ValueError("out must be a writable C-contiguous float64 array matching the input shape.")
        if any(np.shares_memory(out, v) for v in (array, operator.weights, operator.mass)):
            raise ValueError("out must not overlap input gradients or prepared operator arrays.")
        output = out
    before = shared_node_discontinuity(mapping, array, chunk_size=chunk_size) if diagnostics else None
    rhs = np.empty(mapping.global_node_count, dtype=np.float64)
    for i in range(3):
        for j in range(3):
            rhs.fill(0)
            for start in range(0, mapping.element_count, chunk_size):
                stop = min(start + chunk_size, mapping.element_count)
                ids = mapping.node_ids(start, stop).ravel()
                with np.errstate(over="ignore", invalid="ignore"):
                    weighted = np.multiply(
                        array[start:stop, ..., i, j].reshape(-1),
                        operator.weights[start:stop].ravel(), dtype=np.float64,
                    )
                    np.add.at(rhs, ids, weighted)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                rhs /= operator.mass
            if not np.all(np.isfinite(rhs)):
                raise FloatingPointError("Projected gradient contains non-finite values (float64 overflow).")
            for start in range(0, mapping.element_count, chunk_size):
                stop = min(start + chunk_size, mapping.element_count)
                output[start:stop, ..., i, j] = rhs[mapping.node_ids(start, stop)]
    del rhs
    report = None
    if diagnostics:
        after = shared_node_discontinuity(mapping, output, chunk_size=chunk_size)
        report = L2ProjectionDiagnostics(
            mapping.global_node_count, mapping.shared_node_count,
            float(operator.mass.min()), float(operator.mass.max()), float(operator.mass.sum()),
            before, after,
        )
    return L2ProjectionResult(output, report)
