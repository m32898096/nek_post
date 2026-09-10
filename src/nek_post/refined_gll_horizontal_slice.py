"""Non-uniform GLL horizontal planes of the original spectral solution.

Only complete, rectangular, separable tensor meshes are accepted. Coordinates
are evaluated from their stored source polynomials, never fitted to an affine
mesh. The target nodal set adds sampling locations, not solution information.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import Legendre
from numpy.typing import NDArray
from scipy.optimize import brentq

from nek_post.fields import get_concentration
from nek_post.gll import (
    barycentric_basis_and_derivative,
    barycentric_weights,
    gll_interpolation_matrix,
    gll_nodes,
)
from nek_post.gll_directional_integration import (
    _validate_complete_topology,
    _validated_geometry,
)
from nek_post.gll_refinement import refine_gll_tensor3
from nek_post.spectral_interpolation import SpectralGeometryMismatchError


# Coordinate matching allows float64 arithmetic roundoff, not float32 storage
# noise or geometric warping. Scale is max(abs(coordinates), domain span).
GEOMETRY_ROUNDOFF_FACTOR = 64.0
INTERFACE_VALUE_RTOL = 1.0e-10
INTERFACE_VALUE_ATOL = 1.0e-12


def _immutable(values: object, dtype: type = np.float64) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class RefinedGLLHorizontalSlicePlan:
    """Stationary geometry plan; selected elements retain their input order.

    Local-to-global maps refer to physically increasing local nodes. The last
    y index equals ``y.size`` when periodic and is discarded during application.
    Duplicate-node counts describe each 1-D coordinate vector, not full faces.
    ``reference_q_z`` uses each selected element's original reference orientation.
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z_target: float
    source_node_count: int
    target_node_count: int
    periodic_y: bool
    y_period: float
    source_element_count: int
    element_coordinate_signatures: tuple[str, ...]
    physical_to_reference_axes: tuple[int, int, int]
    element_interval_counts: tuple[int, int, int]
    element_indices: NDArray[np.int64]
    element_cell_indices: NDArray[np.int64]
    physical_axis_increasing: NDArray[np.bool_]
    x_local_to_global: NDArray[np.int64]
    y_local_to_global: NDArray[np.int64]
    reference_q_z: NDArray[np.float64]
    vertical_basis: NDArray[np.float64]
    coordinate_tolerances: tuple[float, float, float]
    maximum_tensor_separability_deviation: float
    maximum_repeated_profile_mismatch: float
    maximum_shared_interface_coordinate_mismatch: float
    duplicated_internal_x_nodes_removed: int
    duplicated_internal_y_nodes_removed: int


@dataclass(frozen=True)
class RefinedGLLHorizontalSlice:
    """Original source polynomial sampled as ``concentration[iy, ix]``.

    ``target_node_count`` identifies a sampling set, not a new solution order.
    For periodic y, the lower seam row is retained and the upper one excluded.
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    concentration: NDArray[np.float64]
    z_target: float
    source_node_count: int
    target_node_count: int
    periodic_y: bool
    y_period: float
    maximum_shared_interface_concentration_mismatch: float

    def __post_init__(self) -> None:
        x, y, concentration = (
            _immutable(array) for array in (self.x, self.y, self.concentration)
        )
        for name, coordinate in (("x", x), ("y", y)):
            if (
                coordinate.ndim != 1 or coordinate.size == 0
                or not np.all(np.isfinite(coordinate))
                or np.any(np.diff(coordinate) <= 0.0)
            ):
                raise ValueError(f"{name} must be a finite strictly increasing vector.")
        if concentration.shape != (y.size, x.size):
            raise ValueError("concentration must have shape (y.size, x.size).")
        if not np.all(np.isfinite(concentration)):
            raise ValueError("concentration must contain only finite values.")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "concentration", concentration)


def _separable_profiles(
    geometry: tuple[tuple[NDArray[np.float64], ...], ...],
    node_count: int,
) -> tuple[
    tuple[int, int, int],
    list[tuple[NDArray[np.float64], ...]],
    NDArray[np.bool_],
    NDArray[np.float64],
    tuple[float, float, float],
    float,
]:
    bounds = np.asarray([
        [(float(c.min()), float(c.max())) for c in coordinates]
        for coordinates in geometry
    ])
    axis_tolerances = [
        GEOMETRY_ROUNDOFF_FACTOR * np.finfo(np.float64).eps
        * max(float(np.max(np.abs(bounds[:, p]))), float(np.ptp(bounds[:, p])))
        for p in range(3)
    ]
    tolerances = (axis_tolerances[0], axis_tolerances[1], axis_tolerances[2])
    profiles = []
    increasing = np.empty((len(geometry), 3), dtype=np.bool_)
    mapping = None
    maximum_deviation = 0.0
    for index, coordinates in enumerate(geometry):
        axes, element_profiles = [], []
        for p, coordinate in enumerate(coordinates):
            candidates = []
            for axis in range(3):
                lines = np.moveaxis(coordinate, axis, 0).reshape(node_count, -1)
                profile = lines[:, 0]
                deviation = float(np.max(np.abs(lines - profile[:, None])))
                differences = np.diff(profile)
                if deviation <= tolerances[p] and (
                    np.all(differences > 0.0) or np.all(differences < 0.0)
                ):
                    candidates.append((axis, profile.copy(), deviation))
            if len(candidates) != 1:
                raise ValueError(
                    f"Element {index} coordinate {'xyz'[p]} is not separable and "
                    "monotonic along exactly one reference axis; a rectangular "
                    "tensor grid cannot be formed."
                )
            axis, profile, deviation = candidates[0]
            maximum_deviation = max(maximum_deviation, deviation)
            axes.append(axis)
            element_profiles.append(profile)
            increasing[index, p] = profile[-1] > profile[0]
        if len(set(axes)) != 3 or (mapping is not None and tuple(axes) != mapping):
            raise ValueError("Physical/reference-axis mapping must be consistent and bijective.")
        mapping = (axes[0], axes[1], axes[2])
        profiles.append(tuple(element_profiles))
    assert mapping is not None
    return mapping, profiles, increasing, bounds, tolerances, maximum_deviation


def _intervals(
    bounds: NDArray[np.float64], tolerance: float, name: str,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Cluster endpoint pairs against fixed representatives, without averaging."""
    representatives = []
    identifiers = np.empty(bounds.shape[0], dtype=np.int64)
    for index in np.lexsort((bounds[:, 1], bounds[:, 0])):
        pair = bounds[index]
        if not representatives or np.max(np.abs(pair - representatives[-1])) > tolerance:
            representatives.append(pair.copy())
        identifiers[index] = len(representatives) - 1
    intervals = np.asarray(representatives)
    if np.any(intervals[:, 1] - intervals[:, 0] <= 1024.0 * tolerance):
        raise ValueError(f"{name} intervals are too narrow for reliable coordinate matching.")
    if np.any(np.abs(intervals[1:, 0] - intervals[:-1, 1]) > tolerance):
        raise ValueError(f"{name} intervals contain a gap or overlap.")
    return intervals, identifiers


def _check_monotonic_polynomial(profile: NDArray[np.float64]) -> None:
    # This modal polynomial is used only to certify monotonicity between nodes.
    # All coordinate/field evaluation still uses the existing barycentric code.
    normalized = (profile - profile[0]) / (profile[-1] - profile[0])
    derivative = Legendre.fit(
        gll_nodes(profile.size), normalized, profile.size - 1, domain=(-1.0, 1.0)
    ).deriv()
    roots = derivative.deriv().roots()
    critical = roots[np.isreal(roots)].real
    points = np.r_[-1.0, critical[(critical > -1.0) & (critical < 1.0)], 1.0]
    if np.min(derivative(points)) <= GEOMETRY_ROUNDOFF_FACTOR * np.finfo(float).eps:
        raise ValueError("Coordinate polynomial is not strictly monotonic between GLL nodes.")


def _axis_coordinates(
    profiles: list[tuple[NDArray[np.float64], ...]],
    increasing: NDArray[np.bool_],
    cells: NDArray[np.int64],
    counts: tuple[int, int, int],
    matrix: NDArray[np.float64],
    tolerances: tuple[float, float, float],
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.int64]], float, float]:
    coordinates, maps = [], []
    maximum_repeated = maximum_interface = 0.0
    for p in range(3):
        representatives = []
        for cell in range(counts[p]):
            members = np.flatnonzero(cells[:, p] == cell)
            reference = None
            for index in members:
                profile = profiles[index][p]
                profile = profile if increasing[index, p] else profile[::-1]
                target = matrix @ profile
                if reference is None:
                    _check_monotonic_polynomial(profile)
                    reference = target
                    source_reference = profile
                mismatch = max(
                    float(np.max(np.abs(target - reference))),
                    float(np.max(np.abs(profile - source_reference))),
                )
                maximum_repeated = max(maximum_repeated, mismatch)
                if mismatch > tolerances[p]:
                    raise ValueError(
                        f"Repeated {'xyz'[p]} cell profiles disagree by {mismatch:.6g}; "
                        "one global coordinate vector cannot represent this mesh."
                    )
            assert reference is not None
            representatives.append(reference)
        vector = list(representatives[0])
        for previous, current in zip(representatives[:-1], representatives[1:], strict=True):
            mismatch = abs(float(previous[-1] - current[0]))
            maximum_interface = max(maximum_interface, mismatch)
            if mismatch > tolerances[p]:
                raise ValueError(f"Shared {'xyz'[p]} interface coordinates disagree.")
            vector.extend(current[1:])
        vector = np.asarray(vector)
        if not np.all(np.isfinite(vector)) or np.any(np.diff(vector) <= 0.0):
            raise ValueError(f"Refined {'xyz'[p]} coordinates are not finite and increasing.")
        coordinates.append(vector)
        maps.append(np.arange(counts[p])[:, None] * (matrix.shape[0] - 1)
                    + np.arange(matrix.shape[0])[None, :])
    return coordinates, maps, maximum_repeated, maximum_interface


def build_refined_gll_horizontal_slice_plan(
    data: object,
    *,
    target_node_count: int,
    z_target: float,
    periodic_y: bool = True,
) -> RefinedGLLHorizontalSlicePlan:
    """Plan a physical z plane on a complete separable, isotropic GLL mesh.

    Each coordinate must depend monotonically on a distinct reference axis,
    with one consistent axis permutation. Repeated physical intervals must
    carry matching stored coordinate profiles to float64 roundoff. Unsupported
    curvilinear geometry fails explicitly; there is no remapping or averaging.

    ``target_node_count >= source_node_count`` is required to retain the source
    polynomial through full tensor refinement. Physical z is inverted in the
    ORIGINAL coordinate polynomial, then evaluated spectrally on the refined
    tensor. At a vertical layer boundary, the lower physical layer owns the
    plane. A periodic y grid retains only its lower endpoint.
    """
    target_nodes = gll_nodes(target_node_count)
    target_node_count = int(target_node_count)
    if not isinstance(periodic_y, (bool, np.bool_)):
        raise ValueError("periodic_y must be a boolean.")
    z_target = float(z_target)
    if not np.isfinite(z_target):
        raise ValueError("z_target must be finite.")
    elements, geometry, shape, signatures = _validated_geometry(data)
    if len(set(shape)) != 1:
        raise ValueError("Source elements must have an isotropic GLL shape.")
    source_node_count = shape[0]
    if target_node_count < source_node_count:
        raise ValueError(
            "target_node_count must be >= source_node_count to retain the source polynomial."
        )
    matrix = gll_interpolation_matrix(source_node_count, target_node_count)
    mapping, profiles, increasing, bounds, tolerances, separability = (
        _separable_profiles(geometry, source_node_count)
    )
    intervals, columns = zip(*(
        _intervals(bounds[:, p], tolerances[p], 'xyz'[p]) for p in range(3)
    ), strict=True)
    cells = np.column_stack(columns)
    counts = (len(intervals[0]), len(intervals[1]), len(intervals[2]))
    _validate_complete_topology(cells, counts)
    coordinates, maps, repeated, interface = _axis_coordinates(
        profiles, increasing, cells, counts, matrix, tolerances
    )
    layers = np.flatnonzero(
        (intervals[2][:, 0] <= z_target) & (z_target <= intervals[2][:, 1])
    )
    if not layers.size:
        raise ValueError("z_target lies outside the physical vertical domain.")
    selected = np.flatnonzero(cells[:, 2] == layers[0])
    nodes = gll_nodes(source_node_count)
    weights = barycentric_weights(nodes)
    target_weights = barycentric_weights(target_nodes)
    q_values, bases = [], []
    inverse_cache = {}
    for index in selected:
        profile = profiles[index][2]
        key = tuple(profile)
        if key not in inverse_cache:
            def residual(q: float) -> float:
                basis, _ = barycentric_basis_and_derivative(nodes, weights, q)
                return float(basis @ profile - z_target)

            q = brentq(residual, -1.0, 1.0, xtol=2.0e-15)
            basis, _ = barycentric_basis_and_derivative(target_nodes, target_weights, q)
            if abs(float(basis @ (matrix @ profile)) - z_target) > tolerances[2]:
                raise ValueError("Refined vertical polynomial does not reproduce z_target.")
            inverse_cache[key] = q, basis
        q, basis = inverse_cache[key]
        q_values.append(q)
        bases.append(basis)
    x, full_y, _ = coordinates
    return RefinedGLLHorizontalSlicePlan(
        x=_immutable(x), y=_immutable(full_y[:-1] if periodic_y else full_y),
        z_target=z_target, source_node_count=source_node_count,
        target_node_count=target_node_count, periodic_y=bool(periodic_y),
        y_period=float(full_y[-1] - full_y[0]), source_element_count=len(elements),
        element_coordinate_signatures=signatures, physical_to_reference_axes=mapping,
        element_interval_counts=counts, element_indices=_immutable(selected, np.int64),
        element_cell_indices=_immutable(cells[selected], np.int64),
        physical_axis_increasing=_immutable(increasing[selected], np.bool_),
        x_local_to_global=_immutable(maps[0], np.int64),
        y_local_to_global=_immutable(maps[1], np.int64),
        reference_q_z=_immutable(q_values), vertical_basis=_immutable(bases),
        coordinate_tolerances=tolerances,
        maximum_tensor_separability_deviation=separability,
        maximum_repeated_profile_mismatch=repeated,
        maximum_shared_interface_coordinate_mismatch=interface,
        duplicated_internal_x_nodes_removed=counts[0] - 1,
        duplicated_internal_y_nodes_removed=counts[1] - 1,
    )


def apply_refined_gll_horizontal_slice_plan(
    data: object,
    plan: RefinedGLLHorizontalSlicePlan,
) -> RefinedGLLHorizontalSlice:
    """Refine intersected elements and evaluate the physical horizontal plane.

    Identical geometry and element ordering are required for reuse. Shared
    internal values must agree (rtol=1e-10, atol=1e-12); the first element in
    input order owns each retained value. Neither interfaces nor periodic seam
    rows are averaged. The upper periodic row is discarded unconditionally.
    """
    if not isinstance(plan, RefinedGLLHorizontalSlicePlan):
        raise ValueError("plan must be a RefinedGLLHorizontalSlicePlan.")
    elements, _, shape, signatures = _validated_geometry(data)
    if (len(elements) != plan.source_element_count
            or shape != (plan.source_node_count,) * 3
            or signatures != plan.element_coordinate_signatures):
        raise SpectralGeometryMismatchError(
            "Geometry or element ordering changed since plan construction."
        )
    values = np.full((plan.y.size, plan.x.size), np.nan, dtype=np.float64)
    maximum_mismatch = 0.0
    x_axis, y_axis, z_axis = plan.physical_to_reference_axes
    remaining = [axis for axis in range(3) if axis != z_axis]
    transpose = (remaining.index(y_axis), remaining.index(x_axis))
    for row, index in enumerate(plan.element_indices):
        try:
            raw = get_concentration(elements[index])
            if np.iscomplexobj(raw):
                raise ValueError("Concentration must be real.")
            field = np.asarray(raw, dtype=np.float64)
        except Exception as exc:
            raise ValueError(f"Element {index} has invalid concentration: {exc}") from exc
        if field.shape != shape or not np.all(np.isfinite(field)):
            raise ValueError(f"Element {index} concentration must be finite with shape {shape}.")
        refined = refine_gll_tensor3(field, target_node_count=plan.target_node_count)
        local = np.tensordot(plan.vertical_basis[row], refined, axes=(0, z_axis))
        local = np.transpose(local, transpose)
        if not plan.physical_axis_increasing[row, 1]:
            local = local[::-1, :]
        if not plan.physical_axis_increasing[row, 0]:
            local = local[:, ::-1]
        ix, iy, _ = plan.element_cell_indices[row]
        x_indices = plan.x_local_to_global[ix]
        y_indices = plan.y_local_to_global[iy]
        retained = y_indices < plan.y.size
        local = local[retained]
        output_index = np.ix_(y_indices[retained], x_indices)
        existing = values[output_index]
        overlap = np.isfinite(existing)
        if np.any(overlap):
            mismatch = float(np.max(np.abs(existing[overlap] - local[overlap])))
            maximum_mismatch = max(maximum_mismatch, mismatch)
            if not np.allclose(existing[overlap], local[overlap],
                               rtol=INTERFACE_VALUE_RTOL, atol=INTERFACE_VALUE_ATOL):
                raise ValueError(
                    f"Element {index} concentration disagrees across shared interfaces: "
                    f"maximum mismatch {mismatch:.6g}."
                )
        values[output_index] = np.where(overlap, existing, local)
    if not np.all(np.isfinite(values)):
        raise ValueError("Plane assembly produced non-finite or unfilled concentration values.")
    return RefinedGLLHorizontalSlice(
        x=plan.x, y=plan.y, concentration=values, z_target=plan.z_target,
        source_node_count=plan.source_node_count, target_node_count=plan.target_node_count,
        periodic_y=plan.periodic_y, y_period=plan.y_period,
        maximum_shared_interface_concentration_mismatch=maximum_mismatch,
    )
