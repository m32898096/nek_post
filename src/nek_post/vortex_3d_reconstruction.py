"""Element-bounded P7 visualization sampling on denser GLL nodal sets.

Projection remains on the complete native domain. Only sampling is denser;
neither the solution polynomial degree nor the mesh geometry class changes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from nek_post.gll import gll_interpolation_matrix
from nek_post.gll_refinement import refine_gll_tensor3
from nek_post.l2_projection import StructuredGLLNodeMap, _positive_integer
from nek_post.rectilinear_export import _rectilinear_geometry
from nek_post.swirling_strength import compute_swirling_strength
from nek_post.vortex_3d import _physical_hexes, validate_crop


def sampling_nodes(value):
    """None preserves Phase 1; explicit sampling is isotropic P7, never coarser."""
    return None if value is None else _positive_integer(value, 'sampling_nodes', 8)


def sampling_metadata(count):
    return dict(source_node_count=8, source_polynomial_degree=7,
                target_node_count=count, target_nodal_degree=count - 1,
                target_nodes='Legendre-Gauss-Lobatto on [-1,1]',
                reference_coordinates='endpoints -1,+1; interior roots of d/dq Legendre P_(N-1)(q)',
                interpolation='existing tensor-product barycentric P7 Lagrange',
                increases_simulation_resolution=False,
                lambda_ci='eigenvalues AFTER reconstructing all nine projected gradients',
                omega_x='reconstruct P(A[2,1]-A[1,2]); audit against curl of reconstructed P(A)')


def _canonical(array, mapping, ei):
    """View in increasing physical (z,y,x) order, retaining trailing components."""
    axes = tuple(mapping.physical_to_array_axes[i] for i in (2, 1, 0))
    result = array.transpose((*axes, *range(3, array.ndim)))
    for axis, physical in enumerate((2, 1, 0)):
        if not mapping.element_axis_increasing[ei, physical]:
            result = np.flip(result, axis=axis)
    return result


def _local(array, mapping, ei):
    result = array
    for axis, physical in enumerate((2, 1, 0)):
        if not mapping.element_axis_increasing[ei, physical]:
            result = np.flip(result, axis=axis)
    axes = np.argsort([mapping.physical_to_array_axes[i] for i in (2, 1, 0)])
    return result.transpose((*axes, *range(3, result.ndim)))


@dataclass(frozen=True)
class SamplingPlan:
    """Compact reusable geometry: O(sum(axis elements)*N + element count)."""
    node_map: StructuredGLLNodeMap
    count: int
    profiles: tuple
    # Per element: original-array-axis half-open target node spans, or None.
    spans: tuple
    point_offsets: tuple[int, ...]
    cell_offsets: tuple[int, ...]
    crop: tuple | None


def build_sampling_plan(data, mapping, count, *, crop=None):
    count = sampling_nodes(count)
    if count is None or mapping.element_shape != (8, 8, 8):
        raise ValueError('Reconstruction requires native P7 (8,8,8) elements and sampling_nodes >= 8.')
    mapping.validate_mesh(data)
    coordinates, _ = _rectilinear_geometry(data, mapping)
    bounds = validate_crop(crop)
    domain = np.asarray([(v[0], v[-1]) for v in coordinates])
    if bounds is not None and (np.any(bounds[:, 0] < domain[:, 0]) or
                               np.any(bounds[:, 1] > domain[:, 1])):
        raise ValueError('crop must lie within the physical domain.')
    matrix = gll_interpolation_matrix(8, count)
    profiles = []
    physical_spans = []
    for ax, vector in enumerate(coordinates):
        lines, slices = [], []
        for cell in range(mapping.element_interval_counts[ax]):
            # Exact rectilinearity makes this the factorized P7 geometry map,
            # including the original coordinate-storage roundoff polynomial.
            line = matrix @ vector[cell*7:cell*7+8]
            if not np.isfinite(line).all() or not np.all(np.diff(line) > 0):
                raise ValueError('Reconstructed coordinate profile must be finite and increasing.')
            line.setflags(write=False)
            lines.append(line)
            low, high = (0, count) if bounds is None else (
                int(np.searchsorted(line, bounds[ax, 0], side='left')),
                int(np.searchsorted(line, bounds[ax, 1], side='right')),
            )
            slices.append((low, high))
        profiles.append(tuple(lines))
        physical_spans.append(slices)
    spans, points, cells = [], [0], [0]
    for ei, cell in enumerate(mapping.element_cell_indices):
        local = [None] * 3
        for ax in range(3):
            low, high = physical_spans[ax][cell[ax]]
            if not mapping.element_axis_increasing[ei, ax]:
                low, high = count - high, count - low
            local[mapping.physical_to_array_axes[ax]] = (low, high)
        shape = tuple(high-low for low, high in local)
        active = min(shape) >= 2
        spans.append(tuple(local) if active else None)
        points.append(points[-1] + (int(np.prod(shape)) if active else 0))
        cells.append(cells[-1] + (int(np.prod(np.asarray(shape)-1)) if active else 0))
    if not cells[-1]:
        raise ValueError('crop contains no complete reconstructed 3D subcells.')
    return SamplingPlan(mapping, count, tuple(profiles), tuple(spans), tuple(points), tuple(cells),
                        None if bounds is None else tuple(bounds.ravel()))


def reconstruct_element(gradients, omega_x, mapping, ei, count, *, chunk_size=256):
    """One canonical element tensor, then eigenvalues; never interpolate lambda_ci.

    Canonical physical ordering fixes contraction order across reversed elements.
    The 2e-12 scaled curl tolerance matches the existing shared-trace audit and
    allows the distinct scalar/tensor projection summation orders. Nothing is
    averaged, clipped, filtered, or replaced by the alternate curl route.
    """
    source = _canonical(gradients, mapping, ei)
    tensor = np.empty((count, count, count, 3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            tensor[..., i, j] = refine_gll_tensor3(source[..., i, j], target_node_count=count)
    omega = refine_gll_tensor3(_canonical(omega_x, mapping, ei), target_node_count=count)
    alternate = tensor[..., 2, 1] - tensor[..., 1, 2]
    difference = float(np.max(np.abs(omega - alternate)))
    scale = max(1., float(np.max(np.abs(tensor))), float(np.max(np.abs(omega))))
    if not np.isfinite(omega).all() or difference > 2e-12 * scale:
        raise ValueError('Reconstructed projected-curl and projected-gradient omega_x disagree.')
    strength = compute_swirling_strength(tensor, chunk_size=chunk_size).lambda_ci
    return omega, strength, difference


def _neighbors(mapping):
    lookup = {tuple(cell): ei for ei, cell in enumerate(mapping.element_cell_indices)}
    for ei, cell in enumerate(mapping.element_cell_indices):
        for ax, name in enumerate('xyz'):
            other = list(cell)
            other[ax] += 1
            periodic = other[ax] == mapping.element_interval_counts[ax]
            if periodic:
                if name not in mapping.periodic_axes:
                    continue
                other[ax] = 0
            yield ei, lookup[tuple(other)], ax, periodic


def _face(array, ax, side):
    return np.take(array, side, axis=2-ax)


def _agree(a, b, label):
    if not np.isfinite(a).all() or not np.isfinite(b).all() or a.shape != b.shape:
        raise ValueError(f'Shared-face {label} are nonfinite or mismatched.')
    jump = float(np.max(np.abs(a-b)))
    if jump > 2e-12 * max(1., float(np.max(np.abs(a))), float(np.max(np.abs(b)))):
        raise ValueError(f'Shared-face {label} disagree; no merging or averaging is allowed.')
    return jump


def _check_native_faces(fields):
    """Audit full-domain projected traces, including periodic partners, before crop."""
    mapping = fields.node_map
    maximum = 0.
    for ei, other, ax, _ in _neighbors(mapping):
        for name in ('gradients', 'omega_x'):
            array = getattr(fields, name)
            a = _face(_canonical(array[ei], mapping, ei), ax, -1)
            b = _face(_canonical(array[other], mapping, other), ax, 0)
            maximum = max(maximum, _agree(a, b, name))
    return maximum


def _check_output_faces(plan, points, omega, strength):
    mapping = plan.node_map
    checked = periodic_checked = 0
    jumps = dict(coordinates=0., omega_x=0., lambda_ci=0.)

    def block(array, ei):
        shape = tuple(b-a for a, b in plan.spans[ei])
        local = array[plan.point_offsets[ei]:plan.point_offsets[ei+1]]
        return _canonical(local.reshape((*shape, *array.shape[1:])), mapping, ei)

    def boundary(ei, ax, high):
        low, end = plan.spans[ei][mapping.physical_to_array_axes[ax]]
        if not mapping.element_axis_increasing[ei, ax]:
            low, end = plan.count-end, plan.count-low
        return end == plan.count if high else low == 0

    for ei, other, ax, periodic in _neighbors(mapping):
        if plan.spans[ei] is None or plan.spans[other] is None:
            continue
        if not boundary(ei, ax, True) or not boundary(other, ax, False):
            continue
        a = _face(block(points, ei), ax, -1)
        b = _face(block(points, other), ax, 0)
        if periodic:
            # Normal coordinates intentionally differ by the original period.
            a, b = np.delete(a, ax, axis=-1), np.delete(b, ax, axis=-1)
        jumps['coordinates'] = max(jumps['coordinates'], _agree(a, b, 'coordinates'))
        for name, array in (('omega_x', omega), ('lambda_ci', strength)):
            a = _face(block(array, ei), ax, -1)
            b = _face(block(array, other), ax, 0)
            jumps[name] = max(jumps[name], _agree(a, b, name))
        checked += 1
        periodic_checked += int(periodic)
    return dict(checked_output_faces=checked, checked_periodic_output_faces=periodic_checked,
                maximum_output_face_differences=jumps)


def build_reconstructed_vortex_grid(data, fields, *, sampling_nodes, crop=None,
                                    chunk_size=256, progress=None):
    """Allocate output buffers once; reconstruct at most one N^3 tensor at a time.

    Native full-domain arrays still reside in memory. VTK topology uses separate
    offsets/IDs, avoiding packed nine-ID cells and full-domain list concatenation.
    The existing writer subsequently validates every cell and exact round-trip.
    """
    import pyvista as pv
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkUnstructuredGrid
    from vtkmodules.vtkCommonCore import vtkPoints

    chunk_size = _positive_integer(chunk_size, 'chunk_size')
    mapping = fields.node_map
    shape = (mapping.element_count, *mapping.element_shape)
    for name, expected in (('omega_x', shape), ('gradients', (*shape, 3, 3))):
        array = np.asarray(getattr(fields, name))
        if array.shape != expected or array.dtype != np.float64:
            raise ValueError(f'{name} must be float64 with shape {expected}.')
        for element in array:
            if not np.isfinite(element).all():
                raise ValueError(f'{name} must be finite on the complete native domain.')
    plan = build_sampling_plan(data, mapping, sampling_nodes, crop=crop)
    native_jump = _check_native_faces(fields)
    npnt, ncell = plan.point_offsets[-1], plan.cell_offsets[-1]
    points = np.empty((npnt, 3), dtype=np.float64)
    omega = np.empty(npnt, dtype=np.float64)
    strength = np.empty(npnt, dtype=np.float64)
    connectivity = np.empty((ncell, 8), dtype=np.int64)
    maximum_difference = 0.
    topology_cache = {}
    for ei, spans in enumerate(plan.spans):
        if spans is None:
            continue
        sampled_omega, sampled_strength, difference = reconstruct_element(
            fields.gradients[ei], fields.omega_x[ei], mapping, ei, plan.count,
            chunk_size=chunk_size,
        )
        maximum_difference = max(maximum_difference, difference)
        selection = tuple(slice(a, b) for a, b in spans)
        lo, hi = plan.point_offsets[ei:ei+2]
        omega[lo:hi] = _local(sampled_omega, mapping, ei)[selection].ravel()
        strength[lo:hi] = _local(sampled_strength, mapping, ei)[selection].ravel()
        local_shape = tuple(b-a for a, b in spans)
        for ax in range(3):
            axis = mapping.physical_to_array_axes[ax]
            line = plan.profiles[ax][mapping.element_cell_indices[ei, ax]]
            if not mapping.element_axis_increasing[ei, ax]:
                line = line[::-1]
            broadcast = [1, 1, 1]
            broadcast[axis] = local_shape[axis]
            points[lo:hi, ax] = np.broadcast_to(line[selection[axis]].reshape(broadcast), local_shape).ravel()
        key = (local_shape, tuple(mapping.element_axis_increasing[ei]))
        local_cells = topology_cache.get(key)
        if local_cells is None:
            local_cells = _physical_hexes(replace(mapping, element_shape=local_shape), ei)
            if len(topology_cache) < 8:
                topology_cache[key] = local_cells
        a, b = plan.cell_offsets[ei:ei+2]
        np.add(local_cells, lo, out=connectivity[a:b])
        if progress and (ei + 1) % 4096 == 0:
            progress(f'Reconstructed elements: {ei + 1}/{mapping.element_count}')
    diagnostics = _check_output_faces(plan, points, omega, strength)
    diagnostics.update(maximum_native_projected_face_difference=native_jump,
                       maximum_omega_route_difference=maximum_difference,
                       reconstruction_tensor_elements_at_once=1,
                       sampling=sampling_metadata(plan.count))
    vtk_points = vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points, deep=False))
    vtk_cells = vtkCellArray()
    offsets = np.arange(ncell+1, dtype=np.int64)
    offsets *= 8
    vtk_cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=False),
                      numpy_to_vtkIdTypeArray(connectivity.ravel(), deep=False))
    raw = vtkUnstructuredGrid()
    raw.SetPoints(vtk_points)
    raw.SetCells(pv.CellType.HEXAHEDRON, vtk_cells)
    grid = pv.wrap(raw)
    grid.point_data['omega_x'] = omega
    grid.point_data['lambda_ci'] = strength
    grid.set_active_scalars('omega_x')
    # VTK borrows these buffers. Retain explicit Python owners for the entire
    # grid lifetime, including serialization after this function has returned.
    object.__setattr__(grid, '_reconstruction_buffers', (points, connectivity, offsets, omega, strength))
    return grid, diagnostics
