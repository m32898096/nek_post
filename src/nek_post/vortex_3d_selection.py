"""Physical volume policies and compact crop plans; no field calculations."""
from dataclasses import dataclass, replace
from math import prod

import numpy as np

from nek_post.rectilinear_export import _rectilinear_geometry
from nek_post.vortex_3d_artifacts import native_mesh_signature
from nek_post.vortex_3d_reconstruction import build_sampling_plan
from nek_post.vortex_3d import validate_crop


POSITIVE_X_BOUNDS = (0., 17., 0., 1.5, 0., 1.)


@dataclass(frozen=True)
class VolumeSelection:
    mode: str
    bounds: tuple | None

    def __post_init__(self):
        if self.mode not in ('full', 'crop', 'positive_x'):
            raise ValueError('Unsupported 3D volume selection mode.')
        try:
            values = None if self.bounds is None else tuple(self.bounds)
        except TypeError as exc:
            raise ValueError('Volume bounds must contain six finite values.') from exc
        checked = validate_crop(values)
        if (self.mode == 'full') != (checked is None):
            raise ValueError('Full-domain mode has no crop; other volume modes require six bounds.')
        object.__setattr__(self, 'bounds', None if checked is None else tuple(float(v) for v in checked.ravel()))
        if self.mode == 'positive_x' and self.bounds != POSITIVE_X_BOUNDS:
            raise ValueError('Positive-x volume must use the established physical bounds.')

    def metadata(self):
        return dict(mode=self.mode, bounds=None if self.bounds is None else list(self.bounds),
                    dataset='3D vtkUnstructuredGrid volume', fields=['omega_x', 'lambda_ci'],
                    cells='complete hexahedra; no clipping or crop-edge interpolation',
                    retain_x0_boundary_vertices=self.mode == 'positive_x',
                    require_element_boundaries=self.mode == 'positive_x')


def series_volume_selection(case, config, *, positive_x=False, full_domain=False):
    """N7_H defaults to its complete positive volume; generic cases stay native/full."""
    if type(positive_x) is not bool or type(full_domain) is not bool:
        raise ValueError('positive_x and full_domain must be booleans.')
    if (positive_x and full_domain) or (config.crop is not None and (positive_x or full_domain)):
        raise ValueError('Choose --positive-x, --full-domain, or --crop, not a combination.')
    intended = positive_x or config.crop == POSITIVE_X_BOUNDS or (
        case.name in ('case_N7_H', 'N7_H') and config.crop is None and not full_domain)
    if intended:
        return replace(config, crop=POSITIVE_X_BOUNDS), VolumeSelection('positive_x', POSITIVE_X_BOUNDS)
    return config, VolumeSelection('full' if config.crop is None else 'crop', config.crop)


def _counts_and_bounds(profiles, bounds):
    """Count an element-local tensor partition without constructing 3D points/IDs."""
    nodes, cells, elements, actual = [], [], [], []
    for ax, lines in enumerate(profiles):
        active = []
        for line in lines:
            keep = np.flatnonzero((line >= bounds[ax, 0]) & (line <= bounds[ax, 1]))
            if len(keep) >= 2:
                active.append(line[keep])
        if not active:
            raise ValueError('crop contains no complete 3D hexahedral cells.')
        nodes.append(sum(len(line) for line in active))
        cells.append(sum(len(line)-1 for line in active))
        elements.append(len(active))
        actual.extend((float(active[0][0]), float(active[-1][-1])))
    # Complete structured topology makes the selected set a Cartesian product.
    # Shared SEM faces remain duplicated: summing local profiles counts both ends.
    return prod(nodes), prod(cells), prod(elements), actual


def build_volume_output_plan(data, mapping, sampling_nodes, selection):
    """Reuse exact geometry and the validated P7 plan; return only small metadata."""
    mapping.validate_mesh(data)
    coordinates, _ = _rectilinear_geometry(data, mapping)
    domain = np.asarray([(v[0], v[-1]) for v in coordinates])
    bounds = domain if selection.bounds is None else np.asarray(selection.bounds, dtype=np.float64).reshape(3, 2)
    if np.any(bounds[:, 0] < domain[:, 0]) or np.any(bounds[:, 1] > domain[:, 1]):
        raise ValueError('Requested volume crop lies outside the physical domain.')
    native_xyz = tuple(mapping.element_shape[a] for a in mapping.physical_to_array_axes)
    if selection.mode == 'positive_x':
        if tuple(selection.bounds) != POSITIVE_X_BOUNDS:
            raise ValueError('The positive-x volume must use the established physical bounds.')
        if not np.array_equal(domain[1:], bounds[1:]):
            raise ValueError('Positive-x volume requires the complete y=[0,1.5] and z=[0,1] domain extents.')
        for ax in range(3):
            edges = coordinates[ax][::native_xyz[ax]-1]
            for value in bounds[ax]:
                if not np.any(edges == value):
                    raise ValueError(f'{"xyz"[ax]}={value:g} is not an element boundary; cannot retain the complete '
                                     'positive-x volume without clipping or losing a cell layer.')
    target = sampling_nodes if sampling_nodes is not None else mapping.element_shape[0]
    if mapping.element_shape == (8, 8, 8):
        # This is the same plan used by Phase 2, including original stored
        # coordinate polynomials, axis reversals and wholly contained subcells.
        plan = build_sampling_plan(data, mapping, target, crop=selection.bounds)
        profiles = plan.profiles
        selected_points, selected_cells = plan.point_offsets[-1], plan.cell_offsets[-1]
        points, cells, elements, actual = _counts_and_bounds(profiles, bounds)
        if (points, cells) != (selected_points, selected_cells):
            raise ValueError('Output plan and structured crop counts disagree.')
        target_shape = [target]*3
    else:
        if sampling_nodes is not None:
            raise ValueError('Explicit reconstructed sampling requires native P7 GLL8 elements.')
        profiles = tuple(tuple(v[i*(n-1):i*(n-1)+n] for i in range(mapping.element_interval_counts[ax]))
                         for ax, (v, n) in enumerate(zip(coordinates, native_xyz)))
        points, cells, elements, actual = _counts_and_bounds(profiles, bounds)
        target_shape = list(mapping.element_shape)
    if selection.mode == 'positive_x' and actual != list(POSITIVE_X_BOUNDS):
        raise ValueError('Positive-x selection would lose part of the requested 3D volume.')
    return dict(schema_version=1, selection=selection.metadata(), native_mesh_signature=native_mesh_signature(mapping),
                target_element_shape=target_shape, selected_element_count=elements,
                selected_local_point_count=points, selected_linear_hex_count=cells, actual_bounds=actual)


def apply_volume_output_plan(report, plan, config, selection):
    """Keep the full native projection allowance; reduce only output allocations."""
    expected_shape = [config.sampling_nodes]*3 if config.sampling_nodes is not None else report['element_local_shape']
    if (plan.get('schema_version') != 1 or plan.get('selection') != selection.metadata() or
            plan.get('target_element_shape') != expected_shape or
            plan.get('native_mesh_signature', {}).get('element_shape') != report['element_local_shape'] or
            plan['native_mesh_signature'].get('element_counts') != list(config.expected_element_counts)):
        raise ValueError('Output plan differs from the requested source shape, region or sampling.')
    for name, upper in (('selected_local_point_count', report['full_local_point_count']),
                        ('selected_linear_hex_count', report['full_linear_hex_count']),
                        ('selected_element_count', report['element_count'])):
        if type(plan.get(name)) is not int or not 0 < plan[name] <= upper:
            raise ValueError('Invalid selected output counts.')
    if 'output_plan' in report:
        if report['output_plan'] != plan:
            raise ValueError('Prepared output plans disagree.')
        return report
    previous = report['estimated_output_bytes']
    output = 40*plan['selected_local_point_count']+80*plan['selected_linear_hex_count']+1024**2
    report.update(output_plan=plan, output_selection=selection.metadata(),
                  selected_local_point_count=plan['selected_local_point_count'],
                  selected_linear_hex_count=plan['selected_linear_hex_count'],
                  selected_element_count=plan['selected_element_count'],
                  estimated_output_bytes=output, required_disk_bytes=2*output+1024**2,
                  estimated_ram_bytes=report['estimated_ram_bytes']-6*previous+6*output,
                  output_estimate_scope='selected volume; native gradient/projection allowance remains full-domain')
    report['resources_acceptable'] = (report['estimated_ram_bytes'] < report['available_ram_bytes'] and
                                      report['required_disk_bytes'] < report['available_disk_bytes'])
    return report


def validate_volume_grid(grid, selection, plan=None):
    if plan is not None and (grid.n_points != plan['selected_local_point_count'] or
                             grid.n_cells != plan['selected_linear_hex_count'] or
                             list(grid.bounds) != plan['actual_bounds']):
        raise ValueError('Exported volume differs from the prepared crop plan.')
    if selection.mode == 'positive_x' and list(grid.bounds) != list(POSITIVE_X_BOUNDS):
        raise ValueError('Export must retain the complete 3D positive-x volume and x=0 boundary vertices.')
