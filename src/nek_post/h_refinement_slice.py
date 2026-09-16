"""Exact physical-plane sampling for h-refinement, without convergence metrics.

The Cartesian grid is supplied directly to the spectral evaluator. Thus plane
restriction and Cartesian sampling are one tensor evaluation, with no secondary
scattered interpolation and no extrapolation into unmapped regions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from nek_post.comparison import _pressure_fluctuation
from nek_post.fields import get_concentration, get_coordinates, get_pressure, get_speed, get_velocity
from nek_post.h_refinement import HRefinementStudy
from nek_post.interpolation import create_common_xz_grid, valid_common_mask
from nek_post.io_nek import read_nek_file, read_nek_header
from nek_post.paths import ProjectPaths
from nek_post.spectral_interpolation import (
    apply_spectral_slice_fields, build_spectral_slice_interpolation_plan,
)


@dataclass
class PhysicalPlaneSlice:
    Xi: np.ndarray
    Zi: np.ndarray
    fields: dict[str, dict[str, np.ndarray]]
    geometry_masks: dict[str, np.ndarray]
    common_masks: dict[str, np.ndarray]
    metadata: dict[str, Any]


def inspect_plane_geometry(data: Any, y_target: float) -> dict[str, Any]:
    """Measure bounds and test full stored GLL layers using unrounded coordinates.

    A matching layer requires *every* node on a reference-coordinate layer to
    have y exactly equal to the requested physical value. This is a diagnostic
    only; evaluation always uses the physical inverse map.
    """
    if not np.isfinite(y_target) or not data.elem:
        raise ValueError("A finite y_target and nonempty mesh are required.")
    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    intersecting = exact_layers = 0
    nearest_distance = np.inf
    for element in data.elem:
        pos = np.asarray(get_coordinates(element))
        if pos.ndim != 4 or pos.shape[0] != 3 or not np.all(np.isfinite(pos)):
            raise ValueError("Invalid mesh coordinate arrays.")
        lower = np.minimum(lower, pos.min(axis=(1, 2, 3)))
        upper = np.maximum(upper, pos.max(axis=(1, 2, 3)))
        y = pos[1]
        nearest_distance = min(nearest_distance, float(np.min(np.abs(y - y_target))))
        if float(y.min()) <= y_target <= float(y.max()):
            intersecting += 1
            exact_layers += int(any(np.any(np.all(
                np.moveaxis(y, axis, 0).reshape(y.shape[axis], -1) == y_target, axis=1
            )) for axis in range(3)))
    if not intersecting:
        raise ValueError(f"No elements intersect physical y={y_target}.")
    return {
        "domain_bounds_xyz": np.column_stack((lower, upper)).tolist(),
        "intersecting_element_count": intersecting,
        "elements_with_exact_stored_y_layer": exact_layers,
        "exact_stored_plane_in_all_intersecting_elements": exact_layers == intersecting,
        "minimum_stored_y_distance": nearest_distance,
    }


def common_plane_grid(geometries: Mapping[str, dict], *, y_target: float,
                      nx: int, nz: int, domain_atol: float = 1e-6):
    """Reject bounding-domain mismatches and grid the intersection of bounds."""
    if (not geometries or not np.isfinite(y_target) or not np.isfinite(domain_atol)
            or domain_atol < 0):
        raise ValueError("Finite plane/domain tolerance and nonempty geometries required.")
    if any(type(n) is not int or n < 2 for n in (nx, nz)):
        raise ValueError("nx and nz must be integers >= 2.")
    bounds = [np.asarray(g["domain_bounds_xyz"], dtype=float) for g in geometries.values()]
    for b in bounds:
        if b.shape != (3, 2) or not np.all(np.isfinite(b)) or np.any(b[:, 0] >= b[:, 1]):
            raise ValueError("Invalid domain bounds.")
        if not np.allclose(b, bounds[0], atol=domain_atol, rtol=0):
            raise ValueError("Domain mismatch between h-refinement cases.")
    if not max(b[1, 0] for b in bounds) <= y_target <= min(b[1, 1] for b in bounds):
        raise ValueError("Requested y plane lies outside the common physical domain.")
    slices = {c: {"x": b[0], "z": b[2]} for c, b in zip(geometries, bounds)}
    Xi, Zi, _, _, metadata = create_common_xz_grid(slices, nx, nz)
    return Xi, Zi, metadata


def evaluate_physical_plane(data: Any, Xi: np.ndarray, Zi: np.ndarray,
                            *, y_target: float = .75):
    """Evaluate C, u/v/w and p on physical targets, then derive speed."""
    plan = build_spectral_slice_interpolation_plan(
        data, nx=Xi.shape[1], nz=Xi.shape[0], y_target=y_target,
        target_grid=(Xi, Zi),
    )
    fields = apply_spectral_slice_fields(plan, data, {
        "C": get_concentration, "p": get_pressure,
        "u": lambda e: get_velocity(e)[0], "v": lambda e: get_velocity(e)[1],
        "w": lambda e: get_velocity(e)[2],
    })
    fields["speed"] = get_speed(fields["u"], fields["v"], fields["w"])
    diagnostics = asdict(plan.inverse_mapping_diagnostics)
    diagnostics["geometry_valid_fraction"] = float(np.mean(plan.target_valid_mask))
    diagnostics["extrapolated_point_count"] = 0
    return fields, plan.target_valid_mask, diagnostics


def finish_plane_slices(Xi, Zi, fields, geometry_masks, metadata) -> PhysicalPlaneSlice:
    """Apply common finite masks and the existing unweighted pressure mean rule."""
    masks = {name: valid_common_mask(*(f[name] for f in fields.values()))
             for name in ("C", "speed", "p")}
    masks["geometry"] = np.logical_and.reduce(list(geometry_masks.values()))
    masks["all_fields"] = np.logical_and.reduce(list(masks.values()))
    for name in ("C", "speed", "p"):
        if not np.any(masks[name]):
            raise ValueError(f"No common finite {name} targets on the physical plane.")
    for case, values in fields.items():
        pressure_mean = float(np.mean(values["p"][masks["p"]]))
        values["p_prime"] = _pressure_fluctuation(values["p"], masks["p"])
        metadata["cases"][case]["pressure_mean_on_common_mask"] = pressure_mean
        metadata["cases"][case]["nan_fractions"] = {
            name: float(np.mean(np.isnan(a))) for name, a in values.items()
        }
        metadata["cases"][case]["finite_fractions"] = {
            name: float(np.mean(np.isfinite(a))) for name, a in values.items()
        }
    metadata["common_valid_fractions"] = {k: float(np.mean(v)) for k, v in masks.items()}
    return PhysicalPlaneSlice(Xi, Zi, fields, geometry_masks, masks, metadata)


def sample_h_snapshot(study: HRefinementStudy, paths: ProjectPaths,
                      case_indices: Mapping[str, int], *, nx: int, nz: int,
                      y_target: float = .75, time_atol: float = 1e-8) -> PhysicalPlaneSlice:
    """Read a selected snapshot per case sequentially; reject time/domain mismatch.

    Geometry is read in a preliminary pass to define a genuine common grid;
    solution data is then read one case at a time. No temporal interpolation is
    performed. Call once per selected snapshot group.
    """
    if set(case_indices) != set(study.cases):
        raise ValueError("Snapshot selection must include exactly the study cases.")
    if not np.isfinite(time_atol) or time_atol < 0:
        raise ValueError("time_atol must be finite and nonnegative.")
    if any(type(i) is not int or not 0 <= i <= 99999 for i in case_indices.values()):
        raise ValueError("Snapshot indices must be integers in [0, 99999].")
    sources = {c: paths.case_dir(c) / f"{study.file_prefix}.f{case_indices[c]:05d}" for c in study.cases}
    headers = {c: read_nek_header(p) for c, p in sources.items()}
    for case, h in headers.items():
        if h.nb_files != 1 or h.nb_elems != h.nb_elems_file:
            raise ValueError(f"{case}: split-file snapshots are unsupported.")
        if tuple(h.orders) != (study.expected_polynomial_order + 1,) * 3:
            raise ValueError(f"{case}: snapshot does not have the expected 3D polynomial order.")
        if h.nb_vars[0] != 3 or h.nb_vars[1] != 3 or not h.nb_vars[2] or not (h.nb_vars[3] or h.nb_vars[4]):
            raise ValueError(f"{case}: snapshot lacks coordinates or required solution fields.")
    times = [float(h.time) for h in headers.values()]
    if not np.all(np.isfinite(times)) or max(times) - min(times) > time_atol:
        raise ValueError(f"Snapshot physical time mismatch (tolerance {time_atol}): {times}.")
    geometries = {}
    for case, path in sources.items():
        skip = ("ux", "uy", "uz", "pressure", "temperature") + tuple(
            f"s{i:02d}" for i in range(1, headers[case].nb_vars[4] + 1))
        data = read_nek_file(path, skip_vars=skip)
        geometries[case] = inspect_plane_geometry(data, y_target)
        del data
    Xi, Zi, grid = common_plane_grid(geometries, y_target=y_target, nx=nx, nz=nz,
                                    domain_atol=study.domain_atol)
    metadata = {
        "y_target": y_target, "grid": grid, "domain_atol": study.domain_atol,
        "time_atol": time_atol, "physical_time_spread": max(times) - min(times),
        "expected_polynomial_order": study.expected_polynomial_order,
        "physical_tolerance_factor": 1e-11, "reference_tolerance": 1e-8,
        "plane_evaluation": "physical-to-reference Newton inversion and tensor-product GLL barycentric interpolation",
        "cartesian_grid_sampling": "direct spectral evaluation on common targets; no second interpolation",
        "interface_policy": "minimum inverse residual, then element index; one owner per target",
        "pressure_definition": "p minus its arithmetic mean on the all-case finite-pressure mask",
        "cases": {},
    }
    fields, masks = {}, {}
    for case, path in sources.items():
        data = read_nek_file(path)
        # Check coordinates again: no silent domain change between passes.
        if inspect_plane_geometry(data, y_target) != geometries[case]:
            raise ValueError(f"{case}: geometry changed between inventory and evaluation.")
        fields[case], masks[case], diagnostics = evaluate_physical_plane(data, Xi, Zi, y_target=y_target)
        metadata["cases"][case] = dict(geometries[case], **diagnostics,
            source_file=str(path), file_index=case_indices[case], physical_time=float(headers[case].time))
        del data
    return finish_plane_slices(Xi, Zi, fields, masks, metadata)
