"""Mesh/data inventory for a separately configured h-refinement study.

All headers are inspected; coordinates come from one representative frame.
Equal bounding boxes are necessary, but do not prove identical domain topology
or simulation physics. No solution field comparisons are performed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from nek_post.config import load_yaml
from nek_post.front_detection_io import discover_nek_frame_paths
from nek_post.io_nek import read_nek_file, read_nek_header
from nek_post.paths import ProjectPaths


@dataclass(frozen=True)
class HRefinementStudy:
    cases: tuple[str, ...]
    file_prefix: str
    expected_polynomial_order: int
    provisional_reference_case: str
    domain_atol: float

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> HRefinementStudy:
        cases = config.get("cases")
        if (not isinstance(cases, (list, tuple)) or len(cases) < 2
                or any(not isinstance(c, str) or not c for c in cases)
                or len(set(cases)) != len(cases)):
            raise ValueError("cases must contain at least two distinct case labels.")
        prefix = config.get("file_prefix")
        if not isinstance(prefix, str) or not prefix or any(c in prefix for c in "/\\"):
            raise ValueError("file_prefix must be a nonempty filename prefix.")
        order = config.get("expected_polynomial_order")
        if type(order) is not int or order < 1:
            raise ValueError("expected_polynomial_order must be a positive integer.")
        reference = config.get("provisional_reference_case")
        if reference not in cases:
            raise ValueError("provisional_reference_case must belong to cases.")
        atol = config.get("domain_atol", 1e-6)
        if isinstance(atol, bool) or not isinstance(atol, (int, float)) or not np.isfinite(atol) or atol < 0:
            raise ValueError("domain_atol must be finite and nonnegative.")
        return cls(tuple(cases), prefix, order, reference, float(atol))

    @classmethod
    def from_yaml(cls, path: str | Path) -> HRefinementStudy:
        return cls.from_mapping(load_yaml(path))


def inventory_case(label: str, directory: str | Path, *, file_prefix: str) -> dict[str, Any]:
    """Inventory all exact PREFIX.fNNNNN files and one coordinate-bearing mesh.

    Split-file dumps are rejected: pymech's single-file payload would otherwise
    contain unpopulated elements. Output spacing is not the solver timestep.
    """
    frames = discover_nek_frame_paths(directory, file_prefix=file_prefix)
    headers = [read_nek_header(frame.path) for frame in frames]
    for frame, header in zip(frames, headers):
        if header.nb_files != 1 or header.nb_elems_file != header.nb_elems:
            raise ValueError(f"{frame.path}: split-file dumps are not supported.")
        if header.nb_elems < 1 or any(n < 1 for n in header.orders):
            raise ValueError(f"{frame.path}: invalid mesh dimensions.")
        if not np.isfinite(header.time):
            raise ValueError(f"{frame.path}: nonfinite physical time.")
    signatures = {(tuple(h.orders), h.nb_elems) for h in headers}
    if len(signatures) != 1:
        raise ValueError(f"{label}: mesh order or element count changes across headers.")
    mesh_index = next((i for i, h in enumerate(headers) if h.nb_vars[0]), None)
    if mesh_index is None:
        raise ValueError(f"{label}: no coordinate-bearing field file available.")
    header = headers[mesh_index]
    skip_vars = ("ux", "uy", "uz", "pressure", "temperature") + tuple(
        f"s{i:02d}" for i in range(1, header.nb_vars[4] + 1)
    )
    data = read_nek_file(frames[mesh_index].path, skip_vars=skip_vars)
    shape = tuple(int(n) for n in header.orders[::-1])
    if len(data.elem) != header.nb_elems:
        raise ValueError(f"{label}: payload/header element counts disagree.")
    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    for element in data.elem:
        pos = np.asarray(element.pos)
        if pos.shape != (3, *shape) or not np.all(np.isfinite(pos)):
            raise ValueError(f"{label}: invalid element coordinate array.")
        lower = np.minimum(lower, pos.min(axis=(1, 2, 3)))
        upper = np.maximum(upper, pos.max(axis=(1, 2, 3)))
    del data
    nodes = tuple(int(n) for n in header.orders)
    active_orders = tuple(n - 1 for n in nodes[:header.nb_dims])
    times = np.array([h.time for h in headers], dtype=float)
    spacing = np.diff(times)
    increasing = bool(np.all(spacing > 0))
    warnings = []
    if len(spacing) and not increasing:
        warnings.append("Physical times are not strictly increasing in file-index order.")
    if any(b.index - a.index != 1 for a, b in zip(frames, frames[1:])):
        warnings.append("File indices contain gaps.")
    variable_codes = sorted({h.variables for h in headers})
    if len(variable_codes) > 1:
        warnings.append("Available variables differ across field headers.")
    return {
        "case": label, "data_directory": str(Path(directory)),
        "field_files": [f.path.name for f in frames], "field_file_count": len(frames),
        "first_file_index": frames[0].index, "last_file_index": frames[-1].index,
        "physical_time_range": [float(times.min()), float(times.max())],
        "physical_times": times.tolist(),
        "output_time_spacing": None if not len(spacing) or not increasing else {
            "median": float(np.median(spacing)), "min": float(spacing.min()),
            "max": float(spacing.max()),
            "uniform": bool(np.allclose(spacing, np.median(spacing), rtol=1e-6, atol=1e-12)),
        },
        "mesh_source_file": frames[mesh_index].path.name,
        "mesh_validation_scope": "all headers; coordinates from mesh_source_file only",
        "element_local_scalar_shape_zyx": list(shape),
        "element_coordinate_shape": [3, *shape],
        "gll_nodes_per_axis_xyz": list(nodes), "gll_nodes_per_element": int(np.prod(nodes)),
        "polynomial_orders_per_active_axis": list(active_orders),
        "polynomial_order": active_orders[0] if len(set(active_orders)) == 1 else None,
        "number_of_elements": int(header.nb_elems), "dimension": int(header.nb_dims),
        "domain_extents": {axis: [float(lo), float(hi)] for axis, lo, hi in zip("xyz", lower, upper)},
        "variable_codes": variable_codes,
        "solution_variables": sorted({name for h in headers for name, count in zip(
            ("velocity", "pressure", "temperature", "passive_scalars"), h.nb_vars[1:]) if count}),
        "variable_counts_by_code": {h.variables: list(h.nb_vars) for h in headers},
        "warnings": warnings,
    }


def inventory_study(study: HRefinementStudy, paths: ProjectPaths) -> dict[str, Any]:
    """Evaluate candidate order/reference against measured mesh metadata."""
    directories = [paths.case_dir(case) for case in study.cases]
    inventories = [inventory_case(c, d, file_prefix=study.file_prefix)
                   for c, d in zip(study.cases, directories)]
    first = inventories[0]
    same_order = all(i["polynomial_order"] is not None
                     and i["polynomial_order"] == first["polynomial_order"]
                     and i["dimension"] == first["dimension"] for i in inventories)
    expected = same_order and first["polynomial_order"] == study.expected_polynomial_order
    same_bounds = all(np.allclose(list(i["domain_extents"].values()),
                                 list(first["domain_extents"].values()),
                                 rtol=0, atol=study.domain_atol) for i in inventories)
    counts = [i["number_of_elements"] for i in inventories]
    increasing = all(a < b for a, b in zip(counts, counts[1:]))
    reference_index = study.cases.index(study.provisional_reference_case)
    reference_largest = all(counts[reference_index] > count for j, count in enumerate(counts)
                            if j != reference_index)
    return {"cases": inventories, "validation": {
        "same_polynomial_order": same_order, "matches_expected_order": bool(expected),
        "same_domain_extents": bool(same_bounds), "domain_atol": study.domain_atol,
        "element_counts_strictly_increasing": increasing,
        "provisional_reference_case": study.provisional_reference_case,
        "reference_has_unique_largest_element_count": reference_largest,
        "consistent_with_h_refinement": bool(expected and same_bounds and increasing and reference_largest),
        "limitations": ["Bounding boxes do not establish identical domain topology or local mesh spacing.",
                        "Coordinate invariance over time is assumed, not checked.",
                        "Physical parameters, boundary/initial conditions and solver settings are not verified.",
                        "Largest element count alone does not establish convergence or finest spacing everywhere."],
    }}
