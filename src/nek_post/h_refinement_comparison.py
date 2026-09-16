"""Header-time selection and exact-plane h-refinement field comparisons."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from nek_post.comparison import compare_sampled_grids
from nek_post.front_detection_io import discover_nek_frame_paths
from nek_post.h_refinement import HRefinementStudy
from nek_post.h_refinement_slice import PhysicalPlaneSlice, sample_h_snapshot
from nek_post.io_nek import read_nek_header
from nek_post.paths import ProjectPaths

FIELDS = {"concentration": "C", "velocity_magnitude": "speed", "pressure_fluctuation": "p_prime"}


@dataclass(frozen=True)
class SnapshotTime:
    file_index: int
    physical_time: float
    source_file: str


@dataclass(frozen=True)
class SelectedTime:
    case: str
    target_time: float
    actual_time: float
    file_index: int
    time_error: float
    absolute_time_error: float
    source_file: str


def discover_snapshot_times(directory: Path, file_prefix: str) -> tuple[SnapshotTime, ...]:
    """Read physical times through the existing header adapter and file discovery."""
    return tuple(SnapshotTime(f.index, float(read_nek_header(f.path).time), str(f.path))
                 for f in discover_nek_frame_paths(directory, file_prefix=file_prefix))


def select_snapshot_time(case: str, snapshots: tuple[SnapshotTime, ...], target_time: float,
                         *, max_time_error: float) -> SelectedTime:
    """Nearest header time; ties choose earlier time then lower file index."""
    if not snapshots or not np.isfinite(target_time):
        raise ValueError("Finite target time and nonempty snapshot inventory required.")
    if not np.isfinite(max_time_error) or max_time_error < 0:
        raise ValueError("max_time_error must be finite and nonnegative.")
    times = [s.physical_time for s in snapshots]
    if not np.all(np.isfinite(times)):
        raise ValueError(f"{case}: nonfinite snapshot time.")
    if len({s.file_index for s in snapshots}) != len(snapshots):
        raise ValueError(f"{case}: duplicated snapshot index.")
    if not min(times) <= target_time <= max(times):
        raise ValueError(f"{case}: target time lies outside available physical times.")
    selected = min(snapshots, key=lambda s: (abs(s.physical_time-target_time), s.physical_time, s.file_index))
    error = selected.physical_time - target_time
    if abs(error) > max_time_error:
        raise ValueError(f"{case}: nearest snapshot time error {error} exceeds {max_time_error}.")
    return SelectedTime(case, float(target_time), selected.physical_time, selected.file_index,
                        error, abs(error), selected.source_file)


def validate_inventory_reference(study: HRefinementStudy, inventory: dict) -> dict[str, int]:
    """Require same degree/domain and a uniquely largest reference element count.

    This is a resolution ranking by element count, not proof of uniformly finer
    local spacing, identical physics, or an asymptotic convergence regime.
    """
    cases = inventory["cases"]
    if len(cases) != len(study.cases) or {c["case"] for c in cases} != set(study.cases):
        raise ValueError("Inventory must contain exactly the configured study cases.")
    checks = inventory["validation"]
    if not all(checks[k] for k in ("same_polynomial_order", "matches_expected_order", "same_domain_extents")):
        raise ValueError("Inventory does not establish common polynomial order and domain bounds.")
    counts = {c["case"]: int(c["number_of_elements"]) for c in cases}
    ref = study.provisional_reference_case
    if min(counts.values()) < 1 or any(counts[c] >= counts[ref] for c in counts if c != ref):
        raise ValueError("Provisional reference is not the uniquely largest mesh by element count.")
    return counts


def compare_h_slice(result: PhysicalPlaneSlice, selections: Mapping[str, SelectedTime],
                    reference_case: str, element_counts: Mapping[str, int]):
    """Use a per-field all-case finite mask intersected with all geometry masks."""
    if set(result.fields) != set(selections) or set(result.fields) != set(element_counts):
        raise ValueError("Slice, time selection and inventory cases must agree.")
    geometry_mask = np.logical_and.reduce(list(result.geometry_masks.values()))
    rows, masks = [], {}
    reference = selections[reference_case]
    for field, key in FIELDS.items():
        comparison = compare_sampled_grids(
            {c: f[key] for c, f in result.fields.items()}, reference_case, valid_mask=geometry_mask)
        masks[field] = comparison.common_mask
        for metrics in comparison.error_rows:
            case = metrics["case"]
            rows.append(dict(metrics))
            rows[-1].update(asdict(selections[case]))
            rows[-1].update({
                "field": field, "number_of_elements": element_counts[case],
                "reference_number_of_elements": element_counts[reference_case],
                "reference_actual_time": reference.actual_time,
                "reference_file_index": reference.file_index,
                "time_difference_from_reference": selections[case].actual_time-reference.actual_time,
                "y_target": result.metadata["y_target"], "nx": result.Xi.shape[1], "nz": result.Xi.shape[0],
                "xmin": float(result.Xi.min()), "xmax": float(result.Xi.max()),
                "zmin": float(result.Zi.min()), "zmax": float(result.Zi.max()),
                "mask_definition": "all_case_geometry_and_field_finite",
            })
    return rows, masks


def sample_and_compare(study: HRefinementStudy, paths: ProjectPaths,
                       selections: Mapping[str, SelectedTime], element_counts: Mapping[str, int],
                       *, nx: int, nz: int, max_time_spread: float):
    """Reuse the established physical-plane workflow for one target-time group."""
    result = sample_h_snapshot(study, paths, {c: s.file_index for c, s in selections.items()},
                               nx=nx, nz=nz, y_target=.75, time_atol=max_time_spread)
    for case, selection in selections.items():
        if result.metadata["cases"][case]["physical_time"] != selection.actual_time:
            raise ValueError(f"{case}: selected header time changed before sampling.")
    rows, masks = compare_h_slice(result, selections, study.provisional_reference_case, element_counts)
    return result, rows, masks
