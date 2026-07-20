"""Pure text formatting for polynomial-order comparison summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path


def build_concentration_summary(
    reference_case: str,
    comparison_set_name: str | None,
    target_time: object,
    case_indices: Mapping[str, int],
    case_times: Mapping[str, float],
    time_differences: Mapping[str, float],
    grid_metadata: Mapping[str, float | int],
    valid_point_count: int,
    total_grid_point_count: int,
    error_table: Path,
    front_table: Path,
    error_rows: Sequence[Mapping[str, object]],
    orders: Mapping[str, int],
    warnings: Sequence[str],
) -> str:
    """Build the existing concentration summary without printing or writing it."""
    error_by_case = {row["case"]: row for row in error_rows}
    lines = [
        "Nek5000 polynomial-order concentration comparison",
        "",
        f"Comparison set: {comparison_set_name or 'custom'}",
        f"Target time: {target_time}",
        f"Reference case: {reference_case}",
        f"Grid size: {grid_metadata['nx']} x {grid_metadata['nz']}",
        f"Common x domain: {grid_metadata['xmin']} to {grid_metadata['xmax']}",
        f"Common z domain: {grid_metadata['zmin']} to {grid_metadata['zmax']}",
        f"Valid point count: {valid_point_count} / {total_grid_point_count}",
        f"Error table: {error_table}",
        f"Front position table: {front_table}",
        "",
        "Case time alignment:",
    ]
    for case in case_indices:
        pieces = [
            f"  {case}",
            f"order={orders[case]}",
            f"index={case_indices[case]}",
            f"time={case_times[case]}",
            f"time_difference={time_differences[case]}",
        ]
        if case in error_by_case:
            pieces.append(f"relative_L2_C={error_by_case[case]['relative_L2_C']}")
        else:
            pieces.append("reference")
        lines.append(", ".join(pieces))

    if warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"  {warning}" for warning in warnings)

    return "\n".join(lines)


def build_velocity_summary(
    comparison_set_name: str | None,
    reference_case: str,
    grid_metadata: Mapping[str, float | int],
    valid_point_count: int,
    total_grid_point_count: int,
    error_table: Path,
    error_rows: Sequence[Mapping[str, object]],
) -> str:
    """Build the existing velocity summary without printing or writing it."""
    lines = [
        "Nek5000 polynomial-order velocity comparison",
        "",
        f"Comparison set: {comparison_set_name or 'custom'}",
        "Field: velocity",
        f"Reference case: {reference_case}",
        f"Grid size: {grid_metadata['nx']} x {grid_metadata['nz']}",
        f"Valid point count: {valid_point_count} / {total_grid_point_count}",
        f"Velocity error table: {error_table}",
        "",
        "Speed errors relative to reference:",
    ]
    for row in error_rows:
        lines.append(f"  {row['case']}: relative_L2_speed={row['relative_L2_speed']}")
    return "\n".join(lines)


def build_pressure_summary(
    comparison_set_name: str | None,
    reference_case: str,
    grid_metadata: Mapping[str, float | int],
    valid_point_count: int,
    total_grid_point_count: int,
    error_table: Path,
    error_rows: Sequence[Mapping[str, object]],
) -> str:
    """Build the existing pressure summary without printing or writing it."""
    lines = [
        "Nek5000 polynomial-order pressure comparison",
        "",
        f"Comparison set: {comparison_set_name or 'custom'}",
        "Field: pressure",
        f"Reference case: {reference_case}",
        f"Grid size: {grid_metadata['nx']} x {grid_metadata['nz']}",
        f"Valid point count: {valid_point_count} / {total_grid_point_count}",
        f"Pressure error table: {error_table}",
        "",
        "Pressure fluctuation errors relative to reference:",
    ]
    for row in error_rows:
        lines.append(f"  {row['case']}: relative_L2_p_prime={row['relative_L2_p_prime']}")
    return "\n".join(lines)
