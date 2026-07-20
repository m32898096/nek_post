"""Filesystem paths and artifact I/O for polynomial-order comparisons."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from nek_post.paths import ProjectPaths


CONCENTRATION_ERROR_COLUMNS = (
    "case",
    "order",
    "reference_case",
    "comparison_set",
    "index",
    "reference_index",
    "relative_L2_C",
    "mean_abs_error_C",
    "absolute_Linf_C",
    "relative_Linf_C",
    "valid_point_count",
    "total_grid_point_count",
)
FRONT_POSITION_COLUMNS = ("case", "order", "comparison_set", "index", "threshold", "x_front")
VELOCITY_ERROR_COLUMNS = (
    "case",
    "order",
    "reference_case",
    "comparison_set",
    "index",
    "reference_index",
    "relative_L2_speed",
    "mean_abs_error_speed",
    "absolute_Linf_speed",
    "relative_Linf_speed",
    "relative_L2_u",
    "mean_abs_error_u",
    "relative_L2_v",
    "mean_abs_error_v",
    "relative_L2_w",
    "mean_abs_error_w",
    "valid_point_count",
    "total_grid_point_count",
)
PRESSURE_ERROR_COLUMNS = (
    "case",
    "order",
    "reference_case",
    "comparison_set",
    "index",
    "reference_index",
    "relative_L2_p_prime",
    "mean_abs_error_p_prime",
    "absolute_Linf_p_prime",
    "relative_Linf_p_prime",
    "valid_point_count",
    "total_grid_point_count",
)


def slice_path(paths: ProjectPaths, case: str, index: int) -> Path:
    return paths.slices_dir / case / f"slice_{case}_f{index:05d}.npz"


def concentration_interpolated_path(paths: ProjectPaths, case: str, index: int) -> Path:
    return paths.interpolated_dir / "C" / f"interp_C_{case}_f{index:05d}.npz"


def velocity_interpolated_path(paths: ProjectPaths, case: str, index: int) -> Path:
    return paths.interpolated_dir / "velocity" / f"interp_velocity_{case}_f{index:05d}.npz"


def pressure_interpolated_path(paths: ProjectPaths, case: str, index: int) -> Path:
    return paths.interpolated_dir / "pressure" / f"interp_pressure_{case}_f{index:05d}.npz"


def concentration_error_table_path(paths: ProjectPaths, label: str) -> Path:
    return paths.tables_dir / f"concentration_error_{label}.csv"


def front_position_table_path(paths: ProjectPaths, label: str) -> Path:
    return paths.tables_dir / f"front_position_{label}.csv"


def velocity_error_table_path(paths: ProjectPaths, label: str) -> Path:
    return paths.tables_dir / f"velocity_error_{label}.csv"


def pressure_error_table_path(paths: ProjectPaths, label: str) -> Path:
    return paths.tables_dir / f"pressure_error_{label}.csv"


def comparison_metadata_path(paths: ProjectPaths, comparison_set_name: str) -> Path:
    return paths.tables_dir / f"comparison_set_{comparison_set_name}_metadata.txt"


def comparison_log_path(paths: ProjectPaths) -> Path:
    return paths.logs_dir / "compare_poly_orders.log"


def check_slice_files(paths: ProjectPaths, case_indices: Mapping[str, int]) -> dict[str, Path]:
    """Return required slice paths or raise with per-case regeneration guidance."""
    slice_paths = {case: slice_path(paths, case, index) for case, index in case_indices.items()}
    missing = {case: path for case, path in slice_paths.items() if not path.exists()}
    if missing:
        lines = ["Missing required slice files:"]
        for case, path in missing.items():
            index = case_indices[case]
            lines.append(f"  {case}: {path}")
            lines.append(f"  Run: python scripts/02_extract_midspan_slice.py --case {case} --index {index}")
        raise FileNotFoundError("\n".join(lines))
    return slice_paths


def load_slice_file(path: Path) -> dict[str, np.ndarray]:
    """Load an NPZ slice into a normal dictionary and close the archive."""
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def metadata_scalar(slice_data: Mapping[str, np.ndarray], name: str) -> object:
    if name not in slice_data:
        return None
    value = slice_data[name]
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def slice_time(slice_data: Mapping[str, np.ndarray]) -> float:
    value = metadata_scalar(slice_data, "time")
    if value is None:
        return float("nan")
    return float(value)


def save_concentration_interpolated(
    path: Path,
    Xi: np.ndarray,
    Zi: np.ndarray,
    C_grid: np.ndarray,
    case: str,
    index: int,
    source_slice_file: Path,
    interpolation_method: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        Xi=Xi,
        Zi=Zi,
        C_grid=C_grid,
        case=case,
        index=index,
        source_slice_file=str(source_slice_file),
        interpolation_method=interpolation_method,
    )


def save_velocity_interpolated(
    path: Path,
    Xi: np.ndarray,
    Zi: np.ndarray,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    w_grid: np.ndarray,
    speed_grid: np.ndarray,
    case: str,
    index: int,
    comparison_set: str | None,
    source_slice_file: Path,
    interpolation_method: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        Xi=Xi,
        Zi=Zi,
        u_grid=u_grid,
        v_grid=v_grid,
        w_grid=w_grid,
        speed_grid=speed_grid,
        case=case,
        index=index,
        comparison_set=comparison_set or "",
        source_slice_file=str(source_slice_file),
        interpolation_method=interpolation_method,
    )


def save_pressure_interpolated(
    path: Path,
    Xi: np.ndarray,
    Zi: np.ndarray,
    p_grid: np.ndarray,
    p_prime_grid: np.ndarray,
    case: str,
    index: int,
    comparison_set: str | None,
    source_slice_file: Path,
    interpolation_method: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        Xi=Xi,
        Zi=Zi,
        p_grid=p_grid,
        p_prime_grid=p_prime_grid,
        case=case,
        index=index,
        comparison_set=comparison_set or "",
        source_slice_file=str(source_slice_file),
        interpolation_method=interpolation_method,
    )


def _write_csv(path: Path, columns: tuple[str, ...], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_concentration_error_table(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    _write_csv(path, CONCENTRATION_ERROR_COLUMNS, rows)


def write_front_position_table(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    _write_csv(path, FRONT_POSITION_COLUMNS, rows)


def write_velocity_error_table(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    _write_csv(path, VELOCITY_ERROR_COLUMNS, rows)


def write_pressure_error_table(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    _write_csv(path, PRESSURE_ERROR_COLUMNS, rows)


def write_comparison_set_metadata(
    path: Path,
    comparison_set_name: str,
    target_time: object,
    reference_case: str,
    case_indices: Mapping[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"comparison_set: {comparison_set_name}",
        f"target_time: {target_time}",
        f"reference_case: {reference_case}",
        "case_indices:",
    ]
    for case, index in case_indices.items():
        lines.append(f"  {case}: {index}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_comparison_log(paths: ProjectPaths, text: str) -> None:
    path = comparison_log_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}]\n{text}\n\n")
