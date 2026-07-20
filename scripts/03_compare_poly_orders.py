"""Compare concentration slices across Nek5000 polynomial orders."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

from nek_post.config import load_project_config
from nek_post.interpolation import create_common_xz_grid, interpolate_to_grid, valid_common_mask
from nek_post.metrics import (
    absolute_linf_error,
    front_position,
    mean_absolute_error,
    relative_l2_error,
    relative_linf_error,
)
from nek_post.paths import ProjectPaths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare concentration slices across polynomial orders.")
    parser.add_argument("--comparison-set", help="Named comparison set from config/cases.yaml, for example t19p5.")
    parser.add_argument("--index", type=int, help="File index, for example 80 for slice_*_f00080.npz.")
    parser.add_argument(
        "--case-indices",
        help="Comma-separated per-case indices, for example N5=80,N7=80,N9=80,N11=40. Overrides --index.",
    )
    parser.add_argument(
        "--field",
        default="concentration",
        choices=("concentration", "velocity", "pressure"),
        help="Field to compare.",
    )
    parser.add_argument("--method", default="linear", help="Interpolation method passed to scipy.interpolate.griddata.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing interpolated .npz files.")
    parser.add_argument("--time-tolerance", type=float, default=0.05, help="Allowed physical-time mismatch.")
    parser.add_argument("--strict-time", action="store_true", help="Exit with code 1 when time mismatch exceeds tolerance.")
    parser.add_argument(
        "--front-threshold-ratio",
        type=float,
        default=0.01,
        help="Front threshold as a fraction of the reference concentration maximum.",
    )
    return parser.parse_args()


def _slice_path(paths: ProjectPaths, case: str, index: int) -> Path:
    return paths.slices_dir / case / f"slice_{case}_f{index:05d}.npz"


def _interpolated_path(paths: ProjectPaths, case: str, index: int) -> Path:
    return paths.interpolated_dir / "C" / f"interp_C_{case}_f{index:05d}.npz"


def _velocity_interpolated_path(paths: ProjectPaths, case: str, index: int) -> Path:
    return (
        paths.interpolated_dir
        / "velocity"
        / f"interp_velocity_{case}_f{index:05d}.npz"
    )


def _pressure_interpolated_path(paths: ProjectPaths, case: str, index: int) -> Path:
    return (
        paths.interpolated_dir
        / "pressure"
        / f"interp_pressure_{case}_f{index:05d}.npz"
    )


def _comparison_label(case_indices: dict[str, int], use_case_indices: bool) -> str:
    if not use_case_indices and len(set(case_indices.values())) == 1:
        index = next(iter(case_indices.values()))
        return f"f{index:05d}"

    parts = [f"{case}f{case_indices[case]:05d}" for case in case_indices]
    return "cases_" + "_".join(parts)


def _error_table_path(paths: ProjectPaths, label: str) -> Path:
    return paths.tables_dir / f"concentration_error_{label}.csv"


def _front_table_path(paths: ProjectPaths, label: str) -> Path:
    return paths.tables_dir / f"front_position_{label}.csv"


def _velocity_error_table_path(paths: ProjectPaths, label: str) -> Path:
    return paths.tables_dir / f"velocity_error_{label}.csv"


def _pressure_error_table_path(paths: ProjectPaths, label: str) -> Path:
    return paths.tables_dir / f"pressure_error_{label}.csv"


def _metadata_path(paths: ProjectPaths, comparison_set_name: str) -> Path:
    return paths.tables_dir / f"comparison_set_{comparison_set_name}_metadata.txt"


def _log_path(paths: ProjectPaths) -> Path:
    return paths.logs_dir / "compare_poly_orders.log"


def _append_log(paths: ProjectPaths, text: str) -> None:
    path = _log_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}]\n{text}\n\n")


def _parse_case_indices(raw: str, cases: list[str]) -> dict[str, int]:
    case_indices: dict[str, int] = {}
    for item in raw.split(","):
        if "=" not in item:
            raise ValueError(f"Invalid --case-indices item {item!r}; expected CASE=INDEX.")
        case, index_text = item.split("=", 1)
        case = case.strip()
        index_text = index_text.strip()
        if case not in cases:
            available = ", ".join(cases)
            raise ValueError(f"Unknown case {case!r} in --case-indices. Available cases: {available}")
        if case in case_indices:
            raise ValueError(f"Duplicate case {case!r} in --case-indices.")
        try:
            case_indices[case] = int(index_text)
        except ValueError as exc:
            raise ValueError(f"Invalid index {index_text!r} for case {case!r}.") from exc

    missing_cases = [case for case in cases if case not in case_indices]
    if missing_cases:
        missing = ", ".join(missing_cases)
        raise ValueError(f"--case-indices must include all configured cases. Missing: {missing}")

    return {case: case_indices[case] for case in cases}


def _ordered_case_indices(raw_case_indices: dict[str, int], cases: list[str]) -> dict[str, int]:
    missing_cases = [case for case in cases if case not in raw_case_indices]
    if missing_cases:
        missing = ", ".join(missing_cases)
        raise ValueError(f"Case indices must include all configured cases. Missing: {missing}")

    return {case: int(raw_case_indices[case]) for case in cases}


def _build_comparison_selection(
    args: argparse.Namespace,
    config: dict,
    cases: list[str],
) -> tuple[dict[str, int], str, str | None, object, bool]:
    if args.comparison_set:
        comparison_sets = config["cases"].get("comparison_sets", {})
        if args.comparison_set not in comparison_sets:
            available = ", ".join(sorted(comparison_sets)) or "none"
            raise ValueError(f"Unknown comparison set {args.comparison_set!r}. Available sets: {available}")

        comparison_set = comparison_sets[args.comparison_set]
        case_indices = _ordered_case_indices(comparison_set["case_indices"], cases)
        reference_case = comparison_set.get("reference_case") or config["cases"]["reference_case"]
        target_time = comparison_set.get("target_time")
        return case_indices, reference_case, args.comparison_set, target_time, True

    if args.case_indices:
        case_indices = _parse_case_indices(args.case_indices, cases)
        return case_indices, config["cases"]["reference_case"], None, None, True

    file_indices = config["cases"]["file_indices"]
    index = args.index if args.index is not None else file_indices[-1]
    return {case: index for case in cases}, config["cases"]["reference_case"], None, None, False


def _check_slice_files(project_paths: ProjectPaths, case_indices: dict[str, int]) -> dict[str, Path]:
    paths = {case: _slice_path(project_paths, case, index) for case, index in case_indices.items()}
    missing = {case: path for case, path in paths.items() if not path.exists()}
    if missing:
        lines = ["Missing required slice files:"]
        for case, path in missing.items():
            index = case_indices[case]
            lines.append(f"  {case}: {path}")
            lines.append(f"  Run: python scripts/02_extract_midspan_slice.py --case {case} --index {index}")
        raise FileNotFoundError("\n".join(lines))
    return paths


def _load_slice_file(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def _metadata_scalar(slice_data: dict[str, np.ndarray], name: str) -> object:
    if name not in slice_data:
        return None

    value = slice_data[name]
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _slice_time(slice_data: dict[str, np.ndarray]) -> float:
    value = _metadata_scalar(slice_data, "time")
    if value is None:
        return float("nan")
    return float(value)


def _save_interpolated(
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


def _save_velocity_interpolated(
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


def _save_pressure_interpolated(
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


def _write_error_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
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
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_front_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["case", "order", "comparison_set", "index", "threshold", "x_front"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_velocity_error_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
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
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_pressure_error_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
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
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _safe_relative_l2_error(values, reference, mask=None, denominator_tol: float = 1.0e-14) -> float:
    values_arr = np.asarray(values, dtype=float)
    ref_arr = np.asarray(reference, dtype=float)
    if mask is None:
        mask_arr = np.isfinite(values_arr) & np.isfinite(ref_arr)
    else:
        mask_arr = np.asarray(mask, dtype=bool) & np.isfinite(values_arr) & np.isfinite(ref_arr)

    if not np.any(mask_arr):
        return float("nan")

    denominator = float(np.sum(ref_arr[mask_arr] ** 2))
    if denominator <= denominator_tol:
        return float("nan")

    numerator = float(np.sum((values_arr[mask_arr] - ref_arr[mask_arr]) ** 2))
    return float(np.sqrt(numerator / denominator))


def _safe_relative_linf_error(values, reference, mask=None, denominator_tol: float = 1.0e-14) -> float:
    values_arr = np.asarray(values, dtype=float)
    ref_arr = np.asarray(reference, dtype=float)
    if mask is None:
        mask_arr = np.isfinite(values_arr) & np.isfinite(ref_arr)
    else:
        mask_arr = np.asarray(mask, dtype=bool) & np.isfinite(values_arr) & np.isfinite(ref_arr)

    if not np.any(mask_arr):
        return float("nan")

    denominator = float(np.max(np.abs(ref_arr[mask_arr])))
    if denominator <= denominator_tol:
        return float("nan")

    return float(np.max(np.abs(values_arr[mask_arr] - ref_arr[mask_arr])) / denominator)


def _write_comparison_set_metadata(
    path: Path,
    comparison_set_name: str,
    target_time: object,
    reference_case: str,
    case_indices: dict[str, int],
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


def _build_summary(
    reference_case: str,
    comparison_set_name: str | None,
    target_time: object,
    case_indices: dict[str, int],
    case_times: dict[str, float],
    time_differences: dict[str, float],
    grid_metadata: dict[str, float | int],
    valid_point_count: int,
    total_grid_point_count: int,
    error_table: Path,
    front_table: Path,
    error_rows: list[dict[str, object]],
    orders: dict[str, int],
    warnings: list[str],
) -> str:
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


def _build_velocity_summary(
    comparison_set_name: str | None,
    reference_case: str,
    grid_metadata: dict[str, float | int],
    valid_point_count: int,
    total_grid_point_count: int,
    error_table: Path,
    error_rows: list[dict[str, object]],
) -> str:
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


def _build_pressure_summary(
    comparison_set_name: str | None,
    reference_case: str,
    grid_metadata: dict[str, float | int],
    valid_point_count: int,
    total_grid_point_count: int,
    error_table: Path,
    error_rows: list[dict[str, object]],
) -> str:
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


def _pressure_fluctuation(p_grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    p_arr = np.asarray(p_grid, dtype=float)
    p_prime = p_arr.copy()
    p_prime -= float(np.mean(p_arr[mask]))
    p_prime[~np.isfinite(p_arr)] = np.nan
    return p_prime


def main() -> None:
    """Load slice files, interpolate concentration, and compare to the reference case."""
    args = _parse_args()
    config = load_project_config(
        REPO_ROOT / "config" / "paths.yaml",
        REPO_ROOT / "config" / "cases.yaml",
    )
    paths = ProjectPaths.from_mapping(config["paths"])

    try:
        cases = list(config["cases"]["orders"].keys())
        case_indices, reference_case, comparison_set_name, target_time, use_case_indices = _build_comparison_selection(
            args,
            config,
            cases,
        )
        if reference_case not in cases:
            raise ValueError(f"Reference case {reference_case!r} is not present in cases.orders.")

        nx = int(config["cases"]["grid"]["nx"])
        nz = int(config["cases"]["grid"]["nz"])
        duplicate_decimals = int(config["cases"].get("slice", {}).get("y_round_decimals", 10))

        slice_paths = _check_slice_files(paths, case_indices)
        slice_data_by_case = {case: _load_slice_file(path) for case, path in slice_paths.items()}
        case_times = {case: _slice_time(slice_data_by_case[case]) for case in cases}
        reference_time = case_times[reference_case]
        time_differences = {case: abs(case_times[case] - reference_time) for case in cases}
        warnings = []
        for case in cases:
            time_difference = time_differences[case]
            if not np.isfinite(time_difference):
                warnings.append(f"Missing or invalid time metadata for {case}; time alignment could not be checked.")
            elif time_difference > args.time_tolerance:
                warnings.append(
                    f"{case} time differs from {reference_case} by {time_difference}, "
                    f"exceeding tolerance {args.time_tolerance}."
                )

        if warnings and args.strict_time:
            summary = "\n".join(["Strict time alignment failed:", *warnings])
            print(summary, file=sys.stderr)
            _append_log(paths, summary)
            raise SystemExit(1)

        Xi, Zi, _, _, grid_metadata = create_common_xz_grid(slice_data_by_case, nx, nz)

        if args.field == "velocity":
            velocity_grids: dict[str, dict[str, np.ndarray]] = {}
            for case in cases:
                slice_data = slice_data_by_case[case]
                u_grid = interpolate_to_grid(
                    slice_data["x"],
                    slice_data["z"],
                    slice_data["u"],
                    Xi,
                    Zi,
                    method=args.method,
                    duplicate_decimals=duplicate_decimals,
                )
                v_grid = interpolate_to_grid(
                    slice_data["x"],
                    slice_data["z"],
                    slice_data["v"],
                    Xi,
                    Zi,
                    method=args.method,
                    duplicate_decimals=duplicate_decimals,
                )
                w_grid = interpolate_to_grid(
                    slice_data["x"],
                    slice_data["z"],
                    slice_data["w"],
                    Xi,
                    Zi,
                    method=args.method,
                    duplicate_decimals=duplicate_decimals,
                )
                speed_grid = np.sqrt(u_grid**2 + v_grid**2 + w_grid**2)
                velocity_grids[case] = {
                    "u": u_grid,
                    "v": v_grid,
                    "w": w_grid,
                    "speed": speed_grid,
                }

                interp_path = _velocity_interpolated_path(paths, case, case_indices[case])
                if interp_path.exists() and not args.overwrite:
                    continue
                _save_velocity_interpolated(
                    interp_path,
                    Xi,
                    Zi,
                    u_grid,
                    v_grid,
                    w_grid,
                    speed_grid,
                    case,
                    case_indices[case],
                    comparison_set_name,
                    slice_paths[case],
                    args.method,
                )

            reference_speed = velocity_grids[reference_case]["speed"]
            common_mask = valid_common_mask(*(velocity_grids[case]["speed"] for case in cases))
            valid_point_count = int(np.count_nonzero(common_mask))
            total_grid_point_count = int(common_mask.size)
            if valid_point_count == 0:
                raise ValueError("No finite common grid points are available for velocity comparison.")

            error_rows = []
            for case in cases:
                if case == reference_case:
                    continue
                row = {
                    "case": case,
                    "order": config["cases"]["orders"][case],
                    "reference_case": reference_case,
                    "comparison_set": comparison_set_name or "",
                    "index": case_indices[case],
                    "reference_index": case_indices[reference_case],
                    "relative_L2_speed": relative_l2_error(
                        velocity_grids[case]["speed"],
                        reference_speed,
                        common_mask,
                    ),
                    "mean_abs_error_speed": mean_absolute_error(
                        velocity_grids[case]["speed"] - reference_speed,
                        common_mask,
                    ),
                    "absolute_Linf_speed": absolute_linf_error(
                        velocity_grids[case]["speed"],
                        reference_speed,
                        common_mask,
                    ),
                    "relative_Linf_speed": relative_linf_error(
                        velocity_grids[case]["speed"],
                        reference_speed,
                        common_mask,
                    ),
                    "relative_L2_u": _safe_relative_l2_error(
                        velocity_grids[case]["u"],
                        velocity_grids[reference_case]["u"],
                        common_mask,
                    ),
                    "mean_abs_error_u": mean_absolute_error(
                        velocity_grids[case]["u"] - velocity_grids[reference_case]["u"],
                        common_mask,
                    ),
                    "relative_L2_v": _safe_relative_l2_error(
                        velocity_grids[case]["v"],
                        velocity_grids[reference_case]["v"],
                        common_mask,
                    ),
                    "mean_abs_error_v": mean_absolute_error(
                        velocity_grids[case]["v"] - velocity_grids[reference_case]["v"],
                        common_mask,
                    ),
                    "relative_L2_w": _safe_relative_l2_error(
                        velocity_grids[case]["w"],
                        velocity_grids[reference_case]["w"],
                        common_mask,
                    ),
                    "mean_abs_error_w": mean_absolute_error(
                        velocity_grids[case]["w"] - velocity_grids[reference_case]["w"],
                        common_mask,
                    ),
                    "valid_point_count": valid_point_count,
                    "total_grid_point_count": total_grid_point_count,
                }
                error_rows.append(row)

            output_label = comparison_set_name or _comparison_label(case_indices, use_case_indices)
            error_table = _velocity_error_table_path(paths, output_label)
            _write_velocity_error_table(error_table, error_rows)

            summary = _build_velocity_summary(
                comparison_set_name,
                reference_case,
                grid_metadata,
                valid_point_count,
                total_grid_point_count,
                error_table,
                error_rows,
            )
            print(summary)
            _append_log(paths, summary)
            return

        if args.field == "pressure":
            pressure_grids: dict[str, dict[str, np.ndarray]] = {}
            for case in cases:
                slice_data = slice_data_by_case[case]
                p_grid = interpolate_to_grid(
                    slice_data["x"],
                    slice_data["z"],
                    slice_data["p"],
                    Xi,
                    Zi,
                    method=args.method,
                    duplicate_decimals=duplicate_decimals,
                )
                pressure_grids[case] = {"p": p_grid}

            common_mask = valid_common_mask(*(pressure_grids[case]["p"] for case in cases))
            valid_point_count = int(np.count_nonzero(common_mask))
            total_grid_point_count = int(common_mask.size)
            if valid_point_count == 0:
                raise ValueError("No finite common grid points are available for pressure comparison.")

            for case in cases:
                pressure_grids[case]["p_prime"] = _pressure_fluctuation(pressure_grids[case]["p"], common_mask)
                interp_path = _pressure_interpolated_path(paths, case, case_indices[case])
                if interp_path.exists() and not args.overwrite:
                    continue
                _save_pressure_interpolated(
                    interp_path,
                    Xi,
                    Zi,
                    pressure_grids[case]["p"],
                    pressure_grids[case]["p_prime"],
                    case,
                    case_indices[case],
                    comparison_set_name,
                    slice_paths[case],
                    args.method,
                )

            reference_p_prime = pressure_grids[reference_case]["p_prime"]
            error_rows = []
            for case in cases:
                if case == reference_case:
                    continue
                row = {
                    "case": case,
                    "order": config["cases"]["orders"][case],
                    "reference_case": reference_case,
                    "comparison_set": comparison_set_name or "",
                    "index": case_indices[case],
                    "reference_index": case_indices[reference_case],
                    "relative_L2_p_prime": _safe_relative_l2_error(
                        pressure_grids[case]["p_prime"],
                        reference_p_prime,
                        common_mask,
                    ),
                    "mean_abs_error_p_prime": mean_absolute_error(
                        pressure_grids[case]["p_prime"] - reference_p_prime,
                        common_mask,
                    ),
                    "absolute_Linf_p_prime": absolute_linf_error(
                        pressure_grids[case]["p_prime"],
                        reference_p_prime,
                        common_mask,
                    ),
                    "relative_Linf_p_prime": _safe_relative_linf_error(
                        pressure_grids[case]["p_prime"],
                        reference_p_prime,
                        common_mask,
                    ),
                    "valid_point_count": valid_point_count,
                    "total_grid_point_count": total_grid_point_count,
                }
                error_rows.append(row)

            output_label = comparison_set_name or _comparison_label(case_indices, use_case_indices)
            error_table = _pressure_error_table_path(paths, output_label)
            _write_pressure_error_table(error_table, error_rows)

            summary = _build_pressure_summary(
                comparison_set_name,
                reference_case,
                grid_metadata,
                valid_point_count,
                total_grid_point_count,
                error_table,
                error_rows,
            )
            print(summary)
            _append_log(paths, summary)
            return

        c_grids: dict[str, np.ndarray] = {}
        for case in cases:
            slice_data = slice_data_by_case[case]
            C_grid = interpolate_to_grid(
                slice_data["x"],
                slice_data["z"],
                slice_data["C"],
                Xi,
                Zi,
                method=args.method,
                duplicate_decimals=duplicate_decimals,
            )
            c_grids[case] = C_grid

            interp_path = _interpolated_path(paths, case, case_indices[case])
            if interp_path.exists() and not args.overwrite:
                continue
            _save_interpolated(interp_path, Xi, Zi, C_grid, case, case_indices[case], slice_paths[case], args.method)

        reference_grid = c_grids[reference_case]
        common_mask = valid_common_mask(*(c_grids[case] for case in cases))
        valid_point_count = int(np.count_nonzero(common_mask))
        total_grid_point_count = int(common_mask.size)
        if valid_point_count == 0:
            raise ValueError("No finite common grid points are available for comparison.")

        error_rows = []
        for case in cases:
            if case == reference_case:
                continue
            row = {
                "case": case,
                "order": config["cases"]["orders"][case],
                "reference_case": reference_case,
                "comparison_set": comparison_set_name or "",
                "index": case_indices[case],
                "reference_index": case_indices[reference_case],
                "relative_L2_C": relative_l2_error(c_grids[case], reference_grid, common_mask),
                "mean_abs_error_C": mean_absolute_error(c_grids[case] - reference_grid, common_mask),
                "absolute_Linf_C": absolute_linf_error(c_grids[case], reference_grid, common_mask),
                "relative_Linf_C": relative_linf_error(c_grids[case], reference_grid, common_mask),
                "valid_point_count": valid_point_count,
                "total_grid_point_count": total_grid_point_count,
            }
            error_rows.append(row)

        threshold = float(args.front_threshold_ratio * np.max(reference_grid[common_mask]))
        front_rows = []
        for case in cases:
            front_rows.append(
                {
                    "case": case,
                    "order": config["cases"]["orders"][case],
                    "comparison_set": comparison_set_name or "",
                    "index": case_indices[case],
                    "threshold": threshold,
                    "x_front": front_position(Xi, c_grids[case], threshold, common_mask),
                }
            )

        output_label = comparison_set_name or _comparison_label(case_indices, use_case_indices)
        error_table = _error_table_path(paths, output_label)
        front_table = _front_table_path(paths, output_label)
        _write_error_table(error_table, error_rows)
        _write_front_table(front_table, front_rows)
        if comparison_set_name:
            _write_comparison_set_metadata(
                _metadata_path(paths, comparison_set_name),
                comparison_set_name,
                target_time,
                reference_case,
                case_indices,
            )

        summary = _build_summary(
            reference_case,
            comparison_set_name,
            target_time,
            case_indices,
            case_times,
            time_differences,
            grid_metadata,
            valid_point_count,
            total_grid_point_count,
            error_table,
            front_table,
            error_rows,
            config["cases"]["orders"],
            warnings,
        )
        print(summary)
        _append_log(paths, summary)
    except Exception as exc:
        message = f"ERROR: {exc}"
        print(message, file=sys.stderr)
        try:
            _append_log(paths, message)
        except OSError as log_exc:
            print(f"ERROR: Failed to write log file: {log_exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
