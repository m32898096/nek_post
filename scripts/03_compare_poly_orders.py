"""Compare concentration slices across Nek5000 polynomial orders."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

from nek_post.comparison import (
    compare_concentration_slices,
    compare_pressure_slices,
    compare_velocity_slices,
)
from nek_post.comparison_io import (
    append_comparison_log,
    check_slice_files,
    comparison_metadata_path,
    concentration_error_table_path,
    concentration_interpolated_path,
    front_position_table_path,
    load_slice_file,
    pressure_error_table_path,
    pressure_interpolated_path,
    save_concentration_interpolated,
    save_pressure_interpolated,
    save_velocity_interpolated,
    velocity_error_table_path,
    velocity_interpolated_path,
    write_comparison_set_metadata,
    write_concentration_error_table,
    write_front_position_table,
    write_pressure_error_table,
    write_velocity_error_table,
)
from nek_post.comparison_reporting import (
    build_concentration_summary,
    build_pressure_summary,
    build_velocity_summary,
)
from nek_post.comparison_selection import evaluate_time_alignment, resolve_comparison_selection
from nek_post.config import load_project_config
from nek_post.interpolation import create_common_xz_grid
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


def main() -> None:
    """Load slices, dispatch comparison kernels, and write established artifacts."""
    args = _parse_args()
    config = load_project_config(
        REPO_ROOT / "config" / "paths.yaml",
        REPO_ROOT / "config" / "cases.yaml",
    )
    paths = ProjectPaths.from_mapping(config["paths"])

    try:
        cases_config = config["cases"]
        orders = cases_config["orders"]
        cases = list(orders.keys())
        selection = resolve_comparison_selection(
            comparison_set_name=args.comparison_set,
            raw_case_indices=args.case_indices,
            index=args.index,
            cases=cases,
            cases_config=cases_config,
        )

        nx = int(cases_config["grid"]["nx"])
        nz = int(cases_config["grid"]["nz"])
        duplicate_decimals = int(cases_config.get("slice", {}).get("y_round_decimals", 10))

        slice_paths = check_slice_files(paths, selection.case_indices)
        slice_data_by_case = {case: load_slice_file(path) for case, path in slice_paths.items()}
        alignment = evaluate_time_alignment(
            slice_data_by_case,
            cases,
            selection.reference_case,
            args.time_tolerance,
        )

        if alignment.warnings and args.strict_time:
            summary = "\n".join(["Strict time alignment failed:", *alignment.warnings])
            print(summary, file=sys.stderr)
            append_comparison_log(paths, summary)
            raise SystemExit(1)

        Xi, Zi, _, _, grid_metadata = create_common_xz_grid(slice_data_by_case, nx, nz)

        if args.field == "velocity":
            result = compare_velocity_slices(
                slice_data_by_case=slice_data_by_case,
                cases=cases,
                orders=orders,
                case_indices=selection.case_indices,
                reference_case=selection.reference_case,
                comparison_set_name=selection.comparison_set_name,
                Xi=Xi,
                Zi=Zi,
                interpolation_method=args.method,
                duplicate_decimals=duplicate_decimals,
            )
            for case in cases:
                output_path = velocity_interpolated_path(paths, case, selection.case_indices[case])
                if output_path.exists() and not args.overwrite:
                    continue
                grids = result.grids[case]
                save_velocity_interpolated(
                    output_path,
                    Xi,
                    Zi,
                    grids["u"],
                    grids["v"],
                    grids["w"],
                    grids["speed"],
                    case,
                    selection.case_indices[case],
                    selection.comparison_set_name,
                    slice_paths[case],
                    args.method,
                )

            error_table = velocity_error_table_path(paths, selection.output_label)
            write_velocity_error_table(error_table, result.error_rows)
            summary = build_velocity_summary(
                selection.comparison_set_name,
                selection.reference_case,
                grid_metadata,
                result.valid_point_count,
                result.total_grid_point_count,
                error_table,
                result.error_rows,
            )
            print(summary)
            append_comparison_log(paths, summary)
            return

        if args.field == "pressure":
            result = compare_pressure_slices(
                slice_data_by_case=slice_data_by_case,
                cases=cases,
                orders=orders,
                case_indices=selection.case_indices,
                reference_case=selection.reference_case,
                comparison_set_name=selection.comparison_set_name,
                Xi=Xi,
                Zi=Zi,
                interpolation_method=args.method,
                duplicate_decimals=duplicate_decimals,
            )
            for case in cases:
                output_path = pressure_interpolated_path(paths, case, selection.case_indices[case])
                if output_path.exists() and not args.overwrite:
                    continue
                grids = result.grids[case]
                save_pressure_interpolated(
                    output_path,
                    Xi,
                    Zi,
                    grids["p"],
                    grids["p_prime"],
                    case,
                    selection.case_indices[case],
                    selection.comparison_set_name,
                    slice_paths[case],
                    args.method,
                )

            error_table = pressure_error_table_path(paths, selection.output_label)
            write_pressure_error_table(error_table, result.error_rows)
            summary = build_pressure_summary(
                selection.comparison_set_name,
                selection.reference_case,
                grid_metadata,
                result.valid_point_count,
                result.total_grid_point_count,
                error_table,
                result.error_rows,
            )
            print(summary)
            append_comparison_log(paths, summary)
            return

        result = compare_concentration_slices(
            slice_data_by_case=slice_data_by_case,
            cases=cases,
            orders=orders,
            case_indices=selection.case_indices,
            reference_case=selection.reference_case,
            comparison_set_name=selection.comparison_set_name,
            Xi=Xi,
            Zi=Zi,
            interpolation_method=args.method,
            duplicate_decimals=duplicate_decimals,
            front_threshold_ratio=args.front_threshold_ratio,
        )
        for case in cases:
            output_path = concentration_interpolated_path(paths, case, selection.case_indices[case])
            if output_path.exists() and not args.overwrite:
                continue
            save_concentration_interpolated(
                output_path,
                Xi,
                Zi,
                result.grids[case],
                case,
                selection.case_indices[case],
                slice_paths[case],
                args.method,
            )

        error_table = concentration_error_table_path(paths, selection.output_label)
        front_table = front_position_table_path(paths, selection.output_label)
        write_concentration_error_table(error_table, result.error_rows)
        write_front_position_table(front_table, result.front_rows)
        if selection.comparison_set_name:
            write_comparison_set_metadata(
                comparison_metadata_path(paths, selection.comparison_set_name),
                selection.comparison_set_name,
                selection.target_time,
                selection.reference_case,
                selection.case_indices,
            )

        summary = build_concentration_summary(
            selection.reference_case,
            selection.comparison_set_name,
            selection.target_time,
            selection.case_indices,
            alignment.case_times,
            alignment.time_differences,
            grid_metadata,
            result.valid_point_count,
            result.total_grid_point_count,
            error_table,
            front_table,
            result.error_rows,
            orders,
            alignment.warnings,
        )
        print(summary)
        append_comparison_log(paths, summary)
    except Exception as exc:
        message = f"ERROR: {exc}"
        print(message, file=sys.stderr)
        try:
            append_comparison_log(paths, message)
        except OSError as log_exc:
            print(f"ERROR: Failed to write log file: {log_exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
