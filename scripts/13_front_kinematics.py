"""Analyze front-position kinematics from front_simple.dat files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

from nek_post.front_io import front_simple_path, parse_case_labels, read_front_simple_dat
from nek_post.front_kinematics import compute_kinematics
from nek_post.front_kinematics_io import (
    front_kinematics_summary_path,
    front_kinematics_timeseries_path,
    write_front_kinematics_summary_csv,
    write_front_kinematics_timeseries_csv,
)
from nek_post.front_kinematics_plotting import write_front_kinematics_plots
from nek_post.front_kinematics_reporting import (
    build_front_kinematics_summary_row,
    format_front_kinematics_summary_table,
)
from nek_post.paths import ProjectPaths, load_project_paths


def _parse_args(paths: ProjectPaths) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze front-position kinematics from front_simple.dat files.")
    parser.add_argument("--cases", default="N5,N7,N9", help="Comma-separated cases. Default: N5,N7,N9.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=paths.data_root,
        help=f"Data root. Default: {paths.data_root}",
    )
    parser.add_argument(
        "--filename",
        default="front_simple.dat",
        help="Front-position filename. Default: front_simple.dat.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.front_kinematics_dir,
        help=f"Output directory. Default: {paths.front_kinematics_dir}",
    )
    parser.add_argument("--smooth-window", type=int, default=11, help="Velocity smoothing window. Default: 11.")
    parser.add_argument(
        "--smooth-method",
        choices=("moving_average", "savgol"),
        default="moving_average",
        help="Velocity smoothing method. Default: moving_average.",
    )
    parser.add_argument("--savgol-polyorder", type=int, default=3, help="Savitzky-Golay polynomial order. Default: 3.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files.")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    return parser.parse_args()


def main() -> None:
    paths = load_project_paths(REPO_ROOT / "config" / "paths.yaml")
    args = _parse_args(paths)
    cases = parse_case_labels(args.cases)
    data_root = args.data_root.expanduser()
    output_dir = args.output_dir.expanduser()

    try:
        kinematics_by_case: dict[str, dict] = {}
        summary_rows: list[dict[str, str]] = []
        timeseries_paths: list[Path] = []

        print("Input files:")
        for case in cases:
            input_path = front_simple_path(data_root, case, args.filename)
            print(f"  {case}: {input_path}")
            front = read_front_simple_dat(input_path)
            kinematics = compute_kinematics(
                front,
                args.smooth_method,
                args.smooth_window,
                args.savgol_polyorder,
            )
            kinematics_by_case[case] = kinematics
            summary_rows.append(build_front_kinematics_summary_row(case, kinematics))

            csv_path = front_kinematics_timeseries_path(output_dir, case)
            write_front_kinematics_timeseries_csv(csv_path, kinematics, args.overwrite)
            timeseries_paths.append(csv_path)

        summary_path = front_kinematics_summary_path(output_dir)
        write_front_kinematics_summary_csv(summary_path, summary_rows, args.overwrite)

        figure_paths: list[Path] = []
        if not args.no_plots:
            figure_paths = write_front_kinematics_plots(output_dir, kinematics_by_case, args.overwrite)

        print(format_front_kinematics_summary_table(summary_rows))
        print(f"Output directory: {output_dir}")
        print(f"Summary CSV: {summary_path}")
        print("Timeseries CSV files:")
        for path in timeseries_paths:
            print(f"  {path}")
        if args.no_plots:
            print("Figures: skipped (--no-plots)")
        else:
            print("Figure files:")
            for path in figure_paths:
                print(f"  {path}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
