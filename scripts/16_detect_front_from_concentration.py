"""Detect the N7 gravity-current front and compare it with an external curve."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]

from nek_post.config import load_yaml
from nek_post.front_detection import track_concentration_front
from nek_post.front_detection_compare import (
    build_front_detection_summary,
    compare_detected_front_to_reference,
)
from nek_post.front_detection_io import (
    detected_front_timeseries_path,
    discover_nek_frame_paths,
    format_front_detection_summary_table,
    front_detection_comparison_path,
    front_detection_difference_path,
    front_detection_overlay_path,
    front_detection_summary_path,
    preflight_output_paths,
    write_front_detection_csvs,
)
from nek_post.front_detection_plotting import write_front_detection_plots
from nek_post.front_detection_workflow import build_concentration_sequence
from nek_post.front_io import read_front_simple_dat
from nek_post.paths import ProjectPaths, load_project_paths


def _mapping_value(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(f"Missing required cases.yaml key {context}.{key}.") from exc


def _parse_args(
    paths: ProjectPaths,
    cases_config: Mapping[str, Any],
) -> argparse.Namespace:
    grid = _mapping_value(cases_config, "grid", "cases")
    slice_config = _mapping_value(cases_config, "slice", "cases")
    parser = argparse.ArgumentParser(
        description=(
            "Detect a gravity-current front from concentration, then compare it "
            "post hoc with an external front_simple.dat curve."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--case", default="N7", help="Configured case label.")
    parser.add_argument(
        "--file-prefix",
        default=_mapping_value(cases_config, "file_prefix", "cases"),
        help="Exact Nek filename prefix before .fNNNNN.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        help="Inclusive first Nek file index; omit for no lower bound.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        help="Inclusive last Nek file index; omit for no upper bound.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Fixed absolute concentration threshold.",
    )
    parser.add_argument(
        "--min-component-pixels",
        type=int,
        default=50,
        help="Minimum bottom-connected component size.",
    )
    parser.add_argument(
        "--bottom-rows",
        type=int,
        default=3,
        help="Number of lowest grid z levels treated as bottom contact.",
    )
    parser.add_argument(
        "--max-front-jump",
        type=float,
        default=0.5,
        help="Maximum x-distance from the predicted front, not a velocity.",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(4, 8),
        default=8,
        help="Connected-component neighborhood.",
    )

    parser.add_argument(
        "--nx",
        type=int,
        default=_mapping_value(grid, "nx", "grid"),
        help="Fixed-grid x point count.",
    )
    parser.add_argument(
        "--nz",
        type=int,
        default=_mapping_value(grid, "nz", "grid"),
        help="Fixed-grid z point count.",
    )
    parser.add_argument(
        "--slice-mode",
        choices=("nearest_plane", "slab"),
        default=_mapping_value(slice_config, "mode", "slice"),
        help="Midspan y-slice selection mode.",
    )
    parser.add_argument(
        "--slab-ratio",
        type=float,
        default=_mapping_value(slice_config, "slab_ratio", "slice"),
        help="Relative slab half-width for slab slicing.",
    )
    parser.add_argument(
        "--y-round-decimals",
        type=int,
        default=_mapping_value(slice_config, "y_round_decimals", "slice"),
        help="Y-coordinate rounding precision for nearest-plane slicing.",
    )
    parser.add_argument(
        "--interpolation-method",
        choices=("linear", "nearest"),
        default="linear",
        help="Concentration interpolation method.",
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        default=argparse.SUPPRESS,
        help=(
            "Optional front_simple.dat curve used only for post-hoc comparison; "
            "it does not affect automatic detection. Dynamic default: "
            "CASE_DIR/front_simple.dat."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=argparse.SUPPRESS,
        help=(
            "Output directory. Dynamic default: "
            f"{paths.front_detection_dir}/CASE."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of all requested outputs.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write CSV outputs without reference-comparison figures.",
    )
    return parser.parse_args()


def main() -> None:
    paths = load_project_paths(REPO_ROOT / "config" / "paths.yaml")
    cases_config = load_yaml(REPO_ROOT / "config" / "cases.yaml")
    args = _parse_args(paths, cases_config)

    try:
        case = args.case.strip().upper()
        case_dir = paths.case_dir(case)
        reference_file_arg = getattr(args, "reference_file", None)
        output_dir_arg = getattr(args, "output_dir", None)
        reference_path = (
            reference_file_arg.expanduser()
            if reference_file_arg is not None
            else case_dir / "front_simple.dat"
        )
        output_dir = (
            output_dir_arg.expanduser()
            if output_dir_arg is not None
            else paths.front_detection_dir / case
        )

        frame_paths = discover_nek_frame_paths(
            case_dir,
            file_prefix=args.file_prefix,
            start_index=args.start_index,
            end_index=args.end_index,
        )
        print(
            f"Selected {len(frame_paths)} Nek files: "
            f"indices {frame_paths[0].index}–{frame_paths[-1].index}"
        )

        sequence = build_concentration_sequence(
            frame_paths,
            nx=args.nx,
            nz=args.nz,
            slice_mode=args.slice_mode,
            slab_ratio=args.slab_ratio,
            y_round_decimals=args.y_round_decimals,
            interpolation_method=args.interpolation_method,
        )
        print(
            "Fixed grid: "
            f"{sequence.Xi.shape[1]} x {sequence.Xi.shape[0]}, "
            f"x=[{sequence.grid_metadata['xmin']:.16g}, "
            f"{sequence.grid_metadata['xmax']:.16g}], "
            f"z=[{sequence.grid_metadata['zmin']:.16g}, "
            f"{sequence.grid_metadata['zmax']:.16g}]"
        )
        print(
            f"Time range: {sequence.time[0]:.16g}–{sequence.time[-1]:.16g}"
        )

        tracking_result = track_concentration_front(
            sequence.time,
            sequence.Xi,
            sequence.Zi,
            sequence.C_frames,
            threshold=args.threshold,
            min_component_pixels=args.min_component_pixels,
            bottom_rows=args.bottom_rows,
            max_front_jump=args.max_front_jump,
            connectivity=args.connectivity,
        )
        reference_front = read_front_simple_dat(reference_path)
        comparison = compare_detected_front_to_reference(
            tracking_result,
            sequence.file_indices,
            reference_front,
        )
        summary = build_front_detection_summary(
            case=case,
            sequence=sequence,
            tracking_result=tracking_result,
            comparison=comparison,
            reference_path=reference_path,
        )

        csv_paths = [
            detected_front_timeseries_path(output_dir, case),
            front_detection_comparison_path(output_dir, case),
            front_detection_summary_path(output_dir, case),
        ]
        figure_paths = (
            []
            if args.no_plots
            else [
                front_detection_overlay_path(output_dir, case),
                front_detection_difference_path(output_dir, case),
            ]
        )
        preflight_output_paths([*csv_paths, *figure_paths], args.overwrite)
        written_csv_paths = write_front_detection_csvs(
            output_dir,
            case,
            sequence,
            tracking_result,
            comparison,
            summary,
            overwrite=True,
        )
        written_figure_paths = (
            []
            if args.no_plots
            else write_front_detection_plots(
                output_dir,
                case,
                tracking_result,
                reference_front,
                comparison,
                overwrite=True,
            )
        )

        print(format_front_detection_summary_table(summary))
        print(f"Output directory: {output_dir}")
        print("CSV files:")
        for path in written_csv_paths:
            print(f"  {path}")
        if args.no_plots:
            print("Figures: skipped (--no-plots)")
        else:
            print("Figure files:")
            for path in written_figure_paths:
                print(f"  {path}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
