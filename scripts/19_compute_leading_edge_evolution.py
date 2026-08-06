"""Compute selected leading-edge evolution and write reusable CSV artifacts."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]

from nek_post.config import load_yaml
from nek_post.front_detection_io import (
    discover_nek_frame_paths,
    preflight_output_paths,
)
from nek_post.leading_edge_io import (
    leading_edge_metadata_path,
    leading_edge_timeseries_path,
    write_leading_edge_csvs,
)
from nek_post.leading_edge_methods import (
    SUPPORTED_LEADING_EDGE_METHODS,
    normalize_leading_edge_method,
)
from nek_post.leading_edge_workflow import (
    build_leading_edge_evolution,
    select_leading_edge_times,
)
from nek_post.paths import ProjectPaths, load_project_paths


def _mapping_value(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(f"Missing required cases.yaml key {context}.{key}.") from exc


def _finite_float(text: str) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("value must be a finite number") from exc
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("value must be a finite number")
    return value


def _positive_float(text: str) -> float:
    value = _finite_float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return value


def _integer_at_least(minimum: int):
    def parse(text: str) -> int:
        try:
            value = int(text)
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                f"value must be an integer greater than or equal to {minimum}"
            ) from exc
        if value < minimum:
            raise argparse.ArgumentTypeError(
                f"value must be an integer greater than or equal to {minimum}"
            )
        return value

    return parse


def _default_output_dir(paths: ProjectPaths, case: str) -> Path:
    return paths.results_root / "leading_edge" / case.strip().upper()


def _requested_output_paths(
    output_dir: str | Path,
    case: str,
) -> list[Path]:
    return [
        leading_edge_timeseries_path(output_dir, case),
        leading_edge_metadata_path(output_dir, case),
    ]


def _parse_args(
    paths: ProjectPaths,
    cases_config: Mapping[str, Any],
    argv: list[str] | None = None,
) -> argparse.Namespace:
    leading_edge = _mapping_value(cases_config, "leading_edge", "cases")
    if not isinstance(leading_edge, Mapping):
        raise ValueError("cases.yaml leading_edge must be a mapping.")
    configured_spacing = _mapping_value(
        leading_edge,
        "contour_time_spacing",
        "leading_edge",
    )
    configured_method = normalize_leading_edge_method(
        _mapping_value(leading_edge, "extraction_method", "leading_edge")
    )
    configured_x_min = _mapping_value(leading_edge, "x_min", "leading_edge")
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate horizontal Nek5000 concentration planes, extract the "
            "spanwise leading edge, and write reusable CSV artifacts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--case",
        default=_mapping_value(leading_edge, "case", "leading_edge"),
        help="Configured Nek5000 case label.",
    )
    parser.add_argument(
        "--file-prefix",
        default=_mapping_value(cases_config, "file_prefix", "cases"),
        help="Exact Nek filename prefix before .fNNNNN.",
    )
    parser.add_argument(
        "--start-index",
        type=_integer_at_least(0),
        help="Inclusive first Nek file index; omit for no lower bound.",
    )
    parser.add_argument(
        "--end-index",
        type=_integer_at_least(0),
        help="Inclusive last Nek file index; omit for no upper bound.",
    )
    parser.add_argument(
        "--nx",
        type=_integer_at_least(2),
        default=_mapping_value(leading_edge, "nx", "leading_edge"),
        help="Uniform post-processing target-grid point count in physical x.",
    )
    parser.add_argument(
        "--z-target",
        type=_finite_float,
        default=_mapping_value(leading_edge, "z_target", "leading_edge"),
        help="Fixed physical z coordinate of the horizontal plane.",
    )
    parser.add_argument(
        "--threshold",
        type=_finite_float,
        default=_mapping_value(leading_edge, "threshold", "leading_edge"),
        help="Concentration contour defining the leading edge.",
    )
    parser.add_argument(
        "--y-upsample-factor",
        type=_integer_at_least(1),
        default=_mapping_value(
            leading_edge,
            "y_upsample_factor",
            "leading_edge",
        ),
        help="Multiplier from native_ny to the uniform periodic target count.",
    )
    parser.add_argument(
        "--extraction-method",
        choices=SUPPORTED_LEADING_EDGE_METHODS,
        default=configured_method,
        help="Leading-edge extraction method.",
    )
    parser.add_argument(
        "--x-min",
        type=_finite_float,
        default=configured_x_min,
        help="Strict physical lower bound for extraction (retains x > x_min).",
    )
    parser.add_argument(
        "--workers",
        type=_integer_at_least(1),
        default=_mapping_value(leading_edge, "workers", "leading_edge"),
        help=(
            "Process count for later frames; the first frame remains serial "
            "to define the reusable interpolation plan."
        ),
    )
    parser.add_argument(
        "--contour-time-spacing",
        type=_positive_float,
        default=argparse.SUPPRESS,
        help=(
            "Regular target-time spacing used for nearest-snapshot selection. "
            f"Configured default: {configured_spacing}."
        ),
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Select every processed frame instead of regular target times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=argparse.SUPPRESS,
        help=(
            "CSV artifact directory. Dynamic default: "
            f"{paths.results_root}/leading_edge/CASE."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of both CSV artifacts.",
    )
    args = parser.parse_args(argv)
    spacing_was_explicit = hasattr(args, "contour_time_spacing")
    if args.all_frames and spacing_was_explicit:
        parser.error(
            "--all-frames cannot be combined with --contour-time-spacing."
        )
    if not spacing_was_explicit:
        args.contour_time_spacing = configured_spacing
    if not hasattr(args, "output_dir"):
        args.output_dir = _default_output_dir(paths, args.case)
    return args


def main(argv: list[str] | None = None) -> None:
    paths = load_project_paths(REPO_ROOT / "config" / "paths.yaml")
    cases_config = load_yaml(REPO_ROOT / "config" / "cases.yaml")
    args = _parse_args(paths, cases_config, argv)
    try:
        case = args.case.strip().upper()
        if not case:
            raise ValueError("--case must not be empty.")
        case_dir = paths.case_dir(case)
        if not case_dir.is_dir():
            raise FileNotFoundError(f"Configured case directory not found: {case_dir}")
        output_dir = args.output_dir.expanduser()

        frames = discover_nek_frame_paths(
            case_dir,
            file_prefix=args.file_prefix,
            start_index=args.start_index,
            end_index=args.end_index,
        )
        requested_paths = _requested_output_paths(output_dir, case)
        preflight_output_paths(requested_paths, args.overwrite)

        evolution = build_leading_edge_evolution(
            frames,
            nx=args.nx,
            z_target=args.z_target,
            threshold=args.threshold,
            y_upsample_factor=args.y_upsample_factor,
            workers=args.workers,
            extraction_method=args.extraction_method,
            x_min=args.x_min,
        )
        expected_dense_ny = evolution.y_upsample_factor * evolution.native_ny
        if evolution.dense_ny != expected_dense_ny:
            raise ValueError(
                "Inconsistent spanwise target resolution: "
                f"dense_ny={evolution.dense_ny}, but "
                f"y_upsample_factor * native_ny={expected_dense_ny}."
            )
        selection = select_leading_edge_times(
            evolution,
            spacing=None if args.all_frames else args.contour_time_spacing,
        )
        csv_paths = write_leading_edge_csvs(
            output_dir,
            case,
            evolution,
            selection,
            overwrite=args.overwrite,
        )

        print(f"Case: {case}")
        print(f"Case directory: {case_dir}")
        print(f"Nek file prefix: {args.file_prefix}")
        print(f"Input frame count: {len(frames)}")
        print(f"Workers: {args.workers}")
        print(f"Extraction method: {evolution.extraction_method}")
        print(f"Extraction x condition: x > {evolution.extraction_x_min:.16g}")
        print(f"File-index range: {frames[0].index} to {frames[-1].index}")
        print(
            "Actual time range: "
            f"{evolution.time[0]:.16g} to {evolution.time[-1]:.16g}"
        )
        print(f"Selected frame count: {selection.actual_time.size}")
        if args.all_frames:
            print("Time-selection mode: all processed frames (spacing=None)")
        else:
            print("Time-selection mode: nearest snapshots to regular targets")
            print(
                "Requested contour spacing: "
                f"{args.contour_time_spacing:.16g}"
            )
        print(f"nx: {evolution.nx}")
        print(f"native_ny: {evolution.native_ny}")
        print(f"dense_ny: {evolution.dense_ny}")
        print(f"y_upsample_factor: {evolution.y_upsample_factor}")
        print(
            "Spanwise resolution check: "
            f"dense_ny ({evolution.dense_ny}) == "
            f"y_upsample_factor ({evolution.y_upsample_factor}) * "
            f"native_ny ({evolution.native_ny})"
        )
        print(f"z_target: {evolution.z_target:.16g}")
        print(f"threshold: {evolution.threshold:.16g}")
        print(
            "Periodic endpoint included: "
            f"{evolution.periodic_endpoint_included}"
        )
        print(f"Output directory: {output_dir}")
        print("Written CSV artifacts:")
        for path in csv_paths:
            print(f"  {path}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
