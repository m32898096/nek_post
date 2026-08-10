"""Compute Cantero mean-front positions from configured Nek5000 snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

from nek_post.cantero_mean_front import (
    DEFAULT_CANTERO_MEAN_FRONT_REFERENCE_X,
    DEFAULT_CANTERO_MEAN_FRONT_THRESHOLD,
    STATUS_SUCCESS,
    cantero_mean_front_timeseries_path,
    compute_cantero_mean_front_timeseries,
    write_cantero_mean_front_timeseries_csv,
)
from nek_post.config import load_project_config
from nek_post.front_detection_io import discover_nek_frame_paths
from nek_post.paths import ProjectPaths


def _nonnegative_integer(text: str) -> int:
    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return value


def _finite_float(text: str) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("value must be finite") from exc
    if not np.isfinite(value):
        raise argparse.ArgumentTypeError("value must be finite")
    return value


def _positive_float(text: str) -> float:
    value = _finite_float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return value


def _required_case_value(cases: Mapping[str, Any], key: str) -> Any:
    try:
        return cases[key]
    except KeyError as exc:
        raise ValueError(f"Missing required cases.yaml key {key!r}.") from exc


def _parse_args(
    paths: ProjectPaths,
    cases: Mapping[str, Any],
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the Cantero delta threshold definition to span-averaged "
            "equivalent height for configured Nek5000 snapshots."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--case",
        default=_required_case_value(cases, "reference_case"),
        help="Configured Nek5000 case label.",
    )
    parser.add_argument(
        "--start-index",
        type=_nonnegative_integer,
        help="Inclusive first Nek field-file index.",
    )
    parser.add_argument(
        "--end-index",
        type=_nonnegative_integer,
        help="Inclusive final Nek field-file index.",
    )
    parser.add_argument(
        "--threshold",
        type=_positive_float,
        default=DEFAULT_CANTERO_MEAN_FRONT_THRESHOLD,
        help="Cantero equivalent-height threshold delta.",
    )
    parser.add_argument(
        "--reference-x",
        type=_finite_float,
        default=DEFAULT_CANTERO_MEAN_FRONT_REFERENCE_X,
        help="Physical interior reference used to start the positive-x search.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.cantero_mean_front_dir,
        help="CSV root; the case subdirectory is added.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing CSV artifact.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Compute one absolute Cantero mean-front position per selected snapshot."""
    config = load_project_config(
        REPO_ROOT / "config" / "paths.yaml",
        REPO_ROOT / "config" / "cases.yaml",
    )
    paths = ProjectPaths.from_mapping(config["paths"])
    try:
        args = _parse_args(paths, config["cases"], argv)
        case = str(args.case)
        output_path = cantero_mean_front_timeseries_path(args.output_dir, case)
        if output_path.exists() and not args.overwrite:
            print(
                f"Output file already exists, skipping: {output_path}\n"
                "Use --overwrite to regenerate it."
            )
            return
        frames = discover_nek_frame_paths(
            paths.case_dir(case),
            file_prefix=str(_required_case_value(config["cases"], "file_prefix")),
            start_index=args.start_index,
            end_index=args.end_index,
        )
        series = compute_cantero_mean_front_timeseries(
            frames,
            case=case,
            threshold=args.threshold,
            reference_x=args.reference_x,
        )
        written_path = write_cantero_mean_front_timeseries_csv(
            output_path, series, overwrite=args.overwrite
        )
        successful = series.successful_mask
        success_count = int(np.count_nonzero(successful))
        failure_count = int(successful.size - success_count)
        print(f"Case: {series.case}")
        print(f"Input frame count: {series.time.size}")
        print(f"Successful crossings: {success_count}")
        print(f"Failed crossings: {failure_count}")
        print(f"Time range: {series.time[0]:.16g} to {series.time[-1]:.16g}")
        print(f"Threshold: {series.threshold:.16g}")
        print(f"Reference x: {series.reference_x:.16g}")
        print(f"First successful x_front: {series.x_front[successful][0]:.16g}")
        print(f"Last successful x_front: {series.x_front[successful][-1]:.16g}")
        print(f"Output CSV: {written_path}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
