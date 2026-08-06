"""Plot leading-edge evolution exclusively from existing CSV artifacts."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]

from nek_post.config import load_yaml
from nek_post.front_detection_io import preflight_output_paths
from nek_post.leading_edge_artifacts import read_leading_edge_artifacts
from nek_post.leading_edge_io import (
    leading_edge_evolution_pdf_path,
    leading_edge_evolution_png_path,
    leading_edge_metadata_path,
    leading_edge_timeseries_path,
)
from nek_post.leading_edge_plotting import write_leading_edge_artifact_plots
from nek_post.paths import ProjectPaths, load_project_paths


def _mapping_value(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(f"Missing required cases.yaml key {context}.{key}.") from exc


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("value must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("value must be a finite positive number")
    return value


def _default_output_dir(paths: ProjectPaths, case: str) -> Path:
    return paths.results_root / "leading_edge" / case.strip().upper()


def _requested_output_paths(
    output_dir: str | Path,
    case: str,
) -> list[Path]:
    return [
        leading_edge_evolution_png_path(output_dir, case),
        leading_edge_evolution_pdf_path(output_dir, case),
    ]


def _parse_args(
    paths: ProjectPaths,
    cases_config: Mapping[str, Any],
    argv: list[str] | None = None,
) -> argparse.Namespace:
    leading_edge = _mapping_value(cases_config, "leading_edge", "cases")
    if not isinstance(leading_edge, Mapping):
        raise ValueError("cases.yaml leading_edge must be a mapping.")
    parser = argparse.ArgumentParser(
        description=(
            "Read leading-edge CSV artifacts and write the PNG/PDF evolution "
            "figure without accessing Nek5000 snapshots."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--case",
        default=_mapping_value(leading_edge, "case", "leading_edge"),
        help="Case label stored in the leading-edge artifacts.",
    )
    parser.add_argument(
        "--timeseries-csv",
        type=Path,
        default=argparse.SUPPRESS,
        help="Existing leading-edge timeseries CSV.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=argparse.SUPPRESS,
        help="Existing one-row leading-edge metadata CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=argparse.SUPPRESS,
        help=(
            "Figure output directory. Dynamic default: "
            f"{paths.results_root}/leading_edge/CASE."
        ),
    )
    parser.add_argument(
        "--reynolds-number",
        type=_positive_float,
        default=_mapping_value(
            leading_edge,
            "reynolds_number",
            "leading_edge",
        ),
        help="Reynolds number shown in the figure title.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of both figure outputs.",
    )
    args = parser.parse_args(argv)
    case = args.case.strip().upper()
    if not hasattr(args, "output_dir"):
        args.output_dir = _default_output_dir(paths, case)
    if not hasattr(args, "timeseries_csv"):
        args.timeseries_csv = leading_edge_timeseries_path(args.output_dir, case)
    if not hasattr(args, "metadata_csv"):
        args.metadata_csv = leading_edge_metadata_path(args.output_dir, case)
    return args


def main(argv: list[str] | None = None) -> None:
    paths = load_project_paths(REPO_ROOT / "config" / "paths.yaml")
    cases_config = load_yaml(REPO_ROOT / "config" / "cases.yaml")
    args = _parse_args(paths, cases_config, argv)
    try:
        case = args.case.strip().upper()
        if not case:
            raise ValueError("--case must not be empty.")
        output_dir = args.output_dir.expanduser()
        timeseries_csv = args.timeseries_csv.expanduser()
        metadata_csv = args.metadata_csv.expanduser()
        requested_paths = _requested_output_paths(output_dir, case)
        preflight_output_paths(requested_paths, args.overwrite)

        plot_data = read_leading_edge_artifacts(timeseries_csv, metadata_csv)
        if plot_data.case != case:
            raise ValueError(
                f"Requested case {case!r} does not match artifact case "
                f"{plot_data.case!r}."
            )
        figure_paths = write_leading_edge_artifact_plots(
            output_dir,
            plot_data,
            overwrite=args.overwrite,
            reynolds_number=args.reynolds_number,
        )

        print(f"Case: {case}")
        print(f"Timeseries CSV: {timeseries_csv}")
        print(f"Metadata CSV: {metadata_csv}")
        print(f"Selected frame count: {plot_data.actual_time.size}")
        print(
            "Actual time range: "
            f"{plot_data.actual_time[0]:.16g} to "
            f"{plot_data.actual_time[-1]:.16g}"
        )
        print(f"threshold: {plot_data.threshold:.16g}")
        print(f"z_target: {plot_data.z_target:.16g}")
        print(f"native_ny: {plot_data.native_ny}")
        print(f"dense_ny: {plot_data.dense_ny}")
        print(f"y_upsample_factor: {plot_data.y_upsample_factor}")
        if plot_data.extraction_x_min is None:
            print("Extraction x domain: unrestricted")
        else:
            print(
                "Extraction x condition: "
                f"x > {plot_data.extraction_x_min:.16g}"
            )
        print("Written figure artifacts:")
        for path in figure_paths:
            print(f"  {path}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
