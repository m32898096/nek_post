"""Benchmark leading-edge compute workers in isolated timed subprocesses."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]

from nek_post.config import load_yaml
from nek_post.leading_edge_benchmark import (
    LeadingEdgeBenchmarkError,
    parse_worker_counts,
    run_leading_edge_worker_benchmark,
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


def _worker_counts(text: str) -> tuple[int, ...]:
    try:
        return parse_worker_counts(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _default_output_dir(paths: ProjectPaths, case: str) -> Path:
    return paths.results_root / "leading_edge_benchmarks" / case.strip().upper()


def _parse_args(
    paths: ProjectPaths,
    cases_config: Mapping[str, Any],
    argv: list[str] | None = None,
) -> argparse.Namespace:
    leading_edge = _mapping_value(cases_config, "leading_edge", "cases")
    if not isinstance(leading_edge, Mapping):
        raise ValueError("cases.yaml leading_edge must be a mapping.")
    configured_spacing = _mapping_value(
        leading_edge, "contour_time_spacing", "leading_edge"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark script-19 leading-edge computation with isolated GNU-time "
            "runs and exact CSV equivalence checks."
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
        help="Uniform physical-x target count supplied to every run.",
    )
    parser.add_argument(
        "--z-target",
        type=_finite_float,
        default=_mapping_value(leading_edge, "z_target", "leading_edge"),
        help="Fixed physical horizontal-plane coordinate.",
    )
    parser.add_argument(
        "--threshold",
        type=_finite_float,
        default=_mapping_value(leading_edge, "threshold", "leading_edge"),
        help="Leading-edge concentration contour.",
    )
    parser.add_argument(
        "--y-upsample-factor",
        type=_integer_at_least(1),
        default=_mapping_value(
            leading_edge, "y_upsample_factor", "leading_edge"
        ),
        help="Multiplier defining the uniform periodic y target count.",
    )
    parser.add_argument(
        "--contour-time-spacing",
        type=_positive_float,
        default=argparse.SUPPRESS,
        help=(
            "Regular target-time spacing supplied explicitly to every run. "
            f"Configured default: {configured_spacing}."
        ),
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Select every frame and omit --contour-time-spacing from compute runs.",
    )
    parser.add_argument(
        "--worker-counts",
        type=_worker_counts,
        default=(1, 2, 4),
        help="Comma-separated worker counts including the workers=1 baseline.",
    )
    parser.add_argument(
        "--repeats",
        type=_integer_at_least(1),
        default=3,
        help="Measured repetitions per worker count.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=_integer_at_least(0),
        default=0,
        help="Disposable unreported warmup runs per worker count.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=argparse.SUPPRESS,
        help=(
            "Benchmark-owned output directory. Dynamic default: "
            f"{paths.results_root}/leading_edge_benchmarks/CASE."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only known artifacts in the benchmark output directory.",
    )
    args = parser.parse_args(argv)
    spacing_was_explicit = hasattr(args, "contour_time_spacing")
    if args.all_frames and spacing_was_explicit:
        parser.error("--all-frames cannot be combined with --contour-time-spacing.")
    if not spacing_was_explicit:
        args.contour_time_spacing = configured_spacing
    if not hasattr(args, "output_dir"):
        args.output_dir = _default_output_dir(paths, args.case)
    return args


def _range_text(start_index: int | None, end_index: int | None) -> str:
    if start_index is None and end_index is None:
        return "all discovered frames"
    lower = "first discovered" if start_index is None else str(start_index)
    upper = "last discovered" if end_index is None else str(end_index)
    return f"{lower} to {upper}"


def main(argv: list[str] | None = None) -> None:
    paths = load_project_paths(REPO_ROOT / "config" / "paths.yaml")
    cases_config = load_yaml(REPO_ROOT / "config" / "cases.yaml")
    args = _parse_args(paths, cases_config, argv)
    try:
        case = args.case.strip().upper()
        output_dir = args.output_dir.expanduser()
        production_output_dir = paths.results_root / "leading_edge" / case
        print(f"Case: {case}")
        print(
            "Selected file-index range: "
            f"{_range_text(args.start_index, args.end_index)}"
        )
        print(f"nx: {args.nx}")
        print(f"z_target: {args.z_target:.16g}")
        print(f"threshold: {args.threshold:.16g}")
        print(f"y_upsample_factor: {args.y_upsample_factor}")
        if args.all_frames:
            print("Time selection: all frames")
        else:
            print(f"contour_time_spacing: {args.contour_time_spacing:.16g}")
        print("Worker counts: " + ",".join(str(value) for value in args.worker_counts))
        print(f"Repeats: {args.repeats}")
        print(f"Warmup runs: {args.warmup_runs}")
        print(f"Output directory: {output_dir}")

        result = run_leading_edge_worker_benchmark(
            repo_root=REPO_ROOT,
            results_root=paths.results_root,
            production_output_dir=production_output_dir,
            case=case,
            file_prefix=args.file_prefix,
            start_index=args.start_index,
            end_index=args.end_index,
            nx=args.nx,
            z_target=args.z_target,
            threshold=args.threshold,
            y_upsample_factor=args.y_upsample_factor,
            contour_time_spacing=(
                None if args.all_frames else args.contour_time_spacing
            ),
            all_frames=args.all_frames,
            worker_counts=args.worker_counts,
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            output_dir=output_dir,
            overwrite=args.overwrite,
            reporter=print,
        )
        print("Final benchmark summary:")
        for summary in result.summaries:
            print(
                f"Workers {summary.worker_count}: "
                f"median wall={summary.median_wall_seconds:.6g}s, "
                f"speedup={summary.speedup_relative_to_workers_1:.6g}, "
                f"median peak RSS={summary.median_maximum_rss_kb:.6g} KiB, "
                f"equivalent={summary.all_artifacts_equivalent}"
            )
        print(f"Benchmark CSV: {result.benchmark_csv}")
        print(f"Summary CSV: {result.summary_csv}")
    except (LeadingEdgeBenchmarkError, FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
