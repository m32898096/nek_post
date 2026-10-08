"""Compute and export lambda_ci for exactly one original Nek5000 snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from nek_post.lambda_ci_workflow import run_single_frame_lambda_ci


def _positive_integer(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--case-dir", type=Path, default=Path("/data/Nek5000_data/case_N7_H"))
    parser.add_argument("--index", type=_positive_integer, required=True, help="One GC0.fNNNNN frame index.")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output root; results go under CASE/fNNNNN. Defaults to /data/Nek5000_data/results/h_refinement/lambda_ci.",
    )
    parser.add_argument("--chunk-size", type=_positive_integer, default=256)
    parser.add_argument("--diagnostics", action="store_true", help="Measure shared-node jumps before and after L2 projection.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect header/resources only; create no files.")
    parser.add_argument("--periodic-axes", choices=("x", "y", "z"), nargs="*", default=["x", "y"], help="Explicit confirmed periodic axes; empty list means none.")
    parser.add_argument("--expected-element-counts", type=_positive_integer, nargs=3, default=[272, 12, 8], metavar=("NX", "NY", "NZ"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        report = run_single_frame_lambda_ci(
            args.case_dir, args.index, args.output_dir, chunk_size=args.chunk_size,
            diagnostics=args.diagnostics, dry_run=args.dry_run,
            periodic_axes=tuple(args.periodic_axes), expected_element_counts=tuple(args.expected_element_counts),
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
        print(json.dumps(report, indent=2, allow_nan=False))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
