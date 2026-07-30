"""Compare a GC8950_N7 automatic spectral front with Cantero Re8950."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

from nek_post.fig5a_detected_front import (
    default_detected_front_csv,
    default_detected_front_overlay_dir,
    run_detected_front_overlay,
)
from nek_post.paths import ProjectPaths, load_project_paths


DEFAULT_CASE = "GC8950_N7"


def _parse_args(
    paths: ProjectPaths,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare an automatic spectral front with digitized Cantero "
            "Figure 5a 3D Re8950 data."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--case",
        default=DEFAULT_CASE,
        help="Configured Nek5000 case label.",
    )
    parser.add_argument(
        "--front-csv",
        type=Path,
        default=argparse.SUPPRESS,
        help=(
            "Automatic detected-front timeseries CSV. By default this is "
            "derived from front_detection_dir and --case. Default for "
            f"{DEFAULT_CASE}: "
            f"{default_detected_front_csv(paths, DEFAULT_CASE)}"
        ),
    )
    parser.add_argument(
        "--paper-csv",
        type=Path,
        default=paths.cantero_fig5a_re8950_csv,
        help="Digitized Cantero Figure 5a 3D Re8950 CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=argparse.SUPPRESS,
        help=(
            "Output directory. By default this is derived from "
            "fig5a_paper_overlay_dir and --case. Default for "
            f"{DEFAULT_CASE}: "
            f"{default_detected_front_overlay_dir(paths, DEFAULT_CASE)}"
        ),
    )
    parser.add_argument(
        "--slump-tmin",
        type=float,
        default=3.0,
        help="Inclusive lower bound of the slumping interval.",
    )
    parser.add_argument(
        "--slump-tmax",
        type=float,
        default=12.0,
        help="Inclusive upper bound of the slumping interval.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing outputs.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write CSV outputs without creating figures.",
    )
    args = parser.parse_args(argv)
    if not hasattr(args, "front_csv"):
        args.front_csv = default_detected_front_csv(paths, args.case)
    if not hasattr(args, "output_dir"):
        args.output_dir = default_detected_front_overlay_dir(
            paths, args.case
        )
    return args


def main(argv: list[str] | None = None) -> None:
    paths = load_project_paths(REPO_ROOT / "config" / "paths.yaml")
    args = _parse_args(paths, argv)
    try:
        paths.case_dir(args.case)
        outputs = run_detected_front_overlay(
            case=args.case,
            front_csv=args.front_csv.expanduser(),
            paper_csv=args.paper_csv.expanduser(),
            output_dir=args.output_dir.expanduser(),
            slump_tmin=args.slump_tmin,
            slump_tmax=args.slump_tmax,
            overwrite=args.overwrite,
            no_plots=args.no_plots,
        )
        print(f"Automatic front CSV: {args.front_csv}")
        print(f"Paper CSV: {args.paper_csv}")
        print(f"Comparison CSV: {outputs.comparison_csv}")
        print(f"Summary CSV: {outputs.summary_csv}")
        if args.no_plots:
            print("Figures: skipped (--no-plots)")
        else:
            print("Figure files:")
            for path in outputs.figures:
                print(f"  {path}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
