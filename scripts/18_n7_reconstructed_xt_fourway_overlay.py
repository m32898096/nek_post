"""Reconstruct Re3450/Re8950 N7 automatic fronts and plot four datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

from nek_post.fig5a_detected_front import default_detected_front_csv
from nek_post.paths import ProjectPaths, load_project_paths
from nek_post.reconstructed_xt_comparison import (
    default_n7_reconstructed_xt_output_dir,
    run_n7_reconstructed_xt_fourway,
)


def _parse_args(
    paths: ProjectPaths,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct Re3450 and Re8950 N7 automatic spectral fronts "
            "and overlay them with their Cantero Figure 5a datasets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--re3450-front-csv",
        type=Path,
        default=default_detected_front_csv(paths, "N7"),
        help="Re3450 N7 automatic detected-front CSV.",
    )
    parser.add_argument(
        "--re8950-front-csv",
        type=Path,
        default=default_detected_front_csv(paths, "GC8950_N7"),
        help="Re8950 N7 automatic detected-front CSV.",
    )
    parser.add_argument(
        "--re3450-paper-csv",
        type=Path,
        default=paths.cantero_fig5a_re3450_csv,
        help="Digitized Cantero Figure 5a 3D Re3450 CSV.",
    )
    parser.add_argument(
        "--re8950-paper-csv",
        type=Path,
        default=paths.cantero_fig5a_re8950_csv,
        help="Digitized Cantero Figure 5a 3D Re8950 CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_n7_reconstructed_xt_output_dir(paths),
        help="Directory for reconstructed CSVs and figures.",
    )
    parser.add_argument(
        "--smooth-method",
        choices=("moving_average", "savgol"),
        default="moving_average",
        help="Velocity smoothing method.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=11,
        help="Requested velocity smoothing window.",
    )
    parser.add_argument(
        "--savgol-polyorder",
        type=int,
        default=3,
        help="Requested Savitzky-Golay polynomial order.",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    paths = load_project_paths(REPO_ROOT / "config" / "paths.yaml")
    args = _parse_args(paths, argv)
    try:
        outputs = run_n7_reconstructed_xt_fourway(
            re3450_front_csv=args.re3450_front_csv.expanduser(),
            re8950_front_csv=args.re8950_front_csv.expanduser(),
            re3450_paper_csv=args.re3450_paper_csv.expanduser(),
            re8950_paper_csv=args.re8950_paper_csv.expanduser(),
            output_dir=args.output_dir.expanduser(),
            smooth_method=args.smooth_method,
            smooth_window=args.smooth_window,
            savgol_polyorder=args.savgol_polyorder,
            slump_tmin=args.slump_tmin,
            slump_tmax=args.slump_tmax,
            overwrite=args.overwrite,
            no_plots=args.no_plots,
        )
        print(f"Re3450 automatic front: {args.re3450_front_csv}")
        print(f"Re8950 automatic front: {args.re8950_front_csv}")
        print(f"Re3450 paper CSV: {args.re3450_paper_csv}")
        print(f"Re8950 paper CSV: {args.re8950_paper_csv}")
        print(f"Output directory: {args.output_dir}")
        print("CSV files:")
        for path in outputs.all_paths()[:5]:
            print(f"  {path}")
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
