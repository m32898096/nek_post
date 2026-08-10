"""Reconstruct Cantero-definition mean fronts and compare only after reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

from nek_post.cantero_front_reconstruction import (
    DEFAULT_SAVGOL_POLYORDER,
    DEFAULT_SLUMP_TMAX,
    DEFAULT_SLUMP_TMIN,
    DEFAULT_SMOOTH_METHOD,
    DEFAULT_SMOOTH_WINDOW,
    cantero_front_reconstruction_output_paths,
    run_cantero_front_reconstruction,
)
from nek_post.cantero_mean_front import cantero_mean_front_timeseries_path
from nek_post.paths import ProjectPaths, load_project_paths


def _parse_args(paths: ProjectPaths, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the formal N7/Re3450 Cantero mean-front reconstruction, then compare "
            "only reconstructed relative displacement with Cantero Figure 5a."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--case",
        choices=("N7",),
        default="N7",
        help="Fixed formal N7 case for the Re3450 comparison workflow.",
    )
    parser.add_argument(
        "--front-csv",
        type=Path,
        default=cantero_mean_front_timeseries_path(paths.cantero_mean_front_dir, "N7"),
        help="Phase-2 Cantero mean-front CSV.",
    )
    parser.add_argument(
        "--paper-csv",
        type=Path,
        default=paths.cantero_fig5a_re3450_csv,
        help="Digitized Cantero Figure 5a Re3450 CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.cantero_front_reconstruction_dir,
        help="Output root; the case subdirectory is added.",
    )
    parser.add_argument(
        "--smooth-method",
        choices=("moving_average", "savgol"),
        default=DEFAULT_SMOOTH_METHOD,
        help="Established temporal velocity-smoothing method.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW,
        help="Requested temporal velocity-smoothing window.",
    )
    parser.add_argument(
        "--savgol-polyorder",
        type=int,
        default=DEFAULT_SAVGOL_POLYORDER,
        help="Requested Savitzky-Golay polynomial order.",
    )
    parser.add_argument(
        "--slump-tmin",
        type=float,
        default=DEFAULT_SLUMP_TMIN,
        help="Inclusive lower slumping-interval bound.",
    )
    parser.add_argument(
        "--slump-tmax",
        type=float,
        default=DEFAULT_SLUMP_TMAX,
        help="Inclusive upper slumping-interval bound.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs.")
    parser.add_argument("--no-plots", action="store_true", help="Write CSV outputs only.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    paths = load_project_paths(REPO_ROOT / "config" / "paths.yaml")
    try:
        args = _parse_args(paths, argv)
        outputs = cantero_front_reconstruction_output_paths(
            args.output_dir, args.case, include_plots=not args.no_plots
        )
        if not args.overwrite and all(path.exists() for path in outputs.all_paths()):
            print(
                "Output artifacts already exist, skipping:\n"
                + "\n".join(str(path) for path in outputs.all_paths())
                + "\nUse --overwrite to regenerate them."
            )
            return
        run = run_cantero_front_reconstruction(
            case=args.case,
            front_csv=args.front_csv,
            paper_csv=args.paper_csv,
            output_dir=args.output_dir,
            smooth_method=args.smooth_method,
            smooth_window=args.smooth_window,
            savgol_polyorder=args.savgol_polyorder,
            slump_tmin=args.slump_tmin,
            slump_tmax=args.slump_tmax,
            overwrite=args.overwrite,
            no_plots=args.no_plots,
        )
        print(f"Phase-2 front CSV: {args.front_csv}")
        print(f"Paper CSV: {args.paper_csv}")
        print(f"Input front points: {run.reconstruction.time.size}")
        print(
            "Reconstructed relative endpoint: "
            f"{run.reconstruction.x_reconstructed_relative[-1]:.16g}"
        )
        print(f"Comparison points: {run.comparison.time.size}")
        print(f"Timeseries CSV: {run.outputs.timeseries_csv}")
        print(f"Comparison CSV: {run.outputs.comparison_csv}")
        print(f"Summary CSV: {run.outputs.summary_csv}")
        if run.outputs.overlay_figure is None:
            print("Overlay figure: skipped (--no-plots)")
        else:
            print(f"Overlay figure: {run.outputs.overlay_figure}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
