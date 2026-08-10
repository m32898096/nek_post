"""Overlay reconstructed N5/N7/N9 Re3450 fronts with Cantero Figure 5a."""

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
)
from nek_post.cantero_re3450_multicase import (
    FORMAL_RE3450_CASES,
    cantero_re3450_front_csvs,
    cantero_re3450_multicase_output_paths,
    run_cantero_re3450_multicase,
)
from nek_post.paths import ProjectPaths, load_project_paths


def _parse_args(paths: ProjectPaths, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the formal four-dataset Cantero Re3450 overlays from independently "
            "reconstructed N5, N7, and N9 mean fronts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cases",
        nargs=3,
        choices=FORMAL_RE3450_CASES,
        default=list(FORMAL_RE3450_CASES),
        help="Formal case set; must contain N5, N7, and N9.",
    )
    parser.add_argument(
        "--paper-csv",
        type=Path,
        default=paths.cantero_fig5a_re3450_csv,
        help="Digitized Cantero Figure 5a Re3450 CSV shared by all three cases.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.cantero_re3450_multicase_dir,
        help="Directory for per-case CSVs, combined summary, and combined figures.",
    )
    parser.add_argument(
        "--smooth-method",
        choices=("moving_average", "savgol"),
        default=DEFAULT_SMOOTH_METHOD,
    )
    parser.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW)
    parser.add_argument(
        "--savgol-polyorder", type=int, default=DEFAULT_SAVGOL_POLYORDER
    )
    parser.add_argument("--slump-tmin", type=float, default=DEFAULT_SLUMP_TMIN)
    parser.add_argument("--slump-tmax", type=float, default=DEFAULT_SLUMP_TMAX)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    paths = load_project_paths(REPO_ROOT / "config" / "paths.yaml")
    try:
        args = _parse_args(paths, argv)
        front_csvs = cantero_re3450_front_csvs(paths.cantero_mean_front_dir)
        outputs = cantero_re3450_multicase_output_paths(
            args.output_dir, include_plots=not args.no_plots
        )
        if not args.overwrite and all(path.exists() for path in outputs.all_paths()):
            print(
                "Output artifacts already exist, skipping:\n"
                + "\n".join(str(path) for path in outputs.all_paths())
                + "\nUse --overwrite to regenerate them."
            )
            return
        run = run_cantero_re3450_multicase(
            cases=args.cases,
            front_csvs=front_csvs,
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
        for case in run.cases:
            print(f"{case} Phase-2 CSV: {run.front_csvs[case]}")
        print(f"Paper CSV: {run.paper_csv}")
        print(f"Combined summary: {run.outputs.summary_csv}")
        if run.outputs.linear_overlay is None:
            print("Combined figures: skipped (--no-plots)")
        else:
            print(f"Linear overlay: {run.outputs.linear_overlay}")
            print(f"Log-log overlay: {run.outputs.loglog_overlay}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
