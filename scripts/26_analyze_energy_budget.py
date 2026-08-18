"""Analyze derivative-based closure of an existing energy_budget.dat artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

from nek_post.energy import (
    compute_energy_budget_analysis,
    read_energy_budget,
    summarize_energy_budget,
)
from nek_post.energy.io import (
    ensure_outputs_available,
    output_paths,
    write_summary_csv,
    write_timeseries_csv,
)
from nek_post.energy.plotting import write_energy_plots
from nek_post.paths import ProjectPaths, load_project_paths


def _parse_args(paths: ProjectPaths, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze dE_total/dt + epsilon from an existing energy_budget.dat.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="Explicit canonical energy_budget.dat path.")
    source.add_argument("--case", help="Configured case whose energy_budget.dat should be analyzed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.energy_budget_closure_dir / "derivative_closure",
        help="Directory for CSV and figure outputs.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs.")
    args = parser.parse_args(argv)
    if args.input is None and args.case is None:
        args.case = "N7"
    return args


def _resolve_input(paths: ProjectPaths, args: argparse.Namespace) -> tuple[Path, str | None]:
    if args.input is None:
        case = str(args.case).upper()
        return paths.case_dir(case) / "energy_budget.dat", case

    input_path = args.input.expanduser().resolve()
    for case, case_dir in paths.case_dirs.items():
        if input_path == (case_dir / "energy_budget.dat").resolve():
            return input_path, case
    return input_path, None


def main(argv: list[str] | None = None) -> None:
    paths = load_project_paths(REPO_ROOT / "config" / "paths.yaml")
    try:
        args = _parse_args(paths, argv)
        input_path, case = _resolve_input(paths, args)
        output_dir = args.output_dir.expanduser()
        outputs = output_paths(output_dir)
        if all(path.exists() for path in outputs) and not args.overwrite:
            print("Output artifacts already exist, skipping:")
            for path in outputs:
                print(f"  {path}")
            return
        ensure_outputs_available(outputs, args.overwrite)

        analysis = compute_energy_budget_analysis(read_energy_budget(input_path))
        summary = summarize_energy_budget(analysis)
        write_timeseries_csv(outputs[0], analysis)
        write_summary_csv(outputs[1], summary, case)
        write_energy_plots(output_dir, analysis)

        print(f"Case: {case or '(not resolved)'}")
        print(f"Input: {input_path}")
        print(f"Output directory: {output_dir}")
        print(f"n_points: {summary.n_points}")
        print(f"time_min: {summary.time_min:.16g}")
        print(f"time_max: {summary.time_max:.16g}")
        print(f"energy_consistency_max_abs: {summary.energy_consistency_max_abs:.16g}")
        print(f"closure_rms: {summary.closure_rms:.16g}")
        print(f"closure_max_abs: {summary.closure_max_abs:.16g}")
        relative = "undefined (RMS(epsilon) is zero)"
        if summary.relative_closure_rms is not None:
            relative = f"{summary.relative_closure_rms:.16g}"
        print(f"relative_closure_rms (diagnostic): {relative}")
        print("Outputs:")
        for path in outputs:
            print(f"  {path}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
