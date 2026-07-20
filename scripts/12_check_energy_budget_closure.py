"""Check energy-budget closure from teacher-provided energy_budget.dat files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

from nek_post.energy_budget import (
    EnergyBudgetDiagnostics,
    EnergySummaryRow,
    build_energy_summary_row,
    compute_energy_diagnostics,
)
from nek_post.energy_budget_io import (
    energy_budget_input_path,
    format_energy_summary_table,
    load_energy_budget,
    summary_csv_path,
    timeseries_csv_path,
    write_energy_summary_csv,
    write_energy_timeseries_csv,
)
from nek_post.energy_budget_plotting import write_energy_budget_plots
from nek_post.paths import ProjectPaths, load_project_paths


def _parse_args(paths: ProjectPaths) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Re3450 energy-budget closure from energy_budget.dat files.")
    parser.add_argument("--cases", default="N5,N7,N9", help="Comma-separated cases. Default: N5,N7,N9.")
    parser.add_argument("--data-root", type=Path, default=paths.data_root, help=f"Data root. Default: {paths.data_root}")
    parser.add_argument("--filename", default="energy_budget.dat", help="Energy budget filename. Default: energy_budget.dat.")
    parser.add_argument("--target", type=float, default=12.0, help="Nominal conserved energy target. Default: 12.0.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.energy_budget_closure_dir,
        help=f"Output directory. Default: {paths.energy_budget_closure_dir}",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files.")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    return parser.parse_args()


def _parse_cases(raw: str) -> list[str]:
    cases = [case.strip().upper() for case in raw.split(",") if case.strip()]
    if not cases:
        raise ValueError("--cases must include at least one case.")
    for case in cases:
        if not case.startswith("N") or not case[1:].isdigit():
            raise ValueError(f"Invalid case {case!r}; expected labels such as N5, N7, N9.")
    return cases


def main() -> None:
    paths = load_project_paths(REPO_ROOT / "config" / "paths.yaml")
    args = _parse_args(paths)
    cases = _parse_cases(args.cases)
    data_root = args.data_root.expanduser()
    output_dir = args.output_dir.expanduser()

    try:
        diagnostics_by_case: dict[str, EnergyBudgetDiagnostics] = {}
        summary_rows: list[EnergySummaryRow] = []
        timeseries_paths: list[Path] = []

        print("Input files:")
        for case in cases:
            input_path = energy_budget_input_path(data_root, case, args.filename)
            print(f"  {case}: {input_path}")
            diagnostics = compute_energy_diagnostics(load_energy_budget(input_path), args.target)
            diagnostics_by_case[case] = diagnostics
            summary_rows.append(build_energy_summary_row(case, diagnostics, args.target))

            csv_path = timeseries_csv_path(output_dir, case)
            write_energy_timeseries_csv(csv_path, diagnostics, args.overwrite)
            timeseries_paths.append(csv_path)

        summary_path = summary_csv_path(output_dir)
        write_energy_summary_csv(summary_path, summary_rows, args.overwrite)
        figure_paths: list[Path] = []
        if not args.no_plots:
            figure_paths = write_energy_budget_plots(
                output_dir,
                diagnostics_by_case,
                args.target,
                args.overwrite,
            )

        print(format_energy_summary_table(summary_rows))
        print(f"Output directory: {output_dir}")
        print(f"Summary CSV: {summary_path}")
        print("Timeseries CSV files:")
        for path in timeseries_paths:
            print(f"  {path}")
        if args.no_plots:
            print("Figures: skipped (--no-plots)")
        else:
            print("Figure files:")
            for path in figure_paths:
                print(f"  {path}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
