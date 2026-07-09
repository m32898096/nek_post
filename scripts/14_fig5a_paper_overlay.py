"""Overlay Cantero et al. Figure 5a paper data against front_simple.dat results."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nek-post")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from nek_post.front_compare import (  # noqa: E402
    compare_front_to_paper,
    finite_mean,
    max_abs,
    mean_abs,
    rms,
    slumping_velocity_metrics,
)
from nek_post.front_io import (  # noqa: E402
    front_relative_to_initial,
    front_simple_path,
    parse_case_labels,
    read_digitized_paper_csv,
    read_front_simple_dat,
)

DEFAULT_DATA_ROOT = Path("/data/Nek5000_data")
DEFAULT_PAPER_CSV = Path("/data/Nek5000_data/cantero/cantero_fig5a_3D_Re3450.csv")
DEFAULT_OUTPUT_DIR = Path("/data/Nek5000_data/results/poly_order_compare/fig5a_paper_overlay")
SLUMP_TMIN = 3.0
SLUMP_TMAX = 12.0
COMPARISON_COLUMNS = (
    "time",
    "paper_x",
    "simulation_x_interp",
    "x_error",
    "relative_error",
    "log_error",
)
SUMMARY_COLUMNS = (
    "case",
    "n_comparison_points",
    "time_min_compared",
    "time_max_compared",
    "mean_signed_error",
    "mean_abs_error",
    "rms_error",
    "mean_signed_relative_error",
    "mean_abs_relative_error",
    "max_abs_relative_error",
    "rms_log_error",
    "n_slumping_points",
    "paper_slumping_velocity",
    "simulation_slumping_velocity",
    "slumping_velocity_difference",
    "slumping_velocity_relative_difference",
    "slumping_mean_abs_relative_error",
    "slumping_max_abs_relative_error",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay digitized Cantero Fig. 5a Re3450 data with simulation fronts.")
    parser.add_argument("--cases", default="N5,N7,N9", help="Comma-separated cases. Default: N5,N7,N9.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help=f"Data root. Default: {DEFAULT_DATA_ROOT}")
    parser.add_argument("--front-filename", default="front_simple.dat", help="Front-position filename. Default: front_simple.dat.")
    parser.add_argument("--paper-csv", type=Path, default=DEFAULT_PAPER_CSV, help=f"Paper CSV path. Default: {DEFAULT_PAPER_CSV}")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing outputs.")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    return parser.parse_args()


def _ensure_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Pass --overwrite to replace it.")


def _format(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.16g}"


def _slumping_metrics(comparison: dict[str, np.ndarray]) -> dict[str, float | int]:
    return slumping_velocity_metrics(
        comparison,
        compared_x_key="simulation_x_interp",
        compared_label="simulation",
        tmin=SLUMP_TMIN,
        tmax=SLUMP_TMAX,
        include_relative_error_stats=True,
    )


def _summary_row(case: str, comparison: dict[str, np.ndarray]) -> dict[str, str]:
    time = comparison["time"]
    x_error = comparison["x_error"]
    relative_error = comparison["relative_error"]
    log_error = comparison["log_error"]
    slumping = _slumping_metrics(comparison)
    values: dict[str, float | int | str] = {
        "case": case,
        "n_comparison_points": int(time.size),
        "time_min_compared": float(time[0]),
        "time_max_compared": float(time[-1]),
        "mean_signed_error": finite_mean(x_error),
        "mean_abs_error": mean_abs(x_error, finite_only=True),
        "rms_error": rms(x_error, finite_only=True),
        "mean_signed_relative_error": finite_mean(relative_error),
        "mean_abs_relative_error": mean_abs(relative_error, finite_only=True),
        "max_abs_relative_error": max_abs(relative_error, finite_only=True),
        "rms_log_error": rms(log_error, finite_only=True),
        **slumping,
    }
    return {key: str(value) if isinstance(value, str) else _format(value) for key, value in values.items()}


def _write_comparison_csv(path: Path, comparison: dict[str, np.ndarray], overwrite: bool) -> None:
    _ensure_writable(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
        for index in range(comparison["time"].size):
            writer.writerow({column: _format(float(comparison[column][index])) for column in COMPARISON_COLUMNS})


def _write_summary_csv(path: Path, rows: list[dict[str, str]], overwrite: bool) -> None:
    _ensure_writable(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig, path: Path, overwrite: bool) -> None:
    _ensure_writable(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_overlay(
    path: Path,
    paper: dict[str, np.ndarray],
    fronts_by_case: dict[str, dict[str, np.ndarray]],
    overwrite: bool,
    *,
    loglog: bool,
    slump_only: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    paper_time = paper["time"]
    paper_x = paper["x"]
    paper_mask = np.ones_like(paper_time, dtype=bool)
    if loglog:
        paper_mask &= (paper_time > 0.0) & (paper_x > 0.0)
    if slump_only:
        paper_mask &= (paper_time >= SLUMP_TMIN) & (paper_time <= SLUMP_TMAX)
    ax.plot(paper_time[paper_mask], paper_x[paper_mask], marker="o", linestyle="None", label="Cantero Fig. 5a Re3450")

    for case, front in fronts_by_case.items():
        sim_time = front["time"]
        sim_x = front["x_front_minus_x0"]
        sim_mask = np.ones_like(sim_time, dtype=bool)
        if loglog:
            sim_mask &= (sim_time > 0.0) & (sim_x > 0.0)
        if slump_only:
            sim_mask &= (sim_time >= SLUMP_TMIN) & (sim_time <= SLUMP_TMAX)
        ax.plot(sim_time[sim_mask], sim_x[sim_mask], label=case)

    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    title = "Cantero Fig. 5a Re3450 overlay"
    if loglog:
        title += " (log-log)"
    if slump_only:
        title += " slumping region"
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("x_front - x_0")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    _save_figure(fig, path, overwrite)


def _plot_relative_error(
    path: Path,
    comparisons_by_case: dict[str, dict[str, np.ndarray]],
    overwrite: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for case, comparison in comparisons_by_case.items():
        ax.plot(comparison["time"], comparison["relative_error"], label=case)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title("Figure 5a relative error")
    ax.set_xlabel("t")
    ax.set_ylabel("(simulation - paper) / paper")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_figure(fig, path, overwrite)


def _write_plots(
    output_dir: Path,
    paper: dict[str, np.ndarray],
    fronts_by_case: dict[str, dict[str, np.ndarray]],
    comparisons_by_case: dict[str, dict[str, np.ndarray]],
    overwrite: bool,
) -> list[Path]:
    paths = [
        output_dir / "fig5a_overlay_linear.png",
        output_dir / "fig5a_overlay_loglog.png",
        output_dir / "fig5a_relative_error.png",
        output_dir / "fig5a_slumping_region_overlay.png",
    ]
    _plot_overlay(paths[0], paper, fronts_by_case, overwrite, loglog=False)
    _plot_overlay(paths[1], paper, fronts_by_case, overwrite, loglog=True)
    _plot_relative_error(paths[2], comparisons_by_case, overwrite)
    _plot_overlay(paths[3], paper, fronts_by_case, overwrite, loglog=False, slump_only=True)
    return paths


def _print_summary(rows: list[dict[str, str]]) -> None:
    columns = (
        "case",
        "n_comparison_points",
        "mean_abs_relative_error",
        "max_abs_relative_error",
        "rms_log_error",
        "n_slumping_points",
        "paper_slumping_velocity",
        "simulation_slumping_velocity",
        "slumping_mean_abs_relative_error",
    )
    widths = {column: max(len(column), *(len(row[column]) for row in rows)) for column in columns}
    print("Figure 5a paper overlay summary:")
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(row[column].ljust(widths[column]) for column in columns))


def main() -> None:
    args = _parse_args()
    cases = parse_case_labels(args.cases)
    data_root = args.data_root.expanduser()
    output_dir = args.output_dir.expanduser()
    paper_csv = args.paper_csv.expanduser()

    try:
        paper = read_digitized_paper_csv(paper_csv)
        fronts_by_case: dict[str, dict[str, np.ndarray]] = {}
        comparisons_by_case: dict[str, dict[str, np.ndarray]] = {}
        summary_rows: list[dict[str, str]] = []
        comparison_paths: list[Path] = []

        print(f"Paper CSV: {paper_csv}")
        print("Simulation front files:")
        for case in cases:
            front_path = front_simple_path(data_root, case, args.front_filename)
            print(f"  {case}: {front_path}")
            front = front_relative_to_initial(read_front_simple_dat(front_path))
            comparison = compare_front_to_paper(
                front,
                paper,
                front_x_key="x_front_minus_x0",
                interpolated_key="simulation_x_interp",
                error_key="x_error",
            )
            fronts_by_case[case] = front
            comparisons_by_case[case] = comparison
            summary_rows.append(_summary_row(case, comparison))

            comparison_path = output_dir / f"{case}_fig5a_paper_comparison.csv"
            _write_comparison_csv(comparison_path, comparison, args.overwrite)
            comparison_paths.append(comparison_path)

        summary_path = output_dir / "fig5a_paper_overlay_summary.csv"
        _write_summary_csv(summary_path, summary_rows, args.overwrite)
        figure_paths: list[Path] = []
        if not args.no_plots:
            figure_paths = _write_plots(output_dir, paper, fronts_by_case, comparisons_by_case, args.overwrite)

        _print_summary(summary_rows)
        print(f"Output directory: {output_dir}")
        print(f"Summary CSV: {summary_path}")
        print("Comparison CSV files:")
        for path in comparison_paths:
            print(f"  {path}")
        if args.no_plots:
            print("Figures: skipped (--no-plots)")
        else:
            print("Figure files:")
            for path in figure_paths:
                print(f"  {path}")
    except Exception as exc:
        print(f"ERROR: {exc}")  # noqa: T201
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
