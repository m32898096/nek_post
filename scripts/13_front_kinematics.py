"""Analyze front-position kinematics from front_simple.dat files."""

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

from nek_post.front_compare import max_abs, mean_abs, rms, slumping_region_linear_fit  # noqa: E402
from nek_post.front_io import front_simple_path, parse_case_labels, read_front_simple_dat  # noqa: E402
from nek_post.front_kinematics import compute_kinematics  # noqa: E402
from nek_post.paths import ProjectPaths, load_project_paths  # noqa: E402

SLUMPING_TMIN = 3.0
SLUMPING_TMAX = 12.0
TIMESERIES_COLUMNS = (
    "time",
    "x_front",
    "v_raw",
    "v_smooth",
    "x_reconstructed",
    "x_reconstruction_error",
)
SUMMARY_COLUMNS = (
    "case",
    "n_points",
    "time_start",
    "time_end",
    "x_start",
    "x_end",
    "v_raw_min",
    "v_raw_max",
    "v_smooth_min",
    "v_smooth_max",
    "mean_abs_reconstruction_error",
    "max_abs_reconstruction_error",
    "rms_reconstruction_error",
    "final_reconstruction_error",
    "slumping_velocity_raw_position_fit",
    "slumping_velocity_reconstructed_position_fit",
)


def _parse_args(paths: ProjectPaths) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze front-position kinematics from front_simple.dat files.")
    parser.add_argument("--cases", default="N5,N7,N9", help="Comma-separated cases. Default: N5,N7,N9.")
    parser.add_argument("--data-root", type=Path, default=paths.data_root, help=f"Data root. Default: {paths.data_root}")
    parser.add_argument("--filename", default="front_simple.dat", help="Front-position filename. Default: front_simple.dat.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.front_kinematics_dir,
        help=f"Output directory. Default: {paths.front_kinematics_dir}",
    )
    parser.add_argument("--smooth-window", type=int, default=11, help="Velocity smoothing window. Default: 11.")
    parser.add_argument(
        "--smooth-method",
        choices=("moving_average", "savgol"),
        default="moving_average",
        help="Velocity smoothing method. Default: moving_average.",
    )
    parser.add_argument("--savgol-polyorder", type=int, default=3, help="Savitzky-Golay polynomial order. Default: 3.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files.")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    return parser.parse_args()


def _slumping_velocity(time: np.ndarray, values: np.ndarray) -> float:
    _n_points, slope = slumping_region_linear_fit(time, values, SLUMPING_TMIN, SLUMPING_TMAX)
    return slope


def _format(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.16g}"


def _summary_row(case: str, kinematics: dict[str, np.ndarray]) -> dict[str, str]:
    time = kinematics["time"]
    x_front = kinematics["x_front"]
    v_raw = kinematics["v_raw"]
    v_smooth = kinematics["v_smooth"]
    x_reconstructed = kinematics["x_reconstructed"]
    error = kinematics["x_reconstruction_error"]
    values: dict[str, float | int | str] = {
        "case": case,
        "n_points": int(time.size),
        "time_start": float(time[0]),
        "time_end": float(time[-1]),
        "x_start": float(x_front[0]),
        "x_end": float(x_front[-1]),
        "v_raw_min": float(np.min(v_raw)),
        "v_raw_max": float(np.max(v_raw)),
        "v_smooth_min": float(np.min(v_smooth)),
        "v_smooth_max": float(np.max(v_smooth)),
        "mean_abs_reconstruction_error": mean_abs(error),
        "max_abs_reconstruction_error": max_abs(error),
        "rms_reconstruction_error": rms(error),
        "final_reconstruction_error": float(error[-1]),
        "slumping_velocity_raw_position_fit": _slumping_velocity(time, x_front),
        "slumping_velocity_reconstructed_position_fit": _slumping_velocity(time, x_reconstructed),
    }
    return {key: str(value) if isinstance(value, str) else _format(value) for key, value in values.items()}


def _ensure_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Pass --overwrite to replace it.")


def _write_timeseries_csv(path: Path, kinematics: dict[str, np.ndarray], overwrite: bool) -> None:
    _ensure_writable(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMESERIES_COLUMNS)
        writer.writeheader()
        for index in range(kinematics["time"].size):
            writer.writerow({column: _format(float(kinematics[column][index])) for column in TIMESERIES_COLUMNS})


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
    kinematics_by_case: dict[str, dict[str, np.ndarray]],
    y_key: str,
    title: str,
    ylabel: str,
    overwrite: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for case, kinematics in kinematics_by_case.items():
        ax.plot(kinematics["time"], kinematics[y_key], label=case)
    if y_key == "x_reconstruction_error":
        ax.axhline(0.0, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_figure(fig, path, overwrite)


def _plot_reconstruction(path: Path, case: str, kinematics: dict[str, np.ndarray], overwrite: bool) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(kinematics["time"], kinematics["x_front"], label="original x_front")
    ax.plot(kinematics["time"], kinematics["x_reconstructed"], label="reconstructed x_front")
    ax.set_title(f"Front reconstruction: {case}")
    ax.set_xlabel("time")
    ax.set_ylabel("front position")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_figure(fig, path, overwrite)


def _write_plots(output_dir: Path, kinematics_by_case: dict[str, dict[str, np.ndarray]], overwrite: bool) -> list[Path]:
    plot_specs = [
        ("front_position_xt_vs_time.png", "x_front", "Front position vs time", "x_front"),
        ("front_velocity_raw_vs_time.png", "v_raw", "Raw front velocity vs time", "v_raw"),
        ("front_velocity_smoothed_vs_time.png", "v_smooth", "Smoothed front velocity vs time", "v_smooth"),
        (
            "front_reconstruction_error_vs_time.png",
            "x_reconstruction_error",
            "Front reconstruction error vs time",
            "x_reconstructed - x_front",
        ),
    ]
    figure_paths: list[Path] = []
    for filename, y_key, title, ylabel in plot_specs:
        path = output_dir / filename
        _plot_overlay(path, kinematics_by_case, y_key, title, ylabel, overwrite)
        figure_paths.append(path)

    for case, kinematics in kinematics_by_case.items():
        path = output_dir / f"front_position_reconstructed_vs_original_{case}.png"
        _plot_reconstruction(path, case, kinematics, overwrite)
        figure_paths.append(path)
    return figure_paths


def _print_summary_table(rows: list[dict[str, str]]) -> None:
    columns = (
        "case",
        "n_points",
        "time_start",
        "time_end",
        "x_start",
        "x_end",
        "mean_abs_reconstruction_error",
        "max_abs_reconstruction_error",
        "rms_reconstruction_error",
        "final_reconstruction_error",
        "slumping_velocity_raw_position_fit",
        "slumping_velocity_reconstructed_position_fit",
    )
    widths = {column: max(len(column), *(len(row[column]) for row in rows)) for column in columns}
    print("Front kinematics summary:")
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(row[column].ljust(widths[column]) for column in columns))


def main() -> None:
    paths = load_project_paths(REPO_ROOT / "config" / "paths.yaml")
    args = _parse_args(paths)
    cases = parse_case_labels(args.cases)
    data_root = args.data_root.expanduser()
    output_dir = args.output_dir.expanduser()

    try:
        kinematics_by_case: dict[str, dict[str, np.ndarray]] = {}
        summary_rows: list[dict[str, str]] = []
        timeseries_paths: list[Path] = []

        print("Input files:")
        for case in cases:
            input_path = front_simple_path(data_root, case, args.filename)
            print(f"  {case}: {input_path}")
            front = read_front_simple_dat(input_path)
            kinematics = compute_kinematics(front, args.smooth_method, args.smooth_window, args.savgol_polyorder)
            kinematics_by_case[case] = kinematics
            summary_rows.append(_summary_row(case, kinematics))

            csv_path = output_dir / f"{case}_front_kinematics_timeseries.csv"
            _write_timeseries_csv(csv_path, kinematics, args.overwrite)
            timeseries_paths.append(csv_path)

        summary_path = output_dir / "front_kinematics_summary.csv"
        _write_summary_csv(summary_path, summary_rows, args.overwrite)

        figure_paths: list[Path] = []
        if not args.no_plots:
            figure_paths = _write_plots(output_dir, kinematics_by_case, args.overwrite)

        _print_summary_table(summary_rows)
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
