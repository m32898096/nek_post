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

DEFAULT_DATA_ROOT = Path("/data/Nek5000_data")
DEFAULT_OUTPUT_DIR = Path("/data/Nek5000_data/results/poly_order_compare/front_kinematics")
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze front-position kinematics from front_simple.dat files.")
    parser.add_argument("--cases", default="N5,N7,N9", help="Comma-separated cases. Default: N5,N7,N9.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help=f"Data root. Default: {DEFAULT_DATA_ROOT}")
    parser.add_argument("--filename", default="front_simple.dat", help="Front-position filename. Default: front_simple.dat.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}")
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


def _parse_cases(raw: str) -> list[str]:
    cases = [case.strip().upper() for case in raw.split(",") if case.strip()]
    if not cases:
        raise ValueError("--cases must include at least one case.")
    for case in cases:
        if not case.startswith("N") or not case[1:].isdigit():
            raise ValueError(f"Invalid case {case!r}; expected labels such as N5, N7, N9.")
    return cases


def _input_path(data_root: Path, case: str, filename: str) -> Path:
    return data_root / f"case_{case}" / filename


def _load_front(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Front-position file not found: {path}")

    data = np.loadtxt(path, comments="#")
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError(f"{path} must contain at least two columns: time and x_front.")

    data = data[:, :2]
    data = data[np.argsort(data[:, 0])]
    time = np.asarray(data[:, 0], dtype=float)
    x_front = np.asarray(data[:, 1], dtype=float)

    if time.size < 2:
        raise ValueError(f"{path} must contain at least two time samples.")
    if not np.all(np.isfinite(time)):
        raise ValueError(f"{path} contains non-finite time values.")
    if not np.all(np.isfinite(x_front)):
        raise ValueError(f"{path} contains non-finite x_front values.")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError(f"{path} contains duplicate time values after sorting. Remove duplicates before analysis.")

    return {"time": time, "x_front": x_front}


def _odd_window(window: int, n_points: int) -> int:
    if n_points < 1:
        raise ValueError("Cannot smooth an empty array.")
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window > n_points:
        window = n_points if n_points % 2 == 1 else n_points - 1
    return max(1, window)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = _odd_window(window, values.size)
    if window == 1:
        return values.copy()
    kernel = np.ones(window, dtype=float)
    numerator = np.convolve(values, kernel, mode="same")
    denominator = np.convolve(np.ones_like(values, dtype=float), kernel, mode="same")
    return numerator / denominator


def _smooth_velocity(values: np.ndarray, method: str, window: int, polyorder: int) -> np.ndarray:
    window = _odd_window(window, values.size)
    if method == "moving_average" or window == 1:
        return _moving_average(values, window)

    try:
        from scipy.signal import savgol_filter
    except ImportError:
        print("WARNING: scipy is unavailable; falling back to moving_average smoothing.")
        return _moving_average(values, window)

    polyorder = min(max(0, int(polyorder)), window - 1)
    if polyorder < 1:
        print("WARNING: Savitzky-Golay window is too small for requested polyorder; falling back to moving_average.")
        return _moving_average(values, window)
    return savgol_filter(values, window_length=window, polyorder=polyorder, mode="interp")


def _integrate_velocity(time: np.ndarray, velocity: np.ndarray, x0: float) -> np.ndarray:
    reconstructed = np.empty_like(velocity, dtype=float)
    reconstructed[0] = x0
    increments = 0.5 * (velocity[1:] + velocity[:-1]) * np.diff(time)
    reconstructed[1:] = x0 + np.cumsum(increments)
    return reconstructed


def _slumping_velocity(time: np.ndarray, values: np.ndarray) -> float:
    mask = (time >= SLUMPING_TMIN) & (time <= SLUMPING_TMAX)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    slope, _intercept = np.polyfit(time[mask], values[mask], deg=1)
    return float(slope)


def _compute_kinematics(front: dict[str, np.ndarray], method: str, window: int, polyorder: int) -> dict[str, np.ndarray]:
    time = front["time"]
    x_front = front["x_front"]
    edge_order = 2 if time.size >= 3 else 1
    v_raw = np.gradient(x_front, time, edge_order=edge_order)
    v_smooth = _smooth_velocity(v_raw, method, window, polyorder)
    x_reconstructed = _integrate_velocity(time, v_smooth, float(x_front[0]))
    x_reconstruction_error = x_reconstructed - x_front
    return {
        "time": time,
        "x_front": x_front,
        "v_raw": v_raw,
        "v_smooth": v_smooth,
        "x_reconstructed": x_reconstructed,
        "x_reconstruction_error": x_reconstruction_error,
    }


def _mean_abs(values: np.ndarray) -> float:
    return float(np.mean(np.abs(values)))


def _max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values)))


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


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
        "mean_abs_reconstruction_error": _mean_abs(error),
        "max_abs_reconstruction_error": _max_abs(error),
        "rms_reconstruction_error": _rms(error),
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
    args = _parse_args()
    cases = _parse_cases(args.cases)
    data_root = args.data_root.expanduser()
    output_dir = args.output_dir.expanduser()

    try:
        kinematics_by_case: dict[str, dict[str, np.ndarray]] = {}
        summary_rows: list[dict[str, str]] = []
        timeseries_paths: list[Path] = []

        print("Input files:")
        for case in cases:
            input_path = _input_path(data_root, case, args.filename)
            print(f"  {case}: {input_path}")
            front = _load_front(input_path)
            kinematics = _compute_kinematics(front, args.smooth_method, args.smooth_window, args.savgol_polyorder)
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
