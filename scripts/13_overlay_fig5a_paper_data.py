"""Overlay Cantero et al. Figure 5a paper data against front_simple.dat results."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nek-post")

import matplotlib.pyplot as plt
import numpy as np

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


def _parse_cases(raw: str) -> list[str]:
    cases = [case.strip().upper() for case in raw.split(",") if case.strip()]
    if not cases:
        raise ValueError("--cases must include at least one case.")
    for case in cases:
        if not case.startswith("N") or not case[1:].isdigit():
            raise ValueError(f"Invalid case {case!r}; expected labels such as N5, N7, N9.")
    return cases


def _front_path(data_root: Path, case: str, filename: str) -> Path:
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
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(x_front)):
        raise ValueError(f"{path} contains non-finite time or front-position values.")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError(f"{path} contains duplicate time values after sorting.")

    x_front_minus_x0 = x_front - x_front[0]
    return {"time": time, "x_front_minus_x0": x_front_minus_x0, "x0": np.array([x_front[0]], dtype=float)}


def _load_paper_csv(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Paper CSV not found: {path}")

    rows: list[tuple[float, float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for line_number, row in enumerate(reader, start=1):
            if not row or not row[0].strip() or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 2:
                continue
            try:
                time = float(row[0])
                x_value = float(row[1])
            except ValueError:
                if line_number == 1:
                    continue
                raise ValueError(f"{path}:{line_number} has non-numeric first or second column: {row}") from None
            if np.isfinite(time) and np.isfinite(x_value):
                rows.append((time, x_value))

    if len(rows) < 2:
        raise ValueError(f"{path} must contain at least two finite paper data rows.")

    data = np.asarray(rows, dtype=float)
    data = data[np.argsort(data[:, 0])]
    time = data[:, 0]
    x_value = data[:, 1]
    if np.any(np.diff(time) <= 0.0):
        raise ValueError(f"{path} contains duplicate paper time values after sorting.")
    positive = (time > 0.0) & (x_value > 0.0)
    if not np.any(positive):
        raise ValueError(f"{path} has no finite positive rows for log-log plotting.")
    return {"time": time, "x": x_value, "positive_mask": positive}


def _ensure_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Pass --overwrite to replace it.")


def _format(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.16g}"


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _mean_abs(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(np.abs(finite)))


def _max_abs(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(np.abs(finite)))


def _rms(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite**2)))


def _linear_fit_slope(time: np.ndarray, values: np.ndarray) -> float:
    if time.size < 2:
        return float("nan")
    slope, _intercept = np.polyfit(time, values, deg=1)
    return float(slope)


def _relative_difference(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or reference == 0.0:
        return float("nan")
    return float((value - reference) / reference)


def _comparison_for_case(front: dict[str, np.ndarray], paper: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    sim_time = front["time"]
    sim_x = front["x_front_minus_x0"]
    paper_time = paper["time"]
    paper_x = paper["x"]
    overlap = (paper_time >= sim_time[0]) & (paper_time <= sim_time[-1])
    if not np.any(overlap):
        raise ValueError("Simulation and paper time ranges do not overlap.")

    time = paper_time[overlap]
    paper_x_overlap = paper_x[overlap]
    sim_x_interp = np.interp(time, sim_time, sim_x)
    x_error = sim_x_interp - paper_x_overlap
    relative_error = x_error / paper_x_overlap
    positive = (sim_x_interp > 0.0) & (paper_x_overlap > 0.0)
    log_error = np.full_like(x_error, np.nan, dtype=float)
    log_error[positive] = np.log(sim_x_interp[positive]) - np.log(paper_x_overlap[positive])
    return {
        "time": time,
        "paper_x": paper_x_overlap,
        "simulation_x_interp": sim_x_interp,
        "x_error": x_error,
        "relative_error": relative_error,
        "log_error": log_error,
    }


def _slumping_metrics(comparison: dict[str, np.ndarray]) -> dict[str, float | int]:
    time = comparison["time"]
    paper_x = comparison["paper_x"]
    sim_x = comparison["simulation_x_interp"]
    relative_error = comparison["relative_error"]
    mask = (time >= SLUMP_TMIN) & (time <= SLUMP_TMAX)
    n_points = int(np.count_nonzero(mask))
    if n_points < 2:
        return {
            "n_slumping_points": n_points,
            "paper_slumping_velocity": float("nan"),
            "simulation_slumping_velocity": float("nan"),
            "slumping_velocity_difference": float("nan"),
            "slumping_velocity_relative_difference": float("nan"),
            "slumping_mean_abs_relative_error": float("nan"),
            "slumping_max_abs_relative_error": float("nan"),
        }

    paper_velocity = _linear_fit_slope(time[mask], paper_x[mask])
    sim_velocity = _linear_fit_slope(time[mask], sim_x[mask])
    return {
        "n_slumping_points": n_points,
        "paper_slumping_velocity": paper_velocity,
        "simulation_slumping_velocity": sim_velocity,
        "slumping_velocity_difference": sim_velocity - paper_velocity,
        "slumping_velocity_relative_difference": _relative_difference(sim_velocity, paper_velocity),
        "slumping_mean_abs_relative_error": _mean_abs(relative_error[mask]),
        "slumping_max_abs_relative_error": _max_abs(relative_error[mask]),
    }


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
        "mean_signed_error": _finite_mean(x_error),
        "mean_abs_error": _mean_abs(x_error),
        "rms_error": _rms(x_error),
        "mean_signed_relative_error": _finite_mean(relative_error),
        "mean_abs_relative_error": _mean_abs(relative_error),
        "max_abs_relative_error": _max_abs(relative_error),
        "rms_log_error": _rms(log_error),
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
    cases = _parse_cases(args.cases)
    data_root = args.data_root.expanduser()
    output_dir = args.output_dir.expanduser()
    paper_csv = args.paper_csv.expanduser()

    try:
        paper = _load_paper_csv(paper_csv)
        fronts_by_case: dict[str, dict[str, np.ndarray]] = {}
        comparisons_by_case: dict[str, dict[str, np.ndarray]] = {}
        summary_rows: list[dict[str, str]] = []
        comparison_paths: list[Path] = []

        print(f"Paper CSV: {paper_csv}")
        print("Simulation front files:")
        for case in cases:
            front_path = _front_path(data_root, case, args.front_filename)
            print(f"  {case}: {front_path}")
            front = _load_front(front_path)
            comparison = _comparison_for_case(front, paper)
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
