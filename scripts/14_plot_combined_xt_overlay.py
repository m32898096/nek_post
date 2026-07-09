"""Plot processed reconstructed fronts over digitized Cantero Fig. 5a data."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nek-post")

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_PROCESSED_DIR = Path("/data/Nek5000_data/results/poly_order_compare/front_kinematics")
DEFAULT_PAPER_CSV = Path("/data/Nek5000_data/cantero/cantero_fig5a_3D_Re3450.csv")
DEFAULT_OUTPUT_DIR = Path("/data/Nek5000_data/results/poly_order_compare/combined_xt_overlay")
SLUMP_TMIN = 3.0
SLUMP_TMAX = 12.0
FRONT_KINEMATICS_COMMAND = (
    "python scripts/12_analyze_front_kinematics.py \\\n"
    "  --cases N5,N7,N9 \\\n"
    "  --overwrite"
)
SUMMARY_COLUMNS = (
    "case",
    "n_comparison_points",
    "time_min_compared",
    "time_max_compared",
    "processed_x0_shift",
    "mean_abs_error",
    "rms_error",
    "mean_abs_relative_error",
    "max_abs_relative_error",
    "rms_log_error",
    "n_slumping_points",
    "processed_slumping_velocity",
    "paper_slumping_velocity",
    "slumping_velocity_difference",
    "slumping_velocity_relative_difference",
)
OBSOLETE_OUTPUT_NAMES = (
    "combined_xt_overlay_linear.png",
    "combined_xt_overlay_loglog.png",
    "combined_xt_overlay_slumping_region.png",
    "combined_xt_overlay_summary.csv",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot processed front reconstructions over Cantero Fig. 5a data.")
    parser.add_argument("--cases", default="N5,N7,N9", help="Comma-separated cases. Default: N5,N7,N9.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help=f"Processed front kinematics directory. Default: {DEFAULT_PROCESSED_DIR}",
    )
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


def _processed_path(processed_dir: Path, case: str) -> Path:
    return processed_dir / f"{case}_front_kinematics_timeseries.csv"


def _missing_processed_message(case: str, path: Path) -> str:
    return (
        f"Processed front kinematics CSV is missing for {case}: {path}\n"
        "Run:\n"
        f"{FRONT_KINEMATICS_COMMAND}"
    )


def _load_processed_front(path: Path, case: str) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(_missing_processed_message(case, path))

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"time", "x_reconstructed"}
        available = set(reader.fieldnames or [])
        missing = sorted(required - available)
        if missing:
            raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")
        rows = [(float(row["time"]), float(row["x_reconstructed"])) for row in reader]

    if len(rows) < 2:
        raise ValueError(f"{path} must contain at least two processed samples.")
    data = np.asarray(rows, dtype=float)
    data = data[np.argsort(data[:, 0])]
    time = data[:, 0]
    x_reconstructed = data[:, 1]
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(x_reconstructed)):
        raise ValueError(f"{path} contains non-finite time or x_reconstructed values.")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError(f"{path} contains duplicate time values after sorting.")

    x0 = float(x_reconstructed[0])
    return {
        "time": time,
        "x_processed": x_reconstructed - x0,
        "processed_x0_shift": np.array([x0], dtype=float),
    }


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
        raise ValueError(f"{path} must contain at least two finite paper rows.")
    data = np.asarray(rows, dtype=float)
    data = data[np.argsort(data[:, 0])]
    time = data[:, 0]
    x_value = data[:, 1]
    if np.any(np.diff(time) <= 0.0):
        raise ValueError(f"{path} contains duplicate paper time values after sorting.")
    return {"time": time, "paper_x": x_value}


def _ensure_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Pass --overwrite to replace it.")


def _cleanup_obsolete_outputs(output_dir: Path, overwrite: bool) -> None:
    if not overwrite:
        return
    for name in OBSOLETE_OUTPUT_NAMES:
        path = output_dir / name
        if path.exists():
            path.unlink()


def _format(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.16g}"


def _mean_abs(values: np.ndarray) -> float:
    return float(np.mean(np.abs(values)))


def _max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values)))


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def _linear_fit_slope(time: np.ndarray, values: np.ndarray) -> float:
    if time.size < 2:
        return float("nan")
    slope, _intercept = np.polyfit(time, values, deg=1)
    return float(slope)


def _relative_difference(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or reference == 0.0:
        return float("nan")
    return float((value - reference) / reference)


def _comparison_for_case(processed: dict[str, np.ndarray], paper: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    processed_time = processed["time"]
    processed_x = processed["x_processed"]
    paper_time = paper["time"]
    paper_x = paper["paper_x"]
    overlap = (paper_time >= processed_time[0]) & (paper_time <= processed_time[-1])
    if not np.any(overlap):
        raise ValueError("Processed and paper time ranges do not overlap.")

    time = paper_time[overlap]
    paper_overlap_x = paper_x[overlap]
    processed_interp = np.interp(time, processed_time, processed_x)
    error = processed_interp - paper_overlap_x
    relative_error = error / paper_overlap_x
    positive = (processed_interp > 0.0) & (paper_overlap_x > 0.0)
    log_error = np.full_like(error, np.nan, dtype=float)
    log_error[positive] = np.log(processed_interp[positive]) - np.log(paper_overlap_x[positive])
    return {
        "time": time,
        "paper_x": paper_overlap_x,
        "processed_x_interp": processed_interp,
        "error": error,
        "relative_error": relative_error,
        "log_error": log_error,
    }


def _slumping_metrics(comparison: dict[str, np.ndarray]) -> dict[str, float | int]:
    time = comparison["time"]
    paper_x = comparison["paper_x"]
    processed_x = comparison["processed_x_interp"]
    mask = (time >= SLUMP_TMIN) & (time <= SLUMP_TMAX)
    n_points = int(np.count_nonzero(mask))
    if n_points < 2:
        return {
            "n_slumping_points": n_points,
            "processed_slumping_velocity": float("nan"),
            "paper_slumping_velocity": float("nan"),
            "slumping_velocity_difference": float("nan"),
            "slumping_velocity_relative_difference": float("nan"),
        }

    processed_velocity = _linear_fit_slope(time[mask], processed_x[mask])
    paper_velocity = _linear_fit_slope(time[mask], paper_x[mask])
    return {
        "n_slumping_points": n_points,
        "processed_slumping_velocity": processed_velocity,
        "paper_slumping_velocity": paper_velocity,
        "slumping_velocity_difference": processed_velocity - paper_velocity,
        "slumping_velocity_relative_difference": _relative_difference(processed_velocity, paper_velocity),
    }


def _finite_rms(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return _rms(finite)


def _summary_row(case: str, processed: dict[str, np.ndarray], comparison: dict[str, np.ndarray]) -> dict[str, str]:
    relative_error = comparison["relative_error"]
    values: dict[str, float | int | str] = {
        "case": case,
        "n_comparison_points": int(comparison["time"].size),
        "time_min_compared": float(comparison["time"][0]),
        "time_max_compared": float(comparison["time"][-1]),
        "processed_x0_shift": float(processed["processed_x0_shift"][0]),
        "mean_abs_error": _mean_abs(comparison["error"]),
        "rms_error": _rms(comparison["error"]),
        "mean_abs_relative_error": _mean_abs(relative_error),
        "max_abs_relative_error": _max_abs(relative_error),
        "rms_log_error": _finite_rms(comparison["log_error"]),
        **_slumping_metrics(comparison),
    }
    return {key: _format(value) for key, value in values.items()}


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
    processed_by_case: dict[str, dict[str, np.ndarray]],
    overwrite: bool,
    *,
    loglog: bool = False,
    slumping_only: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    paper_time = paper["time"]
    paper_x = paper["paper_x"]
    paper_mask = np.ones_like(paper_time, dtype=bool)
    if loglog:
        paper_mask &= (paper_time > 0.0) & (paper_x > 0.0)
    if slumping_only:
        paper_mask &= (paper_time >= SLUMP_TMIN) & (paper_time <= SLUMP_TMAX)
    ax.plot(paper_time[paper_mask], paper_x[paper_mask], marker="o", linestyle="None", label="Cantero Fig. 5a Re3450")

    for case, processed in processed_by_case.items():
        time = processed["time"]
        x_value = processed["x_processed"]
        mask = np.ones_like(time, dtype=bool)
        if loglog:
            mask &= (time > 0.0) & (x_value > 0.0)
        if slumping_only:
            mask &= (time >= SLUMP_TMIN) & (time <= SLUMP_TMAX)
        ax.plot(time[mask], x_value[mask], linestyle="-", label=f"{case} processed")

    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    title = "Processed front-position overlay"
    if loglog:
        title += " (log-log)"
    if slumping_only:
        title += " slumping region"
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("front position")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    _save_figure(fig, path, overwrite)


def _write_plots(
    output_dir: Path,
    paper: dict[str, np.ndarray],
    processed_by_case: dict[str, dict[str, np.ndarray]],
    overwrite: bool,
) -> list[Path]:
    paths = [
        output_dir / "processed_xt_paper_overlay_linear.png",
        output_dir / "processed_xt_paper_overlay_loglog.png",
        output_dir / "processed_xt_paper_overlay_slumping_region.png",
    ]
    _plot_overlay(paths[0], paper, processed_by_case, overwrite)
    _plot_overlay(paths[1], paper, processed_by_case, overwrite, loglog=True)
    _plot_overlay(paths[2], paper, processed_by_case, overwrite, slumping_only=True)
    return paths


def _print_summary(rows: list[dict[str, str]]) -> None:
    columns = (
        "case",
        "n_comparison_points",
        "mean_abs_error",
        "rms_error",
        "mean_abs_relative_error",
        "max_abs_relative_error",
        "rms_log_error",
        "processed_slumping_velocity",
        "paper_slumping_velocity",
    )
    widths = {column: max(len(column), *(len(row[column]) for row in rows)) for column in columns}
    print("Processed x-t paper overlay summary:")
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(row[column].ljust(widths[column]) for column in columns))


def main() -> None:
    args = _parse_args()
    cases = _parse_cases(args.cases)
    processed_dir = args.processed_dir.expanduser()
    paper_csv = args.paper_csv.expanduser()
    output_dir = args.output_dir.expanduser()

    try:
        _cleanup_obsolete_outputs(output_dir, args.overwrite)
        paper = _load_paper_csv(paper_csv)
        processed_by_case: dict[str, dict[str, np.ndarray]] = {}
        summary_rows: list[dict[str, str]] = []

        print(f"Paper CSV: {paper_csv}")
        print("Processed front files:")
        for case in cases:
            path = _processed_path(processed_dir, case)
            print(f"  {case}: {path}")
            processed = _load_processed_front(path, case)
            comparison = _comparison_for_case(processed, paper)
            processed_by_case[case] = processed
            summary_rows.append(_summary_row(case, processed, comparison))

        summary_path = output_dir / "processed_xt_paper_overlay_summary.csv"
        _write_summary_csv(summary_path, summary_rows, args.overwrite)
        figure_paths: list[Path] = []
        if not args.no_plots:
            figure_paths = _write_plots(output_dir, paper, processed_by_case, args.overwrite)

        _print_summary(summary_rows)
        print(f"Output directory: {output_dir}")
        print(f"Summary CSV: {summary_path}")
        if args.no_plots:
            print("Figures: skipped (--no-plots)")
        else:
            print("Figure files:")
            for path in figure_paths:
                print(f"  {path}")
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
