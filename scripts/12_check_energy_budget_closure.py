"""Check energy-budget closure from teacher-provided energy_budget.dat files."""

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

from nek_post.paths import ProjectPaths, load_project_paths

TIMESERIES_COLUMNS = (
    "time",
    "E_k",
    "E_p",
    "E_total",
    "epsilon",
    "E_total_minus_Ek_Ep",
    "cumulative_epsilon",
    "energy_closure",
    "closure_residual_from_target",
    "closure_drift_from_initial",
    "dEtotal_dt",
    "differential_closure_residual",
    "energy_closure_minus",
)
SUMMARY_COLUMNS = (
    "case",
    "n_points",
    "time_start",
    "time_end",
    "E_total_initial",
    "E_total_final",
    "epsilon_min",
    "epsilon_max",
    "cumulative_epsilon_final",
    "energy_closure_initial",
    "energy_closure_final",
    "target",
    "max_abs_closure_residual_from_target",
    "mean_abs_closure_residual_from_target",
    "rms_closure_residual_from_target",
    "max_abs_closure_drift_from_initial",
    "final_closure_drift_from_initial",
    "max_abs_differential_closure_residual",
    "rms_differential_closure_residual",
    "max_abs_E_total_minus_Ek_Ep",
)


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


def _input_path(data_root: Path, case: str, filename: str) -> Path:
    return data_root / f"case_{case}" / filename


def _load_energy_budget(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Energy budget file not found: {path}")

    data = np.loadtxt(path, comments="#")
    data = np.atleast_2d(data)
    if data.shape[1] < 5:
        raise ValueError(f"{path} must contain at least 5 columns: time, E_k, E_p, E_total, epsilon.")

    data = data[:, :5]
    sort_order = np.argsort(data[:, 0])
    data = data[sort_order]

    time = np.asarray(data[:, 0], dtype=float)
    E_k = np.asarray(data[:, 1], dtype=float)
    E_p = np.asarray(data[:, 2], dtype=float)
    E_total = np.asarray(data[:, 3], dtype=float)
    epsilon = np.asarray(data[:, 4], dtype=float)

    if time.size < 2:
        raise ValueError(f"{path} must contain at least two time samples.")
    if not np.all(np.isfinite(time)):
        raise ValueError(f"{path} contains non-finite time values.")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError(f"{path} time values must be unique after sorting.")
    for name, values in (("E_k", E_k), ("E_p", E_p), ("E_total", E_total), ("epsilon", epsilon)):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path} contains non-finite {name} values.")

    return {
        "time": time,
        "E_k": E_k,
        "E_p": E_p,
        "E_total": E_total,
        "epsilon": epsilon,
    }


def _cumulative_trapezoid(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    cumulative = np.zeros_like(values, dtype=float)
    increments = 0.5 * (values[1:] + values[:-1]) * np.diff(time)
    cumulative[1:] = np.cumsum(increments)
    return cumulative


def _compute_diagnostics(data: dict[str, np.ndarray], target: float) -> dict[str, np.ndarray]:
    time = data["time"]
    E_k = data["E_k"]
    E_p = data["E_p"]
    E_total = data["E_total"]
    epsilon = data["epsilon"]

    E_total_minus_Ek_Ep = E_total - (E_k + E_p)
    cumulative_epsilon = _cumulative_trapezoid(time, epsilon)
    energy_closure = E_total + cumulative_epsilon
    closure_residual_from_target = energy_closure - target
    closure_drift_from_initial = energy_closure - energy_closure[0]
    dEtotal_dt = np.gradient(E_total, time)
    differential_closure_residual = dEtotal_dt + epsilon
    energy_closure_minus = E_total - cumulative_epsilon

    return {
        **data,
        "E_total_minus_Ek_Ep": E_total_minus_Ek_Ep,
        "cumulative_epsilon": cumulative_epsilon,
        "energy_closure": energy_closure,
        "closure_residual_from_target": closure_residual_from_target,
        "closure_drift_from_initial": closure_drift_from_initial,
        "dEtotal_dt": dEtotal_dt,
        "differential_closure_residual": differential_closure_residual,
        "energy_closure_minus": energy_closure_minus,
    }


def _mean_abs(values: np.ndarray) -> float:
    return float(np.mean(np.abs(values)))


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def _max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values)))


def _format(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.16g}"


def _summary_row(case: str, diagnostics: dict[str, np.ndarray], target: float) -> dict[str, str]:
    time = diagnostics["time"]
    E_total = diagnostics["E_total"]
    epsilon = diagnostics["epsilon"]
    cumulative_epsilon = diagnostics["cumulative_epsilon"]
    energy_closure = diagnostics["energy_closure"]
    closure_residual_from_target = diagnostics["closure_residual_from_target"]
    closure_drift_from_initial = diagnostics["closure_drift_from_initial"]
    differential_closure_residual = diagnostics["differential_closure_residual"]
    E_total_minus_Ek_Ep = diagnostics["E_total_minus_Ek_Ep"]

    values: dict[str, float | int | str] = {
        "case": case,
        "n_points": int(time.size),
        "time_start": float(time[0]),
        "time_end": float(time[-1]),
        "E_total_initial": float(E_total[0]),
        "E_total_final": float(E_total[-1]),
        "epsilon_min": float(np.min(epsilon)),
        "epsilon_max": float(np.max(epsilon)),
        "cumulative_epsilon_final": float(cumulative_epsilon[-1]),
        "energy_closure_initial": float(energy_closure[0]),
        "energy_closure_final": float(energy_closure[-1]),
        "target": float(target),
        "max_abs_closure_residual_from_target": _max_abs(closure_residual_from_target),
        "mean_abs_closure_residual_from_target": _mean_abs(closure_residual_from_target),
        "rms_closure_residual_from_target": _rms(closure_residual_from_target),
        "max_abs_closure_drift_from_initial": _max_abs(closure_drift_from_initial),
        "final_closure_drift_from_initial": float(closure_drift_from_initial[-1]),
        "max_abs_differential_closure_residual": _max_abs(differential_closure_residual),
        "rms_differential_closure_residual": _rms(differential_closure_residual),
        "max_abs_E_total_minus_Ek_Ep": _max_abs(E_total_minus_Ek_Ep),
    }
    return {key: str(value) if isinstance(value, str) else _format(value) for key, value in values.items()}


def _write_timeseries_csv(path: Path, diagnostics: dict[str, np.ndarray], overwrite: bool) -> None:
    _ensure_writable(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMESERIES_COLUMNS)
        writer.writeheader()
        n_points = diagnostics["time"].size
        for index in range(n_points):
            writer.writerow({column: _format(float(diagnostics[column][index])) for column in TIMESERIES_COLUMNS})


def _write_summary_csv(path: Path, rows: list[dict[str, str]], overwrite: bool) -> None:
    _ensure_writable(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _ensure_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Pass --overwrite to replace it.")


def _save_figure(fig, path: Path, overwrite: bool) -> None:
    _ensure_writable(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_components(path: Path, case: str, diagnostics: dict[str, np.ndarray], overwrite: bool) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(diagnostics["time"], diagnostics["E_k"], label="E_k")
    ax.plot(diagnostics["time"], diagnostics["E_p"], label="E_p")
    ax.plot(diagnostics["time"], diagnostics["E_total"], label="E_total")
    ax.set_title(f"Energy budget components: {case}")
    ax.set_xlabel("time")
    ax.set_ylabel("energy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_figure(fig, path, overwrite)


def _plot_overlay(
    path: Path,
    diagnostics_by_case: dict[str, dict[str, np.ndarray]],
    y_key: str,
    title: str,
    ylabel: str,
    overwrite: bool,
    *,
    target: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for case, diagnostics in diagnostics_by_case.items():
        ax.plot(diagnostics["time"], diagnostics[y_key], label=case)
    if target is not None:
        ax.axhline(target, linestyle="--", linewidth=1.0, label=f"target {target:g}")
    if y_key in {"closure_residual_from_target", "closure_drift_from_initial", "differential_closure_residual"}:
        ax.axhline(0.0, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_figure(fig, path, overwrite)


def _write_plots(
    output_dir: Path,
    diagnostics_by_case: dict[str, dict[str, np.ndarray]],
    target: float,
    overwrite: bool,
) -> list[Path]:
    figure_paths: list[Path] = []
    for case, diagnostics in diagnostics_by_case.items():
        path = output_dir / f"energy_budget_components_{case}.png"
        _plot_components(path, case, diagnostics, overwrite)
        figure_paths.append(path)

    plot_specs = [
        ("epsilon_vs_time.png", "epsilon", "Dissipation rate by case", "epsilon", None),
        ("energy_closure_vs_time.png", "energy_closure", "Energy closure by case", "E_total + integral epsilon dt", target),
        (
            "closure_residual_from_target_vs_time.png",
            "closure_residual_from_target",
            "Closure residual from target by case",
            "energy_closure - target",
            None,
        ),
        (
            "closure_drift_from_initial_vs_time.png",
            "closure_drift_from_initial",
            "Closure drift from initial by case",
            "energy_closure - energy_closure[0]",
            None,
        ),
        (
            "differential_closure_residual_vs_time.png",
            "differential_closure_residual",
            "Differential closure residual by case",
            "dE_total/dt + epsilon",
            None,
        ),
    ]
    for filename, y_key, title, ylabel, target_line in plot_specs:
        path = output_dir / filename
        _plot_overlay(path, diagnostics_by_case, y_key, title, ylabel, overwrite, target=target_line)
        figure_paths.append(path)
    return figure_paths


def _print_summary_table(rows: list[dict[str, str]]) -> None:
    columns = (
        "case",
        "time_start",
        "time_end",
        "E_total_initial",
        "E_total_final",
        "cumulative_epsilon_final",
        "energy_closure_initial",
        "energy_closure_final",
        "max_abs_closure_residual_from_target",
        "max_abs_closure_drift_from_initial",
        "max_abs_differential_closure_residual",
        "max_abs_E_total_minus_Ek_Ep",
    )
    widths = {column: max(len(column), *(len(row[column]) for row in rows)) for column in columns}
    print("Energy budget closure summary:")
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(row[column].ljust(widths[column]) for column in columns))


def main() -> None:
    paths = load_project_paths(REPO_ROOT / "config" / "paths.yaml")
    args = _parse_args(paths)
    cases = _parse_cases(args.cases)
    data_root = args.data_root.expanduser()
    output_dir = args.output_dir.expanduser()

    try:
        diagnostics_by_case: dict[str, dict[str, np.ndarray]] = {}
        summary_rows: list[dict[str, str]] = []
        timeseries_paths: list[Path] = []

        print("Input files:")
        for case in cases:
            path = _input_path(data_root, case, args.filename)
            print(f"  {case}: {path}")
            diagnostics = _compute_diagnostics(_load_energy_budget(path), args.target)
            diagnostics_by_case[case] = diagnostics
            summary_rows.append(_summary_row(case, diagnostics, args.target))

            csv_path = output_dir / f"{case}_energy_budget_closure_timeseries.csv"
            _write_timeseries_csv(csv_path, diagnostics, args.overwrite)
            timeseries_paths.append(csv_path)

        summary_path = output_dir / "energy_budget_closure_summary.csv"
        _write_summary_csv(summary_path, summary_rows, args.overwrite)
        figure_paths: list[Path] = []
        if not args.no_plots:
            figure_paths = _write_plots(output_dir, diagnostics_by_case, args.target, args.overwrite)

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
