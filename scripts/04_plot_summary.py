"""Generate summary plots for the polynomial-order comparison study."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from nek_post.config import load_project_config  # noqa: E402
from nek_post.paths import ProjectPaths  # noqa: E402
from nek_post.plotting import plot_contour, plot_difference, plot_error_vs_order, plot_front_position  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate concentration summary plots.")
    parser.add_argument("--comparison-set", default="t19p5", help="Named comparison set from config/cases.yaml.")
    parser.add_argument(
        "--field",
        default="concentration",
        choices=("concentration", "velocity", "pressure"),
        help="Field to plot.",
    )
    return parser.parse_args()


def _interpolated_path(paths: ProjectPaths, case: str, index: int) -> Path:
    return paths.interpolated_dir / "C" / f"interp_C_{case}_f{index:05d}.npz"


def _velocity_interpolated_path(paths: ProjectPaths, case: str, index: int) -> Path:
    return (
        paths.interpolated_dir
        / "velocity"
        / f"interp_velocity_{case}_f{index:05d}.npz"
    )


def _pressure_interpolated_path(paths: ProjectPaths, case: str, index: int) -> Path:
    return (
        paths.interpolated_dir
        / "pressure"
        / f"interp_pressure_{case}_f{index:05d}.npz"
    )


def _error_table_path(paths: ProjectPaths, comparison_set: str) -> Path:
    return paths.tables_dir / f"concentration_error_{comparison_set}.csv"


def _velocity_error_table_path(paths: ProjectPaths, comparison_set: str) -> Path:
    return paths.tables_dir / f"velocity_error_{comparison_set}.csv"


def _pressure_error_table_path(paths: ProjectPaths, comparison_set: str) -> Path:
    return paths.tables_dir / f"pressure_error_{comparison_set}.csv"


def _front_table_path(paths: ProjectPaths, comparison_set: str) -> Path:
    return paths.tables_dir / f"front_position_{comparison_set}.csv"


def _figure_dir(paths: ProjectPaths, field: str, comparison_set: str) -> Path:
    return paths.figures_dir / field / comparison_set


def _log_path(paths: ProjectPaths) -> Path:
    return paths.logs_dir / "plot_summary.log"


def _append_log(paths: ProjectPaths, text: str) -> None:
    path = _log_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}]\n{text}\n\n")


def _load_npz_grid(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Interpolated grid file not found: {path}")

    with np.load(path) as data:
        missing_grid = [name for name in ("Xi", "Zi") if name not in data.files]
        if missing_grid:
            missing = ", ".join(missing_grid)
            raise KeyError(f"{path} is missing required grid key(s): {missing}")

        value_key = next((name for name in ("C_grid", "C", "values") if name in data.files), None)
        if value_key is None:
            raise KeyError(f"{path} is missing concentration grid key: expected C_grid, C, or values")

        return {
            "X": data["Xi"],
            "Z": data["Zi"],
            "C": data[value_key],
        }


def _load_velocity_grid(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Interpolated velocity file not found: {path}")

    with np.load(path) as data:
        missing = [name for name in ("Xi", "Zi", "speed_grid") if name not in data.files]
        if missing:
            missing_text = ", ".join(missing)
            raise KeyError(f"{path} is missing required key(s): {missing_text}")

        return {
            "X": data["Xi"],
            "Z": data["Zi"],
            "speed": data["speed_grid"],
        }


def _load_pressure_grid(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Interpolated pressure file not found: {path}")

    with np.load(path) as data:
        missing = [name for name in ("Xi", "Zi", "p_prime_grid") if name not in data.files]
        if missing:
            missing_text = ", ".join(missing)
            raise KeyError(f"{path} is missing required key(s): {missing_text}")

        return {
            "X": data["Xi"],
            "Z": data["Zi"],
            "p_prime": data["p_prime_grid"],
        }


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV table not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _case_order(config: dict, case: str) -> int:
    return int(config["cases"]["orders"][case])


def _finite_min_max(arrays: list[np.ndarray]) -> tuple[float, float]:
    finite_values = [array[np.isfinite(array)] for array in arrays]
    finite_values = [values for values in finite_values if values.size]
    if not finite_values:
        raise ValueError("No finite values found for plotting color scale.")

    return min(float(np.min(values)) for values in finite_values), max(float(np.max(values)) for values in finite_values)


def _finite_max_abs(arrays: list[np.ndarray]) -> float:
    finite_values = [np.abs(array[np.isfinite(array)]) for array in arrays]
    finite_values = [values for values in finite_values if values.size]
    if not finite_values:
        raise ValueError("No finite values found for plotting color scale.")

    max_abs = max(float(np.max(values)) for values in finite_values)
    return max_abs if max_abs > 0.0 else 1.0


def _build_summary(field: str, comparison_set: str, figure_dir: Path, saved_paths: list[Path]) -> str:
    lines = [
        f"Nek5000 {field} plot summary",
        "",
        f"Comparison set: {comparison_set}",
        f"Output figure directory: {figure_dir}",
        "",
        "Saved figures:",
    ]
    lines.extend(f"  {path}" for path in saved_paths)
    return "\n".join(lines)


def main() -> None:
    """Load comparison outputs and generate summary figures."""
    args = _parse_args()
    config = load_project_config(
        REPO_ROOT / "config" / "paths.yaml",
        REPO_ROOT / "config" / "cases.yaml",
    )
    paths = ProjectPaths.from_mapping(config["paths"])

    try:
        comparison_sets = config["cases"].get("comparison_sets", {})
        if args.comparison_set not in comparison_sets:
            available = ", ".join(sorted(comparison_sets)) or "none"
            raise ValueError(f"Unknown comparison set {args.comparison_set!r}. Available sets: {available}")

        comparison_set = comparison_sets[args.comparison_set]
        case_indices = {case: int(index) for case, index in comparison_set["case_indices"].items()}
        reference_case = comparison_set.get("reference_case") or config["cases"]["reference_case"]
        cases = list(config["cases"]["orders"].keys())
        figure_dir = _figure_dir(paths, args.field, args.comparison_set)
        figure_dir.mkdir(parents=True, exist_ok=True)

        if args.field == "velocity":
            grids = {
                case: _load_velocity_grid(_velocity_interpolated_path(paths, case, case_indices[case]))
                for case in cases
            }
            speed_vmin, speed_vmax = _finite_min_max([grids[case]["speed"] for case in cases])
            saved_paths: list[Path] = []

            for case in cases:
                path = figure_dir / f"speed_field_{args.comparison_set}_{case}.png"
                plot_contour(
                    grids[case]["X"],
                    grids[case]["Z"],
                    grids[case]["speed"],
                    path,
                    title=f"{args.comparison_set} {case} speed",
                    label="speed",
                    vmin=speed_vmin,
                    vmax=speed_vmax,
                )
                saved_paths.append(path)

            reference_speed = grids[reference_case]["speed"]
            diffs = {
                case: np.abs(grids[case]["speed"] - reference_speed)
                for case in cases
                if case != reference_case
            }
            _, diff_vmax = _finite_min_max(list(diffs.values()))

            for case, diff in diffs.items():
                path = figure_dir / f"speed_absdiff_{args.comparison_set}_{case}_vs_{reference_case}.png"
                plot_difference(
                    grids[case]["X"],
                    grids[case]["Z"],
                    diff,
                    path,
                    title=f"{args.comparison_set} |speed_{case} - speed_{reference_case}|",
                    label="|difference|",
                    vmin=0,
                    vmax=diff_vmax,
                )
                saved_paths.append(path)

            error_rows = _read_csv(_velocity_error_table_path(paths, args.comparison_set))
            error_rows.sort(key=lambda row: _case_order(config, row["case"]))
            error_orders = [_case_order(config, row["case"]) for row in error_rows]
            errors = [float(row["relative_L2_speed"]) for row in error_rows]
            error_plot = figure_dir / f"relative_L2_speed_vs_order_{args.comparison_set}.png"
            plot_error_vs_order(
                error_orders,
                errors,
                error_plot,
                title=f"{args.comparison_set} relative L2 error of speed",
                ylabel="Relative L2 error of speed",
            )
            saved_paths.append(error_plot)

            summary = _build_summary(args.field, args.comparison_set, figure_dir, saved_paths)
            print(summary)
            _append_log(paths, summary)
            return

        if args.field == "pressure":
            grids = {
                case: _load_pressure_grid(_pressure_interpolated_path(paths, case, case_indices[case]))
                for case in cases
            }
            p_prime_absmax = _finite_max_abs([grids[case]["p_prime"] for case in cases])
            saved_paths: list[Path] = []

            for case in cases:
                path = figure_dir / f"p_prime_field_{args.comparison_set}_{case}.png"
                plot_contour(
                    grids[case]["X"],
                    grids[case]["Z"],
                    grids[case]["p_prime"],
                    path,
                    title=f"{args.comparison_set} {case} pressure fluctuation",
                    label="p_prime",
                    vmin=-p_prime_absmax,
                    vmax=p_prime_absmax,
                )
                saved_paths.append(path)

            reference_p_prime = grids[reference_case]["p_prime"]
            diffs = {
                case: np.abs(grids[case]["p_prime"] - reference_p_prime)
                for case in cases
                if case != reference_case
            }
            _, diff_vmax = _finite_min_max(list(diffs.values()))
            if diff_vmax == 0.0:
                diff_vmax = 1.0

            for case, diff in diffs.items():
                path = figure_dir / f"p_prime_absdiff_{args.comparison_set}_{case}_vs_{reference_case}.png"
                plot_difference(
                    grids[case]["X"],
                    grids[case]["Z"],
                    diff,
                    path,
                    title=f"{args.comparison_set} |p_prime_{case} - p_prime_{reference_case}|",
                    label="|difference|",
                    vmin=0,
                    vmax=diff_vmax,
                )
                saved_paths.append(path)

            error_rows = _read_csv(_pressure_error_table_path(paths, args.comparison_set))
            error_rows.sort(key=lambda row: _case_order(config, row["case"]))
            error_orders = [_case_order(config, row["case"]) for row in error_rows]
            errors = [float(row["relative_L2_p_prime"]) for row in error_rows]
            error_plot = figure_dir / f"relative_L2_p_prime_vs_order_{args.comparison_set}.png"
            plot_error_vs_order(
                error_orders,
                errors,
                error_plot,
                title=f"{args.comparison_set} relative L2 error of p_prime",
                ylabel="Relative L2 error of p_prime",
            )
            saved_paths.append(error_plot)

            summary = _build_summary(args.field, args.comparison_set, figure_dir, saved_paths)
            print(summary)
            _append_log(paths, summary)
            return

        grids = {
            case: _load_npz_grid(_interpolated_path(paths, case, case_indices[case]))
            for case in cases
        }

        concentration_vmin = 0.0
        concentration_vmax = 1.0
        colorbar_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        saved_paths: list[Path] = []

        for case in cases:
            path = figure_dir / f"C_field_{args.comparison_set}_{case}.png"
            plot_contour(
                grids[case]["X"],
                grids[case]["Z"],
                grids[case]["C"],
                path,
                title=f"{args.comparison_set} {case} concentration",
                label="C",
                vmin=concentration_vmin,
                vmax=concentration_vmax,
                ticks=colorbar_ticks,
            )
            saved_paths.append(path)

        reference_grid = grids[reference_case]["C"]
        diffs = {
            case: np.abs(grids[case]["C"] - reference_grid)
            for case in cases
            if case != reference_case
        }
        diff_vmin = 0.0
        diff_vmax = 1.0

        for case, diff in diffs.items():
            path = figure_dir / f"C_absdiff_{args.comparison_set}_{case}_vs_{reference_case}.png"
            plot_difference(
                grids[case]["X"],
                grids[case]["Z"],
                diff,
                path,
                title=f"{args.comparison_set} |C_{case} - C_{reference_case}|",
                label="|difference|",
                vmin=diff_vmin,
                vmax=diff_vmax,
                ticks=colorbar_ticks,
            )
            saved_paths.append(path)

        error_rows = _read_csv(_error_table_path(paths, args.comparison_set))
        error_rows.sort(key=lambda row: _case_order(config, row["case"]))
        error_orders = [_case_order(config, row["case"]) for row in error_rows]
        errors = [float(row["relative_L2_C"]) for row in error_rows]
        error_plot = figure_dir / f"relative_L2_C_vs_order_{args.comparison_set}.png"
        plot_error_vs_order(error_orders, errors, error_plot, title=f"{args.comparison_set} relative L2 error of C")
        saved_paths.append(error_plot)

        front_rows = _read_csv(_front_table_path(paths, args.comparison_set))
        front_rows.sort(key=lambda row: _case_order(config, row["case"]))
        front_orders = [_case_order(config, row["case"]) for row in front_rows]
        x_front = [float(row["x_front"]) for row in front_rows]
        front_plot = figure_dir / f"front_position_vs_order_{args.comparison_set}.png"
        plot_front_position(front_orders, x_front, front_plot, title=f"{args.comparison_set} concentration front position")
        saved_paths.append(front_plot)

        summary = _build_summary(args.field, args.comparison_set, figure_dir, saved_paths)
        print(summary)
        _append_log(paths, summary)
    except Exception as exc:
        message = f"ERROR: {exc}"
        print(message, file=sys.stderr)
        try:
            _append_log(paths, message)
        except OSError as log_exc:
            print(f"ERROR: Failed to write log file: {log_exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
