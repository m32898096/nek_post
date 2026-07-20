"""File paths, parsing, CSV output, and reporting for energy budgets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from numbers import Integral
from pathlib import Path

import numpy as np

from nek_post.energy_budget import EnergyBudgetData, EnergyBudgetDiagnostics, EnergySummaryValue


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
SUMMARY_TABLE_COLUMNS = (
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


def energy_budget_input_path(data_root: Path, case: str, filename: str) -> Path:
    return data_root / f"case_{case}" / filename


def timeseries_csv_path(output_dir: Path, case: str) -> Path:
    return output_dir / f"{case}_energy_budget_closure_timeseries.csv"


def summary_csv_path(output_dir: Path) -> Path:
    return output_dir / "energy_budget_closure_summary.csv"


def component_figure_path(output_dir: Path, case: str) -> Path:
    return output_dir / f"energy_budget_components_{case}.png"


def epsilon_figure_path(output_dir: Path) -> Path:
    return output_dir / "epsilon_vs_time.png"


def energy_closure_figure_path(output_dir: Path) -> Path:
    return output_dir / "energy_closure_vs_time.png"


def closure_residual_figure_path(output_dir: Path) -> Path:
    return output_dir / "closure_residual_from_target_vs_time.png"


def closure_drift_figure_path(output_dir: Path) -> Path:
    return output_dir / "closure_drift_from_initial_vs_time.png"


def differential_closure_residual_figure_path(output_dir: Path) -> Path:
    return output_dir / "differential_closure_residual_vs_time.png"


def load_energy_budget(path: Path) -> EnergyBudgetData:
    """Load, sort, and validate the first five columns of an energy-budget file."""
    if not path.exists():
        raise FileNotFoundError(f"Energy budget file not found: {path}")

    raw_data = np.loadtxt(path, comments="#")
    raw_data = np.atleast_2d(raw_data)
    if raw_data.shape[1] < 5:
        raise ValueError(f"{path} must contain at least 5 columns: time, E_k, E_p, E_total, epsilon.")

    raw_data = raw_data[:, :5]
    raw_data = raw_data[np.argsort(raw_data[:, 0])]
    data = EnergyBudgetData(
        time=np.asarray(raw_data[:, 0], dtype=float),
        E_k=np.asarray(raw_data[:, 1], dtype=float),
        E_p=np.asarray(raw_data[:, 2], dtype=float),
        E_total=np.asarray(raw_data[:, 3], dtype=float),
        epsilon=np.asarray(raw_data[:, 4], dtype=float),
    )

    if data.time.size < 2:
        raise ValueError(f"{path} must contain at least two time samples.")
    if not np.all(np.isfinite(data.time)):
        raise ValueError(f"{path} contains non-finite time values.")
    if np.any(np.diff(data.time) <= 0.0):
        raise ValueError(f"{path} time values must be unique after sorting.")
    for name, values in (
        ("E_k", data.E_k),
        ("E_p", data.E_p),
        ("E_total", data.E_total),
        ("epsilon", data.epsilon),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path} contains non-finite {name} values.")
    return data


def ensure_writable_output(path: Path, overwrite: bool) -> None:
    """Reject an existing output unless replacement was explicitly enabled."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Pass --overwrite to replace it.")


def format_numeric_value(value: EnergySummaryValue) -> str:
    """Format output values using the established CSV precision."""
    if isinstance(value, str):
        return value
    if isinstance(value, Integral):
        return str(value)
    return f"{float(value):.16g}"


def write_energy_timeseries_csv(
    path: Path,
    diagnostics: EnergyBudgetDiagnostics,
    overwrite: bool,
) -> None:
    ensure_writable_output(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMESERIES_COLUMNS)
        writer.writeheader()
        for index in range(diagnostics.time.size):
            writer.writerow(
                {
                    column: format_numeric_value(float(getattr(diagnostics, column)[index]))
                    for column in TIMESERIES_COLUMNS
                }
            )


def write_energy_summary_csv(
    path: Path,
    rows: Sequence[Mapping[str, EnergySummaryValue]],
    overwrite: bool,
) -> None:
    ensure_writable_output(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: format_numeric_value(row[column]) for column in SUMMARY_COLUMNS})


def format_energy_summary_table(rows: Sequence[Mapping[str, EnergySummaryValue]]) -> str:
    """Format the established deterministic terminal summary table."""
    formatted_rows = [
        {column: format_numeric_value(row[column]) for column in SUMMARY_TABLE_COLUMNS} for row in rows
    ]
    widths = {
        column: max(len(column), *(len(row[column]) for row in formatted_rows))
        for column in SUMMARY_TABLE_COLUMNS
    }
    lines = [
        "Energy budget closure summary:",
        "  ".join(column.ljust(widths[column]) for column in SUMMARY_TABLE_COLUMNS),
        "  ".join("-" * widths[column] for column in SUMMARY_TABLE_COLUMNS),
    ]
    lines.extend(
        "  ".join(row[column].ljust(widths[column]) for column in SUMMARY_TABLE_COLUMNS)
        for row in formatted_rows
    )
    return "\n".join(lines)
