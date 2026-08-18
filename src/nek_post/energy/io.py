"""Parsing and CSV output for derivative-based energy-budget analysis."""

from __future__ import annotations

import csv
from dataclasses import fields
from pathlib import Path

import numpy as np

from nek_post.energy.analysis import (
    EnergyBudget,
    EnergyBudgetAnalysis,
    EnergyBudgetSummary,
    validate_energy_budget,
)


TIMESERIES_COLUMNS = (
    "time",
    "E_k",
    "E_p",
    "E_total",
    "epsilon",
    "dE_k_dt",
    "dE_p_dt",
    "dE_total_dt",
    "minus_epsilon",
    "closure_residual",
)
SUMMARY_COLUMNS = (
    "case",
    "n_points",
    "time_min",
    "time_max",
    "energy_consistency_max_abs",
    "closure_rms",
    "closure_max_abs",
    "relative_closure_rms",
)


def read_energy_budget(path: str | Path) -> EnergyBudget:
    """Read and validate an exact canonical five-column ``energy_budget.dat``."""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Energy-budget file not found: {input_path}")
    try:
        values = np.loadtxt(input_path, comments="#", dtype=float, ndmin=2)
    except ValueError as exc:
        raise ValueError(f"Could not parse canonical energy-budget file {input_path}: {exc}") from exc
    if values.shape[1] != 5:
        raise ValueError(
            f"{input_path} must contain exactly five columns: "
            "time, E_k, E_p, E_total, epsilon."
        )
    budget = EnergyBudget(
        time=values[:, 0],
        E_k=values[:, 1],
        E_p=values[:, 2],
        E_total=values[:, 3],
        epsilon=values[:, 4],
    )
    validate_energy_budget(budget)
    return budget


def output_paths(output_dir: Path) -> tuple[Path, ...]:
    """Return every output path in deterministic order."""
    return (
        output_dir / "energy_timeseries.csv",
        output_dir / "summary.csv",
        output_dir / "energy_evolution.png",
        output_dir / "energy_derivatives.png",
        output_dir / "energy_closure.png",
        output_dir / "energy_closure_residual.png",
    )


def ensure_outputs_available(paths: tuple[Path, ...], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(f"Output files already exist:\n  {formatted}\nPass --overwrite to replace them.")


def _format(value: float | int | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.16g}"


def write_timeseries_csv(path: Path, analysis: EnergyBudgetAnalysis) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(TIMESERIES_COLUMNS)
        for index in range(analysis.time.size):
            writer.writerow(
                [_format(float(getattr(analysis, column)[index])) for column in TIMESERIES_COLUMNS]
            )


def write_summary_csv(path: Path, summary: EnergyBudgetSummary, case: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {field.name: getattr(summary, field.name) for field in fields(summary)}
    values["case"] = case or ""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SUMMARY_COLUMNS)
        writer.writerow([_format(values[column]) for column in SUMMARY_COLUMNS])
