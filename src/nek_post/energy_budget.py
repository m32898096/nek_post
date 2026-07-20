"""Pure numerical calculations for energy-budget closure diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnergyBudgetData:
    """Input energy-budget samples in strictly increasing time order."""

    time: np.ndarray
    E_k: np.ndarray
    E_p: np.ndarray
    E_total: np.ndarray
    epsilon: np.ndarray


@dataclass(frozen=True)
class EnergyBudgetDiagnostics:
    """Input samples and derived energy-budget closure diagnostics."""

    time: np.ndarray
    E_k: np.ndarray
    E_p: np.ndarray
    E_total: np.ndarray
    epsilon: np.ndarray
    E_total_minus_Ek_Ep: np.ndarray
    cumulative_epsilon: np.ndarray
    energy_closure: np.ndarray
    closure_residual_from_target: np.ndarray
    closure_drift_from_initial: np.ndarray
    dEtotal_dt: np.ndarray
    differential_closure_residual: np.ndarray
    energy_closure_minus: np.ndarray


EnergySummaryValue = str | int | float
EnergySummaryRow = dict[str, EnergySummaryValue]


def validate_energy_budget_data(data: EnergyBudgetData) -> None:
    """Validate array lengths, sample count, finite values, and time ordering."""
    arrays = {
        "time": data.time,
        "E_k": data.E_k,
        "E_p": data.E_p,
        "E_total": data.E_total,
        "epsilon": data.epsilon,
    }
    lengths = {name: np.asarray(values).size for name, values in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError("Energy budget arrays must have matching lengths.")
    if lengths["time"] < 2:
        raise ValueError("Energy budget data must contain at least two time samples.")
    for name, values in arrays.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Energy budget data contains non-finite {name} values.")
    if np.any(np.diff(data.time) <= 0.0):
        raise ValueError("Energy budget time values must be strictly increasing and unique.")


def cumulative_trapezoid(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return the cumulative trapezoidal integral with an initial value of zero."""
    cumulative = np.zeros_like(values, dtype=float)
    increments = 0.5 * (values[1:] + values[:-1]) * np.diff(time)
    cumulative[0] = 0.0
    cumulative[1:] = np.cumsum(increments)
    return cumulative


def compute_energy_diagnostics(data: EnergyBudgetData, target: float) -> EnergyBudgetDiagnostics:
    """Calculate the established energy-budget closure diagnostics."""
    validate_energy_budget_data(data)

    E_total_minus_Ek_Ep = data.E_total - (data.E_k + data.E_p)
    cumulative_epsilon = cumulative_trapezoid(data.time, data.epsilon)
    energy_closure = data.E_total + cumulative_epsilon
    closure_residual_from_target = energy_closure - target
    closure_drift_from_initial = energy_closure - energy_closure[0]
    dEtotal_dt = np.gradient(data.E_total, data.time)
    differential_closure_residual = dEtotal_dt + data.epsilon
    energy_closure_minus = data.E_total - cumulative_epsilon

    return EnergyBudgetDiagnostics(
        time=data.time,
        E_k=data.E_k,
        E_p=data.E_p,
        E_total=data.E_total,
        epsilon=data.epsilon,
        E_total_minus_Ek_Ep=E_total_minus_Ek_Ep,
        cumulative_epsilon=cumulative_epsilon,
        energy_closure=energy_closure,
        closure_residual_from_target=closure_residual_from_target,
        closure_drift_from_initial=closure_drift_from_initial,
        dEtotal_dt=dEtotal_dt,
        differential_closure_residual=differential_closure_residual,
        energy_closure_minus=energy_closure_minus,
    )


def mean_absolute_value(values: np.ndarray) -> float:
    """Return the mean absolute value as a Python float."""
    return float(np.mean(np.abs(values)))


def root_mean_square(values: np.ndarray) -> float:
    """Return the root-mean-square value as a Python float."""
    return float(np.sqrt(np.mean(values**2)))


def maximum_absolute_value(values: np.ndarray) -> float:
    """Return the maximum absolute value as a Python float."""
    return float(np.max(np.abs(values)))


def build_energy_summary_row(
    case: str,
    diagnostics: EnergyBudgetDiagnostics,
    target: float,
) -> EnergySummaryRow:
    """Build one numeric summary row for a case."""
    return {
        "case": case,
        "n_points": int(diagnostics.time.size),
        "time_start": float(diagnostics.time[0]),
        "time_end": float(diagnostics.time[-1]),
        "E_total_initial": float(diagnostics.E_total[0]),
        "E_total_final": float(diagnostics.E_total[-1]),
        "epsilon_min": float(np.min(diagnostics.epsilon)),
        "epsilon_max": float(np.max(diagnostics.epsilon)),
        "cumulative_epsilon_final": float(diagnostics.cumulative_epsilon[-1]),
        "energy_closure_initial": float(diagnostics.energy_closure[0]),
        "energy_closure_final": float(diagnostics.energy_closure[-1]),
        "target": float(target),
        "max_abs_closure_residual_from_target": maximum_absolute_value(
            diagnostics.closure_residual_from_target
        ),
        "mean_abs_closure_residual_from_target": mean_absolute_value(
            diagnostics.closure_residual_from_target
        ),
        "rms_closure_residual_from_target": root_mean_square(diagnostics.closure_residual_from_target),
        "max_abs_closure_drift_from_initial": maximum_absolute_value(diagnostics.closure_drift_from_initial),
        "final_closure_drift_from_initial": float(diagnostics.closure_drift_from_initial[-1]),
        "max_abs_differential_closure_residual": maximum_absolute_value(
            diagnostics.differential_closure_residual
        ),
        "rms_differential_closure_residual": root_mean_square(diagnostics.differential_closure_residual),
        "max_abs_E_total_minus_Ek_Ep": maximum_absolute_value(diagnostics.E_total_minus_Ek_Ep),
    }
