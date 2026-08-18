"""Numerical analysis for the differential energy-budget closure relation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

# The canonical text artifacts retain about ten significant decimal digits.
# These tolerances accept their round-off while rejecting material inconsistency.
ENERGY_CONSISTENCY_RTOL = 1.0e-9
ENERGY_CONSISTENCY_ATOL = 1.0e-10
EPSILON_ATOL = 1.0e-12


@dataclass(frozen=True)
class EnergyBudget:
    """Canonical five-column energy-budget samples."""

    time: FloatArray
    E_k: FloatArray
    E_p: FloatArray
    E_total: FloatArray
    epsilon: FloatArray


@dataclass(frozen=True)
class EnergyBudgetAnalysis:
    """Canonical samples plus derivative-based closure diagnostics."""

    time: FloatArray
    E_k: FloatArray
    E_p: FloatArray
    E_total: FloatArray
    epsilon: FloatArray
    dE_k_dt: FloatArray
    dE_p_dt: FloatArray
    dE_total_dt: FloatArray
    minus_epsilon: FloatArray
    closure_residual: FloatArray
    energy_consistency_max_abs: float


@dataclass(frozen=True)
class EnergyBudgetSummary:
    """Scalar diagnostics for one derivative-based closure analysis."""

    n_points: int
    time_min: float
    time_max: float
    energy_consistency_max_abs: float
    closure_rms: float
    closure_max_abs: float
    relative_closure_rms: float | None


def _arrays(budget: EnergyBudget) -> dict[str, FloatArray]:
    return {
        "time": np.asarray(budget.time, dtype=float),
        "E_k": np.asarray(budget.E_k, dtype=float),
        "E_p": np.asarray(budget.E_p, dtype=float),
        "E_total": np.asarray(budget.E_total, dtype=float),
        "epsilon": np.asarray(budget.epsilon, dtype=float),
    }


def validate_energy_budget(
    budget: EnergyBudget,
    *,
    energy_rtol: float = ENERGY_CONSISTENCY_RTOL,
    energy_atol: float = ENERGY_CONSISTENCY_ATOL,
    epsilon_atol: float = EPSILON_ATOL,
) -> float:
    """Validate canonical samples and return the maximum energy inconsistency."""
    arrays = _arrays(budget)
    for name, values in arrays.items():
        if values.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array.")
    lengths = {values.size for values in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("Energy-budget arrays must have matching lengths.")
    if arrays["time"].size < 3:
        raise ValueError("Energy-budget data must contain at least three time samples.")
    for name, values in arrays.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Energy-budget data contains non-finite {name} values.")
    if np.any(np.diff(arrays["time"]) <= 0.0):
        raise ValueError("Energy-budget time values must be strictly increasing.")

    expected_total = arrays["E_k"] + arrays["E_p"]
    difference = arrays["E_total"] - expected_total
    max_abs_difference = float(np.max(np.abs(difference)))
    if not np.allclose(
        arrays["E_total"],
        expected_total,
        rtol=energy_rtol,
        atol=energy_atol,
    ):
        raise ValueError(
            "E_total is inconsistent with E_k + E_p: "
            f"maximum absolute difference is {max_abs_difference:.16g}."
        )
    if np.any(arrays["epsilon"] < -epsilon_atol):
        minimum = float(np.min(arrays["epsilon"]))
        raise ValueError(
            "epsilon must be non-negative within tolerance "
            f"{epsilon_atol:.16g}; minimum is {minimum:.16g}."
        )
    return max_abs_difference


def compute_energy_budget_analysis(budget: EnergyBudget) -> EnergyBudgetAnalysis:
    """Compute second-order derivatives on the supplied, potentially nonuniform times."""
    energy_consistency_max_abs = validate_energy_budget(budget)
    arrays = _arrays(budget)
    time = arrays["time"]
    dE_k_dt = np.gradient(arrays["E_k"], time, edge_order=2)
    dE_p_dt = np.gradient(arrays["E_p"], time, edge_order=2)
    dE_total_dt = np.gradient(arrays["E_total"], time, edge_order=2)
    minus_epsilon = -arrays["epsilon"]
    closure_residual = dE_total_dt + arrays["epsilon"]
    return EnergyBudgetAnalysis(
        **arrays,
        dE_k_dt=np.asarray(dE_k_dt, dtype=float),
        dE_p_dt=np.asarray(dE_p_dt, dtype=float),
        dE_total_dt=np.asarray(dE_total_dt, dtype=float),
        minus_epsilon=np.asarray(minus_epsilon, dtype=float),
        closure_residual=np.asarray(closure_residual, dtype=float),
        energy_consistency_max_abs=energy_consistency_max_abs,
    )


def _rms(values: FloatArray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def summarize_energy_budget(analysis: EnergyBudgetAnalysis) -> EnergyBudgetSummary:
    """Compute scalar closure diagnostics without assigning convergence meaning."""
    closure_rms = _rms(analysis.closure_residual)
    epsilon_rms = _rms(analysis.epsilon)
    relative_closure_rms = None if epsilon_rms == 0.0 else closure_rms / epsilon_rms
    return EnergyBudgetSummary(
        n_points=int(analysis.time.size),
        time_min=float(np.min(analysis.time)),
        time_max=float(np.max(analysis.time)),
        energy_consistency_max_abs=analysis.energy_consistency_max_abs,
        closure_rms=closure_rms,
        closure_max_abs=float(np.max(np.abs(analysis.closure_residual))),
        relative_closure_rms=relative_closure_rms,
    )
