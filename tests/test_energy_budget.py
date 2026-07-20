import numpy as np
import pytest

from nek_post.energy_budget import (
    EnergyBudgetData,
    build_energy_summary_row,
    compute_energy_diagnostics,
    cumulative_trapezoid,
    validate_energy_budget_data,
)


def _data() -> EnergyBudgetData:
    return EnergyBudgetData(
        time=np.array([0.0, 1.0, 3.0]),
        E_k=np.array([2.0, 2.5, 3.0]),
        E_p=np.array([3.0, 2.5, 2.0]),
        E_total=np.array([5.2, 4.2, 2.2]),
        epsilon=np.array([1.0, 1.0, 1.0]),
    )


def test_cumulative_trapezoid_nonuniform_intervals_and_zero_start() -> None:
    actual = cumulative_trapezoid(
        np.array([0.0, 1.0, 3.0]),
        np.array([1.0, 3.0, 2.0]),
    )

    np.testing.assert_allclose(actual, np.array([0.0, 2.0, 7.0]), rtol=0.0, atol=1e-15)
    assert actual[0] == 0.0


def test_compute_energy_diagnostics_preserves_all_definitions() -> None:
    diagnostics = compute_energy_diagnostics(_data(), target=5.0)

    np.testing.assert_allclose(diagnostics.E_total_minus_Ek_Ep, [0.2, -0.8, -2.8], atol=1e-15)
    np.testing.assert_allclose(diagnostics.cumulative_epsilon, [0.0, 1.0, 3.0], atol=1e-15)
    np.testing.assert_allclose(diagnostics.energy_closure, [5.2, 5.2, 5.2], atol=1e-15)
    np.testing.assert_allclose(diagnostics.closure_residual_from_target, [0.2, 0.2, 0.2], atol=1e-15)
    np.testing.assert_allclose(diagnostics.closure_drift_from_initial, [0.0, 0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(diagnostics.dEtotal_dt, [-1.0, -1.0, -1.0], atol=1e-15)
    np.testing.assert_allclose(diagnostics.differential_closure_residual, [0.0, 0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(diagnostics.energy_closure_minus, [5.2, 3.2, -0.8], atol=1e-15)


def test_exact_closure_has_zero_target_residual() -> None:
    data = _data()
    exact_data = EnergyBudgetData(
        time=data.time,
        E_k=data.E_k,
        E_p=data.E_p,
        E_total=np.array([5.0, 4.0, 2.0]),
        epsilon=data.epsilon,
    )

    diagnostics = compute_energy_diagnostics(exact_data, target=5.0)

    np.testing.assert_allclose(diagnostics.energy_closure, 5.0, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(diagnostics.closure_residual_from_target, 0.0, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(diagnostics.differential_closure_residual, 0.0, rtol=0.0, atol=1e-15)


def test_summary_row_contains_exact_numeric_metrics() -> None:
    diagnostics = compute_energy_diagnostics(_data(), target=4.0)

    row = build_energy_summary_row("N5", diagnostics, target=4.0)

    assert row["case"] == "N5"
    assert row["n_points"] == 3
    assert isinstance(row["n_points"], int)
    expected = {
        "time_start": 0.0,
        "time_end": 3.0,
        "E_total_initial": 5.2,
        "E_total_final": 2.2,
        "epsilon_min": 1.0,
        "epsilon_max": 1.0,
        "cumulative_epsilon_final": 3.0,
        "energy_closure_initial": 5.2,
        "energy_closure_final": 5.2,
        "target": 4.0,
        "max_abs_closure_residual_from_target": 1.2,
        "mean_abs_closure_residual_from_target": 1.2,
        "rms_closure_residual_from_target": 1.2,
        "max_abs_closure_drift_from_initial": 0.0,
        "final_closure_drift_from_initial": 0.0,
        "max_abs_differential_closure_residual": 0.0,
        "rms_differential_closure_residual": 0.0,
        "max_abs_E_total_minus_Ek_Ep": 2.8,
    }
    for name, value in expected.items():
        assert float(row[name]) == pytest.approx(value, rel=0.0, abs=1e-14)


def test_validation_rejects_mismatched_array_lengths() -> None:
    data = _data()
    invalid = EnergyBudgetData(data.time, data.E_k[:-1], data.E_p, data.E_total, data.epsilon)

    with pytest.raises(ValueError, match="matching lengths"):
        validate_energy_budget_data(invalid)


def test_validation_requires_at_least_two_samples() -> None:
    one = np.array([0.0])
    invalid = EnergyBudgetData(one, one, one, one, one)

    with pytest.raises(ValueError, match="at least two"):
        validate_energy_budget_data(invalid)


@pytest.mark.parametrize("field", ["time", "E_k", "E_p", "E_total", "epsilon"])
def test_validation_rejects_nonfinite_values(field: str) -> None:
    values = {
        "time": np.array([0.0, 1.0]),
        "E_k": np.array([1.0, 1.0]),
        "E_p": np.array([2.0, 2.0]),
        "E_total": np.array([3.0, 3.0]),
        "epsilon": np.array([0.0, 0.0]),
    }
    values[field] = np.array([values[field][0], np.nan])

    with pytest.raises(ValueError, match=f"non-finite {field}"):
        validate_energy_budget_data(EnergyBudgetData(**values))


@pytest.mark.parametrize("time", [np.array([0.0, 0.0]), np.array([1.0, 0.0])])
def test_validation_requires_strictly_increasing_unique_time(time: np.ndarray) -> None:
    values = np.ones(2)
    invalid = EnergyBudgetData(time, values, values, values, values)

    with pytest.raises(ValueError, match="strictly increasing and unique"):
        validate_energy_budget_data(invalid)
