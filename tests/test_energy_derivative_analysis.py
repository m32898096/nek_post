from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pytest

from nek_post.energy import (
    EnergyBudget,
    compute_energy_budget_analysis,
    read_energy_budget,
    summarize_energy_budget,
    validate_energy_budget,
)
from nek_post.energy.io import (
    SUMMARY_COLUMNS,
    TIMESERIES_COLUMNS,
    output_paths,
    write_summary_csv,
    write_timeseries_csv,
)
from nek_post.energy.plotting import write_energy_plots


def _synthetic_budget() -> EnergyBudget:
    time = np.array([0.0, 0.4, 1.1, 2.0, 3.5])
    E_k = time**2
    E_p = 10.0 - time**2 - 2.0 * time
    E_total = E_k + E_p
    epsilon = np.full_like(time, 2.0)
    return EnergyBudget(time, E_k, E_p, E_total, epsilon)


def test_reads_canonical_five_column_file(tmp_path: Path) -> None:
    path = tmp_path / "energy_budget.dat"
    path.write_text(
        "# time  E_k  E_p  E_total  epsilon\n"
        "0.0 0.0 10.0 10.0 2.0\n"
        "0.4 0.16 9.04 9.2 2.0\n"
        "1.1 1.21 6.59 7.8 2.0\n",
        encoding="utf-8",
    )

    budget = read_energy_budget(path)

    np.testing.assert_array_equal(budget.time, [0.0, 0.4, 1.1])
    np.testing.assert_array_equal(budget.E_k, [0.0, 0.16, 1.21])
    np.testing.assert_array_equal(budget.E_p, [10.0, 9.04, 6.59])
    np.testing.assert_array_equal(budget.E_total, [10.0, 9.2, 7.8])
    np.testing.assert_array_equal(budget.epsilon, [2.0, 2.0, 2.0])


def test_reader_requires_exactly_five_columns(tmp_path: Path) -> None:
    path = tmp_path / "energy_budget.dat"
    np.savetxt(path, np.ones((3, 6)))

    with pytest.raises(ValueError, match="exactly five columns"):
        read_energy_budget(path)


def test_nonuniform_coordinates_and_quadratic_derivatives_are_analytical() -> None:
    budget = _synthetic_budget()

    analysis = compute_energy_budget_analysis(budget)

    np.testing.assert_allclose(analysis.dE_k_dt, 2.0 * budget.time, atol=2e-14, rtol=0.0)
    np.testing.assert_allclose(
        analysis.dE_p_dt,
        -2.0 * budget.time - 2.0,
        atol=2e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(analysis.dE_total_dt, -2.0, atol=2e-14, rtol=0.0)


@pytest.mark.parametrize(
    "time",
    [np.array([0.0, 0.4, 0.4]), np.array([0.0, 0.5, 0.4])],
)
def test_duplicate_or_nonincreasing_time_is_rejected(time: np.ndarray) -> None:
    budget = EnergyBudget(time, time, 1.0 - time, np.ones(3), np.zeros(3))

    with pytest.raises(ValueError, match="strictly increasing"):
        validate_energy_budget(budget)


def test_at_least_three_samples_are_required() -> None:
    values = np.array([0.0, 1.0])
    budget = EnergyBudget(values, values, values, values + values, values)

    with pytest.raises(ValueError, match="at least three"):
        validate_energy_budget(budget)


def test_inconsistent_total_energy_is_rejected() -> None:
    budget = _synthetic_budget()
    invalid = EnergyBudget(
        budget.time,
        budget.E_k,
        budget.E_p,
        budget.E_total + np.array([0.0, 0.0, 1.0e-4, 0.0, 0.0]),
        budget.epsilon,
    )

    with pytest.raises(ValueError, match=r"E_total is inconsistent with E_k \+ E_p"):
        validate_energy_budget(invalid)


def test_negative_epsilon_is_rejected_beyond_roundoff_tolerance() -> None:
    budget = _synthetic_budget()
    epsilon = budget.epsilon.copy()
    epsilon[2] = -1.0e-6

    with pytest.raises(ValueError, match="epsilon must be non-negative"):
        validate_energy_budget(
            EnergyBudget(budget.time, budget.E_k, budget.E_p, budget.E_total, epsilon)
        )


def test_tiny_negative_epsilon_is_accepted_as_floating_point_roundoff() -> None:
    budget = _synthetic_budget()
    epsilon = budget.epsilon.copy()
    epsilon[2] = -5.0e-13

    validate_energy_budget(
        EnergyBudget(budget.time, budget.E_k, budget.E_p, budget.E_total, epsilon)
    )


def test_closure_residual_and_minus_epsilon_are_calculated() -> None:
    analysis = compute_energy_budget_analysis(_synthetic_budget())

    np.testing.assert_array_equal(analysis.minus_epsilon, -2.0)
    np.testing.assert_allclose(analysis.closure_residual, 0.0, atol=2e-14, rtol=0.0)


def test_summary_metrics_are_calculated() -> None:
    budget = _synthetic_budget()
    epsilon = budget.epsilon + np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    analysis = compute_energy_budget_analysis(
        EnergyBudget(budget.time, budget.E_k, budget.E_p, budget.E_total, epsilon)
    )

    summary = summarize_energy_budget(analysis)

    assert summary.n_points == 5
    assert summary.time_min == 0.0
    assert summary.time_max == 3.5
    assert summary.energy_consistency_max_abs == pytest.approx(0.0, abs=2e-15)
    assert summary.closure_rms == pytest.approx(1.0)
    assert summary.closure_max_abs == pytest.approx(1.0)
    expected_relative = 1.0 / np.sqrt(np.mean(epsilon**2))
    assert summary.relative_closure_rms == pytest.approx(expected_relative)


def test_relative_closure_rms_is_undefined_for_zero_epsilon() -> None:
    time = np.array([0.0, 0.5, 1.5])
    E_k = time**2
    E_p = 2.0 - time**2
    analysis = compute_energy_budget_analysis(
        EnergyBudget(time, E_k, E_p, E_k + E_p, np.zeros(3))
    )

    assert summarize_energy_budget(analysis).relative_closure_rms is None


def test_csv_headers_and_all_four_figures(tmp_path: Path) -> None:
    analysis = compute_energy_budget_analysis(_synthetic_budget())
    summary = summarize_energy_budget(analysis)
    paths = output_paths(tmp_path)

    write_timeseries_csv(paths[0], analysis)
    write_summary_csv(paths[1], summary, "N7")
    figure_paths = write_energy_plots(tmp_path, analysis)

    with paths[0].open(encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == list(TIMESERIES_COLUMNS)
    with paths[1].open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == list(SUMMARY_COLUMNS)
    assert rows[1][0] == "N7"
    assert figure_paths == paths[2:]
    assert all(path.is_file() for path in paths)
