import csv
from pathlib import Path

import numpy as np
import pytest

from nek_post.energy_budget import EnergyBudgetData, build_energy_summary_row, compute_energy_diagnostics
from nek_post.energy_budget_io import (
    SUMMARY_COLUMNS,
    TIMESERIES_COLUMNS,
    closure_drift_figure_path,
    closure_residual_figure_path,
    component_figure_path,
    differential_closure_residual_figure_path,
    energy_budget_input_path,
    energy_closure_figure_path,
    epsilon_figure_path,
    format_energy_summary_table,
    format_numeric_value,
    load_energy_budget,
    summary_csv_path,
    timeseries_csv_path,
    write_energy_summary_csv,
    write_energy_timeseries_csv,
)


def _diagnostics():
    data = EnergyBudgetData(
        time=np.array([0.0, 1.0]),
        E_k=np.array([1.0 / 3.0, 0.25]),
        E_p=np.array([0.5, 0.75]),
        E_total=np.array([1.0, 0.0]),
        epsilon=np.array([1.0, 1.0]),
    )
    return compute_energy_diagnostics(data, target=1.0)


def _write_data(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data)


def test_input_path_construction(tmp_path: Path) -> None:
    assert energy_budget_input_path(tmp_path, "N7", "budget.dat") == tmp_path / "case_N7/budget.dat"


def test_load_valid_five_column_file(tmp_path: Path) -> None:
    path = tmp_path / "energy_budget.dat"
    _write_data(path, np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 5.0, 6.0, 7.0, 8.0]]))

    data = load_energy_budget(path)

    np.testing.assert_array_equal(data.time, [0.0, 1.0])
    np.testing.assert_array_equal(data.E_k, [1.0, 5.0])
    np.testing.assert_array_equal(data.E_p, [2.0, 6.0])
    np.testing.assert_array_equal(data.E_total, [3.0, 7.0])
    np.testing.assert_array_equal(data.epsilon, [4.0, 8.0])


def test_load_valid_file_sorts_time_and_ignores_extra_columns(tmp_path: Path) -> None:
    path = tmp_path / "energy_budget.dat"
    _write_data(
        path,
        np.array(
            [
                [2.0, 20.0, 21.0, 22.0, 23.0, 999.0],
                [0.0, 0.0, 1.0, 2.0, 3.0, 998.0],
                [1.0, 10.0, 11.0, 12.0, 13.0, 997.0],
            ]
        ),
    )

    data = load_energy_budget(path)

    np.testing.assert_array_equal(data.time, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(data.E_k, [0.0, 10.0, 20.0])
    np.testing.assert_array_equal(data.E_p, [1.0, 11.0, 21.0])
    np.testing.assert_array_equal(data.E_total, [2.0, 12.0, 22.0])
    np.testing.assert_array_equal(data.epsilon, [3.0, 13.0, 23.0])


def test_load_rejects_one_row(tmp_path: Path) -> None:
    path = tmp_path / "one.dat"
    _write_data(path, np.array([[0.0, 1.0, 2.0, 3.0, 4.0]]))

    with pytest.raises(ValueError, match="at least two time samples"):
        load_energy_budget(path)


def test_load_rejects_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.dat"

    with pytest.raises(FileNotFoundError, match=f"Energy budget file not found: {path}"):
        load_energy_budget(path)


def test_load_rejects_fewer_than_five_columns(tmp_path: Path) -> None:
    path = tmp_path / "short.dat"
    _write_data(path, np.array([[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]]))

    with pytest.raises(ValueError, match="must contain at least 5 columns"):
        load_energy_budget(path)


def test_load_rejects_duplicate_time_after_sorting(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.dat"
    _write_data(path, np.array([[1.0, 1.0, 2.0, 3.0, 4.0], [1.0, 5.0, 6.0, 7.0, 8.0]]))

    with pytest.raises(ValueError, match="time values must be unique after sorting"):
        load_energy_budget(path)


@pytest.mark.parametrize(("column", "name"), [(1, "E_k"), (2, "E_p"), (3, "E_total"), (4, "epsilon")])
def test_load_rejects_nonfinite_fields(tmp_path: Path, column: int, name: str) -> None:
    path = tmp_path / f"nonfinite-{name}.dat"
    values = np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 5.0, 6.0, 7.0, 8.0]])
    values[1, column] = np.inf
    _write_data(path, values)

    with pytest.raises(ValueError, match=f"contains non-finite {name} values"):
        load_energy_budget(path)


def test_load_rejects_nonfinite_time(tmp_path: Path) -> None:
    path = tmp_path / "nonfinite-time.dat"
    _write_data(path, np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [np.nan, 5.0, 6.0, 7.0, 8.0]]))

    with pytest.raises(ValueError, match="contains non-finite time values"):
        load_energy_budget(path)


def test_timeseries_csv_exact_header_and_numeric_formatting(tmp_path: Path) -> None:
    path = tmp_path / "nested/timeseries.csv"

    write_energy_timeseries_csv(path, _diagnostics(), overwrite=False)

    rows = list(csv.reader(path.open(encoding="utf-8")))
    assert rows[0] == list(TIMESERIES_COLUMNS)
    assert rows[1][TIMESERIES_COLUMNS.index("E_k")] == "0.3333333333333333"


def test_summary_csv_exact_header_and_integer_formatting(tmp_path: Path) -> None:
    path = tmp_path / "summary.csv"
    row = build_energy_summary_row("N5", _diagnostics(), target=1.0)

    write_energy_summary_csv(path, [row], overwrite=False)

    rows = list(csv.reader(path.open(encoding="utf-8")))
    assert rows[0] == list(SUMMARY_COLUMNS)
    assert rows[1][SUMMARY_COLUMNS.index("case")] == "N5"
    assert rows[1][SUMMARY_COLUMNS.index("n_points")] == "2"


def test_numeric_formatter_uses_16_significant_digits_and_preserves_integers() -> None:
    assert format_numeric_value(1.0 / 3.0) == "0.3333333333333333"
    assert format_numeric_value(7) == "7"


@pytest.mark.parametrize("writer", ["timeseries", "summary"])
def test_csv_overwrite_protection(tmp_path: Path, writer: str) -> None:
    path = tmp_path / "existing.csv"
    path.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError, match=r"Output exists: .* Pass --overwrite to replace it\."):
        if writer == "timeseries":
            write_energy_timeseries_csv(path, _diagnostics(), overwrite=False)
        else:
            row = build_energy_summary_row("N5", _diagnostics(), target=1.0)
            write_energy_summary_csv(path, [row], overwrite=False)
    assert path.read_text(encoding="utf-8") == "keep me"


def test_exact_output_filenames(tmp_path: Path) -> None:
    assert timeseries_csv_path(tmp_path, "N5").name == "N5_energy_budget_closure_timeseries.csv"
    assert summary_csv_path(tmp_path).name == "energy_budget_closure_summary.csv"
    assert component_figure_path(tmp_path, "N5").name == "energy_budget_components_N5.png"
    assert epsilon_figure_path(tmp_path).name == "epsilon_vs_time.png"
    assert energy_closure_figure_path(tmp_path).name == "energy_closure_vs_time.png"
    assert closure_residual_figure_path(tmp_path).name == "closure_residual_from_target_vs_time.png"
    assert closure_drift_figure_path(tmp_path).name == "closure_drift_from_initial_vs_time.png"
    assert differential_closure_residual_figure_path(tmp_path).name == "differential_closure_residual_vs_time.png"


def test_summary_table_preserves_title_columns_and_row_order() -> None:
    first = build_energy_summary_row("N7", _diagnostics(), target=1.0)
    second = build_energy_summary_row("N5", _diagnostics(), target=1.0)

    text = format_energy_summary_table([first, second])

    lines = text.splitlines()
    assert lines[0] == "Energy budget closure summary:"
    assert lines[1].split() == [
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
    ]
    assert lines[3].startswith("N7")
    assert lines[4].startswith("N5")
