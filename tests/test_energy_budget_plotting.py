from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest

from nek_post.energy_budget import EnergyBudgetData, compute_energy_diagnostics
from nek_post import energy_budget_plotting
from nek_post.energy_budget_plotting import (
    plot_energy_components,
    plot_energy_overlay,
    write_energy_budget_plots,
)


def _diagnostics(offset: float):
    data = EnergyBudgetData(
        time=np.array([0.0, 1.0, 2.0]),
        E_k=np.array([1.0, 1.5, 2.0]) + offset,
        E_p=np.array([2.0, 1.5, 1.0]),
        E_total=np.array([3.0, 2.0, 1.0]) + offset,
        epsilon=np.array([1.0, 1.0, 1.0]),
    )
    return compute_energy_diagnostics(data, target=3.0)


def _n7_diagnostics():
    data = EnergyBudgetData(
        time=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        E_k=np.array([1.0, 1.2, 1.4, 1.6, 1.8]),
        E_p=np.array([10.0, 9.5, 9.0, 8.5, 8.0]),
        E_total=np.array([11.0, 10.7, 10.4, 10.1, 9.8]),
        epsilon=np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
    )
    return compute_energy_diagnostics(data, target=12.0)


def _capture_saved_figure(monkeypatch: pytest.MonkeyPatch):
    saved = {}

    def capture(fig, path, overwrite):
        saved["fig"] = fig
        saved["path"] = path
        saved["overwrite"] = overwrite

    monkeypatch.setattr(energy_budget_plotting, "save_figure", capture)
    return saved


def test_component_figure_uses_one_shared_axis_and_required_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    diagnostics = _n7_diagnostics()
    saved = _capture_saved_figure(monkeypatch)

    plot_energy_components(
        tmp_path / "energy_budget_components_N7.png",
        "N7",
        diagnostics,
        12.0,
        overwrite=True,
    )

    fig = saved["fig"]
    try:
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_title() == "Energy budget and closure: N7"
        assert ax.get_xlim() == pytest.approx((0.0, 20.0))
        assert ax.get_ylim() == pytest.approx((0.0, 15.0))
        assert ax.get_xticks().tolist() == [0, 5, 10, 15, 20]
        assert ax.get_yticks().tolist() == list(range(16))

        lines = {line.get_label(): line for line in ax.lines}
        assert set(lines) == {
            "E_k",
            "E_p",
            "E_total",
            "epsilon",
            "energy closure",
            "target 12",
        }

        expected_time = diagnostics.time[::2]
        raw_series = {
            "E_k": diagnostics.E_k,
            "E_p": diagnostics.E_p,
            "E_total": diagnostics.E_total,
            "epsilon": diagnostics.epsilon,
        }
        for label, source_values in raw_series.items():
            line = lines[label]
            np.testing.assert_array_equal(line.get_xdata(), expected_time)
            np.testing.assert_array_equal(line.get_ydata(), source_values[::2])
            assert line.get_marker() == "o"
            assert line.get_linestyle() == "None"

        np.testing.assert_array_equal(
            lines["energy closure"].get_xdata(), diagnostics.time
        )
        np.testing.assert_array_equal(
            lines["energy closure"].get_ydata(), diagnostics.energy_closure
        )
        np.testing.assert_array_equal(
            lines["epsilon"].get_ydata(), diagnostics.epsilon[::2]
        )
        np.testing.assert_array_equal(lines["target 12"].get_ydata(), [12.0, 12.0])
        assert saved["overwrite"] is True
    finally:
        plt.close(fig)


def test_energy_closure_overlay_uses_teacher_requested_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    diagnostics = _n7_diagnostics()
    saved = _capture_saved_figure(monkeypatch)

    plot_energy_overlay(
        tmp_path / "energy_closure_vs_time.png",
        {"N7": diagnostics},
        "energy_closure",
        "Energy closure by case",
        "E_total + integral epsilon dt",
        overwrite=True,
        target=12.0,
    )

    fig = saved["fig"]
    try:
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_xlim() == pytest.approx((0.0, 20.0))
        assert ax.get_ylim() == pytest.approx((10.0, 13.0))
        assert ax.get_xticks().tolist() == [0, 5, 10, 15, 20]
        assert ax.get_yticks().tolist() == [
            10.0,
            10.5,
            11.0,
            11.5,
            12.0,
            12.5,
            13.0,
        ]
        assert [line.get_label() for line in ax.lines] == ["N7", "target 12"]
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), diagnostics.energy_closure)
    finally:
        plt.close(fig)


def test_writes_component_and_overlay_figures_in_established_order(tmp_path: Path) -> None:
    diagnostics_by_case = {"N7": _diagnostics(0.1), "N5": _diagnostics(0.0)}

    paths = write_energy_budget_plots(tmp_path, diagnostics_by_case, target=3.0, overwrite=False)

    assert paths == [
        tmp_path / "energy_budget_components_N7.png",
        tmp_path / "energy_budget_components_N5.png",
        tmp_path / "epsilon_vs_time.png",
        tmp_path / "energy_closure_vs_time.png",
        tmp_path / "closure_residual_from_target_vs_time.png",
        tmp_path / "closure_drift_from_initial_vs_time.png",
        tmp_path / "differential_closure_residual_vs_time.png",
    ]
    assert all(path.is_file() for path in paths)
    assert plt.get_fignums() == []


def test_plot_overwrite_protection_closes_rejected_figure(tmp_path: Path) -> None:
    diagnostics_by_case = {"N5": _diagnostics(0.0)}
    existing = tmp_path / "energy_budget_components_N5.png"
    existing.write_bytes(b"keep me")

    with pytest.raises(FileExistsError, match=r"Output exists: .* Pass --overwrite to replace it\."):
        write_energy_budget_plots(tmp_path, diagnostics_by_case, target=3.0, overwrite=False)

    assert existing.read_bytes() == b"keep me"
    assert plt.get_fignums() == []
