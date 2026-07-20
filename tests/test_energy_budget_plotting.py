from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest

from nek_post.energy_budget import EnergyBudgetData, compute_energy_diagnostics
from nek_post.energy_budget_plotting import write_energy_budget_plots


def _diagnostics(offset: float):
    data = EnergyBudgetData(
        time=np.array([0.0, 1.0, 2.0]),
        E_k=np.array([1.0, 1.5, 2.0]) + offset,
        E_p=np.array([2.0, 1.5, 1.0]),
        E_total=np.array([3.0, 2.0, 1.0]) + offset,
        epsilon=np.array([1.0, 1.0, 1.0]),
    )
    return compute_energy_diagnostics(data, target=3.0)


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
