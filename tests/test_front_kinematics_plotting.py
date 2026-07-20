from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest

from nek_post.front_kinematics_plotting import write_front_kinematics_plots


def _kinematics(offset: float) -> dict[str, np.ndarray]:
    time = np.array([0.0, 1.0, 2.0])
    x_front = np.array([1.0, 2.0, 3.0]) + offset
    x_reconstructed = np.array([1.0, 2.1, 2.9]) + offset
    return {
        "time": time,
        "x_front": x_front,
        "v_raw": np.array([1.0, 1.0, 1.0]),
        "v_smooth": np.array([0.9, 1.0, 1.1]),
        "x_reconstructed": x_reconstructed,
        "x_reconstruction_error": x_reconstructed - x_front,
    }


def test_all_figures_are_created_in_established_order_and_closed(tmp_path: Path) -> None:
    kinematics_by_case = {"N7": _kinematics(0.2), "N5": _kinematics(0.0)}

    paths = write_front_kinematics_plots(tmp_path, kinematics_by_case, overwrite=False)

    assert paths == [
        tmp_path / "front_position_xt_vs_time.png",
        tmp_path / "front_velocity_raw_vs_time.png",
        tmp_path / "front_velocity_smoothed_vs_time.png",
        tmp_path / "front_reconstruction_error_vs_time.png",
        tmp_path / "front_position_reconstructed_vs_original_N7.png",
        tmp_path / "front_position_reconstructed_vs_original_N5.png",
    ]
    assert all(path.is_file() for path in paths)
    assert plt.get_fignums() == []


def test_plot_overwrite_protection_closes_rejected_figure(tmp_path: Path) -> None:
    existing = tmp_path / "front_position_xt_vs_time.png"
    existing.write_bytes(b"keep me")

    with pytest.raises(FileExistsError, match=r"Output exists: .* Pass --overwrite to replace it\."):
        write_front_kinematics_plots(tmp_path, {"N5": _kinematics(0.0)}, overwrite=False)

    assert existing.read_bytes() == b"keep me"
    assert plt.get_fignums() == []


def test_plot_overwrite_succeeds_when_enabled(tmp_path: Path) -> None:
    existing = tmp_path / "front_position_xt_vs_time.png"
    existing.write_bytes(b"old")

    paths = write_front_kinematics_plots(tmp_path, {"N5": _kinematics(0.0)}, overwrite=True)

    assert paths[0] == existing
    assert existing.read_bytes().startswith(b"\x89PNG")
    assert all(path.is_file() for path in paths)
    assert plt.get_fignums() == []
