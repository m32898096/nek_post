from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest

from nek_post import fig5a_detected_front_plotting as plotting
from nek_post.fig5a_detected_front_plotting import (
    write_detected_front_overlay_plots,
)


def _automatic() -> dict[str, np.ndarray | float]:
    return {
        "time": np.array([0.0, 1.0, 2.0, 3.0]),
        "x_relative": np.array([0.0, -1.0, 2.0, 3.0]),
        "automatic_x0": 8.0,
    }


def _paper() -> dict[str, np.ndarray]:
    return {
        "time": np.array([0.0, 1.0, 2.0, 3.0]),
        "paper_x": np.array([0.0, 1.0, 1.5, 2.5]),
    }


def _comparison() -> dict[str, np.ndarray]:
    return {
        "time": np.array([0.0, 1.0, 2.0, 3.0]),
        "difference": np.array([0.0, -2.0, 0.5, 0.5]),
    }


def test_all_detected_front_figures_are_written_and_closed(
    tmp_path: Path,
) -> None:
    paths = write_detected_front_overlay_plots(
        output_dir=tmp_path,
        case="GC8950_N7",
        automatic_front=_automatic(),
        paper=_paper(),
        comparison=_comparison(),
        slump_tmin=1.0,
        slump_tmax=3.0,
        overwrite=False,
    )

    assert [path.name for path in paths] == [
        "GC8950_N7_Re8950_fig5a_overlay_linear.png",
        "GC8950_N7_Re8950_fig5a_overlay_loglog.png",
        "GC8950_N7_Re8950_fig5a_difference.png",
        "GC8950_N7_Re8950_fig5a_slumping_overlay.png",
    ]
    assert all(path.is_file() for path in paths)
    assert plt.get_fignums() == []


def test_plot_content_uses_positive_log_points_and_difference_terminology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, dict[str, object]] = {}
    original_save = plotting._save_figure

    def capture(fig, path):
        axis = fig.axes[0]
        captured[path.name] = {
            "title": axis.get_title(),
            "xlabel": axis.get_xlabel(),
            "ylabel": axis.get_ylabel(),
            "lines": {
                line.get_label(): (
                    np.asarray(line.get_xdata()),
                    np.asarray(line.get_ydata()),
                )
                for line in axis.lines
            },
        }
        original_save(fig, path)

    monkeypatch.setattr(plotting, "_save_figure", capture)
    write_detected_front_overlay_plots(
        output_dir=tmp_path,
        case="GC8950_N7",
        automatic_front=_automatic(),
        paper=_paper(),
        comparison=_comparison(),
        slump_tmin=1.0,
        slump_tmax=3.0,
        overwrite=False,
    )

    linear = captured[
        "GC8950_N7_Re8950_fig5a_overlay_linear.png"
    ]
    assert linear["xlabel"] == "t"
    assert linear["ylabel"] == "x_front - x0"
    assert set(linear["lines"]) == {
        "Cantero Figure 5a 3D Re8950",
        "GC8950_N7 automatic spectral front",
    }

    loglog = captured[
        "GC8950_N7_Re8950_fig5a_overlay_loglog.png"
    ]
    paper_xy = loglog["lines"]["Cantero Figure 5a 3D Re8950"]
    automatic_xy = loglog["lines"][
        "GC8950_N7 automatic spectral front"
    ]
    np.testing.assert_array_equal(paper_xy[0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(automatic_xy[0], [2.0, 3.0])

    difference = captured[
        "GC8950_N7_Re8950_fig5a_difference.png"
    ]
    assert "error" not in difference["title"].lower()
    assert "Automatic spectral front - Cantero Re8950" in difference["title"]
    np.testing.assert_array_equal(
        difference["lines"]["difference"][1], [0.0, -2.0, 0.5, 0.5]
    )


def test_plot_preflight_protects_all_outputs(tmp_path: Path) -> None:
    conflict = (
        tmp_path / "GC8950_N7_Re8950_fig5a_difference.png"
    )
    conflict.write_bytes(b"keep")

    with pytest.raises(FileExistsError, match="Pass --overwrite"):
        write_detected_front_overlay_plots(
            output_dir=tmp_path,
            case="GC8950_N7",
            automatic_front=_automatic(),
            paper=_paper(),
            comparison=_comparison(),
            slump_tmin=1.0,
            slump_tmax=3.0,
            overwrite=False,
        )

    assert conflict.read_bytes() == b"keep"
    assert not (
        tmp_path / "GC8950_N7_Re8950_fig5a_overlay_linear.png"
    ).exists()
    assert plt.get_fignums() == []
