from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest

from nek_post.front_detection import (
    FrontTrackingResult,
    STATUS_NO_THRESHOLD_COMPONENT,
    STATUS_SELECTED_INITIAL,
    STATUS_SELECTED_TRACKED,
)
from nek_post.front_detection_compare import FrontDetectionComparison
from nek_post import front_detection_plotting
from nek_post.front_detection_plotting import write_front_detection_plots


def _tracking() -> FrontTrackingResult:
    time = np.array([0.0, 1.0, 2.0])
    x = np.array([1.0, np.nan, 3.0])
    return FrontTrackingResult(
        time=time,
        x_front=x,
        predicted_x=np.array([np.nan, 1.0, 2.0]),
        tracking_error=np.array([np.nan, np.nan, 1.0]),
        selected_component_label=np.array([1, -1, 1]),
        selected_component_pixels=np.array([3, 0, 3]),
        selected_component_xmin=np.array([0.0, np.nan, 2.0]),
        selected_component_xmax=x.copy(),
        selected_bottom_contact=np.array([True, False, True]),
        component_count=np.array([1, 0, 1]),
        spatial_candidate_count=np.array([1, 0, 1]),
        temporal_candidate_count=np.array([0, 0, 1]),
        selected_overlap_pixels=np.array([0, 0, 2]),
        status=(
            STATUS_SELECTED_INITIAL,
            STATUS_NO_THRESHOLD_COMPONENT,
            STATUS_SELECTED_TRACKED,
        ),
        threshold=0.01,
        min_component_pixels=2,
        bottom_rows=1,
        max_front_jump=1.0,
        connectivity=8,
    )


def _comparison() -> FrontDetectionComparison:
    return FrontDetectionComparison(
        time=np.array([0.0, 2.0]),
        file_indices=np.array([1, 3]),
        x_front_auto=np.array([1.0, 3.0]),
        x_front_reference=np.array([1.1, 2.9]),
        difference=np.array([-0.1, 0.1]),
        absolute_difference=np.array([0.1, 0.1]),
    )


def _reference() -> dict[str, np.ndarray]:
    return {
        "time": np.array([0.0, 1.0, 2.0]),
        "x_front": np.array([1.1, 2.0, 2.9]),
    }


def test_both_figures_created_in_exact_order_and_closed(tmp_path: Path) -> None:
    paths = write_front_detection_plots(
        tmp_path,
        "N7",
        _tracking(),
        _reference(),
        _comparison(),
        overwrite=False,
    )

    assert paths == [
        tmp_path / "N7_front_detection_overlay.png",
        tmp_path / "N7_front_detection_difference.png",
    ]
    assert all(path.is_file() for path in paths)
    assert plt.get_fignums() == []


def test_plot_labels_and_failed_detection_is_not_converted_to_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_save = front_detection_plotting.save_figure
    plotted: dict[str, dict[str, np.ndarray]] = {}
    metadata: dict[str, tuple[str, str]] = {}

    def capture_and_save(fig, path, overwrite):
        metadata[path.name] = (
            fig.axes[0].get_title(),
            fig.axes[0].get_ylabel(),
        )
        plotted[path.name] = {
            line.get_label(): np.asarray(line.get_ydata())
            for line in fig.axes[0].lines
        }
        original_save(fig, path, overwrite)

    monkeypatch.setattr(front_detection_plotting, "save_figure", capture_and_save)

    write_front_detection_plots(
        tmp_path,
        "N7",
        _tracking(),
        _reference(),
        _comparison(),
        overwrite=False,
    )

    overlay = plotted["N7_front_detection_overlay.png"]
    difference = plotted["N7_front_detection_difference.png"]
    assert set(overlay) == {"front_simple reference", "automatic front"}
    np.testing.assert_array_equal(overlay["automatic front"], [1.0, 3.0])
    assert not np.any(overlay["automatic front"] == 0.0)
    assert set(difference) == {"difference", "zero"}
    np.testing.assert_allclose(difference["difference"], [-0.1, 0.1])
    assert metadata["N7_front_detection_overlay.png"] == (
        "Automatic front and front_simple reference: N7",
        "front position",
    )
    assert metadata["N7_front_detection_difference.png"] == (
        "Automatic front difference from front_simple: N7",
        "x_auto - x_front_simple",
    )
    assert plt.get_fignums() == []


def test_plot_preflight_rejects_before_creating_other_output(
    tmp_path: Path,
) -> None:
    conflict = tmp_path / "N7_front_detection_difference.png"
    conflict.write_bytes(b"keep")

    with pytest.raises(
        FileExistsError,
        match=r"Output exists: .* Pass --overwrite to replace it\.",
    ):
        write_front_detection_plots(
            tmp_path,
            "N7",
            _tracking(),
            _reference(),
            _comparison(),
            overwrite=False,
        )

    assert not (tmp_path / "N7_front_detection_overlay.png").exists()
    assert conflict.read_bytes() == b"keep"
    assert plt.get_fignums() == []


def test_plot_overwrite_true_replaces_outputs(tmp_path: Path) -> None:
    paths = [
        tmp_path / "N7_front_detection_overlay.png",
        tmp_path / "N7_front_detection_difference.png",
    ]
    for path in paths:
        path.write_bytes(b"old")

    written = write_front_detection_plots(
        tmp_path,
        "N7",
        _tracking(),
        _reference(),
        _comparison(),
        overwrite=True,
    )

    assert written == paths
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in paths)
    assert plt.get_fignums() == []
