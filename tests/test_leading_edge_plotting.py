from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from nek_post.leading_edge_artifacts import LeadingEdgePlotData
from nek_post.leading_edge_io import (
    leading_edge_evolution_pdf_path,
    leading_edge_evolution_png_path,
)
from nek_post.leading_edge_plotting import (
    periodic_leading_edge_plot_arrays,
    write_leading_edge_artifact_plots,
    write_leading_edge_evolution_plots,
)
from nek_post.leading_edge_workflow import (
    LeadingEdgeEvolution,
    select_leading_edge_times,
)


def _readonly(values: object, dtype: object) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _outputs(*, all_nan: bool = False):
    x_front = np.array(
        [
            [0.20, np.nan, 0.40, 0.30],
            [0.35, 0.45, 0.55, 0.50],
        ]
    )
    if all_nan:
        x_front[:] = np.nan
    metadata = MappingProxyType(
        {
            "ymin": 0.0,
            "ymax_periodic_endpoint": 1.0,
            "spectral_horizontal_algorithm_version": 1,
            "inverse_mapping_target_count": 20,
            "inverse_mapping_success_count": 20,
            "inverse_mapping_failure_count": 0,
            "ambiguous_boundary_point_count": 0,
            "maximum_successful_residual": 1.0e-13,
            "maximum_iteration_count": 4,
        }
    )
    evolution = LeadingEdgeEvolution(
        file_indices=_readonly([4, 8], np.int64),
        source_files=(Path("GC0.f00004"), Path("GC0.f00008")),
        time=_readonly([0.7, 1.1], np.float64),
        x=_readonly([0.0, 0.5, 1.0], np.float64),
        y=_readonly([0.0, 0.25, 0.5, 0.75], np.float64),
        x_front=_readonly(x_front, np.float64),
        success_mask=_readonly(np.isfinite(x_front), np.bool_),
        crossing_count=_readonly(np.isfinite(x_front), np.int64),
        finite_leading_edge_fraction=_readonly(
            np.mean(np.isfinite(x_front), axis=1), np.float64
        ),
        successful_y_count=_readonly(
            np.count_nonzero(np.isfinite(x_front), axis=1), np.int64
        ),
        threshold=0.1,
        z_target=0.04,
        nx=3,
        native_ny=2,
        dense_ny=4,
        y_upsample_factor=2,
        horizontal_plan_metadata=metadata,
        periodic_endpoint_included=False,
    )
    return evolution, select_leading_edge_times(evolution, spacing=None)


def test_headless_backend_and_png_pdf_creation(tmp_path: Path) -> None:
    evolution, selection = _outputs()
    before = set(plt.get_fignums())

    paths = write_leading_edge_evolution_plots(
        tmp_path,
        "N7",
        evolution,
        selection,
        overwrite=False,
        reynolds_number=3450,
    )

    assert matplotlib.get_backend().lower() == "agg"
    assert paths == [
        leading_edge_evolution_png_path(tmp_path, "N7"),
        leading_edge_evolution_pdf_path(tmp_path, "N7"),
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    assert set(plt.get_fignums()) == before


def test_artifact_plot_adapter_writes_the_same_png_pdf_definition(
    tmp_path: Path,
) -> None:
    evolution, selection = _outputs()
    plot_data = LeadingEdgePlotData(
        case="N7",
        file_indices=selection.file_indices,
        target_time=selection.target_time,
        actual_time=selection.actual_time,
        time_error=selection.time_error,
        y=evolution.y,
        x_front=selection.x_front,
        success_mask=selection.success_mask,
        crossing_count=selection.crossing_count,
        threshold=evolution.threshold,
        z_target=evolution.z_target,
        nx=evolution.nx,
        native_ny=evolution.native_ny,
        dense_ny=evolution.dense_ny,
        y_upsample_factor=evolution.y_upsample_factor,
        y_min=0.0,
        y_max_periodic_endpoint=1.0,
        periodic_endpoint_included=False,
        target_time_spacing=None,
    )

    paths = write_leading_edge_artifact_plots(
        tmp_path,
        plot_data,
        overwrite=False,
        reynolds_number=3450,
    )

    assert paths == [
        leading_edge_evolution_png_path(tmp_path, "N7"),
        leading_edge_evolution_pdf_path(tmp_path, "N7"),
    ]
    assert all(path.stat().st_size > 100 for path in paths)


def test_periodic_plotting_closure_is_copy_and_preserves_nan_gap() -> None:
    y = np.array([0.0, 0.25, 0.5, 0.75])
    x_front = np.array([0.2, np.nan, 0.4, 0.3])
    original_y = y.copy()
    original_x = x_front.copy()

    y_plot, x_plot = periodic_leading_edge_plot_arrays(y, x_front, 1.0)

    assert_array_equal(y_plot, [0.0, 0.25, 0.5, 0.75, 1.0])
    assert x_plot[0] == 0.2
    assert np.isnan(x_plot[1])
    assert x_plot[-1] == x_front[0]
    assert_array_equal(y, original_y)
    assert_array_equal(x_front, original_x)
    assert not np.shares_memory(y_plot, y)
    assert not np.shares_memory(x_plot, x_front)


def test_plot_passes_raw_nan_curves_and_periodic_copies_to_matplotlib(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evolution, selection = _outputs()
    original_x = evolution.x_front.copy()
    original_y = evolution.y.copy()
    captured: list[tuple[np.ndarray, np.ndarray, dict[str, object]]] = []
    original_plot = Axes.plot

    def capture_plot(self, x, y, *args, **kwargs):
        captured.append((np.asarray(x).copy(), np.asarray(y).copy(), dict(kwargs)))
        return original_plot(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", capture_plot)
    write_leading_edge_evolution_plots(
        tmp_path,
        "N7",
        evolution,
        selection,
        overwrite=False,
    )

    assert len(captured) == 2
    first_x, first_y, first_kwargs = captured[0]
    assert first_x.shape == (evolution.dense_ny + 1,)
    assert first_y.shape == (evolution.dense_ny + 1,)
    assert np.isnan(first_x[1])
    assert first_x[-1] == evolution.x_front[0, 0]
    assert first_y[-1] == 1.0
    assert first_kwargs["linestyle"] == "-"
    assert "marker" not in first_kwargs
    assert_array_equal(evolution.x_front, original_x)
    assert_array_equal(evolution.y, original_y)


@pytest.mark.parametrize("conflicting_suffix", (".png", ".pdf"))
def test_preflight_conflict_writes_neither_plot(
    tmp_path: Path,
    conflicting_suffix: str,
) -> None:
    evolution, selection = _outputs()
    png = leading_edge_evolution_png_path(tmp_path, "N7")
    pdf = leading_edge_evolution_pdf_path(tmp_path, "N7")
    conflict = png if conflicting_suffix == ".png" else pdf
    other = pdf if conflict == png else png
    conflict.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Pass --overwrite"):
        write_leading_edge_evolution_plots(
            tmp_path,
            "N7",
            evolution,
            selection,
            overwrite=False,
        )

    assert conflict.read_text(encoding="utf-8") == "keep"
    assert not other.exists()


def test_overwrite_true_replaces_both_plots(tmp_path: Path) -> None:
    evolution, selection = _outputs()
    paths = [
        leading_edge_evolution_png_path(tmp_path, "N7"),
        leading_edge_evolution_pdf_path(tmp_path, "N7"),
    ]
    for path in paths:
        path.write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_leading_edge_evolution_plots(
            tmp_path,
            "N7",
            evolution,
            selection,
            overwrite=False,
        )

    written = write_leading_edge_evolution_plots(
        tmp_path,
        "N7",
        evolution,
        selection,
        overwrite=True,
    )
    assert written == paths
    assert all(path.stat().st_size > 100 for path in paths)


def test_all_nonfinite_selected_points_raise_without_outputs(tmp_path: Path) -> None:
    evolution, selection = _outputs(all_nan=True)

    with pytest.raises(ValueError, match="no finite leading-edge point"):
        write_leading_edge_evolution_plots(
            tmp_path,
            "N7",
            evolution,
            selection,
            overwrite=False,
        )

    assert not leading_edge_evolution_png_path(tmp_path, "N7").exists()
    assert not leading_edge_evolution_pdf_path(tmp_path, "N7").exists()


def test_figure_is_closed_when_save_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evolution, selection = _outputs()
    before = set(plt.get_fignums())

    def failing_savefig(self, *args, **kwargs):
        raise OSError("synthetic save failure")

    monkeypatch.setattr(Figure, "savefig", failing_savefig)
    with pytest.raises(OSError, match="synthetic save failure"):
        write_leading_edge_evolution_plots(
            tmp_path,
            "N7",
            evolution,
            selection,
            overwrite=False,
        )

    assert set(plt.get_fignums()) == before
