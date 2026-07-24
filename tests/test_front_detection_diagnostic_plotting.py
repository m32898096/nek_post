from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest

from nek_post.front_detection import (
    FrontComponent,
    STATUS_NO_VALID_TEMPORAL_CANDIDATE,
    STATUS_SELECTED_TRACKED,
)
from nek_post.front_detection_diagnostics import FrontFrameDiagnostic
from nek_post import front_detection_diagnostic_plotting
from nek_post.front_detection_diagnostic_plotting import (
    plot_front_frame_diagnostic,
    write_front_frame_diagnostic_plots,
)


def _inputs():
    Xi, Zi = np.meshgrid(np.arange(8.0), np.arange(3.0))
    concentration = np.zeros_like(Xi)
    concentration[0, 0:2] = 1.0
    concentration[0, 3:5] = 1.0
    concentration[2, 6:8] = 1.0
    concentration[1, 7] = np.nan
    selected_mask = np.zeros_like(concentration, dtype=bool)
    selected_mask[0, 0:2] = True
    selected = FrontComponent(
        label=1,
        pixel_count=2,
        x_min=0.0,
        x_max=1.0,
        z_min=0.0,
        z_max=0.0,
        bottom_contact=True,
        mask=selected_mask,
    )
    other_mask = np.zeros_like(concentration, dtype=bool)
    other_mask[0, 3:5] = True
    other_spatial = FrontComponent(
        label=2,
        pixel_count=2,
        x_min=3.0,
        x_max=4.0,
        z_min=0.0,
        z_max=0.0,
        bottom_contact=True,
        mask=other_mask,
    )
    rejected_mask = np.zeros_like(concentration, dtype=bool)
    rejected_mask[2, 6:8] = True
    rejected = FrontComponent(
        label=3,
        pixel_count=2,
        x_min=6.0,
        x_max=7.0,
        z_min=2.0,
        z_max=2.0,
        bottom_contact=False,
        mask=rejected_mask,
    )
    sequence = SimpleNamespace(Xi=Xi, Zi=Zi)
    diagnostic = FrontFrameDiagnostic(
        frame_position=0,
        file_index=41,
        source_file=Path("GC0.f00041"),
        time=10.0,
        concentration=concentration,
        components=(selected, other_spatial, rejected),
        spatial_components=(selected, other_spatial),
        selected_component=selected,
        x_front=1.0,
        predicted_x=0.75,
        tracking_error=0.25,
        status=STATUS_SELECTED_TRACKED,
        component_count=3,
        spatial_candidate_count=2,
        temporal_candidate_count=1,
        selected_component_label=1,
        selected_component_pixels=2,
        selected_overlap_pixels=1,
        threshold=0.01,
        reference_x=1.25,
    )
    return sequence, diagnostic


def test_exact_diagnostics_subdirectory_filename_and_requested_order(
    tmp_path: Path,
) -> None:
    sequence, diagnostic = _inputs()
    earlier = replace(diagnostic, file_index=9, time=2.0)

    paths = write_front_frame_diagnostic_plots(
        tmp_path,
        "N7",
        sequence,
        (diagnostic, earlier),
        overwrite=False,
    )

    assert paths == [
        tmp_path / "diagnostics/N7_front_diagnostic_f00041.png",
        tmp_path / "diagnostics/N7_front_diagnostic_f00009.png",
    ]
    assert all(path.is_file() for path in paths)
    assert plt.get_fignums() == []


def test_plot_structure_lines_annotation_and_external_reference_language(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence, diagnostic = _inputs()
    original_save = front_detection_diagnostic_plotting.save_figure
    original_draw = front_detection_diagnostic_plotting._draw_mask_boundary
    captured = {}
    drawn_components = []

    def capture_boundary(ax, Xi, Zi, mask, **style):
        label = next(
            component.label
            for component in diagnostic.components
            if np.array_equal(component.mask, mask)
        )
        drawn_components.append((label, style))
        original_draw(ax, Xi, Zi, mask, **style)

    def capture_and_save(fig, path, overwrite):
        ax = fig.axes[0]
        legend = ax.get_legend()
        annotation = ax.texts[0]
        captured["title"] = ax.get_title()
        captured["xlabel"] = ax.get_xlabel()
        captured["ylabel"] = ax.get_ylabel()
        captured["colorbar_label"] = fig.axes[1].get_ylabel()
        captured["legend"] = [text.get_text() for text in legend.texts]
        captured["legend_loc"] = legend._loc
        captured["legend_ncols"] = legend._ncols
        captured["legend_anchor"] = legend.get_bbox_to_anchor()._bbox.bounds
        captured["annotation_position"] = annotation.get_position()
        captured["annotation_in_axes"] = annotation.get_transform() is ax.transAxes
        captured["annotation_zorder"] = annotation.get_zorder()
        captured["lines"] = {
            line.get_label(): np.asarray(line.get_xdata(), dtype=float)
            for line in ax.lines
        }
        captured["annotation"] = "\n".join(text.get_text() for text in ax.texts)
        original_save(fig, path, overwrite)

    monkeypatch.setattr(
        front_detection_diagnostic_plotting,
        "save_figure",
        capture_and_save,
    )
    monkeypatch.setattr(
        front_detection_diagnostic_plotting,
        "_draw_mask_boundary",
        capture_boundary,
    )

    plot_front_frame_diagnostic(
        tmp_path / "diagnostic.png",
        "N7",
        sequence,
        diagnostic,
        overwrite=False,
    )

    assert captured["title"] == "N7 front diagnostic — f00041, t=10"
    assert captured["xlabel"] == "x"
    assert captured["ylabel"] == "z"
    assert captured["colorbar_label"] == "concentration C"
    assert captured["legend"] == [
        "C = 0.01 contour",
        "rejected threshold components",
        "other spatially valid components",
        "selected component",
        "automatic front",
        "predicted front",
        "front_simple reference",
    ]
    assert "C > threshold" not in captured["legend"]
    assert len(captured["legend"]) == len(set(captured["legend"]))
    assert captured["legend"].count("selected component") == 1
    assert captured["legend_loc"] == 9
    assert captured["legend_ncols"] == 4
    assert captured["legend_anchor"][:2] == pytest.approx((0.5, -0.14))
    assert captured["legend_anchor"][1] < 0.0
    assert captured["annotation_position"] == pytest.approx((0.01, 0.99))
    assert captured["annotation_in_axes"]
    assert captured["annotation_zorder"] == 20
    assert [label for label, _style in drawn_components] == [3, 2, 1]
    styles_by_label = {label: style for label, style in drawn_components}
    assert styles_by_label[3]["colors"] == "0.45"
    assert styles_by_label[2]["colors"] == "tab:orange"
    assert styles_by_label[1]["colors"] == "lime"
    np.testing.assert_array_equal(captured["lines"]["automatic front"], [1.0, 1.0])
    np.testing.assert_array_equal(captured["lines"]["predicted front"], [0.75, 0.75])
    np.testing.assert_array_equal(
        captured["lines"]["front_simple reference"], [1.25, 1.25]
    )
    assert "tracking difference: 0.25" in captured["annotation"]
    assert "front_simple difference" not in captured["annotation"]
    forbidden = ("truth", "target", "benchmark", "accuracy", "error")
    rendered_labels = " ".join(captured["legend"]).lower()
    assert not any(term in rendered_labels for term in forbidden)
    assert plt.get_fignums() == []


def test_failed_frame_has_no_selected_or_nan_front_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence, diagnostic = _inputs()
    failed = replace(
        diagnostic,
        selected_component=None,
        x_front=np.nan,
        predicted_x=np.nan,
        tracking_error=np.nan,
        status=STATUS_NO_VALID_TEMPORAL_CANDIDATE,
        selected_component_label=-1,
        selected_component_pixels=0,
        selected_overlap_pixels=0,
        temporal_candidate_count=0,
        reference_x=np.nan,
    )
    original_save = front_detection_diagnostic_plotting.save_figure
    captured = {}

    def capture_and_save(fig, path, overwrite):
        ax = fig.axes[0]
        captured["legend"] = [text.get_text() for text in ax.get_legend().texts]
        captured["line_x"] = [
            np.asarray(line.get_xdata(), dtype=float) for line in ax.lines
        ]
        captured["annotation"] = "\n".join(text.get_text() for text in ax.texts)
        original_save(fig, path, overwrite)

    monkeypatch.setattr(
        front_detection_diagnostic_plotting,
        "save_figure",
        capture_and_save,
    )

    plot_front_frame_diagnostic(
        tmp_path / "failed.png",
        "N7",
        sequence,
        failed,
        overwrite=False,
    )

    assert "selected component" not in captured["legend"]
    assert "rejected threshold components" in captured["legend"]
    assert "other spatially valid components" in captured["legend"]
    assert "automatic front" not in captured["legend"]
    assert "predicted front" not in captured["legend"]
    assert "front_simple reference" not in captured["legend"]
    assert captured["line_x"] == []
    assert "automatic x: nan" in captured["annotation"]
    assert "tracking difference: nan" in captured["annotation"]
    assert plt.get_fignums() == []


def test_preflight_rejects_existing_output_before_writing_any_figure(
    tmp_path: Path,
) -> None:
    sequence, diagnostic = _inputs()
    later = replace(diagnostic, file_index=61)
    conflict = tmp_path / "diagnostics/N7_front_diagnostic_f00061.png"
    conflict.parent.mkdir(parents=True)
    conflict.write_bytes(b"keep")

    with pytest.raises(
        FileExistsError,
        match=r"Output exists: .* Pass --overwrite to replace it\.",
    ):
        write_front_frame_diagnostic_plots(
            tmp_path,
            "N7",
            sequence,
            (diagnostic, later),
            overwrite=False,
        )

    assert not (
        tmp_path / "diagnostics/N7_front_diagnostic_f00041.png"
    ).exists()
    assert conflict.read_bytes() == b"keep"
    assert plt.get_fignums() == []


def test_overwrite_true_replaces_all_diagnostic_figures(tmp_path: Path) -> None:
    sequence, diagnostic = _inputs()
    paths = [
        tmp_path / "diagnostics/N7_front_diagnostic_f00041.png",
        tmp_path / "diagnostics/N7_front_diagnostic_f00009.png",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"old")

    written = write_front_frame_diagnostic_plots(
        tmp_path,
        "N7",
        sequence,
        (diagnostic, replace(diagnostic, file_index=9)),
        overwrite=True,
    )

    assert written == paths
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in paths)
    assert plt.get_fignums() == []


def test_direct_overwrite_failure_closes_figure(tmp_path: Path) -> None:
    sequence, diagnostic = _inputs()
    path = tmp_path / "existing.png"
    path.write_bytes(b"keep")

    with pytest.raises(FileExistsError):
        plot_front_frame_diagnostic(
            path,
            "N7",
            sequence,
            diagnostic,
            overwrite=False,
        )

    assert path.read_bytes() == b"keep"
    assert plt.get_fignums() == []


def test_save_uses_tight_bounding_box_for_external_legend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fig = plt.figure()
    saved = {}

    def fake_savefig(path, **kwargs):
        saved["path"] = path
        saved["kwargs"] = kwargs

    monkeypatch.setattr(fig, "savefig", fake_savefig)

    front_detection_diagnostic_plotting.save_figure(
        fig,
        tmp_path / "diagnostic.png",
        overwrite=False,
    )

    assert saved["kwargs"]["dpi"] == 200
    assert saved["kwargs"]["bbox_inches"] == "tight"
    assert plt.get_fignums() == []
