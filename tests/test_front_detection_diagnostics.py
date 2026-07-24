from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from nek_post.front_detection import (
    FrontComponent,
    FrontTrackingResult,
    STATUS_NO_VALID_TEMPORAL_CANDIDATE,
    STATUS_SELECTED_TRACKED,
)
from nek_post import front_detection_diagnostics
from nek_post.front_detection_diagnostics import (
    build_front_frame_diagnostics,
    parse_diagnostic_indices,
)


def _sequence(
    *,
    file_indices: tuple[int, ...] = (1,),
    times: tuple[float, ...] = (5.0,),
):
    Xi, Zi = np.meshgrid(np.arange(10.0), np.arange(3.0))
    frame = np.zeros_like(Xi)
    frame[0, 0:2] = 1.0
    frame[0, 5:9] = 1.0
    frames = np.stack([frame.copy() for _ in file_indices])
    return SimpleNamespace(
        time=np.asarray(times, dtype=float),
        file_indices=np.asarray(file_indices, dtype=np.int64),
        source_files=tuple(
            Path(f"GC0.f{index:05d}") for index in file_indices
        ),
        Xi=Xi,
        Zi=Zi,
        C_frames=frames,
    )


def _tracking(
    sequence,
    *,
    statuses: tuple[str, ...] | None = None,
) -> FrontTrackingResult:
    n = len(sequence.time)
    selected = np.ones(n, dtype=np.int64)
    x_front = np.ones(n, dtype=float)
    status = statuses or tuple(STATUS_SELECTED_TRACKED for _ in range(n))
    failed = np.asarray([item != STATUS_SELECTED_TRACKED for item in status])
    selected[failed] = -1
    x_front[failed] = np.nan
    return FrontTrackingResult(
        time=sequence.time.copy(),
        x_front=x_front,
        predicted_x=np.full(n, 0.5),
        tracking_error=x_front - 0.5,
        selected_component_label=selected,
        selected_component_pixels=np.where(failed, 0, 2).astype(np.int64),
        selected_component_xmin=np.where(failed, np.nan, 0.0),
        selected_component_xmax=np.where(failed, np.nan, 1.0),
        selected_bottom_contact=~failed,
        component_count=np.full(n, 2, dtype=np.int64),
        spatial_candidate_count=np.full(n, 2, dtype=np.int64),
        temporal_candidate_count=np.where(failed, 0, 1).astype(np.int64),
        selected_overlap_pixels=np.where(failed, 0, 1).astype(np.int64),
        status=status,
        threshold=0.5,
        min_component_pixels=2,
        bottom_rows=1,
        max_front_jump=1.0,
        connectivity=8,
    )


def test_parse_diagnostic_indices_preserves_order_and_whitespace() -> None:
    assert parse_diagnostic_indices(" 41, 1 , 9 ") == (41, 1, 9)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "At least one"),
        ("   ", "At least one"),
        ("1,,9", "empty tokens"),
        ("1,x", "not an integer"),
        ("1,-9", "non-negative"),
        ("1,9,1", "Duplicate"),
    ],
)
def test_parse_diagnostic_indices_rejects_invalid_input(
    text: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_diagnostic_indices(text)


def test_requested_indices_map_to_positions_and_preserve_requested_order() -> None:
    sequence = _sequence(file_indices=(1, 9), times=(0.0, 5.0))
    tracking = _tracking(sequence)

    diagnostics = build_front_frame_diagnostics(
        sequence,
        tracking,
        (9, 1),
    )

    assert [item.file_index for item in diagnostics] == [9, 1]
    assert [item.frame_position for item in diagnostics] == [1, 0]
    assert [item.source_file.name for item in diagnostics] == [
        "GC0.f00009",
        "GC0.f00001",
    ]


def test_unavailable_requested_indices_are_listed() -> None:
    sequence = _sequence()

    with pytest.raises(ValueError, match=r"Unavailable.*9, 21"):
        build_front_frame_diagnostics(
            sequence,
            _tracking(sequence),
            (9, 21),
        )


def test_reconstruction_uses_all_tracking_segmentation_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence = _sequence()
    tracking = _tracking(sequence)
    mask = np.zeros_like(sequence.Xi, dtype=bool)
    mask[0, 0:2] = True
    component = FrontComponent(
        label=1,
        pixel_count=2,
        x_min=0.0,
        x_max=1.0,
        z_min=0.0,
        z_max=0.0,
        bottom_contact=True,
        mask=mask,
    )
    calls = {}

    def detect(Xi, Zi, C_grid, **kwargs):
        calls["detect"] = (Xi, Zi, C_grid, kwargs)
        return (component,)

    def spatial(components, **kwargs):
        calls["spatial"] = (components, kwargs)
        return tuple(components)

    monkeypatch.setattr(front_detection_diagnostics, "detect_front_components", detect)
    monkeypatch.setattr(
        front_detection_diagnostics,
        "filter_spatial_components",
        spatial,
    )

    diagnostic = build_front_frame_diagnostics(
        sequence,
        tracking,
        (1,),
    )[0]

    assert calls["detect"][0] is sequence.Xi
    assert calls["detect"][1] is sequence.Zi
    assert np.shares_memory(calls["detect"][2], sequence.C_frames)
    assert_array_equal(calls["detect"][2], sequence.C_frames[0])
    assert calls["detect"][3] == {
        "threshold": tracking.threshold,
        "bottom_rows": tracking.bottom_rows,
        "connectivity": tracking.connectivity,
    }
    assert calls["spatial"][1] == {
        "min_component_pixels": tracking.min_component_pixels
    }
    assert diagnostic.selected_component is component


def test_tracked_label_is_used_even_when_another_component_is_larger_and_rightmost() -> None:
    sequence = _sequence()

    diagnostic = build_front_frame_diagnostics(
        sequence,
        _tracking(sequence),
        (1,),
    )[0]

    assert [component.pixel_count for component in diagnostic.components] == [2, 4]
    assert diagnostic.selected_component is not None
    assert diagnostic.selected_component.label == 1
    assert diagnostic.selected_component.x_max == 1.0
    assert diagnostic.components[1].x_max == 8.0


def test_selected_component_statistics_match_tracking_output() -> None:
    sequence = _sequence()
    tracking = _tracking(sequence)

    diagnostic = build_front_frame_diagnostics(sequence, tracking, (1,))[0]

    selected = diagnostic.selected_component
    assert selected is not None
    assert selected.pixel_count == tracking.selected_component_pixels[0]
    assert selected.x_min == tracking.selected_component_xmin[0]
    assert selected.x_max == tracking.selected_component_xmax[0]
    assert selected.bottom_contact == tracking.selected_bottom_contact[0]


def test_inconsistent_selected_label_fails_clearly() -> None:
    sequence = _sequence()
    tracking = _tracking(sequence)
    tracking.selected_component_label[0] = 99

    with pytest.raises(ValueError, match="tracked label 99.*expected exactly one"):
        build_front_frame_diagnostics(sequence, tracking, (1,))


def test_successful_status_with_negative_selected_label_fails() -> None:
    sequence = _sequence()
    tracking = _tracking(sequence)
    tracking.selected_component_label[0] = -1

    with pytest.raises(ValueError, match="Successful frame.*selected label"):
        build_front_frame_diagnostics(sequence, tracking, (1,))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("selected_component_pixels", 3, "pixel count mismatch"),
        ("selected_component_xmin", 0.25, "x_min mismatch"),
        ("selected_component_xmax", 1.25, "x_max mismatch"),
    ],
)
def test_inconsistent_selected_statistics_fail(
    field: str,
    value: float,
    message: str,
) -> None:
    sequence = _sequence()
    tracking = _tracking(sequence)
    changed = getattr(tracking, field).copy()
    changed[0] = value
    tracking = replace(tracking, **{field: changed})

    with pytest.raises(ValueError, match=message):
        build_front_frame_diagnostics(sequence, tracking, (1,))


def test_failed_status_has_no_selected_component_or_fallback() -> None:
    sequence = _sequence()
    tracking = _tracking(
        sequence,
        statuses=(STATUS_NO_VALID_TEMPORAL_CANDIDATE,),
    )

    diagnostic = build_front_frame_diagnostics(sequence, tracking, (1,))[0]

    assert len(diagnostic.components) == 2
    assert len(diagnostic.spatial_components) == 2
    assert diagnostic.selected_component is None
    assert diagnostic.selected_component_label == -1
    assert np.isnan(diagnostic.x_front)


def test_reference_interpolation_is_absolute_and_does_not_extrapolate() -> None:
    sequence = _sequence(
        file_indices=(1, 9),
        times=(5.0, 15.0),
    )
    tracking = _tracking(sequence)
    reference = {
        "time": np.array([0.0, 10.0]),
        "x_front": np.array([-1.0, -3.0]),
    }

    diagnostics = build_front_frame_diagnostics(
        sequence,
        tracking,
        (1, 9),
        reference_front=reference,
    )

    assert diagnostics[0].reference_x == 2.0
    assert np.isnan(diagnostics[1].reference_x)


def test_missing_reference_produces_nan() -> None:
    sequence = _sequence()

    diagnostic = build_front_frame_diagnostics(
        sequence,
        _tracking(sequence),
        (1,),
        reference_front=None,
    )[0]

    assert np.isnan(diagnostic.reference_x)
    assert_array_equal(diagnostic.concentration, sequence.C_frames[0])
    assert_allclose(diagnostic.tracking_error, 0.5)
