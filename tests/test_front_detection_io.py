import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nek_post.front_detection import track_concentration_front
from nek_post.front_detection_io import (
    COMPARISON_COLUMNS,
    SUMMARY_COLUMNS,
    TIMESERIES_COLUMNS,
    detected_front_timeseries_path,
    discover_nek_frame_paths,
    format_csv_value,
    front_detection_comparison_path,
    front_detection_diagnostic_path,
    front_detection_diagnostics_dir,
    front_detection_difference_path,
    front_detection_overlay_path,
    front_detection_summary_path,
    write_front_detection_csvs,
)


def _outputs():
    Xi, Zi = np.meshgrid(np.arange(3.0), np.arange(2.0))
    frames = np.zeros((2, 2, 3))
    frames[0, 0, 0] = 1.0
    frames[1, 0, 1] = 1.0
    tracking = track_concentration_front(
        np.array([0.0, 1.0]),
        Xi,
        Zi,
        frames,
        threshold=0.5,
        min_component_pixels=1,
        bottom_rows=1,
        max_front_jump=np.inf,
    )
    sequence = SimpleNamespace(
        time=np.array([0.0, 1.0]),
        file_indices=np.array([2, 5]),
        source_files=(Path("GC0.f00002"), Path("GC0.f00005")),
        finite_fraction=np.array([1.0 / 3.0, 1.0]),
        concentration_min=np.array([0.0, 0.0]),
        concentration_max=np.array([1.0, 1.0]),
    )
    comparison = SimpleNamespace(
        file_indices=np.array([2, 5]),
        time=np.array([0.0, 1.0]),
        x_front_auto=np.array([0.0, 1.0]),
        x_front_reference=np.array([0.1, 0.9]),
        difference=np.array([-0.1, 0.1]),
        absolute_difference=np.array([0.1, 0.1]),
    )
    summary = {column: 0 for column in SUMMARY_COLUMNS}
    summary.update(
        {
            "case": "N7",
            "reference_file": "front_simple.dat",
            "reference_role": "external_comparison_only",
            "interpolation_method": "linear",
        }
    )
    return sequence, tracking, comparison, summary


def test_discovery_parses_exact_pattern_sorts_and_ignores_unrelated(
    tmp_path: Path,
) -> None:
    for name in (
        "GC0.f00010",
        "GC0.f00001",
        "GC0.f00002",
        "GC0.f001",
        "OTHER.f00003",
        "GC0.f00003.bak",
    ):
        (tmp_path / name).touch()

    frames = discover_nek_frame_paths(tmp_path, file_prefix="GC0")

    assert [frame.index for frame in frames] == [1, 2, 10]
    assert frames[0].path == tmp_path / "GC0.f00001"


def test_discovery_filters_start_and_end_inclusively(tmp_path: Path) -> None:
    for index in (1, 3, 8, 10):
        (tmp_path / f"GC0.f{index:05d}").touch()

    frames = discover_nek_frame_paths(
        tmp_path,
        file_prefix="GC0",
        start_index=3,
        end_index=8,
    )

    assert [frame.index for frame in frames] == [3, 8]


def test_discovery_fails_when_no_files_remain(tmp_path: Path) -> None:
    (tmp_path / "GC0.f00001").touch()

    with pytest.raises(FileNotFoundError, match="No Nek5000 frame files"):
        discover_nek_frame_paths(
            tmp_path,
            file_prefix="GC0",
            start_index=2,
        )


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (-1, None, "start_index"),
        (None, -1, "end_index"),
        (5, 4, "less than or equal"),
    ],
)
def test_discovery_rejects_invalid_ranges(
    tmp_path: Path,
    start: int | None,
    end: int | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        discover_nek_frame_paths(
            tmp_path,
            file_prefix="GC0",
            start_index=start,
            end_index=end,
        )


def test_exact_output_filenames(tmp_path: Path) -> None:
    assert (
        detected_front_timeseries_path(tmp_path, "N7").name
        == "N7_detected_front_timeseries.csv"
    )
    assert (
        front_detection_comparison_path(tmp_path, "N7").name
        == "N7_front_detection_comparison.csv"
    )
    assert (
        front_detection_summary_path(tmp_path, "N7").name
        == "N7_front_detection_summary.csv"
    )
    assert (
        front_detection_overlay_path(tmp_path, "N7").name
        == "N7_front_detection_overlay.png"
    )
    assert (
        front_detection_difference_path(tmp_path, "N7").name
        == "N7_front_detection_difference.png"
    )
    assert front_detection_diagnostics_dir(tmp_path) == tmp_path / "diagnostics"
    assert front_detection_diagnostic_path(
        tmp_path, "N7", 9
    ) == tmp_path / "diagnostics/N7_front_diagnostic_f00009.png"


def test_diagnostic_path_rejects_negative_file_index(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        front_detection_diagnostic_path(tmp_path, "N7", -1)


def test_csv_headers_formatting_status_boolean_and_row_order(tmp_path: Path) -> None:
    sequence, tracking, comparison, summary = _outputs()

    paths = write_front_detection_csvs(
        tmp_path,
        "N7",
        sequence,
        tracking,
        comparison,
        summary,
        overwrite=False,
    )

    timeseries = list(csv.reader(paths[0].open(encoding="utf-8")))
    comparison_rows = list(csv.reader(paths[1].open(encoding="utf-8")))
    summary_rows = list(csv.reader(paths[2].open(encoding="utf-8")))
    assert timeseries[0] == list(TIMESERIES_COLUMNS)
    assert comparison_rows[0] == list(COMPARISON_COLUMNS)
    assert summary_rows[0] == list(SUMMARY_COLUMNS)
    assert [row[0] for row in timeseries[1:]] == ["2", "5"]
    assert timeseries[1][TIMESERIES_COLUMNS.index("status")] == "selected_initial"
    assert (
        timeseries[1][TIMESERIES_COLUMNS.index("selected_bottom_contact")]
        == "True"
    )
    assert (
        timeseries[1][
            TIMESERIES_COLUMNS.index("concentration_finite_fraction")
        ]
        == "0.3333333333333333"
    )
    assert (
        comparison_rows[1][COMPARISON_COLUMNS.index("difference")] == "-0.1"
    )
    assert summary_rows[1][SUMMARY_COLUMNS.index("case")] == "N7"
    assert (
        summary_rows[1][SUMMARY_COLUMNS.index("reference_role")]
        == "external_comparison_only"
    )
    assert format_csv_value(7) == "7"
    assert format_csv_value(False) == "False"


def test_preflight_conflict_rejects_before_writing_any_csv(tmp_path: Path) -> None:
    sequence, tracking, comparison, summary = _outputs()
    conflict = front_detection_comparison_path(tmp_path, "N7")
    conflict.write_text("keep", encoding="utf-8")

    with pytest.raises(
        FileExistsError,
        match=r"Output exists: .* Pass --overwrite to replace it\.",
    ):
        write_front_detection_csvs(
            tmp_path,
            "N7",
            sequence,
            tracking,
            comparison,
            summary,
            overwrite=False,
        )

    assert not detected_front_timeseries_path(tmp_path, "N7").exists()
    assert conflict.read_text(encoding="utf-8") == "keep"
    assert not front_detection_summary_path(tmp_path, "N7").exists()


def test_overwrite_true_replaces_all_csv_outputs(tmp_path: Path) -> None:
    sequence, tracking, comparison, summary = _outputs()
    paths = [
        detected_front_timeseries_path(tmp_path, "N7"),
        front_detection_comparison_path(tmp_path, "N7"),
        front_detection_summary_path(tmp_path, "N7"),
    ]
    for path in paths:
        path.write_text("old", encoding="utf-8")

    written = write_front_detection_csvs(
        tmp_path,
        "N7",
        sequence,
        tracking,
        comparison,
        summary,
        overwrite=True,
    )

    assert written == paths
    assert all(path.read_text(encoding="utf-8") != "old" for path in paths)
