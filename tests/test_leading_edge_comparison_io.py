from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from nek_post.h_refinement_analysis import tree_hashes
from nek_post.h_refinement_leading_edge import METHODS, method_key
import nek_post.leading_edge_comparison_io as reporting


def write_history(root, case, times, elements, *, shift=0., shape=0., failed=False,
                  method_difference=False, y_offset=0.):
    directory = root / case
    directory.mkdir(parents=True)
    times = np.asarray(times, float)
    indices = np.arange(1, len(times)+1)
    x = np.linspace(-17, 17, 5)
    y = np.linspace(y_offset, 1+y_offset, 4, endpoint=False)
    profile = np.array([-1., 0., 1., 0.])
    front = 5 + times[:, None] + (.2+shape)*profile[None, :] + shift
    if failed:
        front[1, 0] = np.nan
    payload = dict(time=times, file_index=indices, x=x, y=y, z_target=.04,
                   threshold=.1, x_min=0., y_endpoint=1+y_offset)
    rows = []
    for method in METHODS:
        values = front.copy()
        if method_difference and method == "moore-boundary":
            values[0, 2] += .05
        key = method_key(method)
        payload.update({key+"_front": values, key+"_success": np.isfinite(values),
                        key+"_crossing_count": np.isfinite(values).astype(int)})
        for index, time in zip(indices, times):
            rows.append(dict(case=case, file_index=index, actual_time=time,
                             extraction_method=method, plane_nan_fraction=0., plane_nonfinite_fraction=0.))
    np.savez_compressed(directory / "raw_curves.npz", **payload)
    metadata = dict(status="complete", case=case, sampling_mode="uniform-spectral",
        y_grid_selection="explicit-ny", temporal_interpolation=False, methods=list(METHODS),
        periodic_y=True, periodic_endpoint_included=False, extraction_x_condition="strict-greater-than",
        z_target=.04, threshold=.1, extraction_x_min=0., y_endpoint=1+y_offset, nx=5, ny=4,
        frame_count=len(times), native_ny=elements, element_count=elements,
        polynomial_order=[7, 7, 7], physical_time_range=[float(times[0]), float(times[-1])],
        source_files=[dict(time=float(t), file_index=int(i)) for i, t in zip(indices, times)])
    (directory / "metadata.json").write_text(json.dumps(metadata))
    reporting._write_table(directory / "frame_statistics.csv", rows)


@pytest.fixture
def histories(tmp_path):
    root = tmp_path / "raw"
    write_history(root, "H", [0, 1, 2], 4, shift=.5, failed=True)
    write_history(root, "VH", [0, .6, 2], 8, shape=.1, method_difference=True)
    write_history(root, "VVH", [0, .5, 1, 1.5, 2], 12)
    return root


def test_saved_comparison_uses_common_mask_and_preserves_raw(histories, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(reporting, "_plot_histories", lambda *args: calls.append(args))
    before = tree_hashes(histories)
    output = tmp_path / "comparison"
    summary = reporting.analyze_leading_edge_histories(histories, output, ("H", "VH", "VVH"), "VVH")
    assert tree_hashes(histories) == before
    assert summary["raw_hashes_unchanged"]
    assert summary["comparison_time_count"] == 5
    assert summary["physical_time_intersection"] == [0., 2.]
    assert summary["temporal_extrapolation"] is False
    assert summary["common_valid_fraction_min"] == .75
    assert summary["maximum_same_index_time_spread"] == 1.
    assert summary["cases"]["VH"]["exactly_identical_method_frames"] == 2
    assert summary["cases"]["VH"]["different_method_y_points"] == 1
    assert summary["observed_h_order"] is None
    assert len(calls) == 1
    with np.load(output / "aligned_fronts.npz") as data:
        np.testing.assert_array_equal(data["time"], [0, .5, 1, 1.5, 2])
        np.testing.assert_array_equal(data["common_valid_mask"][:, 0], [True, False, False, False, True])
        assert np.isnan(data["H_front"][1:4, 0]).all()
    with (output / "difference_metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {r["case"] for r in rows} == {"H", "VH"}
    for row in rows:
        assert float(row["total_rms"])**2 == pytest.approx(
            float(row["bulk_difference"])**2 + float(row["shape_rms"])**2)
        if row["case"] == "H":
            assert float(row["bulk_difference"]) == pytest.approx(.5)
            assert float(row["shape_rms"]) < 1e-14
    with (output / "method_difference_points.csv").open() as handle:
        points = list(csv.DictReader(handle))
    assert len(points) == 1
    assert points[0]["case"] == "VH"
    assert points[0]["y_position"] == "2"
    with pytest.raises(ValueError, match="new and separate"):
        reporting.analyze_leading_edge_histories(histories, output, ("H", "VH", "VVH"), "VVH")


def test_reader_rejects_incomplete_or_inconsistent_history(histories):
    meta = histories / "H" / "metadata.json"
    payload = json.loads(meta.read_text())
    payload["status"] = "in_progress"
    meta.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="incomplete"):
        reporting.read_raw_history(histories / "H")


def test_nonfinest_reference_rejected(histories, tmp_path):
    with pytest.raises(ValueError, match="unique largest element count"):
        reporting.analyze_leading_edge_histories(histories, tmp_path / "out", ("H", "VH", "VVH"), "H")


def test_empty_common_support_is_reported_without_filling(histories, tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "_plot_histories", lambda *args: None)
    path = histories / "H" / "raw_curves.npz"
    with np.load(path) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    for method in METHODS:
        key = method_key(method)
        arrays[key+"_front"][:] = np.nan
        arrays[key+"_success"][:] = False
        arrays[key+"_crossing_count"][:] = 0
    np.savez_compressed(path, **arrays)
    output = tmp_path / "missing"
    summary = reporting.analyze_leading_edge_histories(histories, output, ("H", "VH", "VVH"), "VVH")
    assert summary["common_valid_fraction_min"] == 0
    assert summary["differences"]["H"]["peak_total_rms"] is None
    assert summary["cases"]["H"]["method_common_valid_fraction_min"] == 0
    saved = json.loads((output / "summary.json").read_text())
    assert saved["differences"]["H"]["final"]["total_rms"] is None
    with np.load(output / "aligned_fronts.npz") as data:
        assert not data["common_valid_mask"].any()
        assert np.isnan(data["H_front"]).all()


def test_cross_case_grid_mismatch_rejected(tmp_path):
    raw = tmp_path / "raw"
    write_history(raw, "A", [0, 1, 2], 4)
    write_history(raw, "R", [0, 1, 2], 8, y_offset=.01)
    with pytest.raises(ValueError, match="identical physical coordinates"):
        reporting.analyze_leading_edge_histories(raw, tmp_path / "out", ("A", "R"), "R")


def test_plot_reports_render_from_saved_curves_only(histories, tmp_path):
    output = tmp_path / "plots"
    reporting.analyze_leading_edge_histories(histories, output, ("H", "VH", "VVH"), "VVH")
    for stem in ("leading_edge_evolution_overlay", "mean_front_history", "total_rms_history",
                 "shape_rms_history", "front_std_history", "extraction_method_difference_history"):
        assert (output / (stem+".png")).stat().st_size > 1000
        assert (output / (stem+".pdf")).stat().st_size > 1000
