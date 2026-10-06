from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from nek_post.leading_edge_io import LEADING_EDGE_TIMESERIES_COLUMNS
from nek_post.refined_gll_leading_edge_artifacts import (
    compare_refined_gll_methods,
    read_refined_gll_run,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "32_plot_h_refinement_refined_gll_leading_edges.py"
SPEC = importlib.util.spec_from_file_location("plot_h_refined_gll", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
plot_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot_script)


def _write_run(root: Path, case: str, method: str, y: np.ndarray,
               *, difference: float = 0.0) -> None:
    directory = root / method / case
    directory.mkdir(parents=True)
    x = np.asarray([-17.0, -16.8, -16.1, 17.0])
    times = (0.0, 19.5)
    indices = (1, 79)
    metadata = {
        "metadata_format_version": 1,
        "case": case,
        "sampling_mode": "refined-gll",
        "interpretation": "Original source polynomial evaluated on a target GLL nodal set; no new solution information.",
        "source_node_count": 8,
        "source_polynomial_order": 7,
        "target_node_count": 10,
        "periodic_y": True,
        "spatial_extrapolation": False,
        "plane_values_finite_required": True,
        "extraction_method": method,
        "extraction_x_min": 0.0,
        "extraction_x_condition": "strict-greater-than",
        "threshold": 0.1,
        "z_target": 0.04,
        "nx": len(x),
        "native_ny": len(y),
        "output_ny": len(y),
        "dense_ny": len(y),
        "y_upsample_factor": None,
        "periodic_endpoint_included": False,
        "y_period": 1.5,
        "x": x.tolist(),
        "y": y.tolist(),
        "n_input_frames": 2,
        "n_selected_frames": 2,
        "actual_time_start": times[0],
        "actual_time_end": times[-1],
        "target_time_spacing": None,
        "source_file_indices": list(indices),
        "source_files": [f"GC0.f{index:05d}" for index in indices],
        "stored_physical_times": list(times),
        "horizontal_plan_metadata": {"ymin": 0.0, "ymax_periodic_endpoint": 1.5},
    }
    (directory / f"{case}_leading_edge_sampling_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    with (directory / f"{case}_leading_edge_timeseries.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=LEADING_EDGE_TIMESERIES_COLUMNS)
        writer.writeheader()
        for frame, (index, time) in enumerate(zip(indices, times, strict=True)):
            for position, coordinate in enumerate(y):
                front = 1.0 + frame + position / 10.0 + difference
                writer.writerow({
                    "case": case, "file_index": index,
                    "source_file": f"GC0.f{index:05d}",
                    "target_time": time, "actual_time": time, "time_error": 0.0,
                    "y": coordinate, "x_front": front, "success": True,
                    "crossing_count": 1, "threshold": 0.1, "z_target": 0.04,
                    "nx": len(x), "native_ny": len(y), "dense_ny": len(y),
                    "y_upsample_factor": "",
                })


@pytest.mark.parametrize("case", ("N7_H", "N7_VH", "N7_VVH"))
def test_reader_accepts_h_cases_and_preserves_refined_coordinates(tmp_path, case):
    y = np.asarray([0.0, 0.04, 0.13, 0.31, 0.9])
    _write_run(tmp_path, case, "rightmost-crossing", y)
    run = read_refined_gll_run(
        tmp_path / "rightmost-crossing" / case,
        case=case, method="rightmost-crossing", expected_frame_count=2,
    )
    assert run["metadata"]["sampling_mode"] == "refined-gll"
    assert run["metadata"]["source_node_count"] == 8
    assert run["metadata"]["target_node_count"] == 10
    assert run["metadata"]["extraction_x_condition"] == "strict-greater-than"
    assert np.array_equal(run["y"], y)
    assert run["x_nonuniform"] and run["y_nonuniform"]
    assert run["y"][-1] < run["y"][0] + run["metadata"]["y_period"]


def test_methods_compare_on_same_case_grid_and_report_differences(tmp_path):
    y = np.asarray([0.0, 0.05, 0.2, 0.7])
    _write_run(tmp_path, "N7_H", "rightmost-crossing", y)
    _write_run(tmp_path, "N7_H", "moore-boundary", y, difference=0.01)
    right = read_refined_gll_run(tmp_path / "rightmost-crossing/N7_H",
                                 case="N7_H", method="rightmost-crossing")
    moore = read_refined_gll_run(tmp_path / "moore-boundary/N7_H",
                                 case="N7_H", method="moore-boundary")
    comparison = compare_refined_gll_methods(right, moore)
    assert comparison["exactly_identical_frame_count"] == 0
    assert comparison["differing_y_count_by_frame"] == [len(y), len(y)]
    assert comparison["rms_by_frame"] == pytest.approx([0.01, 0.01])


def test_different_h_case_grids_are_not_forced_into_pointwise_comparison(tmp_path):
    coarse_y = np.asarray([0.0, 0.06, 0.3])
    fine_y = np.asarray([0.0, 0.03, 0.12, 0.4])
    _write_run(tmp_path, "N7_H", "rightmost-crossing", coarse_y)
    _write_run(tmp_path, "N7_VVH", "rightmost-crossing", fine_y)
    coarse = read_refined_gll_run(tmp_path / "rightmost-crossing/N7_H",
                                  case="N7_H", method="rightmost-crossing")
    fine = read_refined_gll_run(tmp_path / "rightmost-crossing/N7_VVH",
                                case="N7_VVH", method="rightmost-crossing")
    assert coarse["y"].shape != fine["y"].shape
    with pytest.raises(ValueError, match="differ in y"):
        compare_refined_gll_methods(coarse, fine)


def test_plot_report_keeps_case_native_grids_and_omits_cross_case_rms(tmp_path):
    counts = {"N7_H": 3, "N7_VH": 4, "N7_VVH": 5}
    for case, count in counts.items():
        y = np.linspace(0.0, 1.2, count) ** 1.1
        for method in ("rightmost-crossing", "moore-boundary"):
            _write_run(tmp_path, case, method, y)
    output = tmp_path / "figures"
    plot_script.main([
        "--input-root", str(tmp_path), "--output-dir", str(output),
        "--expected-frame-count", "2",
    ])
    report = json.loads(
        (output / "h_refinement_refined_gll_nodes10_report.json").read_text()
    )
    assert report["cross_case_y_grids_identical"] is False
    assert report["cross_case_pointwise_metrics_computed"] is False
    assert {case: report["cases"][case]["y_point_count"] for case in counts} == counts
    assert all(report["cases"][case]["method_comparison"]["exactly_identical_frame_count"] == 2
               for case in counts)
