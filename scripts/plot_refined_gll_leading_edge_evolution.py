"""Plot saved full N7 10-node refined-GLL runs using the legacy evolution style.

Reads only timeseries CSVs and sampling metadata JSONs; never opens snapshots.
Existing figures and the comparison report are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from nek_post.config import load_yaml
from nek_post.front_detection_io import preflight_output_paths
from nek_post.leading_edge_io import LEADING_EDGE_TIMESERIES_COLUMNS
from nek_post.leading_edge_plotting import _write_leading_edge_plot_arrays

REPO_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("rightmost-crossing", "moore-boundary")


def read_run(root: Path, method: str) -> dict:
    """Validate the saved route and reconstruct complete frames in time order."""
    directory = root / method / "nodes_10" / "N7"
    csv_path = directory / "N7_leading_edge_timeseries.csv"
    metadata_path = directory / "N7_leading_edge_sampling_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "metadata_format_version": 1, "case": "N7",
        "sampling_mode": "refined-gll", "target_node_count": 10,
        "extraction_method": method, "threshold": 0.1, "z_target": 0.04,
        "periodic_endpoint_included": False, "n_input_frames": 81,
        "n_selected_frames": 81, "y_upsample_factor": None,
    }
    for key, value in expected.items():
        if key not in metadata or metadata[key] != value:
            raise ValueError(f"{metadata_path}: expected {key}={value!r}")
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != LEADING_EDGE_TIMESERIES_COLUMNS:
            raise ValueError(f"Unexpected timeseries schema: {csv_path}")
        rows = list(reader)
    frames: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        for key in ("case", "threshold", "z_target", "nx", "native_ny", "dense_ny"):
            value = row[key] if key == "case" else float(row[key])
            if value != metadata[key]:
                raise ValueError(f"CSV/metadata mismatch for {key}: {csv_path}")
        if row["y_upsample_factor"] != "":
            raise ValueError("Refined-GLL rows must not specify y upsampling.")
        frames.setdefault(int(row["file_index"]), []).append(row)
    ordered = sorted(frames.items(), key=lambda item: float(item[1][0]["actual_time"]))
    y = np.asarray(metadata["y"], dtype=float)
    if len(ordered) != 81 or len(y) != metadata["output_ny"]:
        raise ValueError("Expected 81 complete frames on the recorded y grid.")
    curves, times, indices = [], [], []
    for index, frame in ordered:
        frame.sort(key=lambda row: float(row["y"]))
        frame_y = np.array([float(row["y"]) for row in frame])
        if frame_y.shape != y.shape or not np.allclose(frame_y, y, rtol=0, atol=1e-14):
            raise ValueError(f"Incomplete or mismatched y grid in frame {index}")
        for key in ("actual_time", "target_time", "time_error", "source_file"):
            if len({row[key] for row in frame}) != 1:
                raise ValueError(f"Inconsistent {key} in frame {index}")
        curve = np.array([float(row["x_front"]) for row in frame])
        if np.any(np.isinf(curve)) or any(
            row["success"] != str(bool(np.isfinite(value)))
            for row, value in zip(frame, curve)
        ):
            raise ValueError(f"Invalid front/success values in frame {index}")
        curves.append(curve)
        times.append(float(frame[0]["actual_time"]))
        indices.append(index)
    if np.any(np.diff(times) <= 0) or times[0] != metadata["actual_time_start"] or times[-1] != metadata["actual_time_end"]:
        raise ValueError("Invalid full-run time coverage.")
    return dict(metadata=metadata, y=y, x_front=np.array(curves),
                times=np.array(times), indices=np.array(indices),
                inputs=[str(csv_path), str(metadata_path)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    stems = {method: f"N7_refined_gll_nodes10_{method}_leading_edge_evolution" for method in METHODS}
    report_path = args.output_dir / "N7_refined_gll_nodes10_evolution_report.json"
    outputs = [args.output_dir / f"{stem}.{ext}" for stem in stems.values() for ext in ("png", "pdf")]
    preflight_output_paths([*outputs, report_path], overwrite=False)
    runs = {method: read_run(args.input_root, method) for method in METHODS}
    right, moore = (runs[method] for method in METHODS)
    aligned = all(np.array_equal(right[key], moore[key]) for key in ("y", "times", "indices"))
    identical = aligned and np.array_equal(right["x_front"], moore["x_front"], equal_nan=True)
    reynolds = load_yaml(REPO_ROOT / "config/cases.yaml")["leading_edge"]["reynolds_number"]
    for method, run in runs.items():
        metadata = run["metadata"]
        plan = metadata["horizontal_plan_metadata"]
        _write_leading_edge_plot_arrays(
            args.output_dir, "N7", run["y"], run["x_front"],
            metadata["threshold"], metadata["z_target"],
            plan["ymin"], plan["ymax_periodic_endpoint"], False,
            reynolds_number=reynolds, extraction_x_min=metadata["extraction_x_min"],
            output_stem=stems[method],
            title_note=f"refined-gll, nodes=10, {method}; 81 frames, t={run['times'][0]:g}–{run['times'][-1]:g}",
        )
    report = {
        "inputs": {method: run["inputs"] for method, run in runs.items()},
        "sampling_mode": "refined-gll", "target_node_count": 10,
        "case": "N7", "z_target": 0.04, "threshold": 0.1,
        "frames_per_method": 81, "y_points_per_frame": len(right["y"]),
        "actual_time_range": [float(right["times"][0]), float(right["times"][-1])],
        "coordinates_times_and_file_indices_identical": aligned,
        "curves_exactly_identical_equal_nan": identical,
        "style": "Legacy black curve overlay; periodic closure only; no smoothing or time resampling.",
        "outputs": [str(path) for path in outputs],
    }
    with report_path.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
