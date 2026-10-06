"""Plot saved full N7 10-node refined-GLL runs using the legacy evolution style.

Reads only timeseries CSVs and sampling metadata JSONs; never opens snapshots.
Existing figures and the comparison report are never overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nek_post.config import load_yaml
from nek_post.front_detection_io import preflight_output_paths
from nek_post.leading_edge_plotting import _write_leading_edge_plot_arrays
from nek_post.refined_gll_leading_edge_artifacts import read_refined_gll_run

REPO_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("rightmost-crossing", "moore-boundary")


def read_run(root: Path, method: str) -> dict:
    """Validate the saved route and reconstruct complete frames in time order."""
    directory = root / method / "nodes_10" / "N7"
    return read_refined_gll_run(
        directory, case="N7", method=method, expected_frame_count=81,
    )


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
