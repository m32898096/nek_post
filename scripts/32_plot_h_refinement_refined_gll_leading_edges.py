"""Validate and plot H/VH/VVH refined-GLL leading-edge artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nek_post.config import load_yaml
from nek_post.front_detection_io import preflight_output_paths
from nek_post.h_refinement import HRefinementStudy
from nek_post.leading_edge_plotting import _write_leading_edge_plot_arrays
from nek_post.paths import ProjectPaths
from nek_post.refined_gll_leading_edge_artifacts import (
    REFINED_GLL_METHODS,
    compare_refined_gll_methods,
    read_refined_gll_run,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=REPO_ROOT / "config/paths.yaml")
    parser.add_argument("--study-config", type=Path, default=REPO_ROOT / "config/h_refinement.yaml")
    parser.add_argument("--input-root", type=Path,
                        help="Default: configured h leading-edge/refined_gll_nodes10 root.")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: INPUT_ROOT/figures.")
    parser.add_argument("--expected-frame-count", type=int, default=81,
                        help="Required complete frame count per case and method.")
    args = parser.parse_args(argv)
    try:
        if args.expected_frame_count < 1:
            raise ValueError("--expected-frame-count must be positive.")
        paths = ProjectPaths.from_yaml(args.paths_config)
        study = HRefinementStudy.from_yaml(args.study_config)
        root = (args.input_root or
                paths.h_refinement_refined_gll_leading_edge_dir(10)).resolve()
        output = (args.output_dir or root / "figures").resolve()
        runs = {
            case: {
                method: read_refined_gll_run(
                    root / method / case,
                    case=case,
                    method=method,
                    expected_frame_count=args.expected_frame_count,
                )
                for method in REFINED_GLL_METHODS
            }
            for case in study.cases
        }
        report_path = output / "h_refinement_refined_gll_nodes10_report.json"
        plot_paths = [
            output / f"{case}_refined_gll_nodes10_{method}_leading_edge_evolution.{suffix}"
            for case in study.cases for method in REFINED_GLL_METHODS
            for suffix in ("png", "pdf")
        ]
        preflight_output_paths([*plot_paths, report_path], overwrite=False)
        reynolds = load_yaml(REPO_ROOT / "config/cases.yaml")["leading_edge"]["reynolds_number"]
        produced = []
        cases_report = {}
        for case in study.cases:
            for method in REFINED_GLL_METHODS:
                run = runs[case][method]
                metadata = run["metadata"]
                produced.extend(_write_leading_edge_plot_arrays(
                    output, case, run["y"], run["x_front"],
                    metadata["threshold"], metadata["z_target"],
                    float(run["y"][0]), float(run["y"][0] + metadata["y_period"]),
                    False, reynolds_number=reynolds,
                    extraction_x_min=metadata["extraction_x_min"],
                    output_stem=f"{case}_refined_gll_nodes10_{method}_leading_edge_evolution",
                    title_note=("refined-gll, source P7 nodes=8, target nodes=10\n"
                                f"{method}; {len(run['times'])} frames, "
                                f"t={run['times'][0]:g}–{run['times'][-1]:g}"),
                ))
            right, moore = (runs[case][method] for method in REFINED_GLL_METHODS)
            comparison = compare_refined_gll_methods(right, moore)
            cases_report[case] = {
                "source_node_count": right["metadata"]["source_node_count"],
                "target_node_count": right["metadata"]["target_node_count"],
                "x_point_count": int(right["x"].size),
                "y_point_count": int(right["y"].size),
                "x_nonuniform": right["x_nonuniform"],
                "y_nonuniform": right["y_nonuniform"],
                "periodic_endpoint_included": False,
                "file_indices": right["indices"].tolist(),
                "stored_physical_times": right["times"].tolist(),
                "successful_y_counts": {
                    method: np.count_nonzero(runs[case][method]["success"], axis=1).tolist()
                    for method in REFINED_GLL_METHODS
                },
                "failed_y_counts": {
                    method: (runs[case][method]["y"].size
                             - np.count_nonzero(runs[case][method]["success"], axis=1)).tolist()
                    for method in REFINED_GLL_METHODS
                },
                "method_comparison": comparison,
            }
        y_vectors = [runs[case][REFINED_GLL_METHODS[0]]["y"] for case in study.cases]
        same_y = all(
            vector.shape == y_vectors[0].shape and np.array_equal(vector, y_vectors[0])
            for vector in y_vectors[1:]
        )
        report = {
            "sampling_mode": "refined-gll",
            "interpretation": "Original P7 solution evaluated at 10 GLL nodes per element; no new solution information.",
            "source_node_count": 8,
            "target_node_count": 10,
            "z_target": 0.04,
            "threshold": 0.1,
            "extraction_x_min": 0.0,
            "extraction_x_condition": "strict-greater-than",
            "periodic_y": True,
            "periodic_endpoint_included": False,
            "cross_case_y_grids_identical": same_y,
            "cross_case_pointwise_metrics_computed": False,
            "cross_case_metric_note": "Case-native refined-GLL y grids differ; no direct pointwise H/VH/VVH RMS is defined here.",
            "cases": cases_report,
            "plots": [str(path) for path in produced],
        }
        output.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
        print(json.dumps(report, indent=2, allow_nan=False))
        print(f"Report: {report_path}")
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
