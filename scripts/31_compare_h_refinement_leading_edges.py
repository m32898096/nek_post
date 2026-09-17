"""Extract full common-grid leading edges and compare saved raw histories."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nek_post.front_detection_io import discover_nek_frame_paths
from nek_post.h_refinement import HRefinementStudy
from nek_post.h_refinement_leading_edge import extract_case
from nek_post.paths import ProjectPaths

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=REPO_ROOT / "config/paths.yaml")
    parser.add_argument("--study-config", type=Path, default=REPO_ROOT / "config/h_refinement.yaml")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "results/h_refinement/leading_edge/step6b")
    parser.add_argument("--phase", choices=("all", "extract", "analyze"), default="all")
    parser.add_argument("--case", action="append", help="Extract only this configured case; repeat as needed.")
    args = parser.parse_args(argv)
    try:
        paths = ProjectPaths.from_yaml(args.paths_config)
        study = HRefinementStudy.from_yaml(args.study_config)
        output = args.output_dir.resolve()
        if ("h_refinement" not in output.parts or "leading_edge" not in output.parts
                or "poly_order_compare" in output.parts
                or any(output.is_relative_to(p.resolve()) for p in paths.case_dirs.values())):
            raise ValueError("Output must be in a separate h_refinement/leading_edge tree outside raw data.")
        cases = study.cases if args.case is None else tuple(args.case)
        if len(set(cases)) != len(cases) or any(c not in study.cases for c in cases):
            raise ValueError("Extraction cases must be distinct configured h-study labels.")
        if args.case and args.phase != "extract":
            raise ValueError("--case is supported only with --phase extract.")
        if args.phase in ("all", "extract"):
            for case in cases:
                if (output / "raw" / case).exists():
                    raise FileExistsError(f"Raw case output already exists: {case}; use --phase analyze for saved outputs.")
            # Exactly the Step 6A physical grid, never a simulation resolution.
            x = np.linspace(-17., 17., 1000)
            y = np.linspace(0., 1.5, 308, endpoint=False)
            for case in cases:
                frames = discover_nek_frame_paths(paths.case_dir(case), file_prefix=study.file_prefix)
                extract_case(case, frames, output / "raw" / case, expected_x=x,
                             expected_y=y, y_endpoint=1.5, z_target=.04,
                             threshold=.1, x_min=0., polynomial_order=study.expected_polynomial_order)
        if args.phase in ("all", "analyze"):
            from nek_post.leading_edge_comparison_io import analyze_leading_edge_histories
            analyze_leading_edge_histories(output / "raw", output / "comparison",
                                          study.cases, study.provisional_reference_case)
        print(f"Step 6B {args.phase} outputs: {output}", flush=True)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
