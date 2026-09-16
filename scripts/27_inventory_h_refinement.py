"""Report h-refinement mesh/data inventory as JSON without modifying datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nek_post.h_refinement import HRefinementStudy, inventory_study
from nek_post.paths import ProjectPaths

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=REPO_ROOT / "config/paths.yaml")
    parser.add_argument("--study-config", type=Path, default=REPO_ROOT / "config/h_refinement.yaml")
    args = parser.parse_args()
    try:
        study = HRefinementStudy.from_yaml(args.study_config)
        report = inventory_study(study, ProjectPaths.from_yaml(args.paths_config))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
