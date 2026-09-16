"""Sample h-study fields at exact physical y=0.75 on one common Cartesian grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nek_post.h_refinement import HRefinementStudy
from nek_post.h_refinement_slice import sample_h_snapshot
from nek_post.paths import ProjectPaths

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_snapshot(value: str) -> dict[str, int]:
    result = {}
    try:
        for item in value.split(","):
            case, index = item.split("=")
            case = case.strip()
            if not case or case in result:
                raise ValueError("Empty or duplicated case")
            result[case] = int(index)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use CASE=INDEX,CASE=INDEX,... with unique labels.") from exc
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=REPO_ROOT / "config/paths.yaml")
    parser.add_argument("--study-config", type=Path, default=REPO_ROOT / "config/h_refinement.yaml")
    parser.add_argument("--snapshot", type=parse_snapshot, action="append", required=True,
                        help="Per-case indices; repeat for multiple snapshot groups.")
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--nz", type=int, required=True)
    parser.add_argument("--time-atol", type=float, default=1e-8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        study = HRefinementStudy.from_yaml(args.study_config)
        paths = ProjectPaths.from_yaml(args.paths_config)
        output = args.output_dir.resolve()
        if any(output.is_relative_to(p.resolve()) for p in paths.case_dirs.values()):
            raise ValueError("Output directory must be outside raw case directories.")
        targets = [(output / f"snapshot_{i:03d}.npz", output / f"snapshot_{i:03d}.json")
                   for i in range(1, len(args.snapshot) + 1)]
        for pair in targets:
            for path in pair:
                if path.exists():
                    raise ValueError(f"Refusing to overwrite existing artifact: {path}")
        for selection, (archive, report) in zip(args.snapshot, targets):
            result = sample_h_snapshot(study, paths, selection, nx=args.nx, nz=args.nz,
                                       time_atol=args.time_atol)
            payload = {"Xi": result.Xi, "Zi": result.Zi, "y_target": np.array(.75)}
            for case, fields in result.fields.items():
                payload.update({f"{case}_{key}": value for key, value in fields.items()})
                payload[f"{case}_geometry_mask"] = result.geometry_masks[case]
            payload.update({f"common_{name}_mask": mask for name, mask in result.common_masks.items()})
            text = json.dumps(result.metadata, indent=2, allow_nan=False) + "\n"
            output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(archive, **payload)
            report.write_text(text, encoding="utf-8")
            print(f"Saved {archive} and {report}")
    except (ValueError, OSError, AttributeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
