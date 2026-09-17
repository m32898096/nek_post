"""Compare exact y=0.75 h-refinement fields selected by physical snapshot time."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from nek_post.config import load_yaml
from nek_post.h_refinement import HRefinementStudy, inventory_study
from nek_post.h_refinement_comparison import (
    discover_snapshot_times, select_snapshot_time, validate_inventory_reference, sample_and_compare,
)
from nek_post.h_refinement_comparison_io import write_csv, write_snapshot_artifacts, plot_error_history
from nek_post.paths import ProjectPaths

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=REPO_ROOT / "config/paths.yaml")
    parser.add_argument("--study-config", type=Path, default=REPO_ROOT / "config/h_refinement.yaml")
    parser.add_argument("--cases-config", type=Path, default=REPO_ROOT / "config/cases.yaml")
    parser.add_argument("--times", type=float, nargs="+", help="Default: target times from project multitime comparison sets.")
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--nz", type=int, required=True)
    parser.add_argument("--max-time-error", type=float, default=.01)
    parser.add_argument("--max-time-spread", type=float, default=.01)
    parser.add_argument("--output-dir", type=Path,
                        help="Default: configured h-refinement result root / field_comparison.")
    args = parser.parse_args()
    try:
        study = HRefinementStudy.from_yaml(args.study_config)
        paths = ProjectPaths.from_yaml(args.paths_config)
        if args.output_dir is None:
            args.output_dir = paths.h_refinement_field_comparison_dir
        config = load_yaml(args.cases_config)
        targets = args.times if args.times is not None else [
            float(config["comparison_sets"][name]["target_time"]) for name in config["multitime_comparison_sets"]]
        if (not targets or not np.all(np.isfinite(targets)) or len(set(targets)) != len(targets)
                or args.nx < 2 or args.nz < 2):
            raise ValueError("Finite distinct target times and grid dimensions >=2 required.")
        if any(not np.isfinite(v) or v < 0 for v in (args.max_time_error, args.max_time_spread)):
            raise ValueError("Time tolerances must be finite and nonnegative.")
        output = args.output_dir.resolve()
        if "h_refinement" not in output.parts or "poly_order_compare" in output.parts:
            raise ValueError("Use a separate h_refinement output tree, outside poly_order_compare.")
        if any(output.is_relative_to(p.resolve()) for p in paths.case_dirs.values()):
            raise ValueError("Output must be outside raw case directories.")
        if output.exists():
            raise ValueError(f"Refusing to overwrite an existing output directory: {output}")
        print("Validating mesh inventory and reference...", flush=True)
        inventory = inventory_study(study, paths)
        counts = validate_inventory_reference(study, inventory)
        time_inventory = {c: discover_snapshot_times(paths.case_dir(c), study.file_prefix) for c in study.cases}
        selections = [{c: select_snapshot_time(c, time_inventory[c], t, max_time_error=args.max_time_error)
                       for c in study.cases} for t in targets]
        for selected in selections:
            times = [s.actual_time for s in selected.values()]
            if max(times)-min(times) > args.max_time_spread:
                raise ValueError("Selected snapshot times exceed max-time-spread.")
        output.mkdir(parents=True)
        (output / "inventory.json").write_text(json.dumps(inventory, indent=2, allow_nan=False)+"\n")
        run = dict(status="running", nx=args.nx, nz=args.nz, y_target=.75,
                   target_times=targets, max_time_error=args.max_time_error, max_time_spread=args.max_time_spread,
                   reference_case=study.provisional_reference_case,
                   reference_validation="unique largest element count at common order/domain bounds; local spacing not proven",
                   temporal_interpolation=False, observed_convergence_order=None)
        manifest = output / "run.json"
        manifest.write_text(json.dumps(run, indent=2)+"\n")
        all_rows, all_times = [], []
        grid = None
        for index, selected in enumerate(selections, 1):
            target = next(iter(selected.values())).target_time
            print(f"Sampling target t={target:g}: " + ", ".join(f"{c}=f{s.file_index:05d} (t={s.actual_time:.12g})" for c, s in selected.items()), flush=True)
            result, rows, masks = sample_and_compare(study, paths, selected, counts,
                nx=args.nx, nz=args.nz, max_time_spread=args.max_time_spread)
            if grid is None:
                grid = (result.Xi.copy(), result.Zi.copy())
            elif not all(np.array_equal(a,b) for a,b in zip(grid,(result.Xi,result.Zi))):
                raise ValueError("Common grid changed across target times.")
            write_snapshot_artifacts(output / f"snapshot_{index:03d}", result, selected, rows, masks, study.provisional_reference_case)
            all_rows.extend(rows)
            all_times.extend(asdict(s) for s in selected.values())
            print(f"Saved t={target:g}; valid fractions: " + str({k: float(m.mean()) for k,m in masks.items()}), flush=True)
        write_csv(output / "field_errors.csv", all_rows)
        write_csv(output / "selected_times.csv", all_times)
        plot_error_history(all_rows, output / "error_history.png")
        run["status"] = "complete"
        manifest.write_text(json.dumps(run, indent=2)+"\n")
        print(f"Saved h-refinement comparison to {output}", flush=True)
    except (ValueError, OSError, KeyError, AttributeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
