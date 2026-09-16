"""CSV/NPZ reporting and h-refinement plots, separate from numerical kernels."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nek_post.h_refinement_comparison import FIELDS
from nek_post.plotting import plot_difference


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty comparison table.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_snapshot_artifacts(directory, result, selections, rows, masks, reference_case):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_csv(directory / "field_errors.csv", rows)
    write_csv(directory / "selected_times.csv", [asdict(s) for s in selections.values()])
    payload = {"Xi": result.Xi, "Zi": result.Zi, "y_target": np.asarray(.75)}
    for case, fields in result.fields.items():
        payload.update({f"{case}_{key}": values for key, values in fields.items()})
        payload[f"{case}_geometry_mask"] = result.geometry_masks[case]
    payload.update({f"common_{field}_mask": mask for field, mask in masks.items()})
    np.savez_compressed(directory / "comparison_grids.npz", **payload)
    metadata = dict(result.metadata, target_time=next(iter(selections.values())).target_time,
                    reference_case=reference_case,
                    selections={c: asdict(s) for c, s in selections.items()},
                    metric_mask_definition="per-field intersection of all-case geometry and finite field values",
                    metric_weighting="unweighted Cartesian grid samples",
                    temporal_interpolation=False,
                    common_metric_valid_fractions={k: float(v.mean()) for k, v in masks.items()})
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    for field, key in FIELDS.items():
        for case in result.fields:
            if case == reference_case:
                continue
            difference = np.where(masks[field], abs(result.fields[case][key]-result.fields[reference_case][key]), np.nan)
            plot_difference(result.Xi, result.Zi, difference, directory / f"{field}_{case}_difference.png",
                            f"{case} vs {reference_case}; target t={metadata['target_time']:g}, y=0.75",
                            label=f"Absolute {field.replace('_', ' ')} difference")


def plot_error_history(rows, output_path):
    """Plot non-reference case errors against target physical time, never order."""
    metrics = ("relative_l2_error", "mean_absolute_error", "maximum_absolute_error")
    cases = list(dict.fromkeys(r["case"] for r in rows if not r["is_reference"]))
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), squeeze=False)
    for i, field in enumerate(FIELDS):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            for case in cases:
                selected = sorted((r for r in rows if r["case"] == case and r["field"] == field),
                                  key=lambda r: r["target_time"])
                ax.plot([r["target_time"] for r in selected], [r[metric] for r in selected], marker="o", label=case)
            ax.set_title(field.replace("_", " "))
            ax.set_xlabel("Target physical time (nearest snapshots)")
            ax.set_ylabel(metric.replace("_", " "))
            ax.grid(alpha=.3)
            ax.legend()
    fig.suptitle("H-refinement errors relative to the reference mesh (reference control omitted)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
