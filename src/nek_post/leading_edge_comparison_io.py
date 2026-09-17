"""Read saved raw leading edges, report comparisons, and render diagnostics."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nek-post")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nek_post.h_refinement_analysis import tree_hashes
from nek_post.h_refinement_leading_edge import METHODS, method_key
from nek_post.leading_edge_comparison import (
    align_fronts, common_reference_timeline, compare_extraction_methods,
    front_difference, front_statistics, validate_front_history,
)
from nek_post.leading_edge_plotting import periodic_leading_edge_plot_arrays


def _write_table(path, rows, columns=None):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value):
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _finite_peak(rows, column):
    usable = [row for row in rows if np.isfinite(row[column])]
    return max(usable, key=lambda row: row[column]) if usable else None


def _finite_range(rows, column):
    usable = [row[column] for row in rows if np.isfinite(row[column])]
    return [min(usable), max(usable)] if usable else None


def read_raw_history(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if (metadata["status"] != "complete" or metadata["sampling_mode"] != "uniform-spectral"
            or metadata["y_grid_selection"] != "explicit-ny"
            or metadata["temporal_interpolation"] is not False
            or tuple(metadata["methods"]) != METHODS
            or metadata["periodic_endpoint_included"] is not False or metadata["periodic_y"] is not True
            or metadata["extraction_x_condition"] != "strict-greater-than"):
        raise ValueError("Raw leading-edge metadata is incomplete or incompatible.")
    with np.load(directory / "raw_curves.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    for key, metadata_key in (("z_target", "z_target"), ("threshold", "threshold"),
                              ("x_min", "extraction_x_min"), ("y_endpoint", "y_endpoint")):
        if float(arrays[key]) != metadata[metadata_key]:
            raise ValueError(f"Raw array/metadata definition mismatch: {key}.")
    if (metadata["z_target"], metadata["threshold"], metadata["extraction_x_min"]) != (.04, .1, 0.):
        raise ValueError("Expected z=.04, C=.1 and strict x>0 leading-edge definition.")
    x, y = arrays["x"], arrays["y"]
    if (x.shape != (metadata["nx"],) or y.shape != (metadata["ny"],)
            or len(x) < 2 or len(y) < 2
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0)
            or not np.array_equal(x, np.linspace(x[0], x[-1], len(x)))
            or not np.array_equal(y, np.linspace(y[0], metadata["y_endpoint"], len(y), endpoint=False))):
        raise ValueError("Raw x/y coordinates must be the uniform endpoint-excluded physical grid.")
    time, indices = arrays["time"], arrays["file_index"]
    if (time.shape != (metadata["frame_count"],) or indices.shape != time.shape
            or indices.dtype.kind not in "iu" or np.any(np.diff(indices) <= 0)
            or not np.array_equal(time, [f["time"] for f in metadata["source_files"]])
            or not np.array_equal(indices, [f["file_index"] for f in metadata["source_files"]])):
        raise ValueError("Raw times/indices differ from the source manifest.")
    for method in METHODS:
        key = method_key(method)
        validate_front_history(time, arrays[key+"_front"], arrays[key+"_success"])
        counts = arrays[key+"_crossing_count"]
        if (arrays[key+"_front"].shape != (len(time), len(y))
                or counts.shape != (len(time), len(y)) or counts.dtype.kind not in "iu"
                or np.any(counts < 0)):
            raise ValueError("Raw curve dimensions/crossing counts are invalid.")
        values, valid = arrays[key+"_front"], arrays[key+"_success"]
        if np.any(valid & ((values <= metadata["extraction_x_min"]) | (values > x[-1]))):
            raise ValueError("A successful front lies outside the strict physical extraction domain.")
    return metadata, arrays


def _save_plot(fig, directory, stem):
    fig.tight_layout()
    fig.savefig(directory / (stem+".png"), dpi=180)
    fig.savefig(directory / (stem+".pdf"))
    plt.close(fig)


def _plot_histories(output, cases, reference, time, aligned, morphology, metrics, method_rows, x, y, endpoint):
    palette = ("#0072B2", "#D55E00", "#009E73")
    colors = {case: palette[i % len(palette)] for i, case in enumerate(cases)}
    subtitle = f"z=0.04, C=0.1, x>0; common {len(x)} × {len(y)} sampling grid"
    fig, ax = plt.subplots(figsize=(13, 4))
    for case in cases:
        for position in range(len(time)):
            yy, xx = periodic_leading_edge_plot_arrays(y, aligned[case].front[position], endpoint)
            ax.plot(xx, yy, color=colors[case], linewidth=.6, alpha=.65,
                    label=case if position == 0 else None)
    ax.set(xlabel="Front position x", ylabel="Physical y",
           title=f"All {len(time)} common-time leading-edge profiles, t={time[0]:g}–{time[-1]:g}\n{subtitle}")
    ax.legend(); ax.grid(alpha=.2)
    _save_plot(fig, output, "leading_edge_evolution_overlay")
    positions = np.unique(np.rint(np.linspace(0, len(time)-1, min(9, len(time)))).astype(int))
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), squeeze=False)
    for ax, position in zip(axes.ravel(), positions):
        for case in cases:
            yy, xx = periodic_leading_edge_plot_arrays(y, aligned[case].front[position], endpoint)
            ax.plot(xx, yy, color=colors[case], label=case, linewidth=1.)
        ax.set(title=f"t={time[position]:.5g}", xlabel="Front position x", ylabel="Physical y")
        ax.grid(alpha=.25)
    for ax in axes.ravel()[len(positions):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Leading-edge evolution on common physical times\n"+subtitle)
    _save_plot(fig, output, "leading_edge_profiles_selected_times")

    for column, label, stem in (("mean_front", "Spanwise mean front x", "mean_front_history"),
                                ("std_front", "Spanwise standard deviation", "front_std_history"),
                                ("peak_to_peak", "Peak-to-peak spanwise amplitude", "front_amplitude_history")):
        fig, ax = plt.subplots(figsize=(8, 4))
        for case in cases:
            rows = [r for r in morphology if r["case"] == case]
            ax.plot(time, [r[column] for r in rows], color=colors[case], label=case)
        ax.set(xlabel="Aligned physical time", ylabel=label, title=subtitle)
        ax.legend(); ax.grid(alpha=.25)
        _save_plot(fig, output, stem)
    for column, label, stem in (("total_rms", "Total spanwise RMS difference", "total_rms_history"),
                                ("shape_rms", "Mean-removed spanwise RMS difference", "shape_rms_history"),
                                ("bulk_difference", "Signed mean-front difference", "bulk_difference_history")):
        fig, ax = plt.subplots(figsize=(8, 4))
        for case in cases:
            if case == reference:
                continue
            rows = [r for r in metrics if r["case"] == case]
            ax.plot(time, [r[column] for r in rows], color=colors[case], label=case)
        ax.set(xlabel="Aligned physical time", ylabel=label,
               title=f"Relative to {reference}, finest available numerical reference")
        ax.legend(); ax.grid(alpha=.25)
        _save_plot(fig, output, stem)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for case in cases:
        rows = [r for r in method_rows if r["case"] == case]
        for ax, key, label in zip(axes, ("rms_difference", "max_abs_difference"),
                                  ("RMS method difference", "Maximum method difference")):
            ax.plot([r["actual_time"] for r in rows], [r[key] for r in rows], color=colors[case], label=case)
            ax.set_ylabel(label); ax.grid(alpha=.25)
    axes[0].legend(); axes[0].set_title("Rightmost crossing vs Moore: extraction-method sensitivity")
    axes[-1].set_xlabel("Original physical time (no temporal alignment)")
    _save_plot(fig, output, "extraction_method_difference_history")


def analyze_leading_edge_histories(raw_directory, output, cases, reference):
    """Analyze saved curves only; never access Nek fields or overwrite raw data."""
    raw_directory, output = Path(raw_directory).resolve(), Path(output).resolve()
    if (output.exists() or output.is_relative_to(raw_directory)
            or raw_directory.is_relative_to(output)):
        raise ValueError("Comparison output must be new and separate from raw histories.")
    if len(cases) < 2 or len(set(cases)) != len(cases) or reference not in cases:
        raise ValueError("Distinct cases including the numerical reference are required.")
    before = tree_hashes(raw_directory)
    metadata, raw = {}, {}
    for case in cases:
        metadata[case], raw[case] = read_raw_history(raw_directory / case)
        if metadata[case]["case"] != case:
            raise ValueError("Raw case label does not match the requested study.")
    first = raw[cases[0]]
    for case in cases:
        if any(not np.array_equal(raw[case][key], first[key]) for key in
               ("x", "y", "z_target", "threshold", "x_min", "y_endpoint")):
            raise ValueError("Cases do not share identical physical coordinates and front definitions.")
        if metadata[case]["polynomial_order"] != metadata[reference]["polynomial_order"]:
            raise ValueError("Polynomial orders differ across the h study.")
    if any(metadata[reference]["element_count"] <= metadata[c]["element_count"] for c in cases if c != reference):
        raise ValueError("The numerical reference must have the unique largest element count.")
    time, intersection = common_reference_timeline({c: raw[c]["time"] for c in cases}, reference)
    primary = method_key(METHODS[0])
    aligned = {c: align_fronts(raw[c]["time"], raw[c][primary+"_front"],
                             raw[c][primary+"_success"], time) for c in cases}
    common = np.logical_and.reduce([a.valid for a in aligned.values()])
    metrics, morphology, brackets, raw_stats, method_rows, points = [], [], [], [], [], []
    for case in cases:
        source, result = raw[case], aligned[case]
        for i, t in enumerate(time):
            lo, hi = result.left_position[i], result.right_position[i]
            brackets.append(dict(case=case, target_time=float(t),
                left_file_index=int(source["file_index"][lo]), right_file_index=int(source["file_index"][hi]),
                left_time=float(source["time"][lo]), right_time=float(source["time"][hi]),
                right_weight=float(result.right_weight[i]), exact_stored_time=bool(lo == hi),
                interpolated_valid_y_count=int(result.valid[i].sum()),
                common_valid_y_count=int(common[i].sum())))
            morphology.append(dict(case=case, time=float(t),
                                   **front_statistics(result.front[i], common[i])))
            if case != reference:
                metrics.append(dict(case=case, reference_case=reference, time=float(t),
                    **front_difference(result.front[i], aligned[reference].front[i], common[i])))
        for i, t in enumerate(source["time"]):
            identity = dict(case=case, file_index=int(source["file_index"][i]), actual_time=float(t))
            for method in METHODS:
                key = method_key(method)
                raw_stats.append(dict(identity, extraction_method=method,
                    **front_statistics(source[key+"_front"][i], source[key+"_success"][i])))
            a, b = (method_key(m) for m in METHODS)
            comparison = compare_extraction_methods(source[a+"_front"][i], source[a+"_success"][i],
                                                    source[b+"_front"][i], source[b+"_success"][i])
            differing = comparison.pop("different_y_positions")
            method_rows.append(dict(identity, **comparison,
                crossing_count_difference_y_count=int(np.count_nonzero(
                    source[a+"_crossing_count"][i] != source[b+"_crossing_count"][i]))))
            for j in differing:
                av, bv = source[a+"_front"][i, j], source[b+"_front"][i, j]
                points.append(dict(identity, y_position=int(j), y=float(source["y"][j]),
                    rightmost_front=float(av), moore_front=float(bv),
                    rightmost_success=bool(source[a+"_success"][i, j]),
                    moore_success=bool(source[b+"_success"][i, j]),
                    signed_difference=float(av-bv) if np.isfinite(av) and np.isfinite(bv) else float("nan")))
    shared_indices = sorted(set.intersection(*(set(raw[c]["file_index"].tolist()) for c in cases)))
    offsets = []
    for index in shared_indices:
        times = {c: float(raw[c]["time"][np.flatnonzero(raw[c]["file_index"] == index)[0]]) for c in cases}
        offsets.append(dict(file_index=index, **{c+"_time": t for c, t in times.items()},
            **{c+"_offset_from_reference": t-times[reference] for c, t in times.items()},
            cross_case_time_spread=max(times.values())-min(times.values())))
    summary = dict(reference_case=reference, reference_role="finest available numerical reference, not an exact solution",
        grid=dict(nx=len(first["x"]), ny=len(first["y"]), xmin=float(first["x"][0]),
                  xmax=float(first["x"][-1]), ymin=float(first["y"][0]),
                  y_endpoint=float(first["y_endpoint"]), periodic_endpoint_included=False,
                  z_target=float(first["z_target"]), threshold=float(first["threshold"]),
                  extraction_x_min=float(first["x_min"]), exact_coordinates_identical=True,
                  interpretation="Fixed common post-processing samples, not simulation resolution"),
        common_time_definition="Reference stored times within intersection; other raw curves linearly interpolated between adjacent stored times",
        physical_time_intersection=list(intersection), comparison_time_range=[float(time[0]), float(time[-1])],
        comparison_time_count=len(time), temporal_extrapolation=False,
        metric_mask="intersection of all cases' finite successful adjacent-frame crossings at each target time",
        failed_crossing_policy="No filling; an interpolated point requires both brackets; exact times use that frame only",
        mean_removal="Each case and reference mean uses the same all-case valid y support",
        weighting="Equal weights on endpoint-excluded uniform periodic y; population standard deviation (ddof=0)",
        smoothing=False, observed_h_order=None,
        observed_h_order_reason="Anisotropic graded source meshes do not justify a single scalar h",
        common_valid_fraction_min=float(common.mean(axis=1).min()),
        maximum_same_index_time_spread=max((r["cross_case_time_spread"] for r in offsets), default=None),
        maximum_same_index_offset_from_reference={c: max((abs(r[c+"_offset_from_reference"]) for r in offsets), default=None)
                                                 for c in cases},
        same_index_shared_count=len(offsets),
        source_hashes=before, cases={}, differences={})
    for case in cases:
        case_methods = [r for r in method_rows if r["case"] == case]
        stats = [r for r in raw_stats if r["case"] == case and r["extraction_method"] == METHODS[0]]
        with (raw_directory / case / "frame_statistics.csv").open() as handle:
            diagnostics = list(csv.DictReader(handle))
        expected_identities = {(case, int(i), float(t), method) for i, t in
                               zip(raw[case]["file_index"], raw[case]["time"]) for method in METHODS}
        actual_identities = {(r["case"], int(r["file_index"]), float(r["actual_time"]),
                              r["extraction_method"]) for r in diagnostics}
        if len(diagnostics) != len(expected_identities) or actual_identities != expected_identities:
            raise ValueError("Raw plane diagnostics do not match the complete history.")
        for row in diagnostics:
            nan, nonfinite = float(row["plane_nan_fraction"]), float(row["plane_nonfinite_fraction"])
            if not 0 <= nan <= nonfinite <= 1:
                raise ValueError("Invalid saved plane nonfinite fractions.")
        peak_method = _finite_peak(case_methods, "max_abs_difference")
        summary["cases"][case] = dict(frame_count=len(stats),
            physical_time_range=metadata[case]["physical_time_range"], native_ny=metadata[case]["native_ny"],
            total_successful_y_crossings=sum(r["successful_y_count"] for r in stats),
            total_failed_y_crossings=sum(r["failed_y_count"] for r in stats),
            frames_with_failed_crossings=sum(r["failed_y_count"] > 0 for r in stats),
            exactly_identical_method_frames=sum(r["exactly_identical"] for r in case_methods),
            exactly_identical_method_fraction=float(np.mean([r["exactly_identical"] for r in case_methods])),
            method_max_abs_difference=None if peak_method is None else peak_method["max_abs_difference"],
            method_common_valid_fraction_min=min(r["common_valid_fraction"] for r in case_methods),
            different_method_y_points=sum(r["different_y_count"] for r in case_methods),
            maximum_plane_nan_fraction=max(float(r["plane_nan_fraction"]) for r in diagnostics),
            maximum_plane_nonfinite_fraction=max(float(r["plane_nonfinite_fraction"]) for r in diagnostics),
            raw_std_range=_finite_range(stats, "std_front"),
            raw_peak_to_peak_range=_finite_range(stats, "peak_to_peak"),
            method_crossing_counts={m: {
                "successful": sum(r["successful_y_count"] for r in raw_stats if r["case"] == case and r["extraction_method"] == m),
                "failed": sum(r["failed_y_count"] for r in raw_stats if r["case"] == case and r["extraction_method"] == m)
            } for m in METHODS},
            final_raw_statistics=stats[-1], interpolation=metadata[case])
        if case != reference:
            rows = [r for r in metrics if r["case"] == case]
            summary["differences"][case] = dict(final=rows[-1],
                finite_comparison_time_count=sum(np.isfinite(r["total_rms"]).item() for r in rows),
                peak_total_rms=_finite_peak(rows, "total_rms"),
                peak_shape_rms=_finite_peak(rows, "shape_rms"))
    if tree_hashes(raw_directory) != before:
        raise ValueError("Raw leading-edge artifacts changed during comparison.")
    output.mkdir(parents=True)
    _write_table(output / "raw_frame_statistics.csv", raw_stats)
    _write_table(output / "aligned_morphology.csv", morphology)
    _write_table(output / "difference_metrics.csv", metrics)
    _write_table(output / "temporal_interpolation.csv", brackets)
    _write_table(output / "same_index_time_offsets.csv", offsets, columns=None if offsets else ["file_index", "cross_case_time_spread"])
    _write_table(output / "method_difference_history.csv", method_rows)
    _write_table(output / "method_difference_points.csv", points, columns=["case", "file_index", "actual_time",
        "y_position", "y", "rightmost_front", "moore_front", "rightmost_success", "moore_success", "signed_difference"])
    payload = dict(time=time, x=first["x"], y=first["y"], common_valid_mask=common)
    for case, result in aligned.items():
        payload[case+"_front"] = result.front
        payload[case+"_valid"] = result.valid
    np.savez_compressed(output / "aligned_fronts.npz", **payload)
    _plot_histories(output, cases, reference, time, aligned, morphology, metrics, method_rows,
                    first["x"], first["y"], float(first["y_endpoint"]))
    if tree_hashes(raw_directory) != before:
        raise ValueError("Raw leading-edge artifacts changed during reporting.")
    summary["raw_hashes_unchanged"] = True
    (output / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2, allow_nan=False)+"\n")
    return summary
