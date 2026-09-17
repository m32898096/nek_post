"""Stream h-study snapshots through the validated leading-edge kernels once.

This module orchestrates existing readers, spectral plans and extractors. It
does not implement interpolation or threshold/front mathematics.
"""
from __future__ import annotations

from dataclasses import asdict
import csv
import gc
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from nek_post.io_nek import get_nek_time, read_nek_file, read_nek_header
from nek_post.leading_edge_comparison import front_statistics, validate_front_history
from nek_post.leading_edge_methods import extract_leading_edge
from nek_post.spectral_horizontal_slice import (
    apply_spectral_horizontal_slice_plan,
    build_spectral_horizontal_slice_plan,
    spectral_horizontal_plan_metadata,
)

METHODS = ("rightmost-crossing", "moore-boundary")


def method_key(method):
    return method.replace("-", "_")


def sample_front_methods(data, plan, *, threshold=.1, x_min=0., source_file=None):
    """Evaluate one plane, apply both existing extractors, release the plane."""
    time = float(get_nek_time(data))
    if not np.isfinite(time):
        raise ValueError(f"{source_file}: nonfinite physical time.")
    plane = apply_spectral_horizontal_slice_plan(plan, data, source_file=source_file)
    curves = {method: extract_leading_edge(
        plan.Xi[0], plan.Yi[:, 0], plane, threshold=threshold, method=method,
        periodic_y=True, y_period=plan.y_max-plan.y_min, x_min=x_min,
    ) for method in METHODS}
    for curve in curves.values():
        if not np.array_equal(curve.y, plan.Yi[:, 0]):
            raise ValueError("An extractor changed the common physical y coordinates.")
    diagnostics = {"plane_finite_count": int(np.isfinite(plane).sum()),
                   "plane_point_count": int(plane.size),
                   "plane_nan_fraction": float(np.isnan(plane).mean()),
                   "plane_nonfinite_fraction": float((~np.isfinite(plane)).mean())}
    return time, curves, diagnostics


def extract_case(case, frames, output, *, expected_x, expected_y,
                 y_endpoint, z_target=.04, threshold=.1, x_min=0., polynomial_order=7,
                 progress=print):
    """Produce raw histories and per-frame checkpoints in a new case directory.

    Every frame's complete coordinate signatures are checked by the established
    plan application. Only the tiny raw curves survive each frame. No time
    alignment is performed here. Existing outputs are never overwritten.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Raw output already exists: {output}")
    frames = tuple(frames)
    if not frames or len({f.index for f in frames}) != len(frames):
        raise ValueError("A nonempty sequence of unique frame indices is required.")
    if any(a.index >= b.index for a, b in zip(frames, frames[1:])):
        raise ValueError("Frames must be ordered by file index.")
    headers = [read_nek_header(f.path) for f in frames]
    times = np.array([h.time for h in headers], dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"{case}: stored physical times must be strictly increasing.")
    for frame, header in zip(frames, headers):
        if (header.nb_files != 1 or header.nb_elems_file != header.nb_elems
                or header.nb_vars[0] != 3 or tuple(header.orders) != (polynomial_order+1,)*3
                or header.nb_elems != headers[0].nb_elems):
            raise ValueError(f"{frame.path}: incompatible order/geometry or split-file header.")
    output.mkdir(parents=True)
    (output / "frames").mkdir()
    metadata = {"schema_version": 1, "status": "in_progress", "case": case,
        "sampling_mode": "uniform-spectral", "y_grid_selection": "explicit-ny",
        "nx": len(expected_x), "ny": len(expected_y), "z_target": z_target,
        "threshold": threshold, "extraction_x_min": x_min,
        "extraction_x_condition": "strict-greater-than", "periodic_y": True,
        "y_endpoint": y_endpoint, "periodic_endpoint_included": False,
        "methods": list(METHODS), "temporal_interpolation": False,
        "interpretation": "Common post-processing samples; no additional simulation modes.",
        "source_files": [{"file_index": f.index, "path": str(f.path.resolve()),
                          "time": float(h.time), "size_bytes": f.path.stat().st_size,
                          "mtime_ns": f.path.stat().st_mtime_ns} for f, h in zip(frames, headers)]}
    meta_path = output / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    started = perf_counter()
    progress(f"{case}: reading first frame and building one reusable spectral plan", flush=True)
    first = read_nek_file(frames[0].path, skip_vars=("ux", "uy", "uz", "pressure"))
    plan = build_spectral_horizontal_slice_plan(
        first, nx=len(expected_x), ny=len(expected_y), z_target=z_target)
    if (not np.array_equal(plan.Xi[0], expected_x)
            or not np.array_equal(plan.Yi[:, 0], expected_y) or plan.y_max != y_endpoint
            or plan.z_target != z_target):
        raise ValueError(f"{case}: physical x/y/z grid differs from the common target.")
    if not plan.target_valid_mask.all():
        raise ValueError(f"{case}: unmapped targets; extrapolation is forbidden.")
    metadata.update(element_count=plan.element_count, native_ny=plan.native_ny,
        polynomial_order=list(plan.polynomial_order),
        plan_metadata=dict(spectral_horizontal_plan_metadata(plan)),
        inverse_mapping=asdict(plan.inverse_mapping_diagnostics),
        mapped_target_count=int(plan.target_valid_mask.sum()),
        plan_build_seconds=perf_counter()-started)
    rows, curves_by_method = [], {method: [] for method in METHODS}
    for position, frame in enumerate(frames):
        data = first if position == 0 else read_nek_file(
            frame.path, skip_vars=("ux", "uy", "uz", "pressure"))
        actual, curves, diagnostics = sample_front_methods(
            data, plan, threshold=threshold, x_min=x_min, source_file=frame.path)
        if actual != times[position]:
            raise ValueError(f"{frame.path}: header and payload physical times differ.")
        payload = {"time": actual, "file_index": frame.index}
        for method, curve in curves.items():
            valid = curve.success_mask & np.isfinite(curve.x_front)
            if np.any(curve.success_mask & ~np.isfinite(curve.x_front)):
                raise ValueError(f"{frame.path}: nonfinite successful front crossing.")
            prefix = method_key(method)
            payload.update({prefix+"_front": curve.x_front,
                            prefix+"_success": curve.success_mask,
                            prefix+"_crossing_count": curve.crossing_count})
            curves_by_method[method].append(curve)
            rows.append(dict(case=case, file_index=frame.index, actual_time=actual,
                extraction_method=method, **front_statistics(curve.x_front, valid), **diagnostics))
        np.savez_compressed(output / "frames" / f"f{frame.index:05d}.npz", **payload)
        del data
        if position == 0:
            del first
        gc.collect()
        progress(f"{case}: {position+1}/{len(frames)}, index={frame.index}, t={actual:.12g}, "
                 f"success=" + "/".join(str(int(curves[m].success_mask.sum())) for m in METHODS), flush=True)
    payload = {"time": times, "file_index": np.array([f.index for f in frames]),
               "x": plan.Xi[0], "y": plan.Yi[:, 0], "z_target": z_target,
               "threshold": threshold, "x_min": x_min, "y_endpoint": y_endpoint}
    for method, curves in curves_by_method.items():
        prefix = method_key(method)
        front = np.stack([c.x_front for c in curves])
        success = np.stack([c.success_mask for c in curves])
        validate_front_history(times, front, success)
        payload.update({prefix+"_front": front, prefix+"_success": success,
                        prefix+"_crossing_count": np.stack([c.crossing_count for c in curves])})
    np.savez_compressed(output / "raw_curves.npz", **payload)
    with (output / "frame_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata.update(status="complete", frame_count=len(frames),
                    physical_time_range=[float(times[0]), float(times[-1])],
                    coordinate_signature_validated_frames=len(frames),
                    runtime_seconds=perf_counter()-started)
    meta_path.write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    return metadata
