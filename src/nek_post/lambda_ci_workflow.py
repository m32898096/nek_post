"""One-frame native-GLL gradient -> L2 projection -> swirling -> VTR workflow."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import os
import resource
import shutil
import time
from typing import Callable

import numpy as np

from nek_post.io_nek import read_nek_file, read_nek_header
from nek_post.lambda_ci_export import (
    assemble_rectilinear_lambda_fields, write_rectilinear_lambda_vtr, write_single_frame_pvd,
)
from nek_post.l2_projection import (
    build_l2_projection_operator, project_velocity_gradient, structured_global_node_shape,
)
from nek_post.spectral_derivatives import element_velocity_gradient
from nek_post.swirling_strength import compute_swirling_strength

DEFAULT_OUTPUT_ROOT = Path("/data/Nek5000_data/results/h_refinement/lambda_ci")


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _nearest_directory(path: Path) -> Path:
    while not path.exists():
        path = path.parent
    if not path.is_dir():
        raise ValueError("Output path or an existing ancestor is not a directory.")
    return path


def available_memory_bytes() -> int:
    """Available Linux memory, limited further by an exposed cgroup limit."""
    values = Path("/proc/meminfo").read_text().splitlines()
    available = next(int(line.split()[1])*1024 for line in values if line.startswith("MemAvailable:"))
    for maximum, current in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        if Path(maximum).is_file() and Path(current).is_file():
            limit = Path(maximum).read_text().strip()
            if limit != "max":
                available = min(available, max(0, int(limit)-int(Path(current).read_text())))
    return available


def inspect_single_frame(
    case_dir: str | Path, index: int, output_dir: str | Path | None = None, *,
    chunk_size: int = 256, periodic_axes: tuple[str, ...] = ("x", "y"),
    expected_element_counts: tuple[int, int, int] = (272, 12, 8),
) -> dict:
    """Header-only inspection and conservative RAM/disk plan; creates nothing."""
    index = _positive_integer(index, "index")
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    case_dir = Path(case_dir).resolve()
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Input case directory not found: {case_dir}")
    source = case_dir / f"GC0.f{index:05d}"
    if not source.is_file():
        raise FileNotFoundError(f"Single snapshot not found: {source}")
    output_root = DEFAULT_OUTPUT_ROOT if output_dir is None else Path(output_dir)
    output_root = output_root.expanduser().resolve()
    output_dir = output_root / case_dir.name / f"f{index:05d}"
    repository_root = Path(__file__).resolve().parents[2]
    if output_dir.is_relative_to(case_dir):
        raise ValueError("Output directory must be outside the original simulation directory.")
    if output_dir.is_relative_to(repository_root):
        raise ValueError("Output directory must be outside the source-code repository.")
    ancestor = _nearest_directory(output_dir)
    if not os.access(ancestor, os.W_OK | os.X_OK):
        raise PermissionError(f"Output ancestor is not writable: {ancestor}")
    stem = f"{case_dir.name}_f{index:05d}_lambda_ci"
    paths = {k: str(output_dir / (stem+suffix)) for k, suffix in (("vtr", ".vtr"), ("pvd", ".pvd"), ("metadata", ".json"))}
    for path in paths.values():
        if Path(path).exists() or Path(path).is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    header = read_nek_header(source)
    if header.nb_dims != 3 or header.nb_vars[0] != 3 or header.nb_vars[1] != 3:
        raise ValueError("Snapshot must contain native 3D coordinates and all three velocities.")
    if header.nb_files != 1 or header.nb_elems_file != header.nb_elems or header.wdsz not in (4, 8):
        raise ValueError("Only complete unsplit float32/float64 snapshots are supported.")
    if not np.isfinite(header.time):
        raise ValueError("Snapshot physical timestamp must be finite.")
    # Shape helper also validates counts, polynomial resolutions and axes.
    projection_shape = structured_global_node_shape(expected_element_counts, header.orders, periodic_axes=periodic_axes)
    if header.nb_elems != int(np.prod(expected_element_counts)):
        raise ValueError("Header element count differs from expected structured dimensions.")
    local_nodes = int(header.nb_elems*np.prod(header.orders))
    # Default confirmed N7_H orientation: physical x/y/z = r/s/t. The actual
    # mapping is checked after load and export counts recomputed from it.
    visual_shape = structured_global_node_shape(expected_element_counts, header.orders, periodic_axes=())
    projection_nodes = int(np.prod(projection_shape))
    visual_nodes = int(np.prod(visual_shape))
    reader_bytes = local_nodes*sum(header.nb_vars)*header.wdsz
    gradient_bytes = local_nodes*9*8
    weights_bytes = local_nodes*8
    mass_bytes = projection_nodes*8
    diagnostic_workspace = mass_bytes*2
    # Two gradient tensors are required by Stage 2's non-overlapping input/
    # output API. Release raw immediately after projection. Include reader
    # file buffers, diagnostics, VTK/output storage and a 1-GiB safety margin.
    estimated_ram = reader_bytes+source.stat().st_size+2*gradient_bytes+weights_bytes+mass_bytes+diagnostic_workspace+1024**3
    output_bytes = 16*visual_nodes+8*sum(visual_shape)+1024**2
    disk_required = 2*output_bytes+1024**2
    memory_available = available_memory_bytes()
    disk_available = shutil.disk_usage(ancestor).free
    return {
        "source_file": str(source), "frame_index": index, "time": float(header.time),
        "istep": int(header.istep), "element_count": int(header.nb_elems),
        "header_orders_xyz": list(header.orders), "element_local_shape": list(reversed(header.orders)),
        "input_word_bytes": int(header.wdsz), "local_node_count": local_nodes,
        "expected_element_counts_xyz": list(expected_element_counts), "periodic_axes": list(periodic_axes),
        "expected_projection_shape_xyz": list(projection_shape), "expected_projection_nodes": projection_nodes,
        "estimated_visual_shape_xyz": list(visual_shape), "estimated_visual_nodes": visual_nodes,
        "estimated_linear_volume_cells": int(np.prod(np.asarray(visual_shape)-1)),
        "chunk_size": chunk_size, "estimated_ram_bytes": estimated_ram,
        "available_ram_bytes": memory_available, "estimated_output_bytes": output_bytes,
        "required_disk_bytes": disk_required, "available_disk_bytes": disk_available,
        "resources_acceptable": bool(estimated_ram < memory_available and disk_required < disk_available),
        "output_paths": paths,
    }


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(8*1024**2):
            digest.update(block)
    return digest.hexdigest()


def run_single_frame_lambda_ci(
    case_dir: str | Path, index: int, output_dir: str | Path | None = None, *,
    chunk_size: int = 256, diagnostics: bool = False, dry_run: bool = False,
    periodic_axes: tuple[str, ...] = ("x", "y"),
    expected_element_counts: tuple[int, int, int] = (272, 12, 8),
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Process exactly one file through existing Stage 1-3 public APIs.

    No frame list/loop, interpolation-before-differentiation or method fallback
    is provided. Dry run reads only the header and writes no files. Fresh output
    paths are required. Original source bytes and stat identity are checked
    before/after. RAM/disk insufficiency stops before loading the snapshot.
    """
    started = time.perf_counter()
    report = inspect_single_frame(case_dir, index, output_dir, chunk_size=chunk_size,
                                  periodic_axes=periodic_axes, expected_element_counts=expected_element_counts)
    if dry_run:
        return report
    if not report["resources_acceptable"]:
        raise MemoryError("Resource estimate exceeds available RAM or disk; snapshot was not loaded.")
    source = Path(report["source_file"])
    original_stat = source.stat()
    source_hash = _file_hash(source)
    timings = {"preflight_and_input_hash": time.perf_counter()-started}
    def notify(message):
        if progress is not None: progress(message)
    notify("Reading one snapshot (coordinates and velocities only)")
    stage = time.perf_counter()
    header = read_nek_header(source)
    skip = ("pressure", "temperature") + tuple(f"s{i:02d}" for i in range(1, header.nb_vars[4]+1))
    data = read_nek_file(source, dtype=f"float{header.wdsz*8}", skip_vars=skip)
    if len(data.elem) != header.nb_elems or data.time != header.time:
        raise ValueError("Reader payload count/time differs from the header.")
    timings["read"] = time.perf_counter()-stage
    notify("Computing native element-local spectral velocity gradients")
    stage = time.perf_counter()
    shape = tuple(reversed(header.orders))
    raw = np.empty((header.nb_elems, *shape, 3, 3), dtype=np.float64)
    for ei, element in enumerate(data.elem):
        raw[ei] = element_velocity_gradient(element)
        if (ei+1) % 2048 == 0: notify(f"Gradients: {ei+1}/{header.nb_elems} elements")
    timings["spectral_gradients"] = time.perf_counter()-stage
    notify("Preparing validated GLL lumped L2 operator")
    stage = time.perf_counter()
    operator = build_l2_projection_operator(data, periodic_axes=periodic_axes,
        chunk_size=chunk_size, expected_element_counts=expected_element_counts)
    timings["projection_preparation"] = time.perf_counter()-stage
    mapping = operator.node_map
    if mapping.physical_to_array_axes != (2, 1, 0):
        raise ValueError("This single-frame workflow requires the verified physical x/y/z to array-axis mapping (2,1,0).")
    if mapping.global_node_count != report["expected_projection_nodes"]:
        raise ValueError("Actual projection node count differs from the preflight plan.")
    report["projection"] = {
        "global_node_count": mapping.global_node_count, "shared_node_count": mapping.shared_node_count,
        "physical_to_array_axes": list(mapping.physical_to_array_axes),
        "mass_min": float(operator.mass.min()), "mass_max": float(operator.mass.max()),
        "mass_sum": float(operator.mass.sum()), "operator_array_bytes": operator.storage_bytes,
        "diagnostics_enabled": bool(diagnostics),
    }
    notify("Projecting all nine gradient components")
    stage = time.perf_counter()
    projected = project_velocity_gradient(operator, raw, diagnostics=diagnostics, chunk_size=chunk_size)
    del raw
    timings["l2_projection"] = time.perf_counter()-stage
    if projected.diagnostics is not None:
        report["projection"].update(
            discontinuity_before=projected.diagnostics.discontinuity_before.tolist(),
            discontinuity_after=projected.diagnostics.discontinuity_after.tolist(),
        )
    del operator
    notify("Computing lambda_ci from projected gradients")
    stage = time.perf_counter()
    swirling = compute_swirling_strength(projected.values, chunk_size=chunk_size)
    del projected
    timings["swirling_strength"] = time.perf_counter()-stage
    report["lambda_ci"] = {
        "min": float(swirling.lambda_ci.min()), "max": float(swirling.lambda_ci.max()),
        "local_nonzero_fraction": float(np.count_nonzero(swirling.lambda_ci)/swirling.lambda_ci.size),
        "squared_min": float(swirling.lambda_ci_squared.min()), "squared_max": float(swirling.lambda_ci_squared.max()),
    }
    notify("Assembling native-node rectilinear visualization fields")
    stage = time.perf_counter()
    visual = assemble_rectilinear_lambda_fields(data, mapping, swirling.lambda_ci, swirling.lambda_ci_squared, chunk_size=chunk_size)
    del swirling, data, mapping
    timings["visualization_assembly"] = time.perf_counter()-stage
    report["visualization"] = {
        "format": "VTK XML RectilinearGrid (.vtr), appended binary, uncompressed",
        "interpolation": "linear volume cells between adjacent native GLL nodes",
        "shape_xyz": [len(v) for v in visual.coordinates], "point_count": int(visual.lambda_ci.size),
        "cell_count": int(np.prod(np.asarray(visual.lambda_ci.shape)-1)),
        "bounds_xyz": [[float(v[0]), float(v[-1])] for v in visual.coordinates],
        "periodic_physical_endpoints_preserved": True,
        "nonzero_fraction": float(np.count_nonzero(visual.lambda_ci)/visual.lambda_ci.size),
        "paraview_visually_verified": False,
    }
    stage = time.perf_counter()
    final_stat = source.stat()
    if (original_stat.st_size, original_stat.st_mtime_ns, original_stat.st_ino) != (final_stat.st_size, final_stat.st_mtime_ns, final_stat.st_ino) or _file_hash(source) != source_hash:
        raise RuntimeError("Source changed during processing; results will not be written.")
    report["source_sha256"] = source_hash
    report["source_unchanged"] = True
    timings["source_integrity_recheck"] = time.perf_counter()-stage
    notify("Writing new VTR/PVD output")
    stage = time.perf_counter()
    paths = report["output_paths"]
    # A collision arising during computation must also stop before writing.
    for value in paths.values():
        if Path(value).exists() or Path(value).is_symlink():
            raise FileExistsError(f"Output appeared during computation: {value}")
    write_rectilinear_lambda_vtr(paths["vtr"], visual, time=report["time"])
    write_single_frame_pvd(paths["pvd"], paths["vtr"], time=report["time"])
    del visual
    timings["export"] = time.perf_counter()-stage
    report["output_file_bytes"] = {key: Path(paths[key]).stat().st_size for key in ("vtr", "pvd")}
    report["stage_seconds"] = timings
    report["total_seconds"] = time.perf_counter()-started
    report["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    with Path(paths["metadata"]).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report
