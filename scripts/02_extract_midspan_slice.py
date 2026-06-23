"""Extract a midspan y-slice from one Nek5000 file."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from nek_post.config import load_project_config  # noqa: E402
from nek_post.io_nek import get_nek_time, read_nek_file  # noqa: E402
from nek_post.slicing import extract_y_slice, save_slice_npz  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract one Nek5000 y-midspan slice.")
    parser.add_argument("--case", help="Case name from config/cases.yaml, for example N11.")
    parser.add_argument("--index", type=int, help="File index, for example 80 for GC0.f00080.")
    parser.add_argument("--slab-ratio", type=float, help="Slab tolerance as a fraction of the y extent.")
    parser.add_argument(
        "--slice-mode",
        choices=("nearest_plane", "slab"),
        help="Extraction mode. Defaults to cases.slice.mode from config.",
    )
    parser.add_argument(
        "--y-round-decimals",
        type=int,
        help="Decimals used to group y levels for nearest-plane extraction.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing slice file.")
    return parser.parse_args()


def _input_file(config: dict, case: str, index: int) -> Path:
    case_dirs = config["paths"]["case_dirs"]
    if case not in case_dirs:
        available = ", ".join(sorted(case_dirs))
        raise ValueError(f"Unknown case {case!r}. Available cases: {available}")

    file_prefix = config["cases"]["file_prefix"]
    return Path(case_dirs[case]) / f"{file_prefix}.f{index:05d}"


def _output_file(config: dict, case: str, index: int) -> Path:
    return Path(config["paths"]["postproc_root"]) / "slices" / case / f"slice_{case}_f{index:05d}.npz"


def _log_path(config: dict) -> Path:
    return Path(config["paths"]["postproc_root"]) / "logs" / "extract_midspan_slice.log"


def _append_log(config: dict, text: str) -> None:
    path = _log_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}]\n{text}\n\n")


def _min_max_line(name: str, values: np.ndarray) -> str:
    return f"{name} min/max: {np.min(values)} / {np.max(values)}"


def _rounded_unique_count(values: np.ndarray, decimals: int) -> int:
    return int(np.unique(np.round(np.asarray(values, dtype=float), decimals)).size)


def _build_summary(
    input_path: Path,
    output_path: Path,
    case: str,
    index: int,
    time: object,
    slice_data: dict,
) -> str:
    y_round_decimals = int(slice_data.get("y_round_decimals", 10))
    lines = [
        "Nek5000 midspan slice extraction",
        "",
        f"Input file: {input_path}",
        f"Output file: {output_path}",
        f"Case: {case}",
        f"Index: {index}",
        f"Time: {time}",
        f"slice_mode: {slice_data['mode']}",
        f"y0: {slice_data['y0']}",
        f"selected_y: {slice_data.get('selected_y', '')}",
        f"dy_tol: {slice_data.get('dy_tol', '')}",
        f"slab_ratio: {slice_data['slab_ratio']}",
        f"y_round_decimals: {y_round_decimals}",
        (
            "rounded_unique_y_count_before_selection: "
            f"{slice_data.get('rounded_unique_y_count_before_selection', '')}"
        ),
        f"point_count: {slice_data['point_count']}",
        _min_max_line("x", slice_data["x"]),
        _min_max_line("y", slice_data["y"]),
        f"extracted rounded unique y count: {_rounded_unique_count(slice_data['y'], y_round_decimals)}",
        _min_max_line("z", slice_data["z"]),
        _min_max_line("C", slice_data["C"]),
        _min_max_line("u", slice_data["u"]),
        _min_max_line("v", slice_data["v"]),
        _min_max_line("w", slice_data["w"]),
        _min_max_line("p", slice_data["p"]),
    ]
    return "\n".join(lines)


def main() -> None:
    """Read one Nek5000 file, extract one y-midspan slab, and save it."""
    args = _parse_args()
    config = load_project_config(
        REPO_ROOT / "config" / "paths.yaml",
        REPO_ROOT / "config" / "cases.yaml",
    )

    case = args.case or config["cases"]["reference_case"]
    file_indices = config["cases"]["file_indices"]
    index = args.index if args.index is not None else file_indices[-1]
    slice_config = config["cases"].get("slice", {})
    slab_ratio = args.slab_ratio if args.slab_ratio is not None else slice_config.get("slab_ratio", 0.01)
    slice_mode = args.slice_mode if args.slice_mode is not None else slice_config.get("mode", "nearest_plane")
    y_round_decimals = (
        args.y_round_decimals
        if args.y_round_decimals is not None
        else int(slice_config.get("y_round_decimals", 10))
    )

    try:
        input_path = _input_file(config, case, index)
        output_path = _output_file(config, case, index)

        if output_path.exists() and not args.overwrite:
            message = f"Output file already exists, skipping: {output_path}\nUse --overwrite to regenerate it."
            print(message)
            _append_log(config, message)
            raise SystemExit(0)

        data = read_nek_file(input_path)
        time = get_nek_time(data)
        slice_data = extract_y_slice(
            data,
            slab_ratio=slab_ratio,
            mode=slice_mode,
            y_round_decimals=y_round_decimals,
        )
        metadata = {
            "case": case,
            "index": index,
            "time": time,
            "source_file": str(input_path),
            "mode": slice_data["mode"],
            "y0": slice_data["y0"],
            "slab_ratio": slice_data["slab_ratio"],
            "y_round_decimals": slice_data["y_round_decimals"],
        }
        for name in ("selected_y", "dy_tol", "rounded_unique_y_count_before_selection"):
            if name in slice_data:
                metadata[name] = slice_data[name]
        save_slice_npz(slice_data, output_path, metadata=metadata)

        summary = _build_summary(input_path, output_path, case, index, time, slice_data)
        print(summary)
        _append_log(config, summary)
    except Exception as exc:
        message = f"ERROR: {exc}"
        print(message, file=sys.stderr)
        try:
            _append_log(config, message)
        except OSError as log_exc:
            print(f"ERROR: Failed to write log file: {log_exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
