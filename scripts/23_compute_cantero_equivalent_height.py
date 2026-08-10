"""Compute one Cantero-equivalent-height diagnostic from a Nek5000 snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

from nek_post.cantero_equivalent_height import (
    cantero_equivalent_height_path,
    compute_cantero_equivalent_height,
    save_cantero_equivalent_height_npz,
)
from nek_post.config import load_project_config
from nek_post.io_nek import get_nek_time, read_nek_file
from nek_post.paths import ProjectPaths


def _nonnegative_integer(text: str) -> int:
    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return value


def _required_case_value(cases: Mapping[str, Any], key: str) -> Any:
    try:
        return cases[key]
    except KeyError as exc:
        raise ValueError(f"Missing required cases.yaml key {key!r}.") from exc


def _parse_args(
    paths: ProjectPaths,
    cases: Mapping[str, Any],
    argv: list[str] | None = None,
) -> argparse.Namespace:
    file_indices = _required_case_value(cases, "file_indices")
    if not isinstance(file_indices, list) or not file_indices:
        raise ValueError("cases.yaml file_indices must be a non-empty list.")
    parser = argparse.ArgumentParser(
        description=(
            "Compute Cantero Eq. (4.1) equivalent height and Eq. (4.2) "
            "spanwise average for one Nek5000 snapshot."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--case",
        default=_required_case_value(cases, "reference_case"),
        help="Configured Nek5000 case label.",
    )
    parser.add_argument(
        "--index",
        type=_nonnegative_integer,
        default=int(file_indices[-1]),
        help="Nek field-file index in PREFIX.fNNNNN.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.cantero_equivalent_height_dir,
        help="Artifact root; the case subdirectory is added.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing artifact.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Execute one single-snapshot Cantero-equivalent-height computation."""
    config = load_project_config(
        REPO_ROOT / "config" / "paths.yaml",
        REPO_ROOT / "config" / "cases.yaml",
    )
    paths = ProjectPaths.from_mapping(config["paths"])
    try:
        args = _parse_args(paths, config["cases"], argv)
        case = str(args.case)
        file_prefix = str(_required_case_value(config["cases"], "file_prefix"))
        input_path = paths.case_dir(case) / f"{file_prefix}.f{args.index:05d}"
        output_path = cantero_equivalent_height_path(
            args.output_dir,
            case=case,
            index=args.index,
        )
        if output_path.exists() and not args.overwrite:
            print(
                f"Output file already exists, skipping: {output_path}\n"
                "Use --overwrite to regenerate it."
            )
            return

        data = read_nek_file(input_path)
        time = get_nek_time(data)
        plan, result = compute_cantero_equivalent_height(
            data, source_file=input_path
        )
        written_path = save_cantero_equivalent_height_npz(
            output_path,
            plan,
            result,
            case=case,
            index=args.index,
            time=time,
            source_file=str(input_path),
        )

        print(f"Case: {case}")
        print(f"Input file: {input_path}")
        print(f"Time: {time}")
        print(f"Local equivalent-height shape: {result.local_equivalent_height.shape}")
        print(f"Span-averaged-height shape: {result.span_averaged_height.shape}")
        print(
            "x range: "
            f"{result.x_coordinates[0]:.16g} to {result.x_coordinates[-1]:.16g}"
        )
        print(
            "y range: "
            f"{result.y_coordinates[0]:.16g} to {result.y_coordinates[-1]:.16g}"
        )
        print(f"Ly: {result.spanwise_length:.16g}")
        print(
            "Local h min/max: "
            f"{np.min(result.local_equivalent_height):.16g} / "
            f"{np.max(result.local_equivalent_height):.16g}"
        )
        print(
            "h_bar min/max: "
            f"{np.min(result.span_averaged_height):.16g} / "
            f"{np.max(result.span_averaged_height):.16g}"
        )
        print(f"Output artifact: {written_path}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
