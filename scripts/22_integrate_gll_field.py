"""Integrate one scalar Nek5000 snapshot along one physical GLL direction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]

from nek_post.config import load_project_config
from nek_post.gll_directional_integration import SUPPORTED_INTEGRATION_DIRECTIONS
from nek_post.gll_directional_workflow import (
    SUPPORTED_GLL_DIRECTIONAL_FIELDS,
    compute_gll_directional_integral,
    gll_directional_integral_path,
    save_gll_directional_integral_npz,
)
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
            "Read one Nek5000 snapshot, integrate one scalar field along one "
            "physical GLL direction, and save an NPZ artifact."
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
        "--field",
        choices=SUPPORTED_GLL_DIRECTIONAL_FIELDS,
        default="concentration",
        help="Scalar field to integrate.",
    )
    parser.add_argument(
        "--direction",
        choices=SUPPORTED_INTEGRATION_DIRECTIONS,
        required=True,
        help="Physical integration direction.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.gll_directional_integrals_dir,
        help="Artifact root; case/field/direction subdirectories are added.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing artifact.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Execute the one-snapshot directional-GLL integration workflow."""
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
        output_path = gll_directional_integral_path(
            args.output_dir,
            case=case,
            index=args.index,
            field=args.field,
            direction=args.direction,
        )
        if output_path.exists() and not args.overwrite:
            print(
                f"Output file already exists, skipping: {output_path}\n"
                "Use --overwrite to regenerate it."
            )
            return

        data = read_nek_file(input_path)
        time = get_nek_time(data)
        integral = compute_gll_directional_integral(
            data,
            field=args.field,
            direction=args.direction,
            source_file=input_path,
        )
        written_path = save_gll_directional_integral_npz(
            output_path,
            integral,
            case=case,
            index=args.index,
            time=time,
            source_file=str(input_path),
        )

        print(f"Case: {case}")
        print(f"Input file: {input_path}")
        print(f"Time: {time}")
        print(f"Field: {integral.field}")
        print(f"Direction: {integral.result.direction}")
        print(
            "Output coordinates: "
            f"{integral.result.horizontal_coordinate_name}, "
            f"{integral.result.vertical_coordinate_name}"
        )
        print(f"Output shape: {integral.result.values.shape}")
        print(f"Element count: {integral.plan.element_count}")
        print(f"Artifact: {written_path}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
