"""Collect per-comparison-set error CSV files into one multi-time summary."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

from nek_post.config import load_project_config
from nek_post.paths import ProjectPaths

from nek_post.multitime_summary_io import (
    FIELD_ORDER, FIELD_MAPPINGS, COMMON_INPUT_COLUMNS, OUTPUT_COLUMNS,
    _require_columns,
    collect_error_file as _collect_file,
    error_row_sort_key as _sort_key,
    write_error_summary as _write_summary,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect multi-time error CSV summaries.")
    parser.add_argument(
        "--comparison-sets",
        help="Comma-separated comparison sets, for example t05,t10,t15,t19p5.",
    )
    parser.add_argument(
        "--fields",
        default="concentration,velocity,pressure",
        help="Comma-separated fields to collect.",
    )
    parser.add_argument("--output", help="Output CSV path.")
    parser.add_argument("--allow-missing", action="store_true", help="Warn and continue if input CSV files are missing.")
    return parser.parse_args()


def _parse_csv_list(raw: str | None, *, option_name: str) -> list[str]:
    if raw is None:
        return []

    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError(f"{option_name} must include at least one value.")
    return values


def _target_time_sort_key(item: tuple[str, dict]) -> tuple[float, str]:
    name, comparison_set = item
    try:
        target_time = float(comparison_set.get("target_time"))
    except (TypeError, ValueError):
        target_time = float("inf")
    return target_time, name


def _default_comparison_sets(config: dict) -> list[str]:
    cases_config = config["cases"]
    configured = cases_config.get("multitime_comparison_sets")
    if configured:
        return [str(name) for name in configured]

    comparison_sets = cases_config.get("comparison_sets", {})
    if comparison_sets:
        return [name for name, _ in sorted(comparison_sets.items(), key=_target_time_sort_key)]

    return ["t19p5"]


def _comparison_set_names(config: dict, raw: str | None) -> list[str]:
    names = _parse_csv_list(raw, option_name="--comparison-sets") if raw else _default_comparison_sets(config)
    comparison_sets = config["cases"].get("comparison_sets", {})
    unknown = [name for name in names if name not in comparison_sets]
    if unknown:
        available = ", ".join(sorted(comparison_sets)) or "none"
        raise ValueError(f"Unknown comparison set(s): {', '.join(unknown)}. Available sets: {available}")
    return names


def _fields(raw: str) -> list[str]:
    fields = _parse_csv_list(raw, option_name="--fields")
    unknown = [field for field in fields if field not in FIELD_MAPPINGS]
    if unknown:
        allowed = ", ".join(FIELD_ORDER)
        raise ValueError(f"Unknown field(s): {', '.join(unknown)}. Allowed fields: {allowed}")
    return fields


def _default_output_path(paths: ProjectPaths) -> Path:
    return paths.tables_dir / "multitime_error_summary.csv"


def _input_path(paths: ProjectPaths, comparison_set: str, field: str) -> Path:
    prefix = FIELD_MAPPINGS[field]["file_prefix"]
    return paths.tables_dir / f"{prefix}_{comparison_set}.csv"


def main() -> None:
    """Collect existing error CSV files without recomputing metrics."""
    args = _parse_args()
    config = load_project_config(
        REPO_ROOT / "config" / "paths.yaml",
        REPO_ROOT / "config" / "cases.yaml",
    )
    paths = ProjectPaths.from_mapping(config["paths"])

    try:
        comparison_sets = _comparison_set_names(config, args.comparison_sets)
        fields = _fields(args.fields)
        output_path = Path(args.output) if args.output else _default_output_path(paths)
        comparison_configs = config["cases"].get("comparison_sets", {})

        rows: list[dict[str, str]] = []
        warnings: list[str] = []
        for comparison_set in comparison_sets:
            comparison_config = comparison_configs[comparison_set]
            if not comparison_config.get("reference_case"):
                comparison_config = {**comparison_config, "reference_case": config["cases"].get("reference_case", "")}
            for field in fields:
                path = _input_path(paths, comparison_set, field)
                if not path.exists():
                    message = f"Missing input CSV: {path}"
                    if args.allow_missing:
                        warnings.append(message)
                        continue
                    raise FileNotFoundError(message)
                rows.extend(_collect_file(path, comparison_set, comparison_config, field))

        rows.sort(key=_sort_key)
        _write_summary(output_path, rows)

        for warning in warnings:
            print(f"WARNING: {warning}")
        print(f"Comparison sets: {','.join(comparison_sets)}")
        print(f"Fields: {','.join(fields)}")
        print(f"Rows written: {len(rows)}")
        print(f"Output CSV: {output_path}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
