"""CSV assembly for saved multi-time field-error tables.

Preserves script 08 schemas, string values, sorting and replacement behavior.
Importing this module performs no I/O and requires only the standard library.
"""
from __future__ import annotations

import csv
from pathlib import Path


FIELD_ORDER = ("concentration", "velocity", "pressure")
FIELD_MAPPINGS = {
    "concentration": {
        "file_prefix": "concentration_error",
        "relative_L2": "relative_L2_C",
        "mean_abs_error": "mean_abs_error_C",
        "max_abs_error": "absolute_Linf_C",
        "relative_Linf": "relative_Linf_C",
    },
    "velocity": {
        "file_prefix": "velocity_error",
        "relative_L2": "relative_L2_speed",
        "mean_abs_error": "mean_abs_error_speed",
        "max_abs_error": "absolute_Linf_speed",
        "relative_Linf": "relative_Linf_speed",
    },
    "pressure": {
        "file_prefix": "pressure_error",
        "relative_L2": "relative_L2_p_prime",
        "mean_abs_error": "mean_abs_error_p_prime",
        "max_abs_error": "absolute_Linf_p_prime",
        "relative_Linf": "relative_Linf_p_prime",
    },
}
COMMON_INPUT_COLUMNS = (
    "case",
    "order",
    "index",
    "reference_index",
    "valid_point_count",
    "total_grid_point_count",
)
OUTPUT_COLUMNS = (
    "comparison_set",
    "target_time",
    "field",
    "case",
    "order",
    "reference_case",
    "index",
    "reference_index",
    "relative_L2",
    "mean_abs_error",
    "max_abs_error",
    "relative_Linf",
    "valid_point_count",
    "total_grid_point_count",
)


def _require_columns(path: Path, fieldnames: list[str] | None, required_columns: list[str]) -> None:
    available = set(fieldnames or [])
    missing = [column for column in required_columns if column not in available]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")


def collect_error_file(path: Path, comparison_set: str, comparison_config: dict, field: str) -> list[dict[str, str]]:
    mapping = FIELD_MAPPINGS[field]
    metric_columns = [
        mapping["relative_L2"],
        mapping["mean_abs_error"],
        mapping["max_abs_error"],
        mapping["relative_Linf"],
    ]
    required_columns = [*COMMON_INPUT_COLUMNS, *metric_columns]
    reference_case = str(comparison_config.get("reference_case") or "")
    target_time = comparison_config.get("target_time", "")

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_columns(path, reader.fieldnames, required_columns)
        rows = []
        for input_row in reader:
            rows.append(
                {
                    "comparison_set": comparison_set,
                    "target_time": str(target_time),
                    "field": field,
                    "case": input_row["case"],
                    "order": input_row["order"],
                    "reference_case": reference_case,
                    "index": input_row["index"],
                    "reference_index": input_row["reference_index"],
                    "relative_L2": input_row[mapping["relative_L2"]],
                    "mean_abs_error": input_row[mapping["mean_abs_error"]],
                    "max_abs_error": input_row[mapping["max_abs_error"]],
                    "relative_Linf": input_row[mapping["relative_Linf"]],
                    "valid_point_count": input_row["valid_point_count"],
                    "total_grid_point_count": input_row["total_grid_point_count"],
                }
            )
    return rows


def error_row_sort_key(row: dict[str, str]) -> tuple[float, int, int, str]:
    try:
        target_time = float(row["target_time"])
    except ValueError:
        target_time = float("inf")
    try:
        order = int(row["order"])
    except ValueError:
        order = 0
    return target_time, FIELD_ORDER.index(row["field"]), order, row["case"]


def write_error_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
