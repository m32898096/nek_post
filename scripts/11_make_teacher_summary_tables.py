"""Create teacher-facing summary tables from the multi-time error summary CSV."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

from nek_post.config import load_project_config
from nek_post.paths import ProjectPaths

FIELD_ORDER = ("concentration", "velocity", "pressure")
ORDER_LABELS = ("N5", "N7", "N9")
METRIC_ORDER = ("relative_L2", "mean_abs_error", "max_abs_error")
FIELD_LABELS = {
    "concentration": "Concentration C",
    "velocity": "Velocity magnitude |u|",
    "pressure": "Pressure fluctuation p'",
}
METRIC_LABELS = {
    "relative_L2": "Relative L2 error",
    "mean_abs_error": "Mean absolute error",
    "max_abs_error": "Maximum absolute error",
}
OUTPUT_NAMES = {
    "relative_L2": "teacher_relative_L2_summary.csv",
    "mean_abs_error": "teacher_mean_abs_error_summary.csv",
    "max_abs_error": "teacher_max_abs_error_summary.csv",
}
REQUIRED_COLUMNS = (
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
SUMMARY_COLUMNS = (
    "target_time",
    "comparison_set",
    "field",
    "N5",
    "N7",
    "N9",
    "best_lower_order",
    "trend",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create teacher-facing error summary tables.")
    parser.add_argument("--summary-csv", help="Input multitime_error_summary.csv path.")
    parser.add_argument("--output-dir", help="Output directory for teacher-facing reports.")
    return parser.parse_args()


def _default_summary_csv(paths: ProjectPaths) -> Path:
    return paths.tables_dir / "multitime_error_summary.csv"


def _default_output_dir(paths: ProjectPaths) -> Path:
    return paths.reports_dir


def _require_columns(path: Path, fieldnames: list[str] | None) -> None:
    available = set(fieldnames or [])
    missing = [column for column in REQUIRED_COLUMNS if column not in available]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")


def _read_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            "Missing multitime_error_summary.csv. Run scripts/08_collect_multitime_error_summary.py first."
        )

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_columns(path, reader.fieldnames)
        return list(reader)


def _order_label(raw_order: str) -> str | None:
    stripped = raw_order.strip()
    if stripped.upper().startswith("N"):
        stripped = stripped[1:]
    if stripped in {"5", "7", "9"}:
        return f"N{stripped}"
    return None


def _float_or_none(raw_value: str) -> float | None:
    value = raw_value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _format_value(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4g}"


def _sort_key(row: dict[str, str]) -> tuple[float, int, str, str]:
    try:
        target_time = float(row["target_time"])
    except ValueError:
        target_time = float("inf")
    field_rank = FIELD_ORDER.index(row["field"]) if row["field"] in FIELD_ORDER else len(FIELD_ORDER)
    return target_time, field_rank, row["comparison_set"], row["field"]


def _best_lower_order(values: dict[str, float | None]) -> str:
    available = [(order, value) for order, value in values.items() if value is not None]
    if not available:
        return "NA"
    return min(available, key=lambda item: item[1])[0]


def _trend(values: dict[str, float | None]) -> str:
    n5 = values["N5"]
    n7 = values["N7"]
    n9 = values["N9"]
    if n5 is None or n7 is None or n9 is None:
        return "NA"
    if n5 > n7 > n9:
        return "N5 > N7 > N9"
    if n5 >= n7 >= n9:
        return "mostly decreasing"
    return "not monotonic"


def _build_summary_rows(rows: list[dict[str, str]], metric: str) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], dict[str, float | None]] = {}
    for row in rows:
        field = row["field"]
        if field not in FIELD_ORDER:
            continue
        order = _order_label(row["order"])
        if order not in ORDER_LABELS:
            continue
        key = (row["target_time"], row["comparison_set"], field)
        grouped.setdefault(key, {label: None for label in ORDER_LABELS})[order] = _float_or_none(row[metric])

    summary_rows: list[dict[str, str]] = []
    for target_time, comparison_set, field in sorted(grouped, key=_group_sort_key):
        values = grouped[(target_time, comparison_set, field)]
        summary_rows.append(
            {
                "target_time": target_time,
                "comparison_set": comparison_set,
                "field": field,
                "N5": _format_value(values["N5"]),
                "N7": _format_value(values["N7"]),
                "N9": _format_value(values["N9"]),
                "best_lower_order": _best_lower_order(values),
                "trend": _trend(values),
            }
        )
    return summary_rows


def _group_sort_key(item: tuple[str, str, str]) -> tuple[float, int, str, str]:
    target_time, comparison_set, field = item
    return _sort_key({"target_time": target_time, "comparison_set": comparison_set, "field": field})


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict[str, str]]) -> str:
    headers = [
        "target_time",
        "comparison_set",
        "field",
        "N5",
        "N7",
        "N9",
        "best_lower_order",
        "trend",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        display_row = dict(row)
        display_row["field"] = FIELD_LABELS.get(row["field"], row["field"])
        lines.append("| " + " | ".join(_markdown_cell(display_row[column]) for column in headers) + " |")
    return "\n".join(lines)


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _write_markdown(path: Path, metric_rows: dict[str, list[dict[str, str]]]) -> None:
    content = f"""# Nek5000 Polynomial-Order Error Summary

## Purpose

This comparison is intended to evaluate the trade-off between numerical accuracy and computational cost. N11 is used as the current highest-order reference solution, while N5, N7, and N9 are compared against N11.

## Method

- selected time snapshots: t approximately 5, 10, 15, and 19.5
- y = 0.75 midspan x-z slice
- common 500 x 200 x-z grid
- fields:
  - concentration C
  - velocity magnitude |u|
  - pressure fluctuation p' = p - mean(p)
- metrics:
  - relative L2 error
  - mean absolute error
  - maximum absolute error

## Table 1. {METRIC_LABELS["relative_L2"]}

{_markdown_table(metric_rows["relative_L2"])}

## Table 2. {METRIC_LABELS["mean_abs_error"]}

{_markdown_table(metric_rows["mean_abs_error"])}

## Supplementary Table. {METRIC_LABELS["max_abs_error"]}

{_markdown_table(metric_rows["max_abs_error"])}

## Interpretation

If N9 gives the smallest error for most fields and times, it can be interpreted as the most promising lower-order candidate relative to N11. However, the final selection should also include computational cost indicators such as wall time, CPU hours, memory usage, and output file size.
"""
    path.write_text(content, encoding="utf-8")


def _print_best_counts(metric: str, rows: list[dict[str, str]]) -> None:
    counts = Counter(row["best_lower_order"] for row in rows)
    print(f"{metric} best order counts:")
    for order in ORDER_LABELS:
        print(f"  {order}: {counts.get(order, 0)}")
    if counts.get("NA", 0):
        print(f"  NA: {counts['NA']}")


def main() -> None:
    """Read collected errors and write compact CSV and Markdown summaries."""
    args = _parse_args()
    config = load_project_config(
        REPO_ROOT / "config" / "paths.yaml",
        REPO_ROOT / "config" / "cases.yaml",
    )
    paths = ProjectPaths.from_mapping(config["paths"])

    try:
        summary_csv = Path(args.summary_csv) if args.summary_csv else _default_summary_csv(paths)
        output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(paths)
        rows = _read_summary(summary_csv)

        metric_rows: dict[str, list[dict[str, str]]] = {}
        generated_paths: list[Path] = []
        for metric in METRIC_ORDER:
            summary_rows = _build_summary_rows(rows, metric)
            metric_rows[metric] = summary_rows
            path = output_dir / OUTPUT_NAMES[metric]
            _write_csv(path, summary_rows)
            generated_paths.append(path)

        markdown_path = output_dir / "teacher_error_summary.md"
        _write_markdown(markdown_path, metric_rows)
        generated_paths.append(markdown_path)

        print(f"Input CSV: {summary_csv}")
        print(f"Output directory: {output_dir}")
        print("Generated files:")
        for path in generated_paths:
            print(f"  {path}")
        print("Rows per summary table:")
        for metric in METRIC_ORDER:
            print(f"  {metric}: {len(metric_rows[metric])}")
        for metric in METRIC_ORDER:
            _print_best_counts(metric, metric_rows[metric])
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
