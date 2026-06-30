"""Plot error-versus-time figures from the multi-time error summary CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from nek_post.config import load_project_config  # noqa: E402

FIELD_ORDER = ("concentration", "velocity", "pressure")
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot multi-time error summary figures.")
    parser.add_argument("--summary-csv", help="Input multitime_error_summary.csv path.")
    parser.add_argument("--output-dir", help="Output directory for error summary figures.")
    parser.add_argument(
        "--fields",
        default="concentration,velocity,pressure",
        help="Comma-separated fields to plot.",
    )
    parser.add_argument(
        "--metrics",
        default="relative_L2,mean_abs_error,max_abs_error",
        help="Comma-separated metrics to plot.",
    )
    parser.add_argument("--orders", default="5,7,9", help="Comma-separated polynomial orders to plot.")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    return parser.parse_args()


def _parse_csv_list(raw: str, *, option_name: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError(f"{option_name} must include at least one value.")
    return values


def _parse_fields(raw: str) -> list[str]:
    fields = _parse_csv_list(raw, option_name="--fields")
    unknown = [field for field in fields if field not in FIELD_ORDER]
    if unknown:
        raise ValueError(f"Unknown field(s): {', '.join(unknown)}. Allowed fields: {', '.join(FIELD_ORDER)}")
    return fields


def _parse_metrics(raw: str) -> list[str]:
    metrics = _parse_csv_list(raw, option_name="--metrics")
    unknown = [metric for metric in metrics if metric not in METRIC_ORDER]
    if unknown:
        raise ValueError(f"Unknown metric(s): {', '.join(unknown)}. Allowed metrics: {', '.join(METRIC_ORDER)}")
    return metrics


def _parse_orders(raw: str) -> list[int]:
    values = _parse_csv_list(raw, option_name="--orders")
    orders: list[int] = []
    for value in values:
        try:
            orders.append(int(value))
        except ValueError as exc:
            raise ValueError(f"Invalid order {value!r} in --orders.") from exc
    return orders


def _default_summary_csv(config: dict) -> Path:
    return Path(config["paths"]["results_root"]) / "tables" / "multitime_error_summary.csv"


def _default_output_dir(config: dict) -> Path:
    return Path(config["paths"]["results_root"]) / "figures" / "error_summary"


def _read_summary(path: Path, metrics: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input summary CSV not found: {path}")

    required_columns = [
        "target_time",
        "field",
        "case",
        "order",
        *metrics,
    ]
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        available = set(reader.fieldnames or [])
        missing = [column for column in required_columns if column not in available]
        if missing:
            raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")
        return list(reader)


def _float_value(row: dict[str, str], column: str, path: Path) -> float:
    try:
        return float(row[column])
    except ValueError as exc:
        raise ValueError(f"{path} has non-numeric value {row[column]!r} in column {column!r}.") from exc


def _plot_metric(
    rows: list[dict[str, str]],
    field: str,
    metric: str,
    orders: list[int],
    output_dir: Path,
    dpi: int,
    summary_csv: Path,
) -> Path | None:
    field_rows = [row for row in rows if row["field"] == field]
    if not field_rows:
        print(f"WARNING: No rows found for field {field!r}; skipping.")
        return None

    unique_times = sorted({_float_value(row, "target_time", summary_csv) for row in field_rows})
    if len(unique_times) == 1:
        print(f"WARNING: Field {field!r} has only one time point: {unique_times[0]}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{metric}_{field}_vs_time.png"

    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for order in orders:
        order_rows = [row for row in field_rows if row["order"] == str(order)]
        if not order_rows:
            continue
        order_rows.sort(key=lambda row: _float_value(row, "target_time", summary_csv))
        times = [_float_value(row, "target_time", summary_csv) for row in order_rows]
        values = [_float_value(row, metric, summary_csv) for row in order_rows]
        ax.plot(times, values, marker="o", label=f"N{order}")
        plotted = True

    if not plotted:
        plt.close(fig)
        print(f"WARNING: No rows found for field {field!r} and requested orders; skipping {metric}.")
        return None

    ax.set_title(f"{METRIC_LABELS[metric]} vs time: {FIELD_LABELS[field]}")
    ax.set_xlabel("Time")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def main() -> None:
    """Read the summary CSV and write one error-vs-time figure per field and metric."""
    args = _parse_args()
    config = load_project_config(
        REPO_ROOT / "config" / "paths.yaml",
        REPO_ROOT / "config" / "cases.yaml",
    )

    try:
        fields = _parse_fields(args.fields)
        metrics = _parse_metrics(args.metrics)
        orders = _parse_orders(args.orders)
        summary_csv = Path(args.summary_csv) if args.summary_csv else _default_summary_csv(config)
        output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(config)
        rows = _read_summary(summary_csv, metrics)

        saved_paths: list[Path] = []
        skipped_fields: set[str] = set()
        for field in fields:
            if not any(row["field"] == field for row in rows):
                print(f"WARNING: No rows found for field {field!r}; skipping.")
                skipped_fields.add(field)
                continue
            for metric in metrics:
                path = _plot_metric(rows, field, metric, orders, output_dir, args.dpi, summary_csv)
                if path is not None:
                    saved_paths.append(path)

        plotted_fields = [field for field in fields if field not in skipped_fields]
        print(f"Input summary CSV: {summary_csv}")
        print(f"Output directory: {output_dir}")
        print(f"Fields plotted: {','.join(plotted_fields)}")
        print(f"Metrics plotted: {','.join(metrics)}")
        print(f"Figures written: {len(saved_paths)}")
        print("Saved figures:")
        for path in saved_paths:
            print(f"  {path}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
