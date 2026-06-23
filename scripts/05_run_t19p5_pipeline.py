"""Run the t19p5 comparison pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from nek_post.config import load_project_config  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the t19p5 comparison pipeline.")
    parser.add_argument("--comparison-set", default="t19p5", help="Named comparison set from config/cases.yaml.")
    parser.add_argument(
        "--fields",
        default="concentration",
        help="Comma-separated fields to compare and plot, for example concentration,velocity.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate extract and comparison outputs.")
    parser.add_argument("--skip-extract", action="store_true", help="Skip midspan slice extraction.")
    parser.add_argument("--skip-compare", action="store_true", help="Skip polynomial-order comparison.")
    parser.add_argument("--skip-plot", action="store_true", help="Skip summary plotting.")
    return parser.parse_args()


def _log_path(config: dict) -> Path:
    return Path(config["paths"]["postproc_root"]) / "logs" / "run_t19p5_pipeline.log"


def _append_log(config: dict, text: str) -> None:
    path = _log_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {text}\n")


def _run_command(command: list[str], config: dict) -> None:
    command_text = " ".join(command)
    print(f"Running: {command_text}", flush=True)
    _append_log(config, f"Running: {command_text}")
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    _append_log(config, f"Completed: {command_text}")


def _comparison_set(config: dict, name: str) -> dict:
    comparison_sets = config["cases"].get("comparison_sets", {})
    if name not in comparison_sets:
        available = ", ".join(sorted(comparison_sets)) or "none"
        raise ValueError(f"Unknown comparison set {name!r}. Available sets: {available}")
    return comparison_sets[name]


def _summary_paths(config: dict, comparison_set: str) -> tuple[Path, Path, Path]:
    results_root = Path(config["paths"]["results_root"])
    error_csv = results_root / "tables" / f"concentration_error_{comparison_set}.csv"
    front_csv = results_root / "tables" / f"front_position_{comparison_set}.csv"
    figure_dir = results_root / "figures" / "concentration" / comparison_set
    return error_csv, front_csv, figure_dir


def _parse_fields(raw: str) -> list[str]:
    fields = [field.strip() for field in raw.split(",") if field.strip()]
    if not fields:
        raise ValueError("--fields must include at least one field.")

    allowed = {"concentration", "velocity"}
    unknown = [field for field in fields if field not in allowed]
    if unknown:
        raise ValueError(f"Unknown field(s): {', '.join(unknown)}. Allowed fields: concentration, velocity")

    return fields


def main() -> None:
    """Run extraction, comparison, and plotting for a configured comparison set."""
    args = _parse_args()
    config = load_project_config(
        REPO_ROOT / "config" / "paths.yaml",
        REPO_ROOT / "config" / "cases.yaml",
    )

    try:
        comparison_set = _comparison_set(config, args.comparison_set)
        case_indices = comparison_set["case_indices"]
        fields = _parse_fields(args.fields)

        _append_log(config, f"Pipeline start: comparison_set={args.comparison_set}, fields={','.join(fields)}")

        if not args.skip_extract:
            for case, index in case_indices.items():
                command = [
                    sys.executable,
                    "scripts/02_extract_midspan_slice.py",
                    "--case",
                    str(case),
                    "--index",
                    str(index),
                ]
                if args.overwrite:
                    command.append("--overwrite")
                _run_command(command, config)

        for field in fields:
            if not args.skip_compare:
                command = [
                    sys.executable,
                    "scripts/03_compare_poly_orders.py",
                    "--comparison-set",
                    args.comparison_set,
                    "--field",
                    field,
                ]
                if args.overwrite:
                    command.append("--overwrite")
                _run_command(command, config)

            if not args.skip_plot:
                command = [
                    sys.executable,
                    "scripts/04_plot_summary.py",
                    "--comparison-set",
                    args.comparison_set,
                    "--field",
                    field,
                ]
                _run_command(command, config)

        error_csv, front_csv, figure_dir = _summary_paths(config, args.comparison_set)
        summary_lines = [
            "Pipeline completed.",
            f"Concentration error CSV: {error_csv}",
            f"Front position CSV: {front_csv}",
            f"Concentration figure directory: {figure_dir}",
        ]
        if "velocity" in fields:
            results_root = Path(config["paths"]["results_root"])
            summary_lines.extend(
                [
                    f"Velocity error CSV: {results_root / 'tables' / f'velocity_error_{args.comparison_set}.csv'}",
                    f"Velocity figure directory: {results_root / 'figures' / 'velocity' / args.comparison_set}",
                ]
            )
        summary = "\n".join(summary_lines)
        print(summary)
        _append_log(config, summary)
    except subprocess.CalledProcessError as exc:
        message = f"ERROR: Command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}"
        print(message, file=sys.stderr)
        _append_log(config, message)
        raise SystemExit(exc.returncode) from exc
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
