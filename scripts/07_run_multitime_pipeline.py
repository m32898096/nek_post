"""Run configured comparison sets across multiple fields."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

from nek_post.config import load_project_config
from nek_post.paths import ProjectPaths

ALLOWED_FIELDS = ("concentration", "velocity", "pressure")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple polynomial-order comparison sets.")
    parser.add_argument(
        "--comparison-sets",
        help="Comma-separated comparison sets, for example t05,t10,t15,t19p5.",
    )
    parser.add_argument(
        "--fields",
        default="concentration,velocity,pressure",
        help="Comma-separated fields to compare and plot.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate slice and comparison outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--skip-slices", action="store_true", help="Skip midspan slice extraction.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip summary plotting.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop immediately on command failure. This is also the default behavior.",
    )
    return parser.parse_args()


def _log_path(paths: ProjectPaths) -> Path:
    return paths.logs_dir / "run_multitime_pipeline.log"


def _append_log(paths: ProjectPaths, text: str) -> None:
    path = _log_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {text}\n")


def _parse_csv_list(raw: str | None, *, option_name: str) -> list[str]:
    if raw is None:
        return []

    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError(f"{option_name} must include at least one value.")
    return values


def _default_comparison_sets(config: dict) -> list[str]:
    configured = config["cases"].get("multitime_comparison_sets")
    if configured:
        return [str(name) for name in configured]
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
    unknown = [field for field in fields if field not in ALLOWED_FIELDS]
    if unknown:
        allowed = ", ".join(ALLOWED_FIELDS)
        raise ValueError(f"Unknown field(s): {', '.join(unknown)}. Allowed fields: {allowed}")
    return fields


def _run_command(command: list[str], paths: ProjectPaths, *, dry_run: bool) -> None:
    command_text = " ".join(command)
    prefix = "Dry run" if dry_run else "Running"
    print(f"{prefix}: {command_text}", flush=True)
    _append_log(paths, f"{prefix}: {command_text}")
    if dry_run:
        _append_log(paths, f"Skipped: {command_text}")
        return

    subprocess.run(command, cwd=REPO_ROOT, check=True)
    _append_log(paths, f"Success: {command_text}")


def _slice_commands(config: dict, comparison_set_names: list[str], overwrite: bool) -> list[list[str]]:
    comparison_sets = config["cases"]["comparison_sets"]
    commands: list[list[str]] = []
    seen: set[tuple[str, int]] = set()
    for name in comparison_set_names:
        case_indices = comparison_sets[name]["case_indices"]
        for case, index in case_indices.items():
            key = (str(case), int(index))
            if key in seen:
                continue
            seen.add(key)
            command = [
                sys.executable,
                "scripts/02_extract_midspan_slice.py",
                "--case",
                key[0],
                "--index",
                str(key[1]),
            ]
            if overwrite:
                command.append("--overwrite")
            commands.append(command)
    return commands


def _comparison_command(comparison_set: str, field: str, overwrite: bool) -> list[str]:
    command = [
        sys.executable,
        "scripts/03_compare_poly_orders.py",
        "--comparison-set",
        comparison_set,
        "--field",
        field,
    ]
    if overwrite:
        command.append("--overwrite")
    return command


def _plot_command(comparison_set: str, field: str) -> list[str]:
    return [
        sys.executable,
        "scripts/04_plot_summary.py",
        "--comparison-set",
        comparison_set,
        "--field",
        field,
    ]


def main() -> None:
    """Run slice extraction, comparison, and plotting across comparison sets."""
    args = _parse_args()
    config = load_project_config(
        REPO_ROOT / "config" / "paths.yaml",
        REPO_ROOT / "config" / "cases.yaml",
    )
    paths = ProjectPaths.from_mapping(config["paths"])

    start_time = datetime.now().isoformat(timespec="seconds")
    try:
        comparison_sets = _comparison_set_names(config, args.comparison_sets)
        fields = _fields(args.fields)
        strict_note = "strict" if args.strict else "strict-default"

        header_lines = [
            "Multi-time pipeline start",
            f"start_time: {start_time}",
            f"comparison_sets: {','.join(comparison_sets)}",
            f"fields: {','.join(fields)}",
            f"overwrite: {args.overwrite}",
            f"dry_run: {args.dry_run}",
            f"skip_slices: {args.skip_slices}",
            f"skip_plots: {args.skip_plots}",
            f"failure_mode: {strict_note}",
        ]
        header = "\n".join(header_lines)
        print(header)
        _append_log(paths, header)

        if not args.skip_slices:
            for command in _slice_commands(config, comparison_sets, args.overwrite):
                _run_command(command, paths, dry_run=args.dry_run)

        for comparison_set in comparison_sets:
            for field in fields:
                _run_command(
                    _comparison_command(comparison_set, field, args.overwrite),
                    paths,
                    dry_run=args.dry_run,
                )
                if not args.skip_plots:
                    _run_command(_plot_command(comparison_set, field), paths, dry_run=args.dry_run)

        end_time = datetime.now().isoformat(timespec="seconds")
        summary = "\n".join(
            [
                "Multi-time pipeline completed.",
                f"start_time: {start_time}",
                f"end_time: {end_time}",
                f"log_file: {_log_path(paths)}",
            ]
        )
        print(summary)
        _append_log(paths, summary)
    except subprocess.CalledProcessError as exc:
        end_time = datetime.now().isoformat(timespec="seconds")
        message = "\n".join(
            [
                f"ERROR: Command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}",
                f"start_time: {start_time}",
                f"end_time: {end_time}",
            ]
        )
        print(message, file=sys.stderr)
        _append_log(paths, message)
        raise SystemExit(exc.returncode) from exc
    except Exception as exc:
        end_time = datetime.now().isoformat(timespec="seconds")
        message = "\n".join([f"ERROR: {exc}", f"start_time: {start_time}", f"end_time: {end_time}"])
        print(message, file=sys.stderr)
        try:
            _append_log(paths, message)
        except OSError as log_exc:
            print(f"ERROR: Failed to write log file: {log_exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
