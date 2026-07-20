"""Inspect a single Nek5000 field file and report basic metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

from nek_post.config import load_project_config
from nek_post.fields import summarize_element_fields
from nek_post.io_nek import describe_nek_data, get_first_element, read_nek_file
from nek_post.paths import ProjectPaths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe one Nek5000 field file.")
    parser.add_argument("--case", help="Case name from config/cases.yaml, for example N11.")
    parser.add_argument("--index", type=int, help="File index, for example 80 for GC0.f00080.")
    return parser.parse_args()


def _target_file(paths: ProjectPaths, cases: dict, case: str, index: int) -> Path:
    file_prefix = cases["file_prefix"]
    return paths.case_dir(case) / f"{file_prefix}.f{index:05d}"


def _format_mapping(mapping: dict, indent: str = "  ") -> list[str]:
    return [f"{indent}{key}: {value}" for key, value in mapping.items()]


def _build_report(target_path: Path, data) -> str:
    data_summary = describe_nek_data(data)
    element = get_first_element(data)
    field_summary = summarize_element_fields(element)

    lines = [
        "Nek5000 single-file probe",
        "",
        f"Target file path: {target_path}",
        f"File exists: {target_path.exists()}",
        f"Nek5000 time: {data_summary['time']}",
        f"Number of elements: {data_summary['element_count']}",
        f"First element type: {data_summary['first_element_type']}",
        f"Available attributes of first element: {data_summary['first_element_attributes']}",
        "",
        "Field summary:",
        *_format_mapping(field_summary),
        "",
        f"Inferred concentration source: {field_summary['concentration_source']}",
    ]
    return "\n".join(lines)


def _write_log(paths: ProjectPaths, text: str) -> None:
    log_dir = paths.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "probe_nek_file.log"
    log_path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    """Load configuration, read one Nek5000 file, and print a concise report."""
    args = _parse_args()
    config = load_project_config(
        REPO_ROOT / "config" / "paths.yaml",
        REPO_ROOT / "config" / "cases.yaml",
    )
    paths = ProjectPaths.from_mapping(config["paths"])

    case = args.case or config["cases"]["reference_case"]
    file_indices = config["cases"]["file_indices"]
    index = args.index if args.index is not None else file_indices[-1]

    try:
        target_path = _target_file(paths, config["cases"], case, index)
        data = read_nek_file(target_path)
        report = _build_report(target_path, data)
        print(report)
        _write_log(paths, report)
    except (FileNotFoundError, ValueError) as exc:
        message = f"ERROR: {exc}"
        print(message, file=sys.stderr)
        try:
            _write_log(paths, message)
        except OSError as log_exc:
            print(f"ERROR: Failed to write log file: {log_exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except Exception as exc:
        message = f"ERROR: Failed to read Nek5000 file: {exc}"
        print(message, file=sys.stderr)
        try:
            _write_log(paths, message)
        except OSError as log_exc:
            print(f"ERROR: Failed to write log file: {log_exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
