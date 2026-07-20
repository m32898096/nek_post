"""Check that the expected Nek5000 input files exist.

This script only checks paths. It does not read Nek5000 binary contents.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from nek_post.config import load_project_config  # noqa: E402
from nek_post.paths import ProjectPaths  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check that the configured Nek5000 input files exist.")
    return parser.parse_args()


def _format_row(case: str, order: int | str, index: int | str, file_path: Path | str, status: str) -> str:
    """Format one fixed-width table row."""
    return f"{case:<6} {str(order):<6} {str(index):<7} {str(file_path):<64} {status}"


def _build_report(paths: ProjectPaths, cases: dict) -> tuple[str, int]:
    """Build the file existence report and return the matching process exit code."""
    orders = cases["orders"]
    file_prefix = cases["file_prefix"]
    file_indices = cases["file_indices"]

    lines = [
        "Nek5000 expected file check",
        "",
        _format_row("Case", "Order", "Index", "File path", "Status"),
        _format_row("------", "------", "-------", "---------", "------"),
    ]

    total_count = 0
    existing_count = 0
    missing_count = 0

    for case, case_dir in paths.case_dirs.items():
        order = orders.get(case, "UNKNOWN")
        for idx in file_indices:
            total_count += 1
            file_path = Path(case_dir) / f"{file_prefix}.f{idx:05d}"
            exists = file_path.exists()
            status = "OK" if exists else "MISSING"
            existing_count += int(exists)
            missing_count += int(not exists)
            lines.append(_format_row(case, order, idx, file_path, status))

    lines.extend(
        [
            "",
            "Summary",
            f"Total expected files: {total_count}",
            f"Existing files: {existing_count}",
            f"Missing files: {missing_count}",
        ]
    )

    return "\n".join(lines), 0 if missing_count == 0 else 1


def main() -> None:
    """Load configuration, check expected files, print and log the report."""
    _parse_args()
    config = load_project_config(
        REPO_ROOT / "config" / "paths.yaml",
        REPO_ROOT / "config" / "cases.yaml",
    )
    paths = ProjectPaths.from_mapping(config["paths"])

    report, exit_code = _build_report(paths, config["cases"])
    print(report)

    log_dir = paths.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "check_files.log"
    log_path.write_text(report + "\n", encoding="utf-8")

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
