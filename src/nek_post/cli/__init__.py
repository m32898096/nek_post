"""Lightweight command-line entrypoint for Nek5000 post-processing."""

from collections.abc import Sequence
import sys


UNAVAILABLE_EXIT_STATUS = 3


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Help exits 0, parser errors 2, unavailable commands 3."""
    from .parser import build_parser

    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("paths_config", "cases_config", "study_config", "output_root", "input_root",
                 "case_dir", "output_dir", "publication_root", "preview_dir", "plan", "dry_run"):
        if not hasattr(args, name):
            setattr(args, name, None)
    if args.plan and args.dry_run:
        parser.error("--plan and --dry-run cannot be combined")
    from pathlib import Path
    from .configuration import Refusal
    from .planning import prepare_plan
    try:
        plan = prepare_plan(args, cwd=Path.cwd())
    except Refusal as error:
        print(f"nek-post: command '{args.command.command_id}' is registered but not implemented "
              f"(status: not_implemented). {error.code}: {error}", file=sys.stderr)
        return UNAVAILABLE_EXIT_STATUS
    if args.plan:
        from dataclasses import asdict
        import json
        print(json.dumps(asdict(plan), indent=2))
        if plan.status == "planned":
            return 0

    print(
        f"nek-post: command '{args.command.command_id}' is registered but not implemented "
        "(status: not_implemented). Execution is unavailable in Phase 4B-2.",
        file=sys.stderr,
    )
    return UNAVAILABLE_EXIT_STATUS
