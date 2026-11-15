"""Lightweight command-line entrypoint for Nek5000 post-processing."""

from collections.abc import Sequence
import sys


UNAVAILABLE_EXIT_STATUS = 3


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Help exits 0, parser errors 2, unavailable commands 3."""
    from .parser import build_parser

    args = build_parser().parse_args(argv)
    print(
        f"nek-post: command '{args.command.command_id}' is registered but not implemented "
        "(status: not_implemented). Execution is unavailable in Phase 4B-1.",
        file=sys.stderr,
    )
    return UNAVAILABLE_EXIT_STATUS
