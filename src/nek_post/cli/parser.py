"""Build root, group and leaf help using only standard-library metadata."""

import argparse

from . import UNAVAILABLE_EXIT_STATUS
from .registry import COMMANDS, GROUP_HELP


def build_parser() -> argparse.ArgumentParser:
    """Build the accepted hierarchy with a required command at every group."""
    epilog = (
        "Phase 4B-2 CLI foundation. All registered commands are not implemented. "
        f"Execution exits with status {UNAVAILABLE_EXIT_STATUS}."
    )
    parser = argparse.ArgumentParser(
        prog="nek-post",
        description="Nek5000 post-processing commands",
        epilog=epilog,
        allow_abbrev=False,
    )
    add_options(parser)
    subparsers = {
        (): parser.add_subparsers(title="commands", metavar="COMMAND", required=True),
    }
    for command in COMMANDS:
        for depth in range(1, len(command.group) + 1):
            group = command.group[:depth]
            if group not in subparsers:
                group_parser = subparsers[group[:-1]].add_parser(
                    group[-1],
                    help=GROUP_HELP[group],
                    description=GROUP_HELP[group],
                    epilog=epilog,
                    allow_abbrev=False,
                )
                add_options(group_parser)
                subparsers[group] = group_parser.add_subparsers(
                    title="commands", metavar="COMMAND", required=True,
                )
        leaf_parser = subparsers[command.group].add_parser(
            command.leaf,
            help=command.help,
            description=f"{command.help}. Status: {command.status}.",
            epilog=epilog,
            allow_abbrev=False,
        )
        add_options(leaf_parser)
        leaf_parser.set_defaults(command=command)
    return parser


def add_options(parser):
    """Accept foundation options at root, group or leaf without default clobbering."""
    for flag in ("paths-config", "cases-config", "study-config", "output-root",
                 "input-root", "case-dir", "output-dir", "publication-root", "preview-dir"):
        parser.add_argument("--" + flag, default=argparse.SUPPRESS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", default=argparse.SUPPRESS)
    mode.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS)
