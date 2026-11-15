"""Build root, group and leaf help using only standard-library metadata."""

import argparse

from . import UNAVAILABLE_EXIT_STATUS
from .registry import COMMANDS, GROUP_HELP


def build_parser() -> argparse.ArgumentParser:
    """Build the accepted hierarchy with a required command at every group."""
    epilog = (
        "Phase 4B-1 CLI foundation. All registered commands are not implemented. "
        f"Execution exits with status {UNAVAILABLE_EXIT_STATUS}."
    )
    parser = argparse.ArgumentParser(
        prog="nek-post",
        description="Nek5000 post-processing commands",
        epilog=epilog,
        allow_abbrev=False,
    )
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
        leaf_parser.set_defaults(command=command)
    return parser
