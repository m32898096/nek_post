"""Reconstruct saved projected omega_x frames; full PVD only after 81 validations."""
import argparse
import json
from pathlib import Path
import sys

from nek_post.omega_x_p7_series import run_omega_x_p7_series


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='N7_H')
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, help='External parent root; case directory is appended.')
    parser.add_argument('--indices', help='Unique comma-separated indices/ranges, e.g. 1,41,79 or 1-81.')
    parser.add_argument('--start-index', type=int)
    parser.add_argument('--end-index', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--recover-staging', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    try:
        report = run_omega_x_p7_series(**vars(parse_args(argv)), progress=lambda message: print(message, file=sys.stderr, flush=True))
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError, MemoryError, KeyError, TypeError) as exc:
        print(json.dumps({'status':'invalid', 'error':f'{type(exc).__name__}: {exc}'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
