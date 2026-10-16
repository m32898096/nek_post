"""Validate saved omega_x production and create a ParaView time-series PVD."""
import argparse
import json
from pathlib import Path
import sys

from nek_post.omega_x_series_pvd import build_omega_x_series_pvd


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='N7_H', help='Source case name or snapshot directory.')
    parser.add_argument('--input-dir', type=Path, help='Existing output tree; searches frame directories recursively.')
    parser.add_argument('--output-dir', type=Path, help='Exact directory for the series PVD; defaults to input directory.')
    parser.add_argument('--dry-run', action='store_true', help='Fully validate without creating any files.')
    return parser.parse_args(argv)


def main(argv=None):
    try:
        report = build_omega_x_series_pvd(**vars(parse_args(argv)),
            progress=lambda message:print(message, file=sys.stderr, flush=True))
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if report['status'] in ('validated', 'completed') else 1
    except (ValueError, OSError, RuntimeError, MemoryError) as exc:
        print(json.dumps({'status':'invalid','errors':[f'{type(exc).__name__}: {exc}']}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
