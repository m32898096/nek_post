"""Export one 3D frame with omega_x and lambda_ci to binary VTU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from nek_post.vortex_3d_workflow import run_single_frame_vortex


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--case', default='N7_H', help='Case alias or external case directory.')
    parser.add_argument('--index', type=int, default=79, help='Exactly one GC0.fNNNNN index.')
    parser.add_argument('--output-dir', type=Path, help='External root; writes CASE/fNNNNN/GC0_fNNNNN.{vtu,json}.')
    parser.add_argument('--sampling-nodes', type=int, metavar='N',
                        help='P7 source only: >=8 target GLL nodes/axis (10 or 12 for denser sampling). '
                             'Omit for unchanged native output. Denser output uses ROOT/gllN/CASE/fNNNNN/.')
    parser.add_argument('--crop', type=float, nargs=6, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX', 'ZMIN', 'ZMAX'),
                        help='Keep only output subcells wholly within these physical bounds; no crop-edge interpolation.')
    parser.add_argument('--periodic-axes', nargs='*', choices=('x', 'y', 'z'), default=['x', 'y'],
                        help='Explicit confirmed translational periodicity; empty list means none.')
    parser.add_argument('--expected-element-counts', type=int, nargs=3, default=[272, 12, 8],
                        metavar=('NX', 'NY', 'NZ'), help='Complete physical element partition to validate.')
    parser.add_argument('--chunk-size', type=int, default=256)
    parser.add_argument('--dry-run', action='store_true', help='Header/map/resource inspection only; no files created.')
    return parser.parse_args(argv)


def main(argv=None):
    try:
        report = run_single_frame_vortex(**vars(parse_args(argv)),
                                        progress=lambda message: print(message, file=sys.stderr, flush=True))
        print(json.dumps(report, indent=2, allow_nan=False))
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == '__main__':
    main()
