"""Sequential, resumable native/P7 3D vortex VTUs and one physical-time PVD."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from nek_post.vortex_3d_series import run_vortex_series


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--case', default='N7_H')
    parser.add_argument('--indices', help='Unique comma-separated indices/ranges; default 1-81.')
    parser.add_argument('--start-index', type=int)
    parser.add_argument('--end-index', type=int)
    parser.add_argument('--max-frames', type=int, help='Cap the sorted selection after validating all supplied indices.')
    parser.add_argument('--sampling-nodes', type=int, help='P7 target GLL nodes/axis, >=8; omit for native.')
    parser.add_argument('--output-dir', type=Path, help='External parent root; default vortex_3d_series.')
    region = parser.add_mutually_exclusive_group()
    region.add_argument('--positive-x', action='store_true',
                        help='Complete 3D [0,17] x [0,1.5] x [0,1] volume; default for N7_H. Requires element boundaries.')
    region.add_argument('--full-domain', action='store_true', help='Explicitly export the full physical domain, including x<0.')
    region.add_argument('--crop', type=float, nargs=6, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX', 'ZMIN', 'ZMAX'))
    parser.add_argument('--periodic-axes', nargs='*', choices=('x', 'y', 'z'), default=['x', 'y'])
    parser.add_argument('--expected-element-counts', type=int, nargs=3, default=[272, 12, 8], metavar=('NX', 'NY', 'NZ'))
    parser.add_argument('--chunk-size', type=int, default=256)
    parser.add_argument('--dry-run', action='store_true', help='Headers/maps, checksums and coordinate-only crop planning; no velocity/VTU decoding, gradients or outputs.')
    parser.add_argument('--resume', action='store_true', help='Reuse only fully validated completed frames.')
    parser.add_argument('--recover-staging', action='store_true', help='With resume: validate/recover complete staging, preserve incomplete staging and recompute.')
    return parser.parse_args(argv)


def main(argv=None):
    try:
        report = run_vortex_series(**vars(parse_args(argv)),
                                  progress=lambda value: print(value, file=sys.stderr, flush=True))
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if report['status'] != 'invalid' else 1
    except (OSError, ValueError, RuntimeError, MemoryError, KeyError, TypeError) as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
