"""Sequential high-order lambda_ci slices with validated resume and mesh caches."""
import argparse
import json
from pathlib import Path
import sys

from nek_post.lambda_ci_series import run_slice_series


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir',type=Path,default=Path('/data/Nek5000_data/case_N7_H'))
    parser.add_argument('--output-root',type=Path,help='Case-level result directory, not the parent lambda_ci root.')
    parser.add_argument('--start-index',type=int,default=1)
    parser.add_argument('--end-index',type=int,default=81)
    parser.add_argument('--indices',help='Comma-separated indices/ranges, e.g. 17,33,49 or 1-81.')
    parser.add_argument('--max-frames',type=int)
    parser.add_argument('--y-midplane',type=float,default=.75)
    parser.add_argument('--x-slice',type=float,default=15.)
    parser.add_argument('--x-min',type=float,default=0.)
    parser.add_argument('--subdivisions-per-element',type=int,default=28)
    parser.add_argument('--vmin',type=float,default=0.)
    parser.add_argument('--vmax',type=float,default=10.)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--recover-staging',action='store_true',help='Validate/publish complete orphan staging; preserve incomplete staging and compute into a new directory.')
    parser.add_argument('--save-samples',action='store_true')
    parser.add_argument('--chunk-size',type=int,default=256)
    parser.add_argument('--eigenvalue-chunk-rows',type=int,default=16)
    parser.add_argument('--log-name',help='Unique log/summary filename stem.')
    return parser.parse_args(argv)


def main(argv=None):
    try:
        report=run_slice_series(**vars(parse_args(argv)),progress=lambda message:print(message,file=sys.stderr,flush=True))
        print(json.dumps(report,indent=2,allow_nan=False))
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        raise SystemExit(1) from exc


if __name__=='__main__':main()
