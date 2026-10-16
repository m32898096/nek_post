"""Compute only native-volume streamwise vorticity, with scalar L2 recovery."""
import argparse
import json
from pathlib import Path
import sys

from nek_post.omega_x_workflow import run_omega_x_series


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',default='N7_H',help='N7_H, case_N7_H, or a case directory path.')
    parser.add_argument('--indices',help='Comma-separated indices/ranges. Default selection is frame 79 only.')
    parser.add_argument('--start-index',type=int)
    parser.add_argument('--end-index',type=int)
    parser.add_argument('--output-dir',type=Path,help='Output root; writes CASE/fNNNNN underneath it.')
    parser.add_argument('--chunk-size',type=int,default=256)
    parser.add_argument('--dry-run',action='store_true')
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument('--resume',action='store_true')
    existing.add_argument('--overwrite',action='store_true',help='Preserve a recognized old omega_x frame in a unique backup, then replace it.')
    return parser.parse_args(argv)


def main(argv=None):
    try:
        report = run_omega_x_series(**vars(parse_args(argv)),progress=lambda m:print(m,file=sys.stderr,flush=True))
        print(json.dumps(report,indent=2,allow_nan=False))
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        raise SystemExit(1) from exc


if __name__ == '__main__':main()
