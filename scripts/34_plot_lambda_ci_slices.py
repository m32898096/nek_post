"""Render two dense high-order lambda_ci slices directly in Python."""
import argparse
import json
from pathlib import Path
import sys

from nek_post.lambda_ci_plotting import run_lambda_ci_slice_plots


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-dir',type=Path,default=Path('/data/Nek5000_data/case_N7_H'))
    p.add_argument('--index',type=int,default=79)
    p.add_argument('--y-midplane',type=float,default=.75)
    p.add_argument('--x-slice',type=float,default=15.)
    p.add_argument('--x-min',type=float,default=0.)
    density=p.add_mutually_exclusive_group()
    density.add_argument('--samples-per-element',type=int,default=None,help='Legacy point-count option; N points = N-1 subdivisions.')
    density.add_argument('--subdivisions-per-element',type=int,default=28,help='Intervals per element edge; default 28 gives 29 points.')
    p.add_argument('--no-paraview',dest='export_paraview',action='store_false',help='Retain the original Python-only output naming.')
    p.set_defaults(export_paraview=True)
    p.add_argument('--vmin',type=float,default=0.)
    p.add_argument('--vmax',type=float,default=10.)
    p.add_argument('--output-dir',type=Path,help='Final plots directory; default ROOT/CASE/fNNNNN/plots.')
    p.add_argument('--chunk-size',type=int,default=256)
    p.add_argument('--eigenvalue-chunk-rows',type=int,default=16)
    p.add_argument('--save-samples',action='store_true',help='Save high-order sampled arrays as NPZ for later figure reuse.')
    p.add_argument('--dry-run',action='store_true')
    return p.parse_args(argv)


def main(argv=None):
    try:
        result=run_lambda_ci_slice_plots(**vars(parse_args(argv)),progress=lambda text:print(text,file=sys.stderr,flush=True))
        print(json.dumps(result,indent=2,allow_nan=False))
    except Exception as exc:
        print(f'ERROR: {exc}',file=sys.stderr);raise SystemExit(1) from exc


if __name__=='__main__':main()
