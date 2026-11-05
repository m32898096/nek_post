"""Reconstruct one saved omega_x midplane with P7 and compare native linear VTR."""
import argparse
import json
from pathlib import Path
from nek_post.omega_x_p7_slice import run_omega_x_p7_slice


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',default='N7_H')
    parser.add_argument('--frame',type=int,default=79)
    parser.add_argument('--input-dir',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--y',type=float,default=.75)
    parser.add_argument('--samples-per-element',type=int,default=29)
    try:
        report=run_omega_x_p7_slice(**vars(parser.parse_args(argv)))
        print(json.dumps(report,indent=2,allow_nan=False))
        return 0
    except (ValueError,OSError,RuntimeError,MemoryError) as exc:
        print(json.dumps({'status':'invalid','error':f'{type(exc).__name__}: {exc}'}))
        return 1


if __name__=='__main__':raise SystemExit(main())
