"""Rewrite one existing VTU for manual Tecplot/ParaView serialization tests."""
import argparse
import json
from pathlib import Path
import sys

from nek_post.vtu_compat import MAX_SUBSET_CELLS, PROFILES, rewrite_vtu


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('source',type=Path,help='Existing external VTU; no Nek calculations.')
    parser.add_argument('--output-dir',type=Path,required=True,help='New external directory, published without replacement.')
    parser.add_argument('--profile',choices=PROFILES,default='reference')
    scope=parser.add_mutually_exclusive_group()
    scope.add_argument('--max-cells',type=int,default=4096,
                       help=f'Bounded whole-cell subset, 1..{MAX_SUBSET_CELLS}; resource preflight required.')
    scope.add_argument('--full-grid',action='store_true',help='Explicitly retain ALL source cells/points; may require substantial RAM.')
    parser.add_argument('--seed-cell',type=int,help='Original cell index; default finds a cell incident to maximum lambda_ci.')
    parser.add_argument('--appended-encoding',choices=('raw','base64'),help='Appended profiles default raw; reference is inline base64.')
    parser.add_argument('--header-bits',type=int,choices=(32,64),help='Default UInt32; UInt64 available for appended-zlib/none.')
    parser.add_argument('--add-time-alias',action='store_true',help='Diagnostic only: copy TIME_VALUE to FieldData time; never replace a conflict.')
    parser.add_argument('--dry-run',action='store_true',help='Stream XML headers and estimate resources only; no payload decoding or files.')
    return parser.parse_args(argv)


def main(argv=None):
    try:
        report=rewrite_vtu(**vars(parse_args(argv)))
        print(json.dumps(report,indent=2,allow_nan=False))
        return 0
    except (OSError,ValueError,RuntimeError,MemoryError) as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
