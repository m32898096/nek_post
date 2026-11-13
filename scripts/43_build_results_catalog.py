"""Build a bounded results catalog in a new directory outside the results tree."""
import argparse
import json
from pathlib import Path
import sys

from nek_post.results_catalog import build_catalog, write_catalog


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_root', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--artifacts', action='store_true', help='Include individual file inventory')
    args = parser.parse_args(argv)
    try:
        catalog = build_catalog(args.results_root, include_artifacts=args.artifacts)
        write_catalog(catalog, args.output_dir, results_root=args.results_root)
    except (OSError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    print(json.dumps({key: catalog[key] for key in ('schema_version', 'file_count', 'logical_bytes', 'warnings')}, indent=2))
    print(f"Datasets: {len(catalog['datasets'])}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
