"""Preview results navigation externally, or publish its two new root files."""
import argparse
from pathlib import Path
import sys

from nek_post.results_catalog_publication import prepare_navigation, publish_navigation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_root', type=Path)
    parser.add_argument('--preview-dir', type=Path, required=True,
                        help='Fresh external directory for preview; reviewed directory for publication')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--readme-source', type=Path, help='Create preview using this UTF-8 guide')
    mode.add_argument('--publish', action='store_true', help='Publish reviewed preview; refuse existing targets')
    args = parser.parse_args(argv)
    try:
        if args.publish:
            catalog = publish_navigation(args.results_root, args.preview_dir)
        else:
            catalog = prepare_navigation(args.results_root, args.readme_source, args.preview_dir)
    except (OSError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    print(f"{'Published' if args.publish else 'Previewed'}: {len(catalog['datasets'])} datasets; "
          f"schema {catalog['schema_version']}; {len(catalog['warnings'])} diagnostics")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
