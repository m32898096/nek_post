"""Reviewed navigation snapshots and no-replace, root-only publication.

Scientific payloads are never opened. Publication compares the preview's full
catalog (including file sizes and mtimes) with a fresh read-only scan.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from nek_post.results_catalog import (
    CATALOG_FILES, DATASET_COLUMNS, SCIENTIFIC_ROOT, _csv, _inside,
    build_catalog, write_catalog,
)


def validate_navigation(readme, results_root):
    """Validate simple inline Markdown links relative to the publication root.

    The curated guide uses local inline links without titles or fragments.
    results_index.csv may be absent during preview. Other targets must exist
    inside the root without symlink aliases. Returns the checked link paths.
    """
    root = Path(results_root).resolve(strict=True)
    links = re.findall(r'!?\[[^\]\n]*\]\(([^)\n]+)\)', readme)
    if not links:
        raise ValueError('README must contain local inline navigation links')
    for name in links:
        url = urlsplit(name)
        if url.scheme or url.netloc or url.query or url.fragment:
            raise ValueError(f'unsupported navigation link: {name!r}')
        relative = Path(unquote(url.path))
        if not url.path or relative.is_absolute():
            raise ValueError(f'navigation link must be root-relative: {name!r}')
        lexical = Path(os.path.abspath(root / relative))
        target = lexical.resolve()
        if not _inside(target, root) or target != lexical:
            raise ValueError(f'unsafe navigation link: {name!r}')
        if relative.as_posix() == 'results_index.csv':
            continue
        if not (target.is_file() or target.is_dir()):
            raise ValueError(f'missing navigation target: {name!r}')
    return links


def _outside_preview(preview_dir, results_root):
    preview = Path(preview_dir).resolve()
    for root in (Path(results_root).resolve(), SCIENTIFIC_ROOT.resolve()):
        if _inside(preview, root):
            raise ValueError('preview must be outside scientific results roots')
    return preview


def prepare_navigation(results_root, readme_source, preview_dir):
    """Create a fresh external Phase 2A catalog plus a reviewed-guide candidate."""
    preview = _outside_preview(preview_dir, results_root)
    readme = Path(readme_source).read_text(encoding='utf-8')
    validate_navigation(readme, results_root)
    catalog = build_catalog(results_root, include_artifacts=True)
    write_catalog(catalog, preview, results_root=results_root)
    with (preview / 'README.md').open('x', encoding='utf-8', newline='') as stream:
        stream.write(readme)
    return catalog


def publish_navigation(results_root, preview_dir):
    """Create exactly two root files from an unchanged, reviewed preview.

    Both targets are checked before writing, then opened exclusively. A failure
    during either write may leave partial catalog files for inspection. The
    pair is not an atomic transaction; existing files are never replaced or
    removed. A retry must stop on any collision, including dangling symlinks.
    """
    preview = _outside_preview(preview_dir, results_root)
    root = Path(results_root).resolve(strict=True)
    protected = SCIENTIFIC_ROOT.resolve()
    if root != protected and _inside(root, protected):
        raise ValueError('publication inside a nested production directory is forbidden')
    # build_catalog also rejects missing/non-directory and direct symlink roots.
    catalog = build_catalog(results_root, include_artifacts=True)
    for name in sorted(CATALOG_FILES):
        if os.path.lexists(root / name):
            raise FileExistsError(f'catalog target already exists: {root / name}')
    expected = json.loads((preview / 'catalog.json').read_text(encoding='utf-8'))
    if catalog != expected:
        raise ValueError('results inventory or metadata changed since preview; create and review a fresh preview')
    csv_bytes = (preview / 'datasets.csv').read_bytes()
    if csv_bytes != _csv(catalog['datasets'], DATASET_COLUMNS).encode('utf-8'):
        raise ValueError('preview CSV differs from the fresh catalog')
    readme_bytes = (preview / 'README.md').read_bytes()
    validate_navigation(readme_bytes.decode('utf-8'), root)
    for name, payload in (('README.md', readme_bytes), ('results_index.csv', csv_bytes)):
        with (root / name).open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    return catalog
