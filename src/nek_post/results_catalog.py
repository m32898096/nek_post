"""Bounded, read-only results discovery and versioned catalog serialization."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

SCHEMA_VERSION = 1
MAX_CONTROL_BYTES = 1_048_576
MAX_JSON_PER_DATASET = 64
SCIENTIFIC_ROOT = Path('/data/Nek5000_data/results')
DATASET_COLUMNS = (
    'dataset_id', 'study', 'analysis', 'case_id', 'variant', 'artifact_class',
    'status', 'field_names', 'polynomial_degree', 'sampling_method',
    'sampling_nodes', 'projection_method', 'frame_count', 'actual_time_start',
    'actual_time_end', 'relative_path', 'entrypoint_file', 'manifest_path',
    'producer_script', 'code_revision', 'validation_level', 'file_count',
    'logical_bytes', 'metadata_sources', 'notes',
)
ARTIFACT_COLUMNS = ('dataset_id', 'relative_path', 'file_type', 'logical_bytes',
                    'mtime_ns', 'role', 'related_controls')
SERIES_ANALYSES = {'lambda_ci', 'omega_x', 'omega_x_highorder_slices',
                   'vortex_3d_series', 'spectral_volume'}


def _walk_error(error):
    raise error


def _inside(path, root):
    return path == root or root in path.parents


def _bounded_read(path):
    with path.open('rb') as stream:
        data = stream.read(MAX_CONTROL_BYTES + 1)
    if len(data) > MAX_CONTROL_BYTES:
        raise ValueError('control exceeds 1 MiB inspection limit')
    return data


def _classification(parts):
    label = '/'.join(parts).lower()
    if any(token in label for token in ('smoke', 'fix_test')):
        return 'smoke_test'
    if any(token in label for token in ('diagnostic', '/logs', '.staging')):
        return 'diagnostic'
    if any(token in label for token in ('archive', 'historical')):
        return 'historical'
    return 'unclassified'


def _role(path):
    if path.suffix.lower() == '.pvd':
        return 'collection'
    if path.suffix.lower() == '.json':
        return 'metadata'
    if path.suffix.lower() in {'.png', '.pdf', '.svg'}:
        return 'figure'
    if path.suffix.lower() == '.csv':
        return 'table'
    if path.suffix.lower() in {'.vtu', '.vtr', '.vti', '.npz', '.npy'}:
        return 'scientific_artifact'
    return 'other'


def _pvd(path, root, known_files):
    """Resolve references relative to the PVD, checking inventory membership."""
    data = _bounded_read(path)
    if b'<!DOCTYPE' in data.upper() or b'<!ENTITY' in data.upper():
        raise ValueError('XML declarations/entities are unsupported')
    tree = ET.fromstring(data)
    if tree.tag != 'VTKFile' or tree.get('type') != 'Collection':
        raise ValueError('expected VTK Collection')
    records = tree.findall('./Collection/DataSet')
    if not records:
        raise ValueError('empty PVD collection')
    records_with_times = []
    for record in records:
        name = record.get('file', '')
        if not name or Path(name).is_absolute():
            raise ValueError('PVD reference must be a relative path')
        target = (path.parent / name).resolve()
        # known_files excludes symlinks, including symlink directory aliases.
        lexical = Path(os.path.abspath(path.parent / name))
        if not _inside(target, root) or target != lexical or target not in known_files:
            raise ValueError(f'unsafe or missing PVD reference {name!r}')
        time = float(record.get('timestep', 'nan'))
        if not math.isfinite(time):
            raise ValueError('missing or nonfinite PVD timestep')
        records_with_times.append((target, time))
    return records_with_times


def _output_reference_issues(document, control, root, known_files):
    """Check explicit file-reference fields relative to their control file.

    output_paths is a scalar or a flat role-to-path mapping. outputs is a
    list, and output_file is a scalar. Null entries denote absent optional
    outputs. Other fields retain their original, potentially non-path meaning.
    """
    issues = []
    references = []
    for field in ('output_paths', 'outputs', 'output_file'):
        if field not in document or document[field] is None:
            continue
        value = document[field]
        if field == 'output_paths' and isinstance(value, dict):
            entries = [(f'{field}.{key}', entry) for key, entry in value.items()]
        elif field == 'outputs' and isinstance(value, list):
            entries = [(f'{field}[{index}]', entry) for index, entry in enumerate(value)]
        elif field in {'output_paths', 'output_file'} and isinstance(value, str):
            entries = [(field, value)]
        else:
            issues.append(f'{field}: unsupported output reference schema')
            continue
        for label, reference in entries:
            if reference is None:
                continue
            if not isinstance(reference, str) or not reference:
                issues.append(f'{label}: unsupported output reference schema')
            else:
                references.append((label, reference))
    for label, reference in references:
        try:
            declared = Path(reference)
            lexical = Path(os.path.abspath(control.parent / declared))
            resolved = lexical.resolve()
        except ValueError as exc:
            issues.append(f'{label}: malformed declared output {reference!r}: {exc}')
            continue
        except (OSError, RuntimeError):
            issues.append(f'{label}: unsafe or unresolvable declared output {reference!r}')
            continue
        if not _inside(resolved, root) or resolved != lexical:
            issues.append(f'{label}: unsafe declared output {reference!r}')
        elif resolved not in known_files:
            issues.append(f'{label}: missing declared output {reference!r}')
    return issues


def _metadata_values(document):
    """Read only explicitly named scientific metadata, with no filename parsing."""
    config = document.get('configuration', {})
    reconstruction = document.get('reconstruction', {})
    selection = document.get('output_selection', {})
    if not isinstance(config, dict):
        config = {}
    if not isinstance(reconstruction, dict):
        reconstruction = {}
    if not isinstance(selection, dict):
        selection = {}
    values = {
        'field_names': selection.get('fields', document.get('field_names',
                         [document['field']] if isinstance(document.get('field'), str) else None)),
        'polynomial_degree': reconstruction.get('source_polynomial_degree', document.get('polynomial_degree')),
        'sampling_nodes': config.get('sampling_nodes', document.get('sampling_nodes')),
        'sampling_method': reconstruction.get('interpolation', document.get('sampling_method')),
        'projection_method': document.get('projection_method'),
        'producer_script': document.get('producer_script'),
        'code_revision': document.get('code_revision'),
    }
    for key, value in values.items():
        if value is None:
            continue
        if key in {'polynomial_degree', 'sampling_nodes'}:
            valid = type(value) is int and value >= 0
        elif key == 'field_names':
            valid = isinstance(value, list) and all(isinstance(v, str) for v in value)
        else:
            valid = isinstance(value, str)
        if not valid:
            raise ValueError(f'invalid metadata type for {key}')
    return values


def build_catalog(results_root, *, include_artifacts=False):
    """Discover bundles without following symlinks or opening scientific payloads.

    Series control roots take precedence over leaf-directory grouping. Root,
    study, and analysis directory names are navigation labels. All scientific
    attributes require explicit metadata. Every indexed file has one owner.
    """
    supplied = Path(results_root).absolute()
    if supplied.is_symlink():
        raise ValueError('results root must not be a symlink')
    root = supplied.resolve(strict=True)
    if not root.is_dir():
        raise ValueError('results root must be a directory')
    files, warnings = [], []
    for directory, dirs, names in os.walk(root, followlinks=False, onerror=_walk_error):
        for name in sorted(dirs + names):
            path = Path(directory) / name
            if path.is_symlink():
                warnings.append(f'skipped symlink {path.relative_to(root)}')
                if name in dirs:
                    dirs.remove(name)
            elif name in names and path.is_file():
                files.append(path)
        dirs.sort()
    files.sort()
    known = set(files)
    anchors = set()
    for path in files:
        parts = path.relative_to(root).parts
        if path.name == 'manifest.json':
            anchors.add(path.parent.parent if path.parent.name == 'series' else path.parent)
        if len(parts) >= 3 and parts[1] in SERIES_ANALYSES:
            if path.suffix.lower() == '.pvd' and not any(re.fullmatch(r'f\d+', p) for p in parts):
                anchors.add(path.parent.parent if path.parent.name == 'series' else path.parent)
    bundles = {}
    for path in files:
        candidates = [a for a in anchors if _inside(path, a)]
        parent = max(candidates, key=lambda a: len(a.parts)) if candidates else path.parent
        # Frame folders belong to the run even when a collection is absent.
        parts = path.relative_to(root).parts
        for index, part in enumerate(parts[:-1]):
            if re.fullmatch(r'f\d+', part) and not candidates:
                parent = root.joinpath(*parts[:index])
                break
        if _classification(parts) in {'diagnostic', 'smoke_test', 'historical'}:
            parent = path.parent
        bundles.setdefault(parent, []).append(path)
    owners = {path: parent for parent, members in bundles.items() for path in members}
    datasets, artifacts = [], []
    for parent, members in sorted(bundles.items()):
        relative = parent.relative_to(root).as_posix()
        parts = parent.relative_to(root).parts
        row = dict.fromkeys(DATASET_COLUMNS)
        row.update(dataset_id='ds_' + hashlib.sha256(relative.encode()).hexdigest()[:20],
                   study=parts[0] if parts else None,
                   analysis=parts[1] if len(parts) > 1 else None,
                   case_id=next((p for p in reversed(parts) if re.fullmatch(r'(?:case_)?N\d+(?:_[A-Z]+)?', p)), None),
                   variant='/'.join(parts[2:]) or None, artifact_class=_classification(parts),
                   status='unknown', validation_level='inventory_only',
                   relative_path=relative, file_count=len(members),
                   logical_bytes=sum(p.stat().st_size for p in members),
                   notes=[], metadata_sources=[])
        notes = row['notes']
        if row['case_id']:
            notes.append('case_id is a directory label; physical case identity unverified')
        if row['analysis'] not in SERIES_ANALYSES:
            notes.append('leaf-directory grouping; cross-directory scientific associations unverified')
        controls = [p for p in members if p.suffix.lower() in {'.pvd', '.json'}]
        pvds = sorted((p for p in controls if p.suffix.lower() == '.pvd'),
                      key=lambda p: (any(re.fullmatch(r'f\d+', part) for part in p.parts), p))
        manifests = [p for p in controls if p.name == 'manifest.json']
        if len(manifests) == 1:
            row['manifest_path'] = manifests[0].relative_to(root).as_posix()
        elif manifests:
            notes.append('multiple manifests in one control scope; manifest_path unknown')
        preferred = pvds or [p for p in members if p.suffix.lower() in {'.csv', '.png', '.pdf', '.npz', '.vtr', '.vtu', '.vti'}] or members
        row['entrypoint_file'] = preferred[0].relative_to(root).as_posix()
        times, broken = [], False
        for path in pvds:
            try:
                records = _pvd(path, root, known)
                times.extend(time for target, time in records)
                if any(owners[target] != parent for target, time in records):
                    notes.append(f'{path.relative_to(root)}: aggregate collection references independently indexed datasets')
                row['metadata_sources'].append(path.relative_to(root).as_posix())
            except (OSError, ValueError, ET.ParseError) as exc:
                broken = True
                notes.append(f'{path.relative_to(root)}: {exc}')
        metadata_times = []
        values = {key: [] for key in _metadata_values({})}
        reported_statuses = set()
        jsons = sorted((p for p in controls if p.suffix.lower() == '.json'),
                       key=lambda p: (p.name not in {'processing_summary.json', 'manifest.json'}, p))
        if len(jsons) > MAX_JSON_PER_DATASET:
            notes.append('JSON inspection capped at 64 files; metadata coverage partial')
        for path in jsons[:MAX_JSON_PER_DATASET]:
            try:
                document = json.loads(_bounded_read(path))
                if not isinstance(document, dict):
                    raise ValueError('expected JSON object')
                row['metadata_sources'].append(path.relative_to(root).as_posix())
                issues = _output_reference_issues(document, path, root, known)
                if issues:
                    broken = True
                    notes.extend(f'{path.relative_to(root)}: {issue}' for issue in issues)
                time = document.get('time')
                if type(time) in (int, float) and math.isfinite(time):
                    metadata_times.append(time)
                for key, value in _metadata_values(document).items():
                    if value is not None:
                        values[key].append(json.dumps(value, sort_keys=True, allow_nan=False))
                if path.name in {'processing_summary.json', 'manifest.json'}:
                    status = document.get('status')
                    if isinstance(status, str):
                        reported_statuses.add(status)
            except RecursionError:
                broken = True
                notes.append(f'{path.relative_to(root)}: JSON nesting exceeds parser depth; control not inspected')
            except (OSError, ValueError, TypeError) as exc:
                notes.append(f'{path.relative_to(root)}: {exc}')
        for key, observations in values.items():
            unique = sorted(set(observations))
            if len(unique) == 1:
                row[key] = json.loads(unique[0])
            elif len(unique) > 1:
                notes.append(f'conflicting {key}; retained as unknown')
        if not pvds and metadata_times:
            times = metadata_times
            notes.append('time range covers inspected JSON observations only; series extent unverified')
        if times:
            notes.append('frame_count counts distinct observed physical timesteps across collections/metadata')
            row.update(frame_count=len(set(times)), actual_time_start=min(times), actual_time_end=max(times))
        if broken or (reported_statuses and reported_statuses != {'completed'}):
            row['status'] = 'incomplete_or_ambiguous'
        elif reported_statuses == {'completed'}:
            row['status'] = 'reported_completed'
            notes.append('completion reported by control metadata; payload validity unverified')
        if not controls:
            notes.append('no JSON/PVD control metadata')
        warnings.extend(f'{relative}: {note}' for note in notes if not note.startswith('case_id'))
        datasets.append(row)
        if include_artifacts:
            for path in members:
                stat = path.stat()
                artifacts.append(dict(dataset_id=row['dataset_id'], relative_path=path.relative_to(root).as_posix(),
                                      file_type=path.suffix.lower().lstrip('.') or None,
                                      logical_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                      role=_role(path), related_controls=[p.relative_to(root).as_posix() for p in controls
                                                        if p.parent == path.parent or p.name in {'manifest.json', 'processing_summary.json'}
                                                        or (p.suffix == '.pvd' and not any(re.fullmatch(r'f\d+', part) for part in p.parts))]))
    return dict(schema_version=SCHEMA_VERSION, datasets=datasets, artifacts=artifacts,
                file_count=len(files), logical_bytes=sum(r['logical_bytes'] for r in datasets),
                warnings=sorted(warnings))


def _csv(rows, columns):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()})
    return buffer.getvalue()


def write_catalog(catalog, output_dir, *, results_root):
    """Publish into an exclusively created directory outside protected roots.

    An interrupted write leaves an explicit partial directory for inspection.
    A retry requires a fresh path. Existing files are never replaced.
    """
    output = Path(output_dir).resolve()
    for protected in (Path(results_root).resolve(), SCIENTIFIC_ROOT.resolve()):
        if _inside(output, protected):
            raise ValueError('catalog output must be outside scientific results roots')
    payloads = {'catalog.json': json.dumps(catalog, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                'datasets.csv': _csv(catalog['datasets'], DATASET_COLUMNS)}
    if catalog['artifacts']:
        payloads['artifacts.csv'] = _csv(catalog['artifacts'], ARTIFACT_COLUMNS)
    output.mkdir(exist_ok=False)
    for name, payload in payloads.items():
        with (output / name).open('x', encoding='utf-8', newline='') as stream:
            stream.write(payload)
