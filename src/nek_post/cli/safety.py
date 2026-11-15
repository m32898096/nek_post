"""Read-only path authorization. Native validators remain additional gates.

Existing output symlinks and lexical '..' are refused before canonicalization.
Nonexistent descendants are checked through their existing ancestors. These
checks assume a cooperative filesystem and require revalidation at future
allocation and native locking boundaries to address concurrent replacement.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .configuration import Configuration, Refusal, freeze, plain


def normalize_path(value, *, anchor: Path, kind='directory', output=False) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise Refusal('invalid_path', 'Path must be a nonempty string')
    supplied = Path(value)
    if output and '..' in supplied.parts:
        raise Refusal('unsafe_path', f'Parent traversal in output: {value}')
    absolute = anchor / supplied
    try:
        for component in (*reversed(absolute.parents), absolute):
            if output and component.is_symlink():
                raise Refusal('unsafe_path', f'Symlink in output: {component}')
            if component != absolute and component.exists() and not component.is_dir():
                raise Refusal('invalid_path', f'Non-directory ancestor: {component}')
        path = absolute.resolve()
        if path.exists() and ((kind == 'directory' and not path.is_dir()) or
                              (kind == 'file' and not path.is_file())):
            raise Refusal('invalid_path', f'Expected {kind}: {path}')
        return path
    except Refusal:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise Refusal('invalid_path', f'Cannot resolve path: {value}') from error


def overlaps(left: Path, right: Path) -> bool:
    return left.is_relative_to(right) or right.is_relative_to(left)


@dataclass(frozen=True)
class ProtectedPath:
    path: Path
    role: str
    tree: bool = True


@dataclass(frozen=True)
class SafetyContext:
    source: Configuration
    input_values: Mapping
    protected: tuple[ProtectedPath, ...]
    data_root: Path

    def check_root(self, value, *, cwd: Path, publication=False) -> Path:
        root = normalize_path(value, anchor=cwd, output=True)
        if self.data_root.is_relative_to(root):
            raise Refusal('protected_overlap', f'Output would contain data_root: {root}')
        for item in self.protected:
            # Explicit run-owned configuration inputs may share R as a container.
            # Individual write/delete targets must still pass destination(). P
            # receives no such exemption from immutable input protection.
            if (not publication and item.role == 'run_owned_configuration'
                    and item.path != root and item.path.is_relative_to(root)):
                continue
            # P alone may create the two navigation files within historical roots.
            if publication and item.role == 'historical':
                continue
            conflict = overlaps(root, item.path) if item.tree else item.path.is_relative_to(root)
            if conflict:
                raise Refusal('protected_overlap', f'Output overlaps {item.role}: {item.path}')
        return root

    def destination(self, root: Path, value, *, cwd: Path, kind='file', fresh=True) -> Path:
        """Check an individual write, replacement or deletion target.

        Deletions/replacements use fresh=False and retain all immutable-overlap
        checks, including checks against ancestors of run-owned input files.
        Root eligibility alone never authorizes any operation.
        """
        path = normalize_path(value, anchor=cwd, kind=kind, output=True)
        if path == root or not path.is_relative_to(root):
            raise Refusal('outside_authorization', f'Destination must be strictly beneath R: {path}')
        for item in self.protected:
            if overlaps(path, item.path):
                raise Refusal('protected_overlap', f'Destination overlaps {item.role}: {item.path}')
        if fresh and path.exists():
            raise Refusal('collision', f'Destination exists: {path}')
        return path

    def output_view(self, root: Path):
        """Separate adapter mapping, leaving the original safety inventory intact."""
        return MappingProxyType({'postproc_root': root/'p-refinement/postproc',
                                 'results_root': root/'p-refinement/results',
                                 'h_refinement_results_root': root/'h_refinement'})

    def effective_paths(self):
        """Native raw-file schema with original normalized input/safety roots."""
        return plain(self.input_values)

    def publication_targets(self, root, preview, *, cwd: Path) -> tuple[Path, Path]:
        publication = self.check_root(root, cwd=cwd, publication=True)
        source = normalize_path(preview, anchor=cwd)
        if not source.is_dir() or overlaps(publication, source):
            raise Refusal('unsafe_preview', 'Reviewed V must be an existing directory disjoint from P')
        targets = (publication/'README.md', publication/'results_index.csv')
        for target in targets:
            normalize_path(target, anchor=cwd, kind='file', output=True)
            if target.exists():
                raise Refusal('collision', f'Publication target exists; preserve partial pair: {target}')
            for item in self.protected:
                if item.role != 'historical' and overlaps(target, item.path):
                    raise Refusal('protected_overlap', f'Publication overlaps {item.role}: {item.path}')
        return targets


def build_context(source: Configuration, *, inputs=(), configurations=(), required_labels=(),
                  run_owned_configurations=()) -> SafetyContext:
    """Build the immutable inventory without allocating any paths.

    run_owned_configurations is an explicit assertion by the trusted invocation
    caller about existing effective configuration inputs. It is never inferred
    from filenames, containment or YAML. An effective paths source itself may
    have this role only when explicitly listed. Other declared input roles,
    including original configurations, are retained even for duplicate paths.
    CLI exposure and effective-file creation remain deferred.
    """
    values = plain(source.values)
    allowed = {'data_root', 'case_dirs', 'postproc_root', 'results_root',
               'h_refinement_results_root', 'paper_data'}
    if set(values) - allowed:
        raise Refusal('invalid_inventory', 'Unknown machine-path fields require explicit schema review')
    if source.role != "paths" or source.canonical is None:
        raise Refusal("missing_safety_source", "Safety context requires explicit or environment machine paths")
    anchor = source.canonical.parent
    def directory(value):
        return str(normalize_path(value, anchor=anchor))
    for key in ('data_root', 'postproc_root', 'results_root'):
        values[key] = directory(values.get(key))
    cases = values.get('case_dirs')
    if not isinstance(cases, dict) or not cases or not all(isinstance(k, str) and k for k in cases):
        raise Refusal('invalid_inventory', 'case_dirs must contain all configured case labels')
    values['case_dirs'] = {key: directory(value) for key, value in cases.items()}
    if set(required_labels) - set(cases):
        raise Refusal('invalid_inventory', 'Machine inventory is missing required scientific case labels')
    values['h_refinement_results_root'] = directory(values.get(
        'h_refinement_results_root', str(Path(values['data_root'])/'results/h_refinement')))
    papers = values.get('paper_data')
    paper_keys = {'cantero_fig5a_re3450_csv', 'cantero_fig5a_re8950_csv'}
    if not isinstance(papers, dict) or set(papers) != paper_keys:
        raise Refusal('invalid_inventory', 'paper_data must declare both reference CSV paths')
    values['paper_data'] = {key: str(normalize_path(value, anchor=anchor, kind='file'))
                            for key, value in papers.items()}
    protected = [ProtectedPath(Path(p), 'raw_case') for p in values['case_dirs'].values()]
    protected += [ProtectedPath(Path(values[key]), 'historical') for key in
                  ('postproc_root', 'results_root', 'h_refinement_results_root')]
    protected += [ProtectedPath(Path(p), 'reference', False) for p in values['paper_data'].values()]
    run_owned = set()
    for path in run_owned_configurations:
        if not isinstance(path, (str, Path)) or not Path(path).is_absolute():
            raise Refusal('invalid_input', 'Run-owned configurations require explicit absolute paths')
        canonical = normalize_path(path, anchor=anchor, kind='file', output=True)
        if not canonical.is_file():
            raise Refusal('invalid_input', f'Run-owned configuration must already exist: {canonical}')
        run_owned.add(canonical)
        protected.append(ProtectedPath(canonical, 'run_owned_configuration', False))
    original_inputs = (*inputs, *configurations)
    if source.canonical not in run_owned:
        original_inputs += (source.canonical,)
    for path in original_inputs:
        if path is not None:
            canonical = Path(path).resolve()
            protected.append(ProtectedPath(canonical, 'input', not canonical.is_file()))
    # Native guards use parents[2] of flat scientific modules, including wheels.
    package = Path(__file__).resolve().parents[1]
    protected += [ProtectedPath(package, 'package'), ProtectedPath(package.parents[1], 'native_ancestor')]
    return SafetyContext(source, freeze(values), tuple(protected), Path(values['data_root']))
