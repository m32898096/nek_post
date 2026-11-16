"""Internal, supervised synthetic Script 43 pilot; public dispatch stays gated.

Construction delegates to the native builder without publication authority.
Publication delegates to the native sequential writer after full B2 rechecks.
Plans are caller-owned data, not authorization tokens. Stable cooperative output
parents and launcher-enforced immutable finite inputs are required. Partial
files remain; neither complete-file atomicity nor transactions are provided.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import stat
import sys

from .capabilities import CapabilityResult, check_prerequisites
from .configuration import Refusal, resolve_configuration
from .contracts import CONTRACTS
from .safety import SafetyContext, build_context, normalize_path, overlaps


_PRODUCTS = ('catalog.json', 'datasets.csv', 'artifacts.csv')
_HANDLED = (OSError, ValueError, TypeError, RecursionError)


@dataclass(frozen=True)
class InventoryPlan:
    supplied_root: Path
    input_root: Path
    cwd: Path
    include_artifacts: bool
    route: str
    capabilities: tuple[CapabilityResult, ...]


@dataclass(frozen=True)
class CatalogBuildPlan:
    inventory: InventoryPlan
    context: SafetyContext
    supplied_output_root: Path
    root: Path
    output_dir: Path
    possible_files: tuple[Path, ...]


@dataclass(frozen=True)
class PublicationResult:
    output_dir: Path
    expected_files: tuple[Path, ...]
    published_files: tuple[Path, ...]
    complete: bool = True


class PublicationFailure(Exception):
    """Preserved cause plus bounded observations; no inferred file ownership."""

    def __init__(self, error, *, phase, output_dir, expected_files=(),
                 writer_invoked=False, publication_complete=False):
        self.code = _code(error)
        self.phase = phase
        self.output_dir = output_dir
        self.expected_files = expected_files
        self.writer_invoked = writer_invoked
        # Absence after delegation does not prove no attempted allocation.
        self.allocation_attempted = None if writer_invoked else False
        self.publication_complete = publication_complete
        self.observed_paths, self.inspection_errors = _observe(output_dir)
        super().__init__(str(error))


def _code(error):
    if isinstance(error, Refusal):
        return error.code
    if isinstance(error, OSError):
        return 'io_error'
    if isinstance(error, UnicodeError):
        return 'encoding_error'
    return 'invalid_catalog'


def _observe(directory):
    """Never follow a symlinked D to inspect children; failed lstat is Unknown."""
    observed, errors = [], []

    def inspect(path):
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            kind = 'absent'
        except OSError as error:
            kind = 'unknown'
            errors.append((path, str(error)))
        else:
            kind = ('directory' if stat.S_ISDIR(mode) else
                    'regular' if stat.S_ISREG(mode) else
                    'symlink' if stat.S_ISLNK(mode) else 'other')
        observed.append((path, kind))
        return kind

    kind = inspect(directory)
    for name in _PRODUCTS:
        path = directory/name
        if kind == 'directory':
            inspect(path)
        else:
            observed.append((path, 'absent' if kind == 'absent' else 'unknown'))
    return tuple(observed), tuple(errors)


def _input(input_root, cwd):
    # This check must precede normalization, which would erase the final alias.
    if not isinstance(input_root, (str, Path)) or not str(input_root).strip():
        raise Refusal('invalid_input', 'Supply an explicit input_root')
    supplied = cwd/Path(input_root)
    if supplied.is_symlink():
        raise Refusal('unsafe_path', f'Input root must not be a symlink: {supplied}')
    canonical = normalize_path(supplied, anchor=cwd)
    if not canonical.is_dir():
        raise Refusal('invalid_input', f'Input root must be an existing directory: {supplied}')
    return supplied, canonical


def _options(cwd, include_artifacts):
    if type(include_artifacts) is not bool:
        raise Refusal('invalid_option', 'include_artifacts must be a bool')
    captured = Path(cwd).resolve(strict=True)
    if not captured.is_dir():
        raise Refusal('invalid_path', 'cwd must be an existing directory')
    return captured


def _capabilities(route):
    results = check_prerequisites(route)
    for result in results:
        if (result.requirement.required and result.requirement.probe != 'deferred'
                and result.status != 'available'):
            raise Refusal('missing_dependency', result.diagnostic)
    return results


def _recheck_input(plan):
    _, canonical = _input(plan.supplied_root, plan.cwd)
    if canonical != plan.input_root:
        raise Refusal('input_changed', f'Input root resolution changed: {plan.supplied_root}')


def prepare_inventory(*, input_root, cwd, include_artifacts=False):
    """Validate explicit input and B3 without YAML, output paths or allocation."""
    cwd = _options(cwd, include_artifacts)
    supplied, canonical = _input(input_root, cwd)
    route = 'results inventory'
    return InventoryPlan(supplied, canonical, cwd, include_artifacts,
                         route, _capabilities(route))


def collect_catalog(plan: InventoryPlan):
    """Return the native plain dict; this operation never calls the writer."""
    _recheck_input(plan)
    _capabilities(plan.route)
    from nek_post.results_catalog import build_catalog
    return build_catalog(plan.input_root, include_artifacts=plan.include_artifacts)


def _validate_output(plan):
    inventory = plan.inventory
    plan.context.source.verify_unchanged()
    _recheck_input(inventory)
    current = build_context(plan.context.source, inputs=(inventory.input_root,))
    if current.input_values != plan.context.input_values:
        raise Refusal('configuration_changed', 'Machine-path resolution changed')
    root = current.check_root(plan.supplied_output_root, cwd=inventory.cwd)
    if root != plan.root or not root.is_dir():
        raise Refusal('invalid_output_root', f'Explicit R must already be a real directory: {root}')
    if overlaps(root, inventory.input_root):
        raise Refusal('protected_overlap', 'R must be disjoint from the scanned input tree')
    if CONTRACTS['results catalog build'].primary != 'R/catalog':
        raise Refusal('invalid_contract', 'Expected the accepted R/catalog contract')
    if plan.output_dir != root/'catalog' or plan.possible_files != tuple(plan.output_dir/n for n in _PRODUCTS):
        raise Refusal('outside_authorization', 'Catalog plan does not match exact destinations')
    current.destination(root, plan.output_dir, cwd=inventory.cwd, kind='directory', fresh=True)
    for path in plan.possible_files:
        current.destination(root, path, cwd=inventory.cwd, fresh=True)


def prepare_catalog_build(*, input_root, output_root, cwd, paths_config=None,
                          environ=None, include_artifacts=False):
    """Full B2 preflight precedes B3 and lazy catalog-handler loading."""
    cwd = _options(cwd, include_artifacts)
    supplied, canonical = _input(input_root, cwd)
    paths = resolve_configuration('paths', paths_config, cwd=cwd, environ=environ)
    context = build_context(paths, inputs=(canonical,))
    if output_root is None:
        raise Refusal('missing_output_root', 'Supply explicit existing output_root R')
    root = context.check_root(output_root, cwd=cwd)
    inventory_plan = InventoryPlan(supplied, canonical, cwd, include_artifacts,
                                   'results catalog build', ())
    directory = root/'catalog'
    plan = CatalogBuildPlan(inventory_plan, context, cwd/Path(output_root), root,
                            directory, tuple(directory/n for n in _PRODUCTS))
    _validate_output(plan)
    inventory_plan = InventoryPlan(supplied, canonical, cwd, include_artifacts,
                                   inventory_plan.route, _capabilities(inventory_plan.route))
    return CatalogBuildPlan(inventory_plan, context, plan.supplied_output_root,
                            root, directory, plan.possible_files)


def _native_guard(plan, scientific_root):
    for protected in (plan.inventory.input_root, scientific_root.resolve()):
        if plan.output_dir.is_relative_to(protected):
            raise Refusal('protected_overlap', f'Native scientific output guard: {protected}')


def publish_catalog(plan: CatalogBuildPlan, catalog: dict):
    """Explicit native publication; retain every partial product on failure.

    Trusted callers supply the unchanged catalog collected for this plan. This
    interface does not authenticate arbitrary dictionaries or historical files.
    """
    invoked = False
    expected = ()
    phase = 'preflight'
    try:
        _validate_output(plan)
        _capabilities(plan.inventory.route)
        from nek_post.results_catalog import SCIENTIFIC_ROOT, write_catalog
        _native_guard(plan, SCIENTIFIC_ROOT)
        expected = plan.possible_files if catalog['artifacts'] else plan.possible_files[:2]
        # Revalidate immediately before delegation, not just at plan creation.
        _validate_output(plan)
        phase = 'publication'
        invoked = True
        write_catalog(catalog, plan.output_dir, results_root=plan.inventory.input_root)
        observed, errors = _observe(plan.output_dir)
        states = dict(observed)
        if (errors or states[plan.output_dir] != 'directory'
                or any(states[p] != 'regular' for p in expected)
                or any(states[p] != 'absent' for p in plan.possible_files if p not in expected)):
            raise Refusal('incomplete_publication', 'Expected regular output files were not established')
        return PublicationResult(plan.output_dir, expected, expected)
    except _HANDLED as error:
        raise PublicationFailure(error, phase=phase, output_dir=plan.output_dir,
                                 expected_files=expected, writer_invoked=invoked) from error


def _diagnostic(stderr, route, phase, error):
    code = error.code if isinstance(error, PublicationFailure) else _code(error)
    message = f'ERROR: {route}: {phase}: {code}: {error}'
    if isinstance(error, PublicationFailure):
        message += (f'; publication_complete={error.publication_complete}'
                    f'; writer_invoked={error.writer_invoked}'
                    f'; allocation_attempted={error.allocation_attempted}'
                    '; observed (unattributed)='+repr([(str(p), s) for p, s in error.observed_paths])
                    + '; inspection_errors='+repr([(str(p), s) for p, s in error.inspection_errors]))
    print(message, file=stderr)


def inventory(*, input_root, cwd, include_artifacts=False, stdout=None, stderr=None):
    """Internal read-only full-JSON presentation with integer 0/1 status."""
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    phase = 'preflight'
    try:
        plan = prepare_inventory(input_root=input_root, cwd=cwd, include_artifacts=include_artifacts)
        phase = 'construction'
        catalog = collect_catalog(plan)
        phase = 'reporting'
        print(json.dumps(catalog, indent=2, ensure_ascii=False, allow_nan=False), file=stdout)
        return 0
    except _HANDLED as error:
        _diagnostic(stderr, 'results inventory', phase, error)
        return 1


def catalog_build(*, input_root, output_root, cwd, paths_config=None, environ=None,
                  include_artifacts=False, dry_run=False, stdout=None, stderr=None):
    """Internal 0/1 wrapper; dry-run constructs but explicitly does not publish."""
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    phase = 'preflight'
    plan = None
    result = None
    try:
        if type(dry_run) is not bool:
            raise Refusal('invalid_option', 'dry_run must be a bool')
        plan = prepare_catalog_build(input_root=input_root, output_root=output_root, cwd=cwd,
                                     paths_config=paths_config, environ=environ,
                                     include_artifacts=include_artifacts)
        from nek_post.results_catalog import SCIENTIFIC_ROOT
        _native_guard(plan, SCIENTIFIC_ROOT)
        phase = 'construction'
        catalog = collect_catalog(plan.inventory)
        if dry_run:
            phase = 'reporting'
            print(f'Validated dry-run; NOT PUBLISHED: {plan.output_dir}; Datasets: {len(catalog["datasets"])}', file=stdout)
        else:
            result = publish_catalog(plan, catalog)
            phase = 'reporting'
            print(json.dumps({key: catalog[key] for key in
                              ('schema_version', 'file_count', 'logical_bytes', 'warnings')}, indent=2), file=stdout)
            print(f'Datasets: {len(catalog["datasets"])}', file=stdout)
        return 0
    except PublicationFailure as error:
        _diagnostic(stderr, 'results catalog build', error.phase, error)
        return 1
    except _HANDLED as error:
        if result is not None:
            error = PublicationFailure(error, phase=phase, output_dir=plan.output_dir,
                                       expected_files=result.expected_files, writer_invoked=True,
                                       publication_complete=True)
        _diagnostic(stderr, 'results catalog build', phase, error)
        return 1
