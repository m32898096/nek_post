"""Internal Script 00 pilot; no public dispatcher imports or enables this module.

Only explicitly authorized synthetic callers may execute this pilot. B3's
deferred native-runtime/G-WRITE requirement and the public route remain gated.
The script-local report helpers are not installed package APIs. Their small
integer/text contract is reproduced here and checked against retained B0 bytes;
the original script is never imported, executed, copied or modified at runtime.
"""

from contextlib import ExitStack
from dataclasses import dataclass
import errno
import os
from pathlib import Path
import sys

from .capabilities import check_prerequisites
from .configuration import Configuration, Refusal, resolve_configuration
from .contracts import CONTRACTS
from .safety import SafetyContext, build_context


@dataclass(frozen=True)
class Inspection:
    context: SafetyContext
    cases: Configuration
    root: Path
    cwd: Path
    log: Path


def _validate_output(plan: Inspection) -> None:
    """Recheck configuration identities, input aliases and all write locations."""
    plan.context.source.verify_unchanged()
    plan.cases.verify_unchanged()
    current = build_context(plan.context.source, configurations=(plan.cases.canonical,),
                            required_labels=plan.cases.values['orders'])
    if current.input_values != plan.context.input_values:
        raise Refusal('configuration_changed', 'Machine-path resolution changed')
    if current.check_root(plan.root, cwd=plan.cwd) != plan.root:
        raise Refusal('unsafe_path', 'Output root resolution changed')
    if not plan.root.parent.is_dir():
        raise Refusal('missing_output_parent', 'The parent of explicit R must already exist')
    for path in (plan.root/'inspect', plan.root/'inspect/logs'):
        current.destination(plan.root, path, cwd=plan.cwd, kind='directory', fresh=False)
    current.destination(plan.root, plan.log, cwd=plan.cwd)


def prepare_inspection(*, output_root, cwd: Path, paths_config=None,
                       cases_config=None, environ=None) -> Inspection:
    """B2 authorization then B3 discovery, with no directory/file allocation.

    This does not grant public execution authority or close a write-set gate.
    Relative YAML paths use B2's canonical configuration parent, not legacy cwd.
    """
    cwd = Path(cwd).resolve(strict=True)
    paths = resolve_configuration('paths', paths_config, cwd=cwd, environ=environ)
    cases = resolve_configuration('cases', cases_config, cwd=cwd, environ=environ)
    prefix, indices = cases.values.get('file_prefix'), cases.values.get('file_indices')
    if (not isinstance(prefix, str) or not prefix or
            any(c in prefix for c in ('/', '\\', '\0')) or prefix in {'.', '..'}):
        raise Refusal('invalid_configuration', 'file_prefix must be one nonempty filename prefix')
    if not isinstance(indices, tuple) or any(type(i) is not int for i in indices):
        raise Refusal('invalid_configuration', 'file_indices must be a list of integers (not booleans)')
    context = build_context(paths, configurations=(cases.canonical,),
                            required_labels=cases.values['orders'])
    if output_root is None:
        raise Refusal('missing_output_root', 'Supply explicit output_root R')
    root = context.check_root(output_root, cwd=cwd)
    # Keep destination authority in the accepted contract rather than a new root.
    primary = CONTRACTS['inspect files'].primary
    log = root/primary.removeprefix('R/')/'check_files.log'
    plan = Inspection(context, cases, root, cwd, log)
    _validate_output(plan)
    if (os.name != 'posix' or not hasattr(os, 'O_NOFOLLOW') or
            os.open not in os.supports_dir_fd or os.mkdir not in os.supports_dir_fd):
        raise Refusal('unsupported_platform', 'Safe creation requires POSIX no-follow directory descriptors')
    for result in check_prerequisites('inspect files'):
        # Deferred is retained, never converted into execution authorization.
        if result.requirement.probe != 'deferred' and result.status != 'available':
            raise Refusal('missing_dependency', result.diagnostic)
    return plan


def build_report(plan: Inspection) -> tuple[str, int]:
    """Preserve Script 00's fixed-width rows, ordering, exists and integer counts."""
    def row(case, order, index, path, status):
        return f'{case:<6} {str(order):<6} {str(index):<7} {str(path):<64} {status}'

    lines = ['Nek5000 expected file check', '',
             row('Case', 'Order', 'Index', 'File path', 'Status'),
             row('------', '------', '-------', '---------', '------')]
    total = existing = 0
    for case, directory in plan.context.input_values['case_dirs'].items():
        order = plan.cases.values['orders'].get(case, 'UNKNOWN')
        for index in plan.cases.values['file_indices']:
            path = Path(directory)/f"{plan.cases.values['file_prefix']}.f{index:05d}"
            present = path.exists()
            total += 1
            existing += int(present)
            lines.append(row(case, order, index, path, 'OK' if present else 'MISSING'))
    missing = total-existing
    lines.extend(['', 'Summary', f'Total expected files: {total}',
                  f'Existing files: {existing}', f'Missing files: {missing}'])
    return '\n'.join(lines), 1 if missing else 0


def _write_log(plan: Inspection, payload: bytes) -> None:
    """Exclusive final open; no-follow dir_fd traversal at each allocation.

    Only R itself, R/inspect, R/inspect/logs and the new log may be created.
    Earlier existence checks are not the collision protection. Failed/partial
    products remain for diagnosis and block retry; no rollback/atomicity claim.
    Directory descriptors prevent following replacement symlinks. Arbitrary
    hostile relocation of already opened directories is outside the B2
    cooperative-filesystem model and remains a public G-WRITE gate limitation.
    """
    _validate_output(plan)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    with ExitStack() as stack:
        def opened(path, flags, *, parent=None, mode=0o600):
            descriptor = os.open(path, flags, mode, dir_fd=parent)
            stack.callback(os.close, descriptor)
            return descriptor

        parent = opened('/', directory_flags)
        # No missing ancestor above R is ever created.
        for component in plan.root.parent.parts[1:]:
            parent = opened(component, directory_flags, parent=parent)
        for component in (plan.root.name, 'inspect', 'logs'):
            try:
                os.mkdir(component, mode=0o700, dir_fd=parent)
            except FileExistsError:
                pass
            parent = opened(component, directory_flags, parent=parent)
        descriptor = opened('check_files.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                            os.O_NOFOLLOW | os.O_CLOEXEC, parent=parent)
        remaining = memoryview(payload)
        while remaining:
            count = os.write(descriptor, remaining)
            if count <= 0:
                raise OSError(errno.EIO, 'Log write made no progress')
            remaining = remaining[count:]


def inspect_files(*, output_root, cwd: Path, paths_config=None, cases_config=None,
                  environ=None, dry_run=False, stdout=None, stderr=None) -> int:
    """Internal synthetic pilot entry point, never a public authorization path.

    Preflight refusals emit no report. After successful preflight, print() is
    called before allocation; buffering supplies no kernel-write-order promise.
    Dry-run validates the same configuration/destination/dependencies and reports
    existence without allocating anything. Missing snapshots and I/O/refusal
    failures return 1; public argparse/unavailable statuses remain 2/3 elsewhere.
    """
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    try:
        plan = prepare_inspection(output_root=output_root, cwd=cwd,
                                  paths_config=paths_config, cases_config=cases_config,
                                  environ=environ)
        report, status = build_report(plan)
        print(report, file=stdout)
        if not dry_run:
            _write_log(plan, (report+'\n').encode('utf-8'))
        return status
    except (Refusal, OSError, UnicodeError) as error:
        code = error.code if isinstance(error, Refusal) else 'io_error'
        print(f'ERROR: inspect files: {code}: {error}', file=stderr)
        return 1
