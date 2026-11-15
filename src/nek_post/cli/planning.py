"""Configuration and authorization before any future allocation or dispatch."""

from dataclasses import dataclass
from pathlib import Path

from .configuration import Refusal, resolve_configuration
from .contracts import CONTRACTS, READ_ONLY
from .safety import build_context, normalize_path


@dataclass(frozen=True)
class Plan:
    command_id: str
    safety: str
    primary_contract: str
    unknown_write_set: bool
    destinations: tuple[str, ...] = ()
    operational_categories: tuple[str, ...] = ()
    status: str = 'incomplete'
    reason: str = 'Route-specific complete write inventory and G-WRITE acceptance remain pending'
    execution_authorized: bool = False


def prepare_plan(args, *, cwd: Path) -> Plan:
    """Resolve source -> inventory -> destinations -> authorization, without writes.

    All scientific handlers remain unavailable. Known primary destinations are
    reported separately from complete transitive write sets. Even non-Unknown
    native companion contracts need route-specific observed G-WRITE acceptance.
    """
    command = args.command.command_id
    contract = CONTRACTS[command]
    source = None
    if command not in READ_ONLY or command == 'h-refinement inventory':
        source = resolve_configuration('paths', args.paths_config, cwd=cwd)
    inputs = []
    for name in ('input_root', 'case_dir'):
        value = getattr(args, name)
        if value is not None:
            path = normalize_path(value, anchor=cwd)
            if not path.is_dir():
                raise Refusal('missing_input', f'Input directory does not exist: {path}')
            inputs.append(path)
    if command in READ_ONLY and command != 'h-refinement inventory':
        if not args.input_root or (command == 'omega-x validate-series' and not args.case_dir):
            raise Refusal('missing_input', 'Supply --input-root and applicable --case-dir')
        return Plan(command, 'RO', contract.primary, False, status='planned',
                    reason='Read-only input plan; scientific execution remains unavailable')
    configurations = []
    scientific_sources = []
    labels = set()
    # Resolve only scientific roles needed by the selected family or overrides.
    if args.cases_config is not None or command.split()[0] in {'inspect', 'p-refinement', 'integrate', 'cantero', 'leading-edge'}:
        cases = resolve_configuration('cases', args.cases_config, cwd=cwd)
        scientific_sources.append(cases)
        configurations.append(cases.canonical)
        labels.update(cases.values['orders'])
    if args.study_config is not None or command.startswith('h-refinement '):
        study = resolve_configuration('study', args.study_config, cwd=cwd)
        scientific_sources.append(study)
        configurations.append(study.canonical)
        labels.update(study.values['cases'])
    context = build_context(source, inputs=inputs, configurations=configurations, required_labels=labels)
    if command in READ_ONLY:
        return Plan(command, 'RO', contract.primary, False, status='planned',
                    reason='Read-only configuration plan; scientific execution remains unavailable')
    if args.output_root is None:
        raise Refusal('missing_output_root', 'Supply explicit --output-root R')
    root = context.check_root(args.output_root, cwd=cwd)
    destinations = []
    if args.output_dir is not None:
        if command != 'leading-edge extract':
            raise Refusal('unsupported_override', '--output-dir is reserved for exact leading-edge extraction D')
        destinations.append(context.destination(root, args.output_dir, cwd=cwd, kind='directory', fresh=False))
    if args.preview_dir is not None and not command.startswith('results navigation '):
        raise Refusal('unsupported_override', '--preview-dir applies only to navigation')
    if args.publication_root is not None and command != 'results navigation publish':
        raise Refusal('unsupported_override', '--publication-root applies only to navigation publication')
    if command == 'results catalog build':
        destinations.append(context.destination(root, root/'catalog', cwd=cwd, kind='directory'))
    if command == 'results navigation preview':
        preview = args.preview_dir if args.preview_dir is not None else root/'navigation-preview'
        destinations.append(context.destination(root, preview, cwd=cwd, kind='directory'))
    if command == 'results navigation publish':
        if args.publication_root is None or args.preview_dir is None:
            raise Refusal('missing_publication', 'Supply explicit --publication-root P and --preview-dir V')
        preview = normalize_path(args.preview_dir, anchor=cwd)
        if preview == root or not preview.is_dir():
            raise Refusal('unsafe_preview', 'Reviewed V must be an existing directory distinct from R')
        # V becomes an immutable input for both operational R and publication P.
        context = build_context(source, inputs=(*inputs, preview), configurations=configurations,
                                required_labels=labels)
        # V may be a previous preview under R. Authorize individual operational
        # destinations against it, without reclassifying V as writable.
        destinations.extend(context.publication_targets(args.publication_root, preview, cwd=cwd))
    operational = tuple(str(context.destination(root, root/'.cli'/name, cwd=cwd,
                                               kind='directory', fresh=False))
                        for name in ('invocations', 'config', 'cwd', 'tmp', 'matplotlib', 'locks'))
    for configuration in (source, *scientific_sources):
        configuration.verify_unchanged()
    return Plan(command, 'FS', contract.primary, contract.unknown,
                tuple(str(path) for path in destinations), operational)
