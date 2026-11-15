"""Whole-file configuration selection with immutable, original input values."""

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
import os


class Refusal(ValueError):
    """A deterministic configuration or authorization refusal."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(freeze(item) for item in value)
    return value


def plain(value):
    """Return native YAML-schema containers without changing scalar values."""
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [plain(item) for item in value]
    return value


@dataclass(frozen=True)
class Configuration:
    role: str
    supplied: str
    canonical: Path | None
    digest: str
    values: Mapping

    def verify_unchanged(self):
        if self.canonical is not None:
            try:
                current = sha256(self.canonical.read_bytes()).hexdigest()
            except OSError as error:
                raise Refusal('configuration_changed', str(error)) from error
            if current != self.digest:
                raise Refusal('configuration_changed', f'Configuration changed: {self.canonical}')


def resolve_configuration(role: str, explicit=None, *, cwd: Path, environ=None) -> Configuration:
    """Explicit file, environment file, packaged scientific default, or refusal.

    Relative configuration filenames use captured invocation cwd. Relative YAML
    path fields are normalized later against the canonical supplying parent.
    Resources are read through Traversable and never extracted to a temporary.
    """
    roles = {'paths': ('NEK_POST_PATHS_CONFIG', None),
             'cases': ('NEK_POST_CASES_CONFIG', 'cases.yaml'),
             'study': ('NEK_POST_STUDY_CONFIG', 'h_refinement.yaml')}
    variable, resource = roles[role]
    environment = os.environ if environ is None else environ
    supplied = explicit if explicit is not None else environment.get(variable)
    canonical = None
    if supplied is not None:
        if not str(supplied).strip():
            raise Refusal('invalid_configuration', f'Empty {role} configuration source')
        path = Path(supplied)
        try:
            canonical = (cwd / path).resolve(strict=True)
            if not canonical.is_file():
                raise Refusal('invalid_configuration', f'Configuration must be a file: {canonical}')
            raw = canonical.read_bytes()
        except (OSError, RuntimeError, ValueError) as error:
            raise Refusal('invalid_configuration', f'Cannot read {role} configuration: {supplied}') from error
    else:
        if resource is None:
            raise Refusal('missing_safety_source', 'Supply --paths-config or NEK_POST_PATHS_CONFIG')
        supplied = f'nek_post.resources/{resource}'
        raw = files('nek_post.resources').joinpath(resource).read_bytes()
    # YAML is needed only after source selection, never during help.
    try:
        import yaml
    except ImportError as error:
        raise Refusal('missing_dependency', 'Configuration parsing requires PyYAML') from error
    try:
        values = yaml.safe_load(raw)
    except (yaml.YAMLError, UnicodeError) as error:
        raise Refusal('invalid_configuration', f'Malformed {role} YAML: {supplied}') from error
    if not isinstance(values, dict) or not values or not all(isinstance(k, str) for k in values):
        raise Refusal('invalid_configuration', f'{role} configuration must be a nonempty string-keyed mapping')
    if role == 'cases':
        if not isinstance(values.get('orders'), dict) or not values['orders']:
            raise Refusal('invalid_configuration', 'cases configuration requires nonempty orders mapping')
        if not isinstance(values.get('reference_case'), str) or values['reference_case'] not in values['orders']:
            raise Refusal('invalid_configuration', 'reference_case must occur in orders')
    if role == 'study':
        cases = values.get('cases')
        if not isinstance(cases, list) or not cases or not all(isinstance(c, str) and c for c in cases):
            raise Refusal('invalid_configuration', 'study configuration requires case labels')
        if not isinstance(values.get('provisional_reference_case'), str) or values['provisional_reference_case'] not in cases:
            raise Refusal('invalid_configuration', 'study reference must occur in cases')
    try:
        frozen = freeze(values)
    except RecursionError as error:
        raise Refusal('invalid_configuration', 'Recursive or excessively nested configuration') from error
    return Configuration(role, str(supplied), canonical, sha256(raw).hexdigest(), frozen)
