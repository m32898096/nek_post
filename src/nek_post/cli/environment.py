"""Declarative environment for future authorized standalone execution."""

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
from types import MappingProxyType

from .configuration import Refusal
from .safety import SafetyContext


@dataclass(frozen=True)
class EnvironmentSpec:
    inherited: Mapping[str, str]
    overrides: Mapping[str, str]
    unset: tuple[str, ...]
    environment: Mapping[str, str]
    writable_paths: tuple[Path, ...]
    readonly_inputs: tuple[Path, ...]
    cwd: Path
    execution_authorized: bool = False


def plan_environment(context: SafetyContext, root, *, cwd: Path,
                     inherited: Mapping[str, str] | None = None,
                     unset: tuple[str, ...] = (), benchmark: bool = False) -> EnvironmentSpec:
    """Authorize modeled locations using B2, without allocation or env mutation.

    Caller-supplied environment is copied. Cache overrides take precedence over
    inherited external values as required by the architecture. Other library
    caches and native same-directory staging require future G-WRITE observation.
    Do not log inherited/environment mappings wholesale: they may hold secrets.
    """
    if set(unset) - {'PYTHONPATH', 'PYTHONHOME', 'PYTHONPYCACHEPREFIX'}:
        raise Refusal('invalid_environment', 'Only interpreter path overrides may be explicitly unset')
    authorized_root = context.check_root(root, cwd=cwd)
    locations = tuple(context.destination(authorized_root, authorized_root/'.cli'/name,
                                        cwd=cwd, kind='directory', fresh=False)
                      for name in ('tmp', 'matplotlib', 'cwd'))
    temporary, matplotlib, execution_cwd = locations
    original = dict(os.environ if inherited is None else inherited)
    if not all(isinstance(key, str) and isinstance(value, str) and key and '=' not in key
               and '\0' not in key + value for key, value in original.items()):
        raise Refusal('invalid_environment', 'Environment requires valid string keys and values')
    overrides = {'TMPDIR': str(temporary), 'TMP': str(temporary), 'TEMP': str(temporary),
                 'MPLCONFIGDIR': str(matplotlib), 'MPLBACKEND': 'Agg',
                 'PYTHONDONTWRITEBYTECODE': '1'}
    if benchmark:
        overrides['LC_ALL'] = 'C'
    effective = {key: value for key, value in original.items() if key not in unset}
    effective.update(overrides)
    inputs = tuple(item.path for item in context.protected)
    return EnvironmentSpec(MappingProxyType(original), MappingProxyType(overrides),
                           tuple(unset), MappingProxyType(effective), locations, inputs,
                           execution_cwd)
