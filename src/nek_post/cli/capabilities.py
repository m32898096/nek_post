"""Passive prerequisite discovery for future adapters, independent of CLI startup.

Presence never proves importability, ABI/version compatibility, tool behavior,
filesystem semantics, scientific correctness or permission to execute.
"""

from dataclasses import dataclass, replace
from importlib.util import find_spec
import shutil
import sys
from types import MappingProxyType


@dataclass(frozen=True)
class Requirement:
    identity: str
    category: str
    probe: str
    required: bool = True
    platforms: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapabilityResult:
    requirement: Requirement
    status: str
    diagnostic: str
    evidence: str = ''
    execution_authorized: bool = False


# These are architecture profile inventories. Branch-specific qualification and
# complete runtime support remain deferred until each route is migrated.
PROFILES = MappingProxyType({
    'core': ('numpy', 'scipy', 'yaml', 'pymech'),
    'plot': ('numpy', 'scipy', 'yaml', 'pymech', 'matplotlib', 'PIL'),
    'vtk': ('numpy', 'scipy', 'yaml', 'pymech', 'vtk'),
    'volume': ('numpy', 'scipy', 'yaml', 'pymech', 'vtk', 'pyvista'),
    'poppler': ('pdftoppm',),
    'time': ('/usr/bin/time',),
})

ROUTE_PROFILES = MappingProxyType({
    'inspect files': 'core',
    'inspect nek-file': 'core',
    'p-refinement extract-slice': 'core',
    'p-refinement compare-fields': 'core',
    'p-refinement plot-summary': 'plot',
    'p-refinement run-t19p5': 'plot',
    'diagnostics velocity-stripes': 'plot',
    'p-refinement run-multitime': 'plot',
    'p-refinement collect-errors': 'core',
    'p-refinement plot-errors': 'plot',
    'p-refinement plot-overlays': 'plot',
    'p-refinement summarize-errors': 'core',
    'energy closure': 'plot*',
    'front history kinematics': 'plot*',
    'front history paper-overlay': 'plot*',
    'front history processed-overlay': 'plot*',
    'front concentration detect': 'plot*',
    'front concentration paper-overlay': 'plot*',
    'front concentration reconstructed-overlay': 'plot*',
    'leading-edge extract': 'core',
    'leading-edge plot': 'plot',
    'leading-edge benchmark': 'core+time',
    'integrate gll-field': 'core',
    'cantero equivalent-height': 'core',
    'cantero mean-front': 'core',
    'cantero reconstruct-front': 'plot*',
    'cantero overlay-re3450': 'plot*',
    'h-refinement inventory': 'core',
    'h-refinement extract-slice': 'core',
    'h-refinement compare-fields': 'plot',
    'h-refinement convergence': 'plot',
    'h-refinement leading-edge compare': 'plot',
    'h-refinement leading-edge plot-refined': 'plot',
    'lambda-ci native': 'vtk',
    'lambda-ci tensor-slices': 'plot+vtk*',
    'lambda-ci tensor-slice-series': 'plot+vtk+poppler',
    'omega-x native': 'vtk',
    'omega-x validate-series': 'vtk',
    'omega-x publish-series-pvd': 'vtk',
    'omega-x saved-scalar-slice': 'plot+vtk+poppler',
    'omega-x saved-scalar-series': 'plot+vtk+poppler',
    'vortex export': 'volume',
    'vortex export-series': 'volume',
    'diagnostics vtu-rewrite': 'volume',
    'results inventory': 'core',
    'results catalog build': 'core',
    'results navigation preview': 'core',
    'results navigation publish': 'core',
    'leading-edge plot-refined': 'plot',
})


def requirements_for(command: str) -> tuple[Requirement, ...]:
    """Return conservative architecture profiles without importing a handler.

    Starred profiles retain eager dependencies. No-plots/export reductions need
    future branch-specific characterization. A profile is not an execution gate
    closure, including for metadata-only operations.
    """
    profiles = ROUTE_PROFILES[command].replace('*', '').split('+')
    identities = dict.fromkeys(item for profile in profiles for item in PROFILES[profile])
    requirements = [Requirement('python>=3.12', 'stdlib', 'python')]
    for identity in identities:
        category = 'optional_scientific'
        if identity in {'matplotlib', 'PIL', 'vtk', 'pyvista'}:
            category = 'optional_visualization'
        if identity == 'yaml':
            category = 'package_dependency'
        probe = 'module'
        if identity in {'pdftoppm', '/usr/bin/time'}:
            category, probe = 'external_tool', 'executable'
        requirements.append(Requirement(identity, category, probe))
    # Tensor-slice ParaView export uses the PVD lock. Omega audit/publication
    # imports the native workflow and its locking modules without publishing frames.
    if command in {'lambda-ci tensor-slices', 'omega-x validate-series',
                   'omega-x publish-series-pvd'}:
        requirements.append(Requirement('fcntl', 'platform', 'module'))
    # These delegates transitively use lambda_ci_series._publish_directory.
    if command in {'lambda-ci tensor-slice-series', 'omega-x native',
                   'omega-x saved-scalar-slice', 'omega-x saved-scalar-series',
                   'vortex export', 'vortex export-series', 'diagnostics vtu-rewrite'}:
        requirements.extend((Requirement('fcntl', 'platform', 'module'),
                             Requirement('renameat2(RENAME_NOREPLACE)', 'native_filesystem',
                                         'deferred', platforms=('linux',))))
    # Direct resource use and imports through the native lambda/omega workflows.
    if command in {'lambda-ci native', 'lambda-ci tensor-slices',
                   'lambda-ci tensor-slice-series', 'omega-x native',
                   'omega-x validate-series', 'omega-x publish-series-pvd',
                   'omega-x saved-scalar-slice', 'omega-x saved-scalar-series',
                   'vortex export', 'vortex export-series', 'diagnostics vtu-rewrite'}:
        requirements.append(Requirement('resource', 'platform', 'module'))
    # Native platform/filesystem checks are route-specific, never inferred from
    # import presence. Selected routes need concrete native checks in Phase 4C.
    requirements.append(Requirement('native-runtime-and-G-WRITE', 'workflow', 'deferred'))
    return tuple(requirements)


def check_requirement(requirement: Requirement, *, platform=None, search_path=None) -> CapabilityResult:
    """Passively discover a top-level module or executable; launch nothing.

    Optional means optional for the requested operation. Scientific packages are
    optional for base CLI startup but required when selected by a route profile.
    Version ranges for scientific dependencies remain an architecture deferral.
    """
    identity = requirement.identity
    current_platform = sys.platform if platform is None else platform
    if requirement.platforms and current_platform not in requirement.platforms:
        return CapabilityResult(requirement, 'unsupported_platform',
                                f'{identity} requires {requirement.platforms}; observed {current_platform}')
    try:
        if requirement.probe == 'python':
            status = 'available' if sys.version_info >= (3, 12) else 'unsupported_version'
            return CapabilityResult(requirement, status, f'{identity}; interpreter {sys.version.split()[0]}')
        if requirement.probe == 'deferred':
            return CapabilityResult(requirement, 'deferred',
                                    f'{identity} requires route-specific runtime/filesystem acceptance')
        if requirement.probe == 'module':
            if not identity.isidentifier():
                return CapabilityResult(requirement, 'indeterminate', 'Only top-level module discovery is supported')
            present = find_spec(identity) is not None
            evidence = 'module specification present' if present else 'module specification absent'
        elif requirement.probe == 'executable':
            executable = shutil.which(identity, path=search_path)
            present = executable is not None
            evidence = executable or 'executable absent'
            if present:
                return CapabilityResult(requirement, 'indeterminate',
                                        f'{identity} is present; tool identity/version/behavior unverified', evidence)
        else:
            return CapabilityResult(requirement, 'indeterminate', f'Unsupported probe {requirement.probe}')
    except (ImportError, AttributeError, OSError, ValueError) as error:
        return CapabilityResult(requirement, 'indeterminate', f'{identity} discovery failed: {error}')
    status = 'available' if present else ('missing_required' if requirement.required else 'missing_optional')
    return CapabilityResult(requirement, status,
                            f'{identity}: {evidence}; import/version/runtime compatibility unverified', evidence)


def check_prerequisites(command: str) -> tuple[CapabilityResult, ...]:
    """Internal read-only API. Call after B2 authorization, before allocation.

    This reports discovery only and cannot enable a route. Current CLI execution
    stays unavailable, so startup/planning never invokes these checks implicitly.
    """
    return tuple(replace(result, diagnostic=f'{command}: {result.diagnostic}')
                 for result in map(check_requirement, requirements_for(command)))
