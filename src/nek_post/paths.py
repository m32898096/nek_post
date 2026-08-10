"""Centralized filesystem paths for the Nek5000 post-processing project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from nek_post.config import load_yaml


DEFAULT_PATHS_FILE = Path(__file__).resolve().parents[2] / "config" / "paths.yaml"
CASE_LABEL_ALIASES = MappingProxyType(
    {
        "GC3450_N5": "N5",
        "GC3450_N7": "N7",
        "GC3450_N9": "N9",
        "GC3450_N11": "N11",
    }
)


class ProjectPathsConfigError(ValueError):
    """Raised when the project path configuration is missing required values."""


def _required(mapping: Mapping[str, Any], key: str, context: str = "path configuration") -> Any:
    try:
        value = mapping[key]
    except KeyError as exc:
        raise ProjectPathsConfigError(f"Missing required key {key!r} in {context}.") from exc
    if value is None or value == "":
        raise ProjectPathsConfigError(f"Required key {key!r} in {context} must not be empty.")
    return value


@dataclass(frozen=True)
class ProjectPaths:
    """Immutable project path configuration plus established derived locations."""

    data_root: Path
    case_dirs: Mapping[str, Path]
    postproc_root: Path
    results_root: Path
    cantero_fig5a_re3450_csv: Path
    cantero_fig5a_re8950_csv: Path

    def __post_init__(self) -> None:
        converted_case_dirs = MappingProxyType(
            {str(label): Path(directory) for label, directory in self.case_dirs.items()}
        )
        object.__setattr__(self, "data_root", Path(self.data_root))
        object.__setattr__(self, "case_dirs", converted_case_dirs)
        object.__setattr__(self, "postproc_root", Path(self.postproc_root))
        object.__setattr__(self, "results_root", Path(self.results_root))
        object.__setattr__(self, "cantero_fig5a_re3450_csv", Path(self.cantero_fig5a_re3450_csv))
        object.__setattr__(self, "cantero_fig5a_re8950_csv", Path(self.cantero_fig5a_re8950_csv))

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> ProjectPaths:
        """Build project paths from the parsed contents of ``paths.yaml``."""
        case_dirs = _required(config, "case_dirs")
        if not isinstance(case_dirs, Mapping):
            raise ProjectPathsConfigError("Required key 'case_dirs' must be a mapping.")

        paper_data = _required(config, "paper_data")
        if not isinstance(paper_data, Mapping):
            raise ProjectPathsConfigError("Required key 'paper_data' must be a mapping.")

        return cls(
            data_root=Path(_required(config, "data_root")),
            case_dirs={str(label): Path(directory) for label, directory in case_dirs.items()},
            postproc_root=Path(_required(config, "postproc_root")),
            results_root=Path(_required(config, "results_root")),
            cantero_fig5a_re3450_csv=Path(
                _required(paper_data, "cantero_fig5a_re3450_csv", "paper_data")
            ),
            cantero_fig5a_re8950_csv=Path(
                _required(paper_data, "cantero_fig5a_re8950_csv", "paper_data")
            ),
        )

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_PATHS_FILE) -> ProjectPaths:
        """Load project paths from a YAML configuration file."""
        return cls.from_mapping(load_yaml(path))

    def case_dir(self, label: str) -> Path:
        """Return a configured case directory, rejecting unknown case labels."""
        resolved_label = CASE_LABEL_ALIASES.get(label, label)
        try:
            return self.case_dirs[resolved_label]
        except KeyError as exc:
            available = ", ".join(sorted(self.case_dirs))
            raise ValueError(f"Unknown case {label!r}. Available cases: {available}") from exc

    @property
    def logs_dir(self) -> Path:
        return self.postproc_root / "logs"

    @property
    def slices_dir(self) -> Path:
        return self.postproc_root / "slices"

    @property
    def interpolated_dir(self) -> Path:
        return self.postproc_root / "interpolated"

    @property
    def tables_dir(self) -> Path:
        return self.results_root / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.results_root / "figures"

    @property
    def reports_dir(self) -> Path:
        return self.results_root / "reports"

    @property
    def energy_budget_closure_dir(self) -> Path:
        return self.results_root / "energy_budget_closure"

    @property
    def front_kinematics_dir(self) -> Path:
        return self.results_root / "front_kinematics"

    @property
    def front_detection_dir(self) -> Path:
        return self.results_root / "front_detection"

    @property
    def front_detection_cache_dir(self) -> Path:
        return self.postproc_root / "front_detection_cache"

    @property
    def fig5a_paper_overlay_dir(self) -> Path:
        return self.results_root / "fig5a_paper_overlay"

    @property
    def combined_xt_overlay_dir(self) -> Path:
        return self.results_root / "combined_xt_overlay"

    @property
    def gll_directional_integrals_dir(self) -> Path:
        """Root directory for single-snapshot directional GLL artifacts."""
        return self.results_root / "gll_directional_integrals"

    @property
    def cantero_equivalent_height_dir(self) -> Path:
        """Root directory for Cantero-equivalent-height artifacts."""
        return self.results_root / "cantero_equivalent_height"

    @property
    def cantero_mean_front_dir(self) -> Path:
        """Root directory for Cantero mean-front CSV artifacts."""
        return self.results_root / "cantero_mean_front"

    @property
    def cantero_front_reconstruction_dir(self) -> Path:
        """Root directory for Cantero reconstructed-front artifacts."""
        return self.results_root / "cantero_front_reconstruction"


def load_project_paths(path: str | Path = DEFAULT_PATHS_FILE) -> ProjectPaths:
    """Load the project's fixed and derived filesystem paths."""
    return ProjectPaths.from_yaml(path)
