"""Static command metadata from the accepted Phase 4A migration matrix."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    """A discoverable leaf, identified by its complete space-separated route."""

    group: tuple[str, ...]
    leaf: str
    help: str
    status: str = "not_implemented"

    @property
    def command_id(self) -> str:
        """Return the stable command key used by the architecture contracts."""
        return " ".join((*self.group, self.leaf))


GROUP_HELP = {
    ("inspect",): "Inspect configured files and Nek snapshots",
    ("p-refinement",): "P-refinement field comparisons and reporting",
    ("diagnostics",): "Velocity and VTU diagnostics",
    ("energy",): "Energy-budget diagnostics",
    ("front",): "Front histories and concentration-derived fronts",
    ("front", "history"): "External front-history analysis",
    ("front", "concentration"): "Concentration-derived front analysis",
    ("leading-edge",): "Leading-edge extraction, plotting and benchmarks",
    ("integrate",): "Directional field integration",
    ("cantero",): "Cantero equivalent height and mean-front analysis",
    ("h-refinement",): "H-refinement inventory and comparisons",
    ("h-refinement", "leading-edge"): "H-study leading-edge comparisons and plots",
    ("lambda-ci",): "Native and tensor-first lambda-ci workflows",
    ("omega-x",): "Native and saved-scalar omega-x workflows",
    ("vortex",): "Native and reconstructed vortex VTU export",
    ("results",): "Results inventory, catalogs and navigation",
    ("results", "catalog"): "Results catalog creation",
    ("results", "navigation"): "Results navigation preview and publication",
}


COMMANDS = (
    Command(("inspect",), "files", "Configured-file existence inspection"),
    Command(("inspect",), "nek-file", "Single-snapshot metadata probe"),
    Command(("p-refinement",), "extract-slice", "Nearest-midspan sampled slice"),
    Command(("p-refinement",), "compare-fields", "Sampled common-grid C/speed/pressure-prime comparison"),
    Command(("p-refinement",), "plot-summary", "Comparison contours/errors/front plots"),
    Command(("p-refinement",), "run-t19p5", "Ordered extraction/comparison/plot pipeline"),
    Command(("diagnostics",), "velocity-stripes", "Sampled velocity stripe/interpolation diagnosis"),
    Command(("p-refinement",), "run-multitime", "Deduplicated multi-time pipeline"),
    Command(("p-refinement",), "collect-errors", "Collect existing per-time error tables"),
    Command(("p-refinement",), "plot-errors", "Error-versus-physical-time reporting"),
    Command(("p-refinement",), "plot-overlays", "Selected-time contours and profiles"),
    Command(("p-refinement",), "summarize-errors", "Teacher-facing summary tables"),
    Command(("energy",), "closure", "Existing P-refinement energy-budget diagnostics"),
    Command(("front", "history"), "kinematics", "External front_simple.dat kinematics"),
    Command(("front", "history"), "paper-overlay", "Raw external front versus digitized paper"),
    Command(("front", "history"), "processed-overlay", "Processed reconstructed external fronts versus paper"),
    Command(("front", "concentration"), "detect", "Concentration-derived front and reference diagnostics"),
    Command(("front", "concentration"), "paper-overlay", "Detected Re8950 front versus matching paper"),
    Command(("front", "concentration"), "reconstructed-overlay", "Two-Reynolds reconstructed automatic fronts"),
    Command(("leading-edge",), "extract", "Spectral uniform or refined-GLL leading-edge extraction"),
    Command(("leading-edge",), "plot", "Render validated saved legacy curves"),
    Command(("leading-edge",), "benchmark", "Isolated worker-count measurement with artifact equality"),
    Command(("integrate",), "gll-field", "One-snapshot directional GLL scalar integral"),
    Command(("cantero",), "equivalent-height", "Cantero z then physical-y composite-GLL preprocessing"),
    Command(("cantero",), "mean-front", "Physical-GLL equivalent-height threshold crossing history"),
    Command(("cantero",), "reconstruct-front", "N7/Re3450 Cantero temporal reconstruction and paper comparison"),
    Command(("cantero",), "overlay-re3450", "Ordered multicase reconstructed Cantero overlay"),
    Command(("h-refinement",), "inventory", "Header/representative-mesh H-study inventory"),
    Command(("h-refinement",), "extract-slice", "Exact physical-y shared-grid H fields"),
    Command(("h-refinement",), "compare-fields", "Header-time selected H field comparison"),
    Command(("h-refinement",), "convergence", "Directional mesh and saved-field convergence diagnostics"),
    Command(("h-refinement", "leading-edge"), "compare", "Common-grid rightmost/Moore extraction and time-aligned comparison"),
    Command(("h-refinement", "leading-edge"), "plot-refined", "Validate case-native ten-node H refined-GLL curves"),
    Command(("lambda-ci",), "native", "Native projected-tensor lambda-ci frame"),
    Command(("lambda-ci",), "tensor-slices", "Tensor-first high-order slices and optional ParaView companions"),
    Command(("lambda-ci",), "tensor-slice-series", "Sequential cached tensor slice-series production"),
    Command(("omega-x",), "native", "Native projected vorticity series"),
    Command(("omega-x",), "validate-series", "Read-only complete existing native production audit"),
    Command(("omega-x",), "publish-series-pvd", "Validate then no-replace native series PVD publication"),
    Command(("omega-x",), "saved-scalar-slice", "P7 reconstruction from saved projected omega scalar"),
    Command(("omega-x",), "saved-scalar-series", "Saved scalar P7 series with controls and full-series PVD"),
    Command(("vortex",), "export", "Native/reconstructed omega and tensor-first lambda VTU"),
    Command(("vortex",), "export-series", "Strict-resume VTU series and actual-time PVD"),
    Command(("diagnostics",), "vtu-rewrite", "Existing-VTU serialization diagnostic"),
    Command(("results",), "inventory", "Bounded provenance-aware catalog discovery"),
    Command(("results", "catalog"), "build", "Write a fresh external inventory catalog"),
    Command(("results", "navigation"), "preview", "Prepare candidate navigation plus fresh catalog"),
    Command(("results", "navigation"), "publish", "Publish reviewed navigation pair only"),
    Command(("leading-edge",), "plot-refined", "N7 saved ten-node refined-GLL evolution plots"),
)
