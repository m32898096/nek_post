"""Plot selected-time overlays from existing interpolated comparison outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from nek_post.config import load_project_config  # noqa: E402
from nek_post.paths import ProjectPaths  # noqa: E402

FIELD_ORDER = ("concentration", "velocity", "pressure")
FIELD_CONFIG = {
    "concentration": {
        "directory": "C",
        "prefix": "interp_C",
        "variable": "C_grid",
        "label": "Concentration C",
        "filename_prefix": "concentration",
    },
    "velocity": {
        "directory": "velocity",
        "prefix": "interp_velocity",
        "variable": "speed_grid",
        "label": "Velocity magnitude |u|",
        "filename_prefix": "velocity",
    },
    "pressure": {
        "directory": "pressure",
        "prefix": "interp_pressure",
        "variable": "p_prime_grid",
        "label": "Pressure fluctuation p'",
        "filename_prefix": "pressure",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot selected-time overlays from interpolated outputs.")
    parser.add_argument("--comparison-sets", help="Comma-separated comparison sets, for example t05,t10,t15,t19p5.")
    parser.add_argument(
        "--fields",
        default="concentration,velocity,pressure",
        help="Comma-separated fields to plot.",
    )
    parser.add_argument("--orders", default="5,7,9,11", help="Comma-separated polynomial orders to plot.")
    parser.add_argument("--profile-z", default="0.5", help="Float or comma-separated z values for x profiles.")
    parser.add_argument("--profile-x", default="0.0", help="Float or comma-separated x values for z profiles.")
    parser.add_argument(
        "--concentration-thresholds",
        default="0.01",
        help="Comma-separated concentration contour thresholds.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    parser.add_argument("--output-dir", help="Output root for overlay figures.")
    return parser.parse_args()


def _parse_csv_list(raw: str | None, *, option_name: str) -> list[str]:
    if raw is None:
        return []
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError(f"{option_name} must include at least one value.")
    return values


def _parse_float_list(raw: str, *, option_name: str) -> list[float]:
    values = _parse_csv_list(raw, option_name=option_name)
    parsed: list[float] = []
    for value in values:
        try:
            parsed.append(float(value))
        except ValueError as exc:
            raise ValueError(f"Invalid float {value!r} in {option_name}.") from exc
    return parsed


def _parse_int_list(raw: str, *, option_name: str) -> list[int]:
    values = _parse_csv_list(raw, option_name=option_name)
    parsed: list[int] = []
    for value in values:
        try:
            parsed.append(int(value))
        except ValueError as exc:
            raise ValueError(f"Invalid integer {value!r} in {option_name}.") from exc
    return parsed


def _parse_fields(raw: str) -> list[str]:
    fields = _parse_csv_list(raw, option_name="--fields")
    unknown = [field for field in fields if field not in FIELD_CONFIG]
    if unknown:
        raise ValueError(f"Unknown field(s): {', '.join(unknown)}. Allowed fields: {', '.join(FIELD_ORDER)}")
    return fields


def _default_comparison_sets(config: dict) -> list[str]:
    configured = config["cases"].get("multitime_comparison_sets")
    if configured:
        return [str(name) for name in configured]
    return ["t19p5"]


def _comparison_set_names(config: dict, raw: str | None) -> list[str]:
    names = _parse_csv_list(raw, option_name="--comparison-sets") if raw else _default_comparison_sets(config)
    comparison_sets = config["cases"].get("comparison_sets", {})
    unknown = [name for name in names if name not in comparison_sets]
    if unknown:
        available = ", ".join(sorted(comparison_sets)) or "none"
        raise ValueError(f"Unknown comparison set(s): {', '.join(unknown)}. Available sets: {available}")
    return names


def _output_root(paths: ProjectPaths, raw: str | None) -> Path:
    if raw:
        return Path(raw)
    return paths.figures_dir / "overlays"


def _format_value(value: float) -> str:
    if float(value).is_integer():
        raw = f"{value:.1f}"
    else:
        raw = f"{value:g}"
    text = raw.replace("-", "m").replace(".", "p")
    return text


def _case_for_order(config: dict, order: int) -> str | None:
    for case, configured_order in config["cases"]["orders"].items():
        if int(configured_order) == order:
            return str(case)
    return None


def _requested_cases(config: dict, orders: list[int], comparison_set: dict) -> list[str]:
    case_indices = comparison_set["case_indices"]
    cases: list[str] = []
    for order in orders:
        case = _case_for_order(config, order)
        if case is None:
            print(f"WARNING: No configured case for order N{order}; skipping.")
            continue
        if case not in case_indices:
            print(f"WARNING: Case {case} for order N{order} is not in this comparison set; skipping.")
            continue
        cases.append(case)
    return cases


def _expected_interpolated_path(paths: ProjectPaths, field: str, case: str, index: int) -> Path:
    field_config = FIELD_CONFIG[field]
    return (
        paths.interpolated_dir
        / str(field_config["directory"])
        / f"{field_config['prefix']}_{case}_f{index:05d}.npz"
    )


def _metadata_scalar(data: np.lib.npyio.NpzFile, key: str) -> object:
    if key not in data.files:
        return None
    value = data[key]
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _metadata_matches(path: Path, case: str, index: int, comparison_set: str) -> bool:
    try:
        with np.load(path) as data:
            file_case = _metadata_scalar(data, "case")
            file_index = _metadata_scalar(data, "index")
            file_set = _metadata_scalar(data, "comparison_set")
    except Exception:
        return False

    if str(file_case) != str(case):
        return False
    try:
        if int(file_index) != int(index):
            return False
    except (TypeError, ValueError):
        return False
    return file_set in (None, "", comparison_set)


def _find_interpolated_path(paths: ProjectPaths, field: str, comparison_set: str, case: str, index: int) -> Path:
    expected = _expected_interpolated_path(paths, field, case, index)
    if expected.exists():
        return expected

    field_config = FIELD_CONFIG[field]
    directory = paths.interpolated_dir / str(field_config["directory"])
    for candidate in sorted(directory.glob("*.npz")):
        if _metadata_matches(candidate, case, index, comparison_set):
            return candidate

    raise FileNotFoundError(
        "\n".join(
            [
                f"Missing interpolated file for comparison_set={comparison_set}, field={field}, case={case}, index={index}.",
                f"Expected path: {expected}",
                "Run first:",
                f"  python scripts/03_compare_poly_orders.py --comparison-set {comparison_set} --field {field} --overwrite",
            ]
        )
    )


def _load_grid(path: Path, field: str) -> dict[str, np.ndarray]:
    expected_var = str(FIELD_CONFIG[field]["variable"])
    with np.load(path) as data:
        missing = [name for name in ("Xi", "Zi", expected_var) if name not in data.files]
        if missing:
            raise KeyError(
                "\n".join(
                    [
                        f"{path} is missing expected variable(s): {', '.join(missing)}",
                        f"Expected field variable: {expected_var}",
                        f"Available npz keys: {', '.join(data.files)}",
                    ]
                )
            )
        return {
            "Xi": np.asarray(data["Xi"], dtype=float),
            "Zi": np.asarray(data["Zi"], dtype=float),
            "values": np.asarray(data[expected_var], dtype=float),
        }


def _load_field_grids(
    paths: ProjectPaths,
    comparison_set_name: str,
    comparison_set: dict,
    field: str,
    cases: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    grids: dict[str, dict[str, np.ndarray]] = {}
    for case in cases:
        index = int(comparison_set["case_indices"][case])
        path = _find_interpolated_path(paths, field, comparison_set_name, case, index)
        grids[case] = _load_grid(path, field)
    return grids


def _warn_if_sparse_profile(case: str, values: np.ndarray, description: str) -> None:
    finite_count = int(np.count_nonzero(np.isfinite(values)))
    if finite_count == 0:
        print(f"WARNING: {description} for {case} has no finite values.")
        return
    nan_fraction = 1.0 - finite_count / values.size
    if nan_fraction > 0.25:
        print(f"WARNING: {description} for {case} is {nan_fraction:.1%} non-finite.")


def _plot_concentration_contour(
    grids: dict[str, dict[str, np.ndarray]],
    threshold: float,
    comparison_set: str,
    output_dir: Path,
    dpi: int,
) -> Path:
    path = output_dir / f"C_contour_overlay_{comparison_set}_C{_format_value(threshold)}.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    handles = []
    for case, grid in grids.items():
        values = grid["values"]
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0 or threshold < np.min(finite_values) or threshold > np.max(finite_values):
            print(f"WARNING: C={threshold:g} is outside finite range for {case}; contour may be absent.")
        contour = ax.contour(grid["Xi"], grid["Zi"], values, levels=[threshold], linewidths=1.5)
        contour_handles, _ = contour.legend_elements()
        if contour_handles:
            handles.append((contour_handles[0], case))
    if handles:
        ax.legend([handle for handle, _ in handles], [case for _, case in handles])
    ax.set_title(f"Concentration contour overlay, C = {threshold:g}, {comparison_set}")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _plot_x_profile(
    grids: dict[str, dict[str, np.ndarray]],
    field: str,
    z_value: float,
    comparison_set: str,
    output_dir: Path,
    dpi: int,
) -> Path:
    path = output_dir / f"{FIELD_CONFIG[field]['filename_prefix']}_x_profile_{comparison_set}_z{_format_value(z_value)}.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    for case, grid in grids.items():
        zi = grid["Zi"]
        row = int(np.nanargmin(np.abs(zi[:, 0] - z_value)))
        x_values = grid["Xi"][row, :]
        q_values = grid["values"][row, :]
        _warn_if_sparse_profile(case, q_values, f"{field} x-profile at z={z_value:g}")
        finite = np.isfinite(x_values) & np.isfinite(q_values)
        ax.plot(x_values[finite], q_values[finite], marker=".", markersize=2, linewidth=1.2, label=case)
    ax.set_title(f"{FIELD_CONFIG[field]['label']} x-profile overlay, z = {z_value:g}, {comparison_set}")
    ax.set_xlabel("x")
    ax.set_ylabel(str(FIELD_CONFIG[field]["label"]))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _plot_z_profile(
    grids: dict[str, dict[str, np.ndarray]],
    field: str,
    x_value: float,
    comparison_set: str,
    output_dir: Path,
    dpi: int,
) -> Path:
    path = output_dir / f"{FIELD_CONFIG[field]['filename_prefix']}_z_profile_{comparison_set}_x{_format_value(x_value)}.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    for case, grid in grids.items():
        xi = grid["Xi"]
        column = int(np.nanargmin(np.abs(xi[0, :] - x_value)))
        z_values = grid["Zi"][:, column]
        q_values = grid["values"][:, column]
        _warn_if_sparse_profile(case, q_values, f"{field} z-profile at x={x_value:g}")
        finite = np.isfinite(z_values) & np.isfinite(q_values)
        ax.plot(z_values[finite], q_values[finite], marker=".", markersize=2, linewidth=1.2, label=case)
    ax.set_title(f"{FIELD_CONFIG[field]['label']} z-profile overlay, x = {x_value:g}, {comparison_set}")
    ax.set_xlabel("z")
    ax.set_ylabel(str(FIELD_CONFIG[field]["label"]))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def main() -> None:
    """Generate overlays from already interpolated grids."""
    args = _parse_args()
    config = load_project_config(
        REPO_ROOT / "config" / "paths.yaml",
        REPO_ROOT / "config" / "cases.yaml",
    )
    paths = ProjectPaths.from_mapping(config["paths"])

    try:
        comparison_set_names = _comparison_set_names(config, args.comparison_sets)
        fields = _parse_fields(args.fields)
        orders = _parse_int_list(args.orders, option_name="--orders")
        profile_z_values = _parse_float_list(args.profile_z, option_name="--profile-z")
        profile_x_values = _parse_float_list(args.profile_x, option_name="--profile-x")
        concentration_thresholds = _parse_float_list(
            args.concentration_thresholds,
            option_name="--concentration-thresholds",
        )
        output_root = _output_root(paths, args.output_dir)
        comparison_sets = config["cases"]["comparison_sets"]

        saved_paths: list[Path] = []
        for comparison_set_name in comparison_set_names:
            comparison_set = comparison_sets[comparison_set_name]
            cases = _requested_cases(config, orders, comparison_set)
            output_dir = output_root / comparison_set_name
            output_dir.mkdir(parents=True, exist_ok=True)

            for field in fields:
                grids = _load_field_grids(paths, comparison_set_name, comparison_set, field, cases)
                if field == "concentration":
                    for threshold in concentration_thresholds:
                        saved_paths.append(
                            _plot_concentration_contour(grids, threshold, comparison_set_name, output_dir, args.dpi)
                        )
                for z_value in profile_z_values:
                    saved_paths.append(_plot_x_profile(grids, field, z_value, comparison_set_name, output_dir, args.dpi))
                for x_value in profile_x_values:
                    saved_paths.append(_plot_z_profile(grids, field, x_value, comparison_set_name, output_dir, args.dpi))

        print(f"Comparison sets: {','.join(comparison_set_names)}")
        print(f"Fields: {','.join(fields)}")
        print(f"Output directory: {output_root}")
        print(f"Figures written: {len(saved_paths)}")
        print("Generated figures:")
        for path in saved_paths:
            print(f"  {path}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
