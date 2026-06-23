"""Diagnose velocity stripe artifacts in extracted x-z slice data."""

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
from nek_post.interpolation import interpolate_to_grid  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose velocity stripe artifacts from extracted slice files.")
    parser.add_argument("--case", default="N11", help="Case name, for example N11.")
    parser.add_argument("--index", type=int, default=40, help="Slice index, for example 40.")
    parser.add_argument("--comparison-set", default="t19p5", help="Comparison set name used for output directory.")
    parser.add_argument("--method", default="linear", help="Interpolation method passed to scipy.interpolate.griddata.")
    parser.add_argument("--round-decimals", type=int, default=10, help="Decimals for near-duplicate x-z counting.")
    parser.add_argument("--y-round-decimals", type=int, default=10, help="Decimals for unique y counting.")
    parser.add_argument("--nx", type=int, help="Diagnostic grid x size. Defaults to config grid nx.")
    parser.add_argument("--nz", type=int, help="Diagnostic grid z size. Defaults to config grid nz.")
    parser.add_argument(
        "--scatter-max-points",
        type=int,
        default=250_000,
        help="Maximum points shown in raw scatter plots; deterministic downsampling is used when needed.",
    )
    parser.add_argument("--scatter-seed", type=int, default=20260623, help="Seed for raw scatter downsampling.")
    return parser.parse_args()


def _slice_path(config: dict, case: str, index: int) -> Path:
    return Path(config["paths"]["postproc_root"]) / "slices" / case / f"slice_{case}_f{index:05d}.npz"


def _diagnostic_dir(config: dict, comparison_set: str) -> Path:
    return Path(config["paths"]["results_root"]) / "figures" / "velocity" / comparison_set / "diagnostics"


def _load_slice(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Slice file not found: {path}")

    with np.load(path) as data:
        missing = [name for name in ("x", "y", "z", "u", "v", "w") if name not in data.files]
        if missing:
            missing_text = ", ".join(missing)
            raise KeyError(f"{path} is missing required key(s): {missing_text}")
        return {name: data[name] for name in data.files}


def _finite_min_max(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.min(finite)), float(np.max(finite))


def _format_min_max(name: str, values: np.ndarray) -> str:
    min_value, max_value = _finite_min_max(values)
    return f"{name} min/max: {min_value:.16g}, {max_value:.16g}"


def _midspan_value(slice_data: dict[str, np.ndarray]) -> float:
    if "y0" in slice_data:
        y0 = np.asarray(slice_data["y0"])
        if y0.shape == ():
            return float(y0.item())
    y = np.asarray(slice_data["y"], dtype=float)
    return float(0.5 * (np.nanmin(y) + np.nanmax(y)))


def _duplicate_counts(x: np.ndarray, z: np.ndarray, round_decimals: int) -> dict[str, int]:
    exact_pairs = np.column_stack((x, z))
    _, exact_counts = np.unique(exact_pairs, axis=0, return_counts=True)
    exact_unique_count = exact_counts.size

    rounded_pairs = np.column_stack((np.round(x, round_decimals), np.round(z, round_decimals)))
    _, rounded_counts = np.unique(rounded_pairs, axis=0, return_counts=True)
    rounded_unique_count = rounded_counts.size

    point_count = int(x.size)
    return {
        "point_count": point_count,
        "exact_unique_pair_count": int(exact_unique_count),
        "exact_duplicate_point_count": point_count - int(exact_unique_count),
        "exact_duplicate_pair_count": int(np.count_nonzero(exact_counts > 1)),
        "exact_max_pair_multiplicity": int(np.max(exact_counts)),
        "rounded_unique_pair_count": int(rounded_unique_count),
        "rounded_duplicate_point_count": point_count - int(rounded_unique_count),
        "rounded_duplicate_pair_count": int(np.count_nonzero(rounded_counts > 1)),
        "rounded_max_pair_multiplicity": int(np.max(rounded_counts)),
    }


def average_projected_duplicates(
    x: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
    round_decimals: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average values sharing the same projected x-z location.

    If ``round_decimals`` is set, x and z are grouped after rounding. The
    returned x and z coordinates are averaged original coordinates for each
    group, and counts contains the number of raw points in each group.
    """
    x_arr = np.asarray(x, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    if x_arr.shape != z_arr.shape or x_arr.shape != values_arr.shape:
        raise ValueError("x, z, and values must have matching shapes.")

    if round_decimals is None:
        keys = np.column_stack((x_arr, z_arr))
    else:
        keys = np.column_stack((np.round(x_arr, round_decimals), np.round(z_arr, round_decimals)))

    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    x_sum = np.bincount(inverse, weights=x_arr)
    z_sum = np.bincount(inverse, weights=z_arr)
    value_sum = np.bincount(inverse, weights=values_arr)

    return x_sum / counts, z_sum / counts, value_sum / counts, counts


def _grid_from_bounds(x: np.ndarray, z: np.ndarray, nx: int, nz: int) -> tuple[np.ndarray, np.ndarray]:
    xi = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), nx)
    zi = np.linspace(float(np.nanmin(z)), float(np.nanmax(z)), nz)
    return np.meshgrid(xi, zi)


def _downsample_indices(point_count: int, max_points: int, seed: int) -> np.ndarray:
    if point_count <= max_points:
        return np.arange(point_count)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(point_count, size=max_points, replace=False))


def _plot_raw_scatter(
    x: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
    output_path: Path,
    title: str,
    label: str,
    max_points: int,
    seed: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    indices = _downsample_indices(x.size, max_points, seed)

    fig, ax = plt.subplots(figsize=(9, 4))
    scatter = ax.scatter(x[indices], z[indices], c=values[indices], s=1.0, cmap="viridis", linewidths=0)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label(label)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_y_distribution(y: np.ndarray, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y[np.isfinite(y)], bins=80)
    ax.set_xlabel("y")
    ax.set_ylabel("point count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_interpolation_comparison(
    Xi: np.ndarray,
    Zi: np.ndarray,
    speed_from_components: np.ndarray,
    speed_direct: np.ndarray,
    output_path: Path,
    case: str,
    index: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diff = speed_from_components - speed_direct

    finite_speeds = np.concatenate(
        [
            speed_from_components[np.isfinite(speed_from_components)],
            speed_direct[np.isfinite(speed_direct)],
        ]
    )
    if finite_speeds.size:
        speed_vmin = float(np.min(finite_speeds))
        speed_vmax = float(np.max(finite_speeds))
    else:
        speed_vmin = None
        speed_vmax = None

    finite_abs_diff = np.abs(diff[np.isfinite(diff)])
    diff_vmax = float(np.max(finite_abs_diff)) if finite_abs_diff.size else None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    panels = [
        (axes[0], speed_from_components, "sqrt(interp u^2 + interp v^2 + interp w^2)", "speed", speed_vmin, speed_vmax),
        (axes[1], speed_direct, "interp raw speed directly", "speed", speed_vmin, speed_vmax),
        (axes[2], np.abs(diff), "absolute difference", "|difference|", 0.0, diff_vmax),
    ]
    for ax, values, title, label, vmin, vmax in panels:
        contour = ax.contourf(Xi, Zi, values, levels=80, vmin=vmin, vmax=vmax)
        colorbar = fig.colorbar(contour, ax=ax)
        colorbar.set_label(label)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_title(title)
    fig.suptitle(f"{case} f{index:05d} speed interpolation diagnostics")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _print_summary(
    slice_path: Path,
    slice_data: dict[str, np.ndarray],
    speed_raw: np.ndarray,
    y_round_decimals: int,
    round_decimals: int,
    duplicate_counts: dict[str, int],
) -> None:
    x = np.asarray(slice_data["x"], dtype=float)
    y = np.asarray(slice_data["y"], dtype=float)
    z = np.asarray(slice_data["z"], dtype=float)
    y0 = _midspan_value(slice_data)
    y_distance = np.abs(y - y0)
    unique_y_count = int(np.unique(np.round(y, y_round_decimals)).size)

    print(f"Slice file: {slice_path}")
    print(f"Point count: {x.size}")
    print(_format_min_max("x", x))
    print(_format_min_max("y", y))
    print(_format_min_max("z", z))
    print(_format_min_max("u", slice_data["u"]))
    print(_format_min_max("v", slice_data["v"]))
    print(_format_min_max("w", slice_data["w"]))
    print(_format_min_max("speed_raw", speed_raw))
    print(f"Rounded unique y count ({y_round_decimals} decimals): {unique_y_count}")
    print(f"Slice midspan y value: {y0:.16g}")
    print(_format_min_max("|y - y_midspan|", y_distance))
    print(f"Exact projected (x,z) unique pairs: {duplicate_counts['exact_unique_pair_count']}")
    print(f"Exact projected (x,z) duplicate points: {duplicate_counts['exact_duplicate_point_count']}")
    print(f"Exact projected (x,z) duplicate pair groups: {duplicate_counts['exact_duplicate_pair_count']}")
    print(f"Exact projected (x,z) max pair multiplicity: {duplicate_counts['exact_max_pair_multiplicity']}")
    print(
        f"Rounded projected (x,z) unique pairs ({round_decimals} decimals): "
        f"{duplicate_counts['rounded_unique_pair_count']}"
    )
    print(
        f"Rounded projected (x,z) duplicate points ({round_decimals} decimals): "
        f"{duplicate_counts['rounded_duplicate_point_count']}"
    )
    print(
        f"Rounded projected (x,z) duplicate pair groups ({round_decimals} decimals): "
        f"{duplicate_counts['rounded_duplicate_pair_count']}"
    )
    print(
        f"Rounded projected (x,z) max pair multiplicity ({round_decimals} decimals): "
        f"{duplicate_counts['rounded_max_pair_multiplicity']}"
    )


def _print_interpolation_comparison(speed_from_components: np.ndarray, speed_direct: np.ndarray) -> None:
    mask = np.isfinite(speed_from_components) & np.isfinite(speed_direct)
    valid_count = int(np.count_nonzero(mask))
    total_count = int(mask.size)
    print(f"Speed interpolation comparison valid grid points: {valid_count} / {total_count}")
    if valid_count == 0:
        return

    diff = speed_from_components[mask] - speed_direct[mask]
    abs_diff = np.abs(diff)
    denom = float(np.sqrt(np.sum(speed_direct[mask] ** 2)))
    relative_l2 = float(np.sqrt(np.sum(diff**2)) / denom) if denom > 0.0 else float("nan")
    print(f"Component-first vs direct speed abs diff min/max: {float(np.min(abs_diff)):.16g}, {float(np.max(abs_diff)):.16g}")
    print(f"Component-first vs direct speed abs diff mean: {float(np.mean(abs_diff)):.16g}")
    print(f"Component-first vs direct speed relative L2: {relative_l2:.16g}")


def main() -> None:
    args = _parse_args()
    config = load_project_config(
        REPO_ROOT / "config" / "paths.yaml",
        REPO_ROOT / "config" / "cases.yaml",
    )

    slice_path = _slice_path(config, args.case, args.index)
    slice_data = _load_slice(slice_path)

    x = np.asarray(slice_data["x"], dtype=float)
    y = np.asarray(slice_data["y"], dtype=float)
    z = np.asarray(slice_data["z"], dtype=float)
    u = np.asarray(slice_data["u"], dtype=float)
    v = np.asarray(slice_data["v"], dtype=float)
    w = np.asarray(slice_data["w"], dtype=float)
    speed_raw = np.sqrt(u**2 + v**2 + w**2)

    duplicate_counts = _duplicate_counts(x, z, args.round_decimals)
    _print_summary(slice_path, slice_data, speed_raw, args.y_round_decimals, args.round_decimals, duplicate_counts)

    nx = args.nx if args.nx is not None else int(config["cases"]["grid"]["nx"])
    nz = args.nz if args.nz is not None else int(config["cases"]["grid"]["nz"])
    Xi, Zi = _grid_from_bounds(x, z, nx, nz)

    u_grid = interpolate_to_grid(x, z, u, Xi, Zi, method=args.method)
    v_grid = interpolate_to_grid(x, z, v, Xi, Zi, method=args.method)
    w_grid = interpolate_to_grid(x, z, w, Xi, Zi, method=args.method)
    speed_from_components = np.sqrt(u_grid**2 + v_grid**2 + w_grid**2)
    speed_direct = interpolate_to_grid(x, z, speed_raw, Xi, Zi, method=args.method)
    _print_interpolation_comparison(speed_from_components, speed_direct)

    output_dir = _diagnostic_dir(config, args.comparison_set)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = [
        output_dir / f"raw_speed_scatter_{args.case}_f{args.index:05d}.png",
        output_dir / f"raw_u_scatter_{args.case}_f{args.index:05d}.png",
        output_dir / f"raw_w_scatter_{args.case}_f{args.index:05d}.png",
        output_dir / f"y_distribution_{args.case}_f{args.index:05d}.png",
        output_dir / f"compare_speed_interpolation_methods_{args.case}_f{args.index:05d}.png",
    ]

    _plot_raw_scatter(
        x,
        z,
        speed_raw,
        plot_paths[0],
        f"{args.case} f{args.index:05d} raw speed on extracted slice",
        "speed_raw",
        args.scatter_max_points,
        args.scatter_seed,
    )
    _plot_raw_scatter(
        x,
        z,
        u,
        plot_paths[1],
        f"{args.case} f{args.index:05d} raw u on extracted slice",
        "u",
        args.scatter_max_points,
        args.scatter_seed,
    )
    _plot_raw_scatter(
        x,
        z,
        w,
        plot_paths[2],
        f"{args.case} f{args.index:05d} raw w on extracted slice",
        "w",
        args.scatter_max_points,
        args.scatter_seed,
    )
    _plot_y_distribution(y, plot_paths[3], f"{args.case} f{args.index:05d} extracted y distribution")
    _plot_interpolation_comparison(
        Xi,
        Zi,
        speed_from_components,
        speed_direct,
        plot_paths[4],
        args.case,
        args.index,
    )

    print("Generated diagnostic figures:")
    for path in plot_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
