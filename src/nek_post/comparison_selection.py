"""Comparison selection and time-alignment metadata helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import numpy as np

from nek_post.comparison_io import slice_time


@dataclass(frozen=True)
class ComparisonSelection:
    """Resolved case indices, reference metadata, and output label."""

    case_indices: Mapping[str, int]
    reference_case: str
    comparison_set_name: str | None
    target_time: object
    use_case_indices: bool
    output_label: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_indices", MappingProxyType(dict(self.case_indices)))


@dataclass(frozen=True)
class TimeAlignmentResult:
    """Per-case slice times, reference differences, and alignment warnings."""

    case_times: Mapping[str, float]
    reference_time: float
    time_differences: Mapping[str, float]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_times", MappingProxyType(dict(self.case_times)))
        object.__setattr__(self, "time_differences", MappingProxyType(dict(self.time_differences)))


def comparison_label(case_indices: Mapping[str, int], use_case_indices: bool) -> str:
    """Return the established single-index or per-case comparison label."""
    if not use_case_indices and len(set(case_indices.values())) == 1:
        index = next(iter(case_indices.values()))
        return f"f{index:05d}"

    parts = [f"{case}f{case_indices[case]:05d}" for case in case_indices]
    return "cases_" + "_".join(parts)


def parse_case_indices(raw: str, cases: Sequence[str]) -> dict[str, int]:
    """Parse ``CASE=INDEX`` values and require every configured case."""
    case_indices: dict[str, int] = {}
    for item in raw.split(","):
        if "=" not in item:
            raise ValueError(f"Invalid --case-indices item {item!r}; expected CASE=INDEX.")
        case, index_text = item.split("=", 1)
        case = case.strip()
        index_text = index_text.strip()
        if case not in cases:
            available = ", ".join(cases)
            raise ValueError(f"Unknown case {case!r} in --case-indices. Available cases: {available}")
        if case in case_indices:
            raise ValueError(f"Duplicate case {case!r} in --case-indices.")
        try:
            case_indices[case] = int(index_text)
        except ValueError as exc:
            raise ValueError(f"Invalid index {index_text!r} for case {case!r}.") from exc

    missing_cases = [case for case in cases if case not in case_indices]
    if missing_cases:
        missing = ", ".join(missing_cases)
        raise ValueError(f"--case-indices must include all configured cases. Missing: {missing}")

    return {case: case_indices[case] for case in cases}


def _ordered_case_indices(raw_case_indices: Mapping[str, object], cases: Sequence[str]) -> dict[str, int]:
    missing_cases = [case for case in cases if case not in raw_case_indices]
    if missing_cases:
        missing = ", ".join(missing_cases)
        raise ValueError(f"Case indices must include all configured cases. Missing: {missing}")
    return {case: int(raw_case_indices[case]) for case in cases}


def resolve_comparison_selection(
    comparison_set_name: str | None,
    raw_case_indices: str | None,
    index: int | None,
    cases: Sequence[str],
    cases_config: Mapping[str, object],
) -> ComparisonSelection:
    """Resolve CLI values using comparison-set, per-case, index, then config precedence."""
    selected_name: str | None = None
    target_time: object = None

    if comparison_set_name:
        comparison_sets = cast(Mapping[str, object], cases_config.get("comparison_sets", {}))
        if comparison_set_name not in comparison_sets:
            available = ", ".join(sorted(comparison_sets)) or "none"
            raise ValueError(f"Unknown comparison set {comparison_set_name!r}. Available sets: {available}")

        comparison_set = cast(Mapping[str, object], comparison_sets[comparison_set_name])
        case_indices = _ordered_case_indices(
            cast(Mapping[str, object], comparison_set["case_indices"]),
            cases,
        )
        reference_case = cast(
            str,
            comparison_set.get("reference_case") or cases_config["reference_case"],
        )
        selected_name = comparison_set_name
        target_time = comparison_set.get("target_time")
        use_case_indices = True
        output_label = comparison_set_name
    elif raw_case_indices:
        case_indices = parse_case_indices(raw_case_indices, cases)
        reference_case = cast(str, cases_config["reference_case"])
        use_case_indices = True
        output_label = comparison_label(case_indices, use_case_indices)
    else:
        file_indices = cast(Sequence[int], cases_config["file_indices"])
        selected_index = index if index is not None else file_indices[-1]
        case_indices = {case: selected_index for case in cases}
        reference_case = cast(str, cases_config["reference_case"])
        use_case_indices = False
        output_label = comparison_label(case_indices, use_case_indices)

    if reference_case not in cases:
        raise ValueError(f"Reference case {reference_case!r} is not present in cases.orders.")

    return ComparisonSelection(
        case_indices=case_indices,
        reference_case=reference_case,
        comparison_set_name=selected_name,
        target_time=target_time,
        use_case_indices=use_case_indices,
        output_label=output_label,
    )


def evaluate_time_alignment(
    slice_data_by_case: Mapping[str, Mapping[str, np.ndarray]],
    cases: Sequence[str],
    reference_case: str,
    tolerance: float,
) -> TimeAlignmentResult:
    """Evaluate slice times relative to the reference without enforcing strict mode."""
    case_times = {case: slice_time(slice_data_by_case[case]) for case in cases}
    reference_time = case_times[reference_case]
    time_differences = {case: abs(case_times[case] - reference_time) for case in cases}
    warnings: list[str] = []
    for case in cases:
        time_difference = time_differences[case]
        if not np.isfinite(time_difference):
            warnings.append(f"Missing or invalid time metadata for {case}; time alignment could not be checked.")
        elif time_difference > tolerance:
            warnings.append(
                f"{case} time differs from {reference_case} by {time_difference}, "
                f"exceeding tolerance {tolerance}."
            )

    return TimeAlignmentResult(
        case_times=case_times,
        reference_time=reference_time,
        time_differences=time_differences,
        warnings=tuple(warnings),
    )
