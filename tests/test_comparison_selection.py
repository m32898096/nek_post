import numpy as np
import pytest

from nek_post.comparison_selection import (
    comparison_label,
    evaluate_time_alignment,
    parse_case_indices,
    resolve_comparison_selection,
)


CASES = ["N5", "N11"]


def _cases_config() -> dict[str, object]:
    return {
        "reference_case": "N11",
        "file_indices": [1, 80],
        "comparison_sets": {
            "t19p5": {
                "target_time": 19.5,
                "reference_case": "N11",
                "case_indices": {"N5": 79, "N11": 40},
            },
            "fallback": {
                "target_time": 10.0,
                "case_indices": {"N5": 39, "N11": 20},
            },
        },
    }


def test_named_comparison_set_resolution() -> None:
    selection = resolve_comparison_selection("t19p5", "N5=1,N11=1", 12, CASES, _cases_config())

    assert dict(selection.case_indices) == {"N5": 79, "N11": 40}
    assert selection.reference_case == "N11"
    assert selection.comparison_set_name == "t19p5"
    assert selection.target_time == 19.5
    assert selection.use_case_indices is True
    assert selection.output_label == "t19p5"


def test_named_comparison_set_uses_global_reference_fallback() -> None:
    selection = resolve_comparison_selection("fallback", None, None, CASES, _cases_config())

    assert selection.reference_case == "N11"
    assert selection.target_time == 10.0


def test_explicit_case_indices_are_parsed_in_configured_order() -> None:
    parsed = parse_case_indices("N11=40,N5=79", CASES)
    selection = resolve_comparison_selection(None, "N11=40,N5=79", 12, CASES, _cases_config())

    assert parsed == {"N5": 79, "N11": 40}
    assert dict(selection.case_indices) == parsed
    assert selection.output_label == "cases_N5f00079_N11f00040"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("N5=79,N9=40", "Unknown case 'N9'"),
        ("N5=79,N5=40", "Duplicate case 'N5'"),
        ("N5=79", "Missing: N11"),
        ("N5=abc,N11=40", "Invalid index 'abc'"),
    ],
)
def test_invalid_explicit_case_indices_are_rejected(raw: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_case_indices(raw, CASES)


def test_single_index_selection() -> None:
    selection = resolve_comparison_selection(None, None, 12, CASES, _cases_config())

    assert dict(selection.case_indices) == {"N5": 12, "N11": 12}
    assert selection.output_label == "f00012"
    assert selection.use_case_indices is False


def test_selection_falls_back_to_final_configured_file_index() -> None:
    selection = resolve_comparison_selection(None, None, None, CASES, _cases_config())

    assert dict(selection.case_indices) == {"N5": 80, "N11": 80}
    assert selection.output_label == "f00080"


def test_comparison_label_preserves_existing_formats() -> None:
    assert comparison_label({"N5": 80, "N11": 80}, False) == "f00080"
    assert comparison_label({"N5": 79, "N11": 40}, True) == "cases_N5f00079_N11f00040"


@pytest.mark.parametrize("config", [{}, {"comparison_sets": {}}])
def test_unknown_or_missing_comparison_set_is_rejected(config: dict[str, object]) -> None:
    config = {"reference_case": "N11", "file_indices": [80], **config}
    with pytest.raises(ValueError, match="Unknown comparison set 'missing'.*Available sets: none"):
        resolve_comparison_selection("missing", None, None, CASES, config)


def test_missing_reference_case_is_rejected() -> None:
    config = _cases_config()
    config["reference_case"] = "N9"

    with pytest.raises(ValueError, match="Reference case 'N9' is not present in cases.orders"):
        resolve_comparison_selection(None, None, 80, CASES, config)


def test_valid_time_alignment_differences() -> None:
    result = evaluate_time_alignment(
        {"N5": {"time": np.array(19.48)}, "N11": {"time": np.array(19.5)}},
        CASES,
        "N11",
        0.05,
    )

    assert result.reference_time == 19.5
    assert result.case_times == {"N5": 19.48, "N11": 19.5}
    assert result.time_differences["N5"] == pytest.approx(0.02)
    assert result.warnings == ()


def test_time_alignment_warns_when_tolerance_is_exceeded() -> None:
    result = evaluate_time_alignment(
        {"N5": {"time": np.array(19.0)}, "N11": {"time": np.array(19.5)}},
        CASES,
        "N11",
        0.05,
    )

    assert result.warnings == ("N5 time differs from N11 by 0.5, exceeding tolerance 0.05.",)


@pytest.mark.parametrize("metadata", [{}, {"time": np.array(np.nan)}])
def test_time_alignment_warns_for_missing_or_nonfinite_metadata(metadata: dict[str, np.ndarray]) -> None:
    result = evaluate_time_alignment(
        {"N5": metadata, "N11": {"time": np.array(19.5)}},
        CASES,
        "N11",
        0.05,
    )

    assert result.warnings == ("Missing or invalid time metadata for N5; time alignment could not be checked.",)
