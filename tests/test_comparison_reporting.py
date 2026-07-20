from pathlib import Path

from nek_post.comparison_reporting import (
    build_concentration_summary,
    build_pressure_summary,
    build_velocity_summary,
)


GRID_METADATA = {"nx": 3, "nz": 2, "xmin": 0.0, "xmax": 2.0, "zmin": 0.0, "zmax": 1.0}


def test_complete_concentration_summary_with_warning_and_custom_name() -> None:
    summary = build_concentration_summary(
        reference_case="N11",
        comparison_set_name=None,
        target_time=None,
        case_indices={"N5": 79, "N11": 40},
        case_times={"N5": 19.48, "N11": 19.5},
        time_differences={"N5": 0.02, "N11": 0.0},
        grid_metadata=GRID_METADATA,
        valid_point_count=5,
        total_grid_point_count=6,
        error_table=Path("/tmp/concentration.csv"),
        front_table=Path("/tmp/front.csv"),
        error_rows=[{"case": "N5", "relative_L2_C": 0.125}],
        orders={"N5": 5, "N11": 11},
        warnings=["N5 time warning"],
    )

    assert summary == (
        "Nek5000 polynomial-order concentration comparison\n"
        "\n"
        "Comparison set: custom\n"
        "Target time: None\n"
        "Reference case: N11\n"
        "Grid size: 3 x 2\n"
        "Common x domain: 0.0 to 2.0\n"
        "Common z domain: 0.0 to 1.0\n"
        "Valid point count: 5 / 6\n"
        "Error table: /tmp/concentration.csv\n"
        "Front position table: /tmp/front.csv\n"
        "\n"
        "Case time alignment:\n"
        "  N5, order=5, index=79, time=19.48, time_difference=0.02, relative_L2_C=0.125\n"
        "  N11, order=11, index=40, time=19.5, time_difference=0.0, reference\n"
        "\n"
        "Warnings:\n"
        "  N5 time warning"
    )
    assert "N11, order=11" in summary
    assert "N11, order=11, index=40, time=19.5, time_difference=0.0, reference" in summary
    assert summary.index("  N5, order=5") < summary.index("  N11, order=11")


def test_complete_velocity_summary() -> None:
    summary = build_velocity_summary(
        comparison_set_name="t19p5",
        reference_case="N11",
        grid_metadata=GRID_METADATA,
        valid_point_count=5,
        total_grid_point_count=6,
        error_table=Path("/tmp/velocity.csv"),
        error_rows=[
            {"case": "N5", "relative_L2_speed": 0.25},
            {"case": "N9", "relative_L2_speed": 0.1},
        ],
    )

    assert summary == (
        "Nek5000 polynomial-order velocity comparison\n"
        "\n"
        "Comparison set: t19p5\n"
        "Field: velocity\n"
        "Reference case: N11\n"
        "Grid size: 3 x 2\n"
        "Valid point count: 5 / 6\n"
        "Velocity error table: /tmp/velocity.csv\n"
        "\n"
        "Speed errors relative to reference:\n"
        "  N5: relative_L2_speed=0.25\n"
        "  N9: relative_L2_speed=0.1"
    )


def test_complete_pressure_summary() -> None:
    summary = build_pressure_summary(
        comparison_set_name="t19p5",
        reference_case="N11",
        grid_metadata=GRID_METADATA,
        valid_point_count=5,
        total_grid_point_count=6,
        error_table=Path("/tmp/pressure.csv"),
        error_rows=[{"case": "N5", "relative_L2_p_prime": 0.375}],
    )

    assert summary == (
        "Nek5000 polynomial-order pressure comparison\n"
        "\n"
        "Comparison set: t19p5\n"
        "Field: pressure\n"
        "Reference case: N11\n"
        "Grid size: 3 x 2\n"
        "Valid point count: 5 / 6\n"
        "Pressure error table: /tmp/pressure.csv\n"
        "\n"
        "Pressure fluctuation errors relative to reference:\n"
        "  N5: relative_L2_p_prime=0.375"
    )
