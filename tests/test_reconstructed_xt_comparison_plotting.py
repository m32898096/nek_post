from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest

from nek_post import reconstructed_xt_comparison_plotting as plotting
from nek_post.reconstructed_xt_comparison_plotting import (
    LEGEND_LABELS,
    write_reconstructed_xt_plots,
)


def _reconstructed() -> dict[str, dict[str, np.ndarray]]:
    return {
        "Re3450": {
            "time": np.array([0.0, 1.0, 2.0, 3.0]),
            "x_detected_relative": np.array([0.0, 0.5, 2.0, 3.0]),
            "x_reconstructed_relative": np.array(
                [0.0, -1.0, 2.0, 3.0]
            ),
        },
        "Re8950": {
            "time": np.array([0.0, 2.0, 4.0, 6.0]),
            "x_detected_relative": np.array([0.0, 2.5, 4.0, 7.0]),
            "x_reconstructed_relative": np.array(
                [0.0, 2.0, 4.0, 6.0]
            ),
        },
    }


def _papers() -> dict[str, dict[str, np.ndarray]]:
    return {
        "Re3450": {
            "time": np.array([0.25, 1.25, 2.25]),
            "paper_x": np.array([0.1, 1.1, 2.1]),
        },
        "Re8950": {
            "time": np.array([0.5, 2.5, 4.5]),
            "paper_x": np.array([0.2, 2.2, 4.2]),
        },
    }


def test_writes_exactly_three_required_figures(tmp_path: Path) -> None:
    paths = write_reconstructed_xt_plots(
        output_dir=tmp_path,
        reconstructed_by_reynolds=_reconstructed(),
        papers_by_reynolds=_papers(),
        slump_tmin=1.0,
        slump_tmax=4.0,
        overwrite=False,
    )

    assert [path.name for path in paths] == [
        "Re3450_N7_reconstructed_vs_Cantero_Re3450.png",
        "Re8950_N7_reconstructed_vs_Cantero_Re8950.png",
        "N7_Re3450_Re8950_reconstructed_fourway_overlay.png",
    ]
    assert all(path.is_file() for path in paths)
    assert len(list(tmp_path.glob("*.png"))) == 3
    assert plt.get_fignums() == []


def test_pair_and_fourway_artists_have_real_data_and_required_styles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, dict[str, object]] = {}
    original_save = plotting._save_figure

    def capture(fig, path):
        axis = fig.axes[0]
        captured[path.name] = {
            "xlabel": axis.get_xlabel(),
            "ylabel": axis.get_ylabel(),
            "legend_labels": tuple(
                text.get_text() for text in axis.get_legend().get_texts()
            ),
            "lines": [
                {
                    "label": line.get_label(),
                    "x": np.asarray(line.get_xdata()),
                    "y": np.asarray(line.get_ydata()),
                    "linestyle": line.get_linestyle(),
                    "marker": line.get_marker(),
                    "markersize": line.get_markersize(),
                    "linewidth": line.get_linewidth(),
                    "zorder": line.get_zorder(),
                    "color": line.get_color(),
                }
                for line in axis.lines
            ],
        }
        original_save(fig, path)

    monkeypatch.setattr(plotting, "_save_figure", capture)
    papers = _papers()
    write_reconstructed_xt_plots(
        output_dir=tmp_path,
        reconstructed_by_reynolds=_reconstructed(),
        papers_by_reynolds=papers,
        slump_tmin=1.0,
        slump_tmax=4.0,
        overwrite=False,
    )

    re3450_pair = captured[
        "Re3450_N7_reconstructed_vs_Cantero_Re3450.png"
    ]
    re8950_pair = captured[
        "Re8950_N7_reconstructed_vs_Cantero_Re8950.png"
    ]
    combined = captured[
        "N7_Re3450_Re8950_reconstructed_fourway_overlay.png"
    ]
    assert re3450_pair["legend_labels"] == LEGEND_LABELS[:2]
    assert re8950_pair["legend_labels"] == LEGEND_LABELS[2:]
    assert combined["legend_labels"] == LEGEND_LABELS
    assert len(re3450_pair["lines"]) == 2
    assert len(re8950_pair["lines"]) == 2
    assert len(combined["lines"]) == 4
    assert combined["xlabel"] == "t"
    assert combined["ylabel"] == "x_front - x0"

    re3450_pair_paper = re3450_pair["lines"][1]
    re8950_pair_paper = re8950_pair["lines"][1]
    np.testing.assert_array_equal(
        re3450_pair_paper["x"], papers["Re3450"]["time"]
    )
    np.testing.assert_array_equal(
        re3450_pair_paper["y"], papers["Re3450"]["paper_x"]
    )
    np.testing.assert_array_equal(
        re8950_pair_paper["x"], papers["Re8950"]["time"]
    )
    np.testing.assert_array_equal(
        re8950_pair_paper["y"], papers["Re8950"]["paper_x"]
    )

    # Combined plotting order is both simulations followed by both papers.
    assert [line["label"] for line in combined["lines"]] == [
        LEGEND_LABELS[0],
        LEGEND_LABELS[2],
        LEGEND_LABELS[1],
        LEGEND_LABELS[3],
    ]
    np.testing.assert_array_equal(
        combined["lines"][2]["x"], papers["Re3450"]["time"]
    )
    np.testing.assert_array_equal(
        combined["lines"][2]["y"], papers["Re3450"]["paper_x"]
    )
    np.testing.assert_array_equal(
        combined["lines"][3]["x"], papers["Re8950"]["time"]
    )
    np.testing.assert_array_equal(
        combined["lines"][3]["y"], papers["Re8950"]["paper_x"]
    )

    for simulation in (
        re3450_pair["lines"][0],
        re8950_pair["lines"][0],
        combined["lines"][0],
        combined["lines"][1],
    ):
        assert simulation["linestyle"] == "-"
        assert simulation["marker"] == "None"

    for paper_artist in (
        re3450_pair_paper,
        re8950_pair_paper,
        combined["lines"][2],
        combined["lines"][3],
    ):
        assert paper_artist["x"].size > 0
        assert paper_artist["y"].size > 0
        assert paper_artist["marker"] == "o"
        assert paper_artist["markersize"] == pytest.approx(2.0)
        assert paper_artist["linestyle"] == "None"
        assert paper_artist["marker"] != "s"

    assert (
        combined["lines"][2]["zorder"]
        > combined["lines"][0]["zorder"]
    )
    assert (
        combined["lines"][2]["zorder"]
        > combined["lines"][3]["zorder"]
    )
    assert combined["lines"][0]["color"] != combined["lines"][1]["color"]


@pytest.mark.parametrize("empty_reynolds", ["Re3450", "Re8950"])
def test_empty_finite_paper_data_is_rejected_clearly(
    tmp_path: Path,
    empty_reynolds: str,
) -> None:
    papers = _papers()
    papers[empty_reynolds] = {
        "time": np.array([np.nan, np.inf]),
        "paper_x": np.array([np.nan, 1.0]),
    }

    with pytest.raises(
        ValueError,
        match=rf"Cantero {empty_reynolds} paper data has zero finite "
        r"plotting points",
    ):
        write_reconstructed_xt_plots(
            output_dir=tmp_path,
            reconstructed_by_reynolds=_reconstructed(),
            papers_by_reynolds=papers,
            slump_tmin=1.0,
            slump_tmax=4.0,
            overwrite=False,
        )

    assert not list(tmp_path.glob("*.png"))
    assert plt.get_fignums() == []


def test_plot_preflight_protects_every_figure(tmp_path: Path) -> None:
    conflict = (
        tmp_path
        / "N7_Re3450_Re8950_reconstructed_fourway_overlay.png"
    )
    conflict.write_bytes(b"keep")

    with pytest.raises(FileExistsError, match="Pass --overwrite"):
        write_reconstructed_xt_plots(
            output_dir=tmp_path,
            reconstructed_by_reynolds=_reconstructed(),
            papers_by_reynolds=_papers(),
            slump_tmin=1.0,
            slump_tmax=4.0,
            overwrite=False,
        )

    assert conflict.read_bytes() == b"keep"
    assert not (
        tmp_path / "Re3450_N7_reconstructed_vs_Cantero_Re3450.png"
    ).exists()
    assert plt.get_fignums() == []
