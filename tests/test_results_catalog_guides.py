"""Portable source links and optional bounded real-control associations."""
import json
from pathlib import Path
import re

import pytest

REPO = Path(__file__).parents[1]
GUIDES = ('h_refinement.md', 'poly_order_compare.md',
          'historical_results.md', 'producer_scripts.md')


def links(document):
    return re.findall(r'\[[^\]\n]*\]\(([^)\n]+)\)', document.read_text(encoding='utf-8'))


@pytest.mark.parametrize('name', GUIDES)
def test_guide_repository_references(name):
    document = REPO / 'docs/results_catalog' / name
    for target in links(document):
        if not Path(target).is_absolute():
            assert (document.parent / target.split('#')[0]).is_file(), target
    text = document.read_text(encoding='utf-8')
    assert text.endswith('\n') and '湍流' not in text


def test_optional_guide_external_links_and_control_associations():
    root = Path('/data/Nek5000_data/results')
    pv_name = ('h_refinement/lambda_ci/case_N7_H/f00079/plots/'
               'case_N7_H_f00079_lambda_ci_high_order_slices_paraview.json')
    if not (root / pv_name).is_file():
        pytest.skip('Optional local N7_H result controls unavailable')
    documents = [REPO / 'docs/results_catalog' / name for name in GUIDES]
    targets = {Path(target) for doc in documents for target in links(doc)
               if Path(target).is_absolute()}
    for path in targets:
        assert path.resolve().is_relative_to(root)
        assert path.exists(), path

    def control(path):
        # Read only bounded metadata; never scientific volume payloads.
        with path.open('rb') as stream:
            data = stream.read(1_048_577)
        assert len(data) <= 1_048_576
        return json.loads(data)

    h_links = {Path(p) for p in links(documents[0]) if Path(p).is_absolute()}
    pv = control(root / pv_name)
    assert Path(pv['output_paths']['y_midplane_vtr']) in h_links
    assert pv['samples_per_element'] == 29
    summary = control(root / 'h_refinement/lambda_ci/case_N7_H/processing_summary.json')
    assert summary['configuration']['subdivisions_per_element'] == pv['samples_per_element'] - 1 == 28
    py = control(root / pv_name.replace('_paraview.json', '.json'))
    assert py['samples_per_element'] == 32
    assert Path(py['output_paths']['y_midplane_png']) in h_links
    assert set(py['output_paths'].values()).isdisjoint(pv['output_paths'].values())
    vortex_root = root / 'h_refinement/vortex_3d_series/gll10/case_N7_H'
    vortex = control(vortex_root / 'f00079/GC0_f00079_gll10.json')
    assert Path(vortex['output_paths']['vtu']) in h_links
    assert vortex['reconstruction']['source_polynomial_degree'] == 7
    assert vortex['configuration']['sampling_nodes'] == 10
    manifest = control(vortex_root / 'series/manifest.json')
    assert vortex['time'] == manifest['frames']['79']['time'] == 19.50240408807
    assert vortex['frame_index'] == 79 and vortex['time'] != vortex['frame_index']
