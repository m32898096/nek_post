"""Read-only Step 3 artifact analysis for conditional h-convergence diagnostics."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from nek_post.comparison import compare_sampled_grids
from nek_post.h_refinement_comparison import FIELDS
from nek_post.h_refinement_convergence import analyze_convergence_triplet
from nek_post.h_refinement_phase import morphology_diagnostics, translation_diagnostics


def tree_hashes(directory: str | Path) -> dict[str, str]:
    """Hash all saved source artifacts so preservation is independently checkable."""
    root = Path(directory)
    hashes = {}
    for path in sorted(root.rglob('*')):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1024*1024), b''):
                    digest.update(block)
            hashes[str(path.relative_to(root))] = digest.hexdigest()
    return hashes


def analyze_saved_comparisons(directory: str | Path, case_order: tuple[str, str, str], *,
                              h_values=None, scalar_h_justified=False,
                              change_tolerance=.05, max_shift=.5) -> dict:
    """Analyze existing common-grid fields without reading raw solution arrays.

    Snapshot-time disagreement is an additional observed-order rejection gate;
    the existing nearest-time fields are not temporally interpolated here.
    """
    root = Path(directory)
    run = json.loads((root/'run.json').read_text())
    if run.get('status') != 'complete' or run.get('y_target') != .75:
        raise ValueError('Step 3 must be complete and use exact physical y=0.75.')
    if len(case_order) != 3 or len(set(case_order)) != 3 or run['reference_case'] != case_order[-1]:
        raise ValueError('Three ordered cases ending with the Step 3 numerical reference are required.')
    shape = (run['nz'],run['nx'])
    targets = run['target_times']
    directories = sorted(root.glob('snapshot_*'))
    if len(directories) != len(targets):
        raise ValueError('Saved snapshot group count does not match the completed manifest.')
    reference_grid = None
    results, morphology, shifts = [], [], []
    for sub, target in zip(directories, targets):
        metadata = json.loads((sub/'metadata.json').read_text())
        if metadata['target_time'] != target or metadata['y_target'] != .75:
            raise ValueError('Snapshot metadata does not match the target manifest.')
        if set(metadata['cases']) != set(case_order):
            raise ValueError('Snapshot metadata cases disagree with study cases.')
        times = [metadata['selections'][c]['actual_time'] for c in case_order]
        time_spread = float(max(times)-min(times))
        if not np.all(np.isfinite(times)):
            raise ValueError('Actual snapshot times must be finite.')
        source_rows = list(csv.DictReader((sub/'field_errors.csv').open()))
        with np.load(sub/'comparison_grids.npz', allow_pickle=False) as archive:
            X, Z = archive['Xi'], archive['Zi']
            if X.shape != shape or Z.shape != shape or float(archive['y_target']) != .75:
                raise ValueError('Saved grid shape/plane disagrees with manifest.')
            if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Z)):
                raise ValueError('Saved physical coordinates must be finite.')
            if reference_grid is None:
                reference_grid = (X.copy(),Z.copy())
            elif not all(np.array_equal(a,b) for a,b in zip(reference_grid,(X,Z))):
                raise ValueError('Saved grid coordinates changed across target times.')
            for field, key in FIELDS.items():
                arrays = [archive[c+'_'+key] for c in case_order]
                if any(a.shape != shape for a in arrays):
                    raise ValueError('Saved field shapes must match the common grid.')
                mask = np.logical_and.reduce([archive[c+'_geometry_mask'].astype(bool)&np.isfinite(a)
                                              for c,a in zip(case_order,arrays)])
                if not np.array_equal(mask,archive['common_'+field+'_mask']):
                    raise ValueError('Stored common mask differs from the all-case finite geometry mask.')
                # Time mismatch alone prevents claiming a spatial observed order.
                result = analyze_convergence_triplet(*arrays, mask,
                    h_values=h_values, scalar_h_justified=scalar_h_justified,
                    relative_change_tolerance=change_tolerance,
                    external_order_rejection_reasons=(
                        ('snapshot_times_not_identical_temporal_error_uncontrolled',)
                        if time_spread > 1e-8 else ()))
                result.update(target_time=target,field=field,actual_times=dict(zip(case_order,times)),
                              time_spread=time_spread,finest_available_numerical_reference=case_order[-1])
                result['source_step3_rows'] = [r for r in source_rows if r['field']==field]
                if len(result['source_step3_rows']) != 3:
                    raise ValueError('Missing source field-error rows.')
                # Check finest-reference norms independently against stored Step 3.
                checked = compare_sampled_grids(dict(zip(case_order, arrays)), case_order[-1], valid_mask=mask)
                for row in checked.error_rows:
                    rows = [r for r in result['source_step3_rows'] if r['case']==row['case']]
                    if len(rows)!=1:
                        raise ValueError('Duplicated/missing Step 3 case metric row.')
                    names = ('relative_l2_error','mean_absolute_error','maximum_absolute_error')
                    expected = tuple(row[k] for k in names)
                    observed = tuple(float(rows[0][k]) for k in names)
                    if not np.allclose(expected,observed,rtol=1e-12,atol=1e-14,equal_nan=True):
                        raise ValueError('Saved Step 3 errors do not match its fields/masks.')
                results.append(result)
                for case, values in zip(case_order[:-1],arrays[:-1]):
                    info=dict(target_time=target,field=field,case=case,reference_case=case_order[-1])
                    morphology.append(dict(info,**morphology_diagnostics(values,arrays[-1],mask)))
                    shifts.extend(dict(info,**d) for d in translation_diagnostics(values,arrays[-1],mask,X[0],max_shift=max_shift))
    return dict(case_order=list(case_order),grid=dict(nx=shape[1],nz=shape[0],y_target=.75,
                xmin=float(reference_grid[0].min()),xmax=float(reference_grid[0].max()),
                zmin=float(reference_grid[1].min()),zmax=float(reference_grid[1].max()),
                interpretation='fixed post-processing grid; not simulation mesh resolution'),
                convergence=results,morphology=morphology,translation=shifts)
