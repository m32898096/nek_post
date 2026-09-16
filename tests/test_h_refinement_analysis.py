import csv
import json
from pathlib import Path

import numpy as np
import pytest

from nek_post.comparison import compare_sampled_grids
from nek_post.h_refinement_analysis import analyze_saved_comparisons, tree_hashes


def saved_step3(tmp_path, *, fine_zero=False, time_spread=0.):
    root=tmp_path/'field_comparison';sub=root/'snapshot_001';sub.mkdir(parents=True)
    X,Z=np.meshgrid(np.linspace(-2,2,21),np.linspace(0,1,7))
    cases=('a','b','ref');fields={'concentration':'C','velocity_magnitude':'speed','pressure_fluctuation':'p_prime'}
    run=dict(status='complete',nx=21,nz=7,y_target=.75,reference_case='ref',target_times=[5.])
    (root/'run.json').write_text(json.dumps(run))
    selections={c:dict(actual_time=5.+(time_spread if c=='b' else 0)) for c in cases}
    (sub/'metadata.json').write_text(json.dumps(dict(target_time=5.,y_target=.75,cases={c:{} for c in cases},selections=selections)))
    payload=dict(Xi=X,Zi=Z,y_target=np.array(.75));rows=[]
    for c in cases:payload[c+'_geometry_mask']=np.ones(X.shape,bool)
    for field,key in fields.items():
        grids={c: (np.zeros_like(X) if fine_zero else 2+X*.1+Z*.1)+(h*h-1)*.01*(1+X**2)
               for c,h in zip(cases,(4.,2.,1.))}
        result=compare_sampled_grids(grids,'ref')
        for row in result.error_rows:rows.append(dict(row,field=field))
        payload.update({c+'_'+key:v for c,v in grids.items()})
        payload['common_'+field+'_mask']=result.common_mask
    np.savez_compressed(sub/'comparison_grids.npz',**payload)
    with (sub/'field_errors.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return root


def test_saved_fields_known_order_and_preservation(tmp_path):
    root=saved_step3(tmp_path);before=tree_hashes(root)
    result=analyze_saved_comparisons(root,('a','b','ref'),h_values=(4,2,1),scalar_h_justified=True)
    assert tree_hashes(root)==before
    assert len(result['convergence'])==3
    for r in result['convergence']:
        assert r['observed_order']==pytest.approx(2)
        assert r['valid_mask_fraction']==1.
    assert len(result['morphology'])==6 and len(result['translation'])==12


def test_time_mismatch_rejects_spatial_order(tmp_path):
    root=saved_step3(tmp_path,time_spread=.001)
    result=analyze_saved_comparisons(root,('a','b','ref'),h_values=(4,2,1),scalar_h_justified=True)
    assert all(r['observed_order'] is None for r in result['convergence'])
    assert all('snapshot_times_not_identical_temporal_error_uncontrolled' in r['order_rejection_reasons'] for r in result['convergence'])


def test_zero_reference_retains_step3_safe_metric_policy(tmp_path):
    root=saved_step3(tmp_path,fine_zero=True)
    result=analyze_saved_comparisons(root,('a','b','ref'))
    assert result['convergence'][0]['finest_reference']['coarse']['relative_l2_error'] is None
    assert result['convergence'][0]['finest_reference']['fine']['relative_l2_error']==0.


def test_inconsistent_saved_mask_rejected(tmp_path):
    root=saved_step3(tmp_path);path=root/'snapshot_001/comparison_grids.npz'
    with np.load(path,allow_pickle=False) as f:payload=dict(f)
    payload['common_concentration_mask'][0,0]=False
    np.savez_compressed(path,**payload)
    with pytest.raises(ValueError,match='mask'):
        analyze_saved_comparisons(root,('a','b','ref'))
