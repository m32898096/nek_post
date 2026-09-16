import numpy as np
import pytest
from nek_post.h_refinement_phase import morphology_diagnostics, translation_diagnostics


def test_manufactured_translation_and_fixed_support():
    x = np.linspace(-5,5,201)
    z = np.linspace(0,1,5)[:,None]
    def field(q):
        return np.exp(-((q-2)/.4)**2) + np.exp(-((q+2)/.4)**2) + z
    b = field(x)
    a = field(x-.1)
    result = translation_diagnostics(a,b,np.ones_like(a,bool),x,max_shift=.3)
    for half in result:
        assert half['best_shift_x'] == pytest.approx(.1)
        assert half['best_rms'] < 1e-14
        assert half['rms_reduction_fraction'] > .999999
        assert half['raw_rms'] > .01
        assert not half['search_boundary_hit']


def test_distribution_distance_discards_spatial_arrangement():
    a = np.arange(12.).reshape(3,4)
    result = morphology_diagnostics(a,a[:,::-1],np.ones_like(a,bool))
    assert result['spatial_rms'] > 0
    assert result['distribution_rms'] == 0
    assert result['distribution_to_spatial_rms'] == 0


def test_nan_masks_and_zero_difference():
    a = np.ones((3,100)); x=np.linspace(-5,5,100)
    a[1,30] = np.nan
    for row in translation_diagnostics(a,a,np.ones_like(a,bool),x,max_shift=.2):
        assert row['best_shift_x']==0
        assert row['raw_rms']==0
        assert row['rms_reduction_fraction'] is None
    assert morphology_diagnostics(a,a,np.ones_like(a,bool))['spatial_correlation'] is None


def test_bad_grid_rejected():
    with pytest.raises(ValueError,match='uniform'):
        translation_diagnostics(np.ones((2,4)),np.ones((2,4)),np.ones((2,4),bool),[-2,-1,1,2])
