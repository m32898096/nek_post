"""Signed scalar recovery, independent assembly and frozen pre-refactor results."""
from copy import deepcopy

import numpy as np
import pytest

from test_l2_projection import mesh, coordinate_oracle
from nek_post.l2_projection import build_l2_projection_operator, project_scalar_field, project_velocity_gradient
from nek_post.swirling_strength import compute_swirling_strength


def legacy_projection(operator, array, chunk_size):
    """Frozen pre-refactor nine-component arithmetic (commit 722e0c8)."""
    mapping = operator.node_map
    output = np.empty(array.shape, dtype=np.float64)
    rhs = np.empty(mapping.global_node_count, dtype=np.float64)
    for i in range(3):
        for j in range(3):
            rhs.fill(0)
            for start in range(0, mapping.element_count, chunk_size):
                stop = min(start + chunk_size, mapping.element_count)
                ids = mapping.node_ids(start, stop).ravel()
                weighted = np.multiply(array[start:stop,...,i,j].reshape(-1),
                                       operator.weights[start:stop].ravel(),dtype=np.float64)
                np.add.at(rhs,ids,weighted)
            rhs /= operator.mass
            for start in range(0,mapping.element_count,chunk_size):
                stop = min(start + chunk_size,mapping.element_count)
                output[start:stop,...,i,j] = rhs[mapping.node_ids(start,stop)]
    return output


@pytest.mark.parametrize('periodic',[(),('x','y')])
@pytest.mark.parametrize('constant',[-2.5,0.,3.25])
def test_scalar_preserves_signed_constant(periodic,constant):
    data=mesh(((0.,1.,4.),(0.,1.,3.),(0.,2.)),shape=(4,5,8))
    operator=build_l2_projection_operator(data,periodic_axes=periodic)
    values=np.full(operator.weights.shape,constant,dtype=np.float32)
    saved=values.copy();out=np.empty(values.shape,dtype=np.float64)
    actual=project_scalar_field(operator,values,mesh=data,out=out,chunk_size=1)
    assert actual is out and actual.dtype==np.float64
    np.testing.assert_allclose(actual,constant,atol=3e-15)
    np.testing.assert_array_equal(values,saved)


@pytest.mark.parametrize('periodic',[(),('x',),('y',),('x','y')])
def test_signed_scalar_independent_bm1_dssum_binvm1_oracle(periodic):
    edges=((0.,1.,3.),(0.,2.,3.),(0.,1.,5.))
    data=mesh(edges,reverse=True)
    operator=build_l2_projection_operator(data,periodic_axes=periodic,chunk_size=2)
    values=np.random.default_rng(43).normal(size=operator.weights.shape)
    tensors=np.zeros(values.shape+(3,3));tensors[...,0,0]=values
    expected,_=coordinate_oracle(data,tensors,periodic,edges)
    actual=project_scalar_field(operator,values,chunk_size=3)
    np.testing.assert_allclose(actual,expected[...,0,0],atol=2e-15,rtol=3e-15)
    assert actual.min()<0<actual.max()
    ids=operator.node_map.node_ids(0,len(data.elem))
    for global_id in np.unique(ids):
        contributors=actual[ids==global_id]
        np.testing.assert_array_equal(contributors,contributors[0])


def test_scalar_shared_face_uses_mass_weights_and_keeps_interior_signs():
    data=mesh(((0.,1.,4.),(0.,1.),(0.,1.)))
    operator=build_l2_projection_operator(data,periodic_axes=())
    values=np.empty(operator.weights.shape);values[0]=-1;values[1]=3
    actual=project_scalar_field(operator,values,chunk_size=1)
    np.testing.assert_allclose(actual[0,:,:,-1],2.)
    np.testing.assert_array_equal(actual[0,:,:,-1],actual[1,:,:,0])
    np.testing.assert_allclose(actual[0,:,:,0],-1.)
    np.testing.assert_allclose(actual[1,:,:,-1],3.)


@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.int32])
@pytest.mark.parametrize('chunk',[1,3])
def test_gradient_and_lambda_ci_exactly_match_pre_refactor(dtype,chunk):
    data=mesh(((0.,1.,3.),(0.,2.,3.),(0.,1.,5.)),reverse=True)
    operator=build_l2_projection_operator(data,periodic_axes=('x','y'))
    values=np.random.default_rng(9).normal(size=operator.weights.shape+(3,3))*3
    values=values.astype(dtype)[...,::-1,:]  # Noncontiguous component views.
    expected=legacy_projection(operator,values,chunk)
    actual=project_velocity_gradient(operator,values,chunk_size=chunk)
    np.testing.assert_array_equal(actual.values,expected)
    np.testing.assert_array_equal(actual.diagnostics.discontinuity_after,0)
    for old,new in zip(compute_swirling_strength(expected).__dict__.values(),
                       compute_swirling_strength(actual.values).__dict__.values()):
        np.testing.assert_array_equal(old,new)


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf,1+2j,True])
def test_invalid_scalar_rejected_before_output_mutation(bad):
    operator=build_l2_projection_operator(mesh(),periodic_axes=())
    values=np.zeros(operator.weights.shape,dtype=complex if np.iscomplexobj(bad) else bool if bad is True else float)
    values.flat[0]=bad;out=np.full(operator.weights.shape,42.)
    with pytest.raises(ValueError):project_scalar_field(operator,values,out=out)
    np.testing.assert_array_equal(out,42.)


def test_scalar_output_alias_shape_chunk_and_mesh_validation():
    data=mesh(((0.,1.,2.),(0.,1.),(0.,1.)));operator=build_l2_projection_operator(data,periodic_axes=())
    values=np.ones(operator.weights.shape)
    readonly=values.copy();readonly.setflags(write=False)
    for out in (values,operator.weights,np.zeros(values.shape,dtype=np.float32),readonly,values[...,::-1],np.zeros(1)):
        with pytest.raises(ValueError,match='out'):project_scalar_field(operator,values,out=out)
    with pytest.raises(ValueError):project_scalar_field(operator,values[0])
    for chunk in (0,True,1.5):
        with pytest.raises(ValueError):project_scalar_field(operator,values,chunk_size=chunk)
    changed=deepcopy(data);changed.elem.reverse()
    with pytest.raises(ValueError,match='ordering'):project_scalar_field(operator,values,mesh=changed)
