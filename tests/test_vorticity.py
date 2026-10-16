"""Analytic curl and physical-coordinate independent assembly checks."""
from types import SimpleNamespace

import numpy as np
import pytest

from test_l2_projection import mesh,coordinate_oracle
from nek_post.l2_projection import build_l2_projection_operator,project_scalar_field
from nek_post.spectral_derivatives import element_velocity_gradient
from nek_post.vorticity import element_streamwise_vorticity,compute_projected_omega_x
from nek_post.rectilinear_export import assemble_rectilinear_scalar_field,write_rectilinear_scalar_vtr
from nek_post.lambda_ci_export import assemble_rectilinear_lambda_fields


@pytest.mark.parametrize('mapping',[(2,1,0),(0,2,1),(1,0,2)])
@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('kind',['rotation','shear','polynomial'])
def test_local_omega_x_analytic_physical_axes(mapping,reverse,kind):
    data=mesh(shape=(8,7,6),mapping=mapping,reverse=reverse)
    e=data.elem[0];x,y,z=e.pos
    if kind=='rotation':
        e.vel=np.stack((np.zeros_like(x),-1.25*z,1.25*y));expected=np.full(x.shape,2.5)
    elif kind=='shear':
        e.vel=np.stack((np.zeros_like(x),2*z,np.zeros_like(x)));expected=np.full(x.shape,-2.)
    else:
        e.vel=np.stack((x**2,2*z+y*z*z,y**3+.2*x*z));expected=3*y*y-2-2*y*z
    np.testing.assert_allclose(element_streamwise_vorticity(e),expected,atol=5e-13,rtol=3e-14)


def test_skew_and_curved_jacobian_and_invalid_geometry():
    e=mesh(shape=(6,6,6)).elem[0]
    x,y,z=e.pos.copy();e.pos=np.stack((x+.1*y*y,y+.15*z,z+.2*x))
    x,y,z=e.pos;e.vel=np.stack((0*x,-2*z,2*y))
    np.testing.assert_allclose(element_streamwise_vorticity(e),4.,atol=8e-14)
    e.pos[2]=0
    with pytest.raises(ValueError,match='Jacobian'):element_streamwise_vorticity(e)


def test_only_two_velocity_derivatives_and_no_eigenvalues(monkeypatch):
    import nek_post.vorticity as module
    import nek_post.spectral_derivatives as derivatives
    e=mesh(shape=(8,8,8)).elem[0];x,y,z=e.pos;e.vel=np.stack((x,y*z,y*y))
    calls=[];original=module.reference_derivatives
    def observe(values):calls.append(values);return original(values)
    monkeypatch.setattr(module,'reference_derivatives',observe)
    monkeypatch.setattr(derivatives,'velocity_gradient_tensor',lambda *a,**k:pytest.fail('full gradient computed'))
    monkeypatch.setattr(np.linalg,'eigvals',lambda *a,**k:pytest.fail('eigenvalues computed'))
    actual=element_streamwise_vorticity(e)
    assert len(calls)==2 and np.shares_memory(calls[0],e.vel[2]) and np.shares_memory(calls[1],e.vel[1])
    np.testing.assert_allclose(actual,y,atol=1e-14)
    e.vel[1,0,0,0]=np.nan
    with pytest.raises(ValueError):element_streamwise_vorticity(e)


@pytest.mark.parametrize('periodic',[(),('x','y')])
def test_projected_curl_linearity_and_independent_mass_assembly(periodic):
    edges=((0.,1.,3.),(0.,1.,2.),(0.,1.,4.));data=mesh(edges,shape=(4,4,4),reverse=True)
    for ei,e in enumerate(data.elem):
        x,y,z=e.pos;e.vel=np.stack((x*x,(ei+1)*z*y,(-1)**ei*y*y))
    gradients=np.stack([element_velocity_gradient(e) for e in data.elem])
    raw=gradients[...,2,1]-gradients[...,1,2]
    operator=build_l2_projection_operator(data,periodic_axes=periodic)
    actual=compute_projected_omega_x(data,operator,chunk_size=3)
    separate=project_scalar_field(operator,gradients[...,2,1])-project_scalar_field(operator,gradients[...,1,2])
    np.testing.assert_allclose(actual,separate,atol=8e-14,rtol=3e-14)
    oracle=np.zeros(raw.shape+(3,3));oracle[...,0,0]=raw
    expected,_=coordinate_oracle(data,oracle,periodic,edges)
    np.testing.assert_allclose(actual,expected[...,0,0],atol=8e-14,rtol=3e-14)
    assert actual.min()<0<actual.max()


def test_signed_vtr_native_coordinates_values_and_lambda_grid_compatibility(tmp_path):
    from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader
    from vtkmodules.util.numpy_support import vtk_to_numpy
    data=mesh(((0.,1.,3.),(0.,2.),(0.,1.)),shape=(4,5,8),mapping=(0,2,1),reverse=True)
    for e in data.elem:
        x,y,z=e.pos;e.vel=np.stack((0*x,z*x,y*y))
    operator=build_l2_projection_operator(data,periodic_axes=('x','y'))
    values=compute_projected_omega_x(data,operator)
    field=assemble_rectilinear_scalar_field(data,operator.node_map,values,name='omega_x',chunk_size=1)
    positive=np.abs(values)
    reference=assemble_rectilinear_lambda_fields(data,operator.node_map,positive,positive**2)
    for a,b in zip(field.coordinates,reference.coordinates):np.testing.assert_array_equal(a,b)
    assert field.values.shape==reference.lambda_ci.shape
    path=write_rectilinear_scalar_vtr(tmp_path/'omega.vtr',field,time=.123456789)
    reader=vtkXMLRectilinearGridReader();reader.SetFileName(str(path));reader.Update();grid=reader.GetOutput()
    assert grid.GetDimensions()==tuple(len(v) for v in field.coordinates)
    for getter,v in zip((grid.GetXCoordinates,grid.GetYCoordinates,grid.GetZCoordinates),field.coordinates):
        np.testing.assert_array_equal(vtk_to_numpy(getter()),v)
    scalars=vtk_to_numpy(grid.GetPointData().GetArray('omega_x'))
    np.testing.assert_array_equal(scalars,field.values.ravel())
    assert scalars.dtype==np.float64 and scalars.min()<0<scalars.max()
    assert grid.GetFieldData().GetArray('TIME_VALUE').GetValue(0)==.123456789
    before=path.read_bytes()
    with pytest.raises(FileExistsError):write_rectilinear_scalar_vtr(path,field,time=1.)
    assert path.read_bytes()==before
    corrupted=values.copy()
    shared_node=tuple(np.argwhere(data.elem[1].pos[0]==1.)[0])
    corrupted[(1,*shared_node)]+=1
    with pytest.raises(ValueError,match='inconsistent'):
        assemble_rectilinear_scalar_field(data,operator.node_map,corrupted,name='omega_x')
