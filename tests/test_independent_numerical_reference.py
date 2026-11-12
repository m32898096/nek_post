"""Independent Phase 1B references on deterministic, bounded synthetic fields.

Analytical expectations and explicit connectivity are defined here without
production test helpers. Float64 tolerances cover polynomial evaluation and
small conditioned solves. See docs/methods/phase1b_results.md for measured
errors, reference build details, and the production-provenance limits.
"""
from itertools import permutations, product
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import shutil
import subprocess

import numpy as np
from numpy.polynomial import Legendre
import pytest

from nek_post.gll import gll_nodes, gll_quadrature_weights, gll_differentiation_matrix
from nek_post.gll_refinement import refine_gll_tensor3
from nek_post.l2_projection import build_l2_projection_operator, project_scalar_field, project_velocity_gradient
from nek_post.spectral_derivatives import element_jacobian, element_velocity_gradient, reference_derivatives
from nek_post.swirling_strength import compute_swirling_strength
from nek_post.vorticity import element_streamwise_vorticity, compute_projected_omega_x
from nek_post.vortex_3d import compute_projected_vortex_fields
from nek_post.vortex_3d_reconstruction import build_reconstructed_vortex_grid


# Independent companion-matrix roots and monomial moment equations, as opposed
# to production scipy Jacobi roots, Legendre weight formula and barycentric D.
def reference_rule(n):
    nodes = np.r_[-1., Legendre.basis(n-1).deriv().roots(), 1.]
    moments = np.array([0. if k % 2 else 2./(k+1) for k in range(n)])
    vandermonde = np.vander(nodes, N=n, increasing=True)
    weights = np.linalg.solve(vandermonde.T, moments)
    derivatives = np.column_stack([np.zeros(n)] + [k*nodes**(k-1) for k in range(1, n)])
    differentiation = np.linalg.solve(vandermonde.T, derivatives.T).T
    return nodes, weights, differentiation


def check(actual, expected, tolerance, record_property, label):
    """Report absolute and relative error, scaling relative error away from zero."""
    actual, expected = np.broadcast_arrays(actual, expected)
    error = float(np.max(np.abs(actual-expected)))
    scale = float(np.max(np.abs(expected)))
    relative = error/scale if scale else error
    record_property(label+'_max_abs', error)
    record_property(label+'_relative_inf', relative)
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('n', [2, 3, 4, 6, 8, 10, 12])
def test_gll_moment_and_polynomial_oracles(n, record_property):
    """Nodal rule, D, exact moments through 2p-1 and derivatives through p."""
    nodes, weights, derivative = reference_rule(n)
    x, w, d = gll_nodes(n), gll_quadrature_weights(n), gll_differentiation_matrix(n)
    check(x, nodes, 5e-14, record_property, 'nodes')
    check(w, weights, 3e-12, record_property, 'weights')
    check(d, derivative, 3e-10, record_property, 'D')
    for k in range(n):
        check(d @ x**k, np.zeros(n) if k == 0 else k*x**(k-1), 3e-12, record_property, f'degree_{k}_derivative')
    for k in range(2*n-2):
        check(w @ x**k, 0. if k % 2 else 2./(k+1), 3e-13, record_property, f'moment_{k}')
    boundary = np.zeros((n,n))
    boundary[0,0], boundary[-1,-1] = -1., 1.
    check(w[:,None]*d + d.T*w[None,:], boundary, 3e-12, record_property, 'SBP')


@pytest.fixture(scope='session')
def nek_reference(tmp_path_factory):
    """Build actual pinned local code, without a solver or MPI initialization."""
    source = Path('/home/dai_lab/Nek5000/core/speclib.f')
    compiler = shutil.which('gfortran')
    if not source.exists() or compiler is None:
        pytest.skip('Optional local Nek5000 source or gfortran unavailable')
    inventory = json.loads(Path('docs/methods/nek5000_evidence.json').read_text())
    pinned = next(r for r in inventory['files'] if r['path'] == str(source))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == pinned['sha256']
    work = tmp_path_factory.mktemp('nek_reference')
    executable = work/'gll_reference'
    command = [compiler, '-std=legacy', '-fallow-argument-mismatch',
               '-fdefault-real-8', '-fdefault-double-8', '-ffunction-sections',
               '-fdata-sections', '-Wl,--gc-sections', str(source),
               str(Path(__file__).parent/'reference/nek_gll_driver.f90'), '-o', str(executable)]
    subprocess.run(command, cwd=work, check=True, capture_output=True, text=True)
    return executable


@pytest.mark.parametrize('n', [2, 3, 4, 6, 8, 10, 12])
def test_actual_nek_gll(nek_reference, n, record_property):
    """Independent implementation comparison to local ZWGLL/DGLL in REAL*8."""
    result = subprocess.run([str(nek_reference)], input=f'{n}\n', capture_output=True, text=True, check=True)
    values = np.loadtxt(result.stdout.splitlines())
    check(gll_nodes(n), values[:,0], 3e-14, record_property, 'fortran_nodes')
    check(gll_quadrature_weights(n), values[:,1], 3e-14, record_property, 'fortran_weights')
    check(gll_differentiation_matrix(n), values[:,2:], 3e-11, record_property, 'fortran_D')


def box_mesh(mapping=(2,1,0), reverse=False, periodic=(), n=3,
             edges=((0., .5, 2.), (0., 1., 3.), (0., .25, 1.))):
    """Explicit entity labels define connectivity before production sees a mesh.

    Each axis label is a vertex number or an element-interior ordinal. Boundary
    partners are explicitly identified by the declared periodic axes. This is
    an asserted conforming translational box mesh, with shuffled element order.
    """
    nodes = reference_rule(n)[0]
    elements, labels = [], []
    for cell in reversed(list(product(*(range(len(a)-1) for a in edges)))):
        axes, entity_axes = [], []
        for axis in range(3):
            lo, hi = edges[axis][cell[axis]:cell[axis]+2]
            axis_nodes = (lo+hi)/2 + (hi-lo)/2*nodes
            # Conforming interfaces use the declared endpoints exactly.
            axis_nodes[0], axis_nodes[-1] = lo, hi
            entities = [('vertex',cell[axis])] + [('inside',cell[axis],j) for j in range(1,n-1)] + [('vertex',cell[axis]+1)]
            if 'xyz'[axis] in periodic and cell[axis]+1 == len(edges[axis])-1:
                entities[-1] = ('vertex',0)
            if reverse and (sum(cell)+axis) % 2:
                axis_nodes = axis_nodes[::-1]
                entities = entities[::-1]
            axes.append(axis_nodes)
            entity_axes.append(entities)
        pos = np.empty((3,n,n,n))
        local_labels = {}
        for node in np.ndindex((n,n,n)):
            for axis in range(3):
                pos[(axis,*node)] = axes[axis][node[mapping[axis]]]
            local_labels[node] = tuple(entity_axes[axis][node[mapping[axis]]] for axis in range(3))
        volume = np.prod([edges[a][cell[a]+1]-edges[a][cell[a]] for a in range(3)])
        elements.append(SimpleNamespace(pos=pos, volume=volume))
        labels.append(local_labels)
    return SimpleNamespace(elem=elements), labels


def explicit_assembly(data, labels, values):
    """P2 rational weights and contributor lists, without production IDs/masses."""
    factors = [1/3, 4/3, 1/3]
    weights = np.empty(values.shape[:4])
    groups = {}
    for ei, element in enumerate(data.elem):
        for node, label in labels[ei].items():
            local = (ei,*node)
            weights[local] = element.volume/8*np.prod([factors[j] for j in node])
            groups.setdefault(label, []).append(local)
    expected = np.empty_like(values)
    for contributors in groups.values():
        denominator = sum(weights[local] for local in contributors)
        numerator = sum(weights[local]*values[local] for local in contributors)
        for local in contributors:
            expected[local] = numerator/denominator
    return expected, weights, groups


@pytest.mark.parametrize('mapping', list(permutations(range(3))))
@pytest.mark.parametrize('periodic', [(), ('x',), ('x','y','z')])
@pytest.mark.parametrize('reverse', [False, True])
def test_explicit_mass_and_connectivity(mapping, periodic, reverse, record_property):
    """Unequal masses, faces/edges/corners, periodic seams and nine components."""
    data, labels = box_mesh(mapping, reverse, periodic)
    shape = (8,3,3,3)
    scalar = np.arange(np.prod(shape), dtype=float).reshape(shape)/17-3
    # Distinct non-proportional tensor components exercise component indexing.
    tensor = np.stack([np.sin(scalar*(j+1)) + j*scalar for j in range(9)], axis=-1).reshape((*shape,3,3))
    expected, weights, groups = explicit_assembly(data, labels, scalar)
    expected_tensor, _, _ = explicit_assembly(data, labels, tensor)
    operator = build_l2_projection_operator(data, periodic_axes=periodic, chunk_size=3)
    actual = project_scalar_field(operator, scalar, chunk_size=3)
    check(operator.weights, weights, 2e-13, record_property, 'local_mass')
    ids = operator.node_map.node_ids(0,8)
    reference_partition = {frozenset(v) for v in groups.values()}
    actual_partition = {frozenset(map(tuple,np.argwhere(ids == i))) for i in np.unique(ids)}
    assert actual_partition == reference_partition
    for contributors in groups.values():
        local = contributors[0]
        check(operator.mass[ids[local]], sum(weights[p] for p in contributors), 2e-13, record_property, 'assembled_mass')
    assert max(map(len,groups.values())) >= 8
    check(actual, expected, 3e-12, record_property, 'scalar_recovery')
    check(project_velocity_gradient(operator,tensor,diagnostics=False).values, expected_tensor, 5e-12, record_property, 'tensor_recovery')
    check(project_scalar_field(operator,np.ones(shape)), 1., 2e-14, record_property, 'constants')
    check(project_scalar_field(operator,actual), actual, 3e-12, record_property, 'idempotence')
    check(np.sum(weights*actual), np.sum(weights*scalar), 3e-12, record_property, 'conservation')
    # Weighted residual orthogonality for every independent continuous basis DOF.
    for contributors in groups.values():
        check(sum(weights[p]*(scalar[p]-actual[p]) for p in contributors), 0., 3e-12, record_property, 'orthogonality')


def polynomial_velocity(pos):
    x,y,z = pos
    velocity = np.stack((x*x+y*z, y*y+x*z-2*z, z*z+x*y+3*y))
    tensor = np.empty((*x.shape,3,3))
    tensor[...,0,:] = np.stack((2*x,z,y),axis=-1)
    tensor[...,1,:] = np.stack((z,2*y,x-2),axis=-1)
    tensor[...,2,:] = np.stack((y,x+3,2*z),axis=-1)
    return velocity,tensor


@pytest.mark.parametrize('mapping', list(permutations(range(3))))
@pytest.mark.parametrize('matrix', [np.diag([.25,1.5,2.]), np.array([[1.,.3,-.2],[.2,1.5,.1],[.1,-.2,.5]])])
@pytest.mark.parametrize('reverse', [False,True])
def test_analytic_physical_gradients(mapping, matrix, reverse, record_property):
    """All nine polynomial gradients on affine boxes/skew maps, analytic J."""
    axes = np.meshgrid(*([reference_rule(4)[0]]*3),indexing='ij')
    reference = np.stack([axes[mapping[a]] for a in range(3)])
    if reverse:
        reference[0] *= -1
    pos = np.einsum('ab,b...->a...',matrix,reference)+np.array([.2,-.3,.7])[:,None,None,None]
    velocity,expected = polynomial_velocity(pos)
    element = SimpleNamespace(pos=pos,vel=velocity)
    expected_j = np.empty((3,3))
    # J columns correspond to array axes 2,1,0 (r,s,t).
    for col, array_axis in enumerate((2,1,0)):
        physical_reference = mapping.index(array_axis)
        expected_j[:,col] = matrix[:,physical_reference]*(-1 if reverse and physical_reference == 0 else 1)
    jacobian,_ = element_jacobian(pos)
    check(jacobian, expected_j, 3e-12, record_property, 'jacobian')
    check(element_velocity_gradient(element), expected, 5e-12, record_property, 'physical_gradient')
    for component in range(3):
        reference_expected = np.einsum('...a,ab->...b',expected[...,component,:],expected_j)
        check(reference_derivatives(velocity[component]), reference_expected, 5e-12, record_property, 'reference_gradient')
    check(element_streamwise_vorticity(element), 5., 4e-12, record_property, 'local_polynomial_curl')
    element.vel = np.full_like(velocity, 2.5)
    check(element_velocity_gradient(element), 0., 3e-12, record_property, 'translation')


@pytest.mark.parametrize('kind', ['rotation','shear','translation','polynomial'])
@pytest.mark.parametrize('periodic', [(),('x','y')])
def test_analytic_curl_and_recovery(kind, periodic, record_property):
    """Local analytic curl then independent mass recovery on a nonuniform mesh."""
    data,labels = box_mesh((0,2,1),True,periodic)
    raw = []
    for ei,element in enumerate(data.elem):
        x,y,z = element.pos
        scale = ei+1.
        velocity, tensor = polynomial_velocity(element.pos)
        if kind == 'rotation':
            velocity = np.stack((0*x,-scale*z,scale*y))
            expected = np.full_like(x,2*scale)
        elif kind == 'shear':
            velocity = np.stack((0*x,scale*z,0*x))
            expected = np.full_like(x,-scale)
        elif kind == 'translation':
            velocity = np.ones_like(element.pos)*scale
            expected = np.zeros_like(x)
        else:
            velocity *= scale
            expected = tensor[...,2,1]*scale-tensor[...,1,2]*scale
        element.vel = velocity
        raw.append(expected)
        check(element_streamwise_vorticity(element), expected, 3e-12, record_property, 'raw_curl')
    expected,_,_ = explicit_assembly(data,labels,np.array(raw))
    check(compute_projected_omega_x(data,periodic_axes=periodic),expected,3e-12,record_property,'projected_curl')
    check(compute_projected_vortex_fields(data,periodic_axes=periodic).omega_x,expected,3e-12,record_property,'combined_curl')


@pytest.mark.parametrize('a,b', [(0.,0.),(0.,3.),(2.,2.),(-3.,-3.),(1e-12,1.),(1e-24,1.),(-1e-12,1.),(2.,.5)])
@pytest.mark.parametrize('axis', [0,1,2])
def test_analytic_eigenvalues(a,b,axis,record_property):
    """2x2 block has lambda=shift +/- sqrt(-a*b), including near defects."""
    tensor = np.array([[.7,-a,0.],[b,.7,0.],[.3,-.2,-1.]])
    tensor = np.roll(np.roll(tensor,axis,axis=0),axis,axis=1)
    expected = np.sqrt(max(0.,a*b))
    result = compute_swirling_strength(tensor)
    check(result.lambda_ci,expected,2e-14,record_property,'lambda_ci')
    # Relative accuracy also protects tiny nonzero imaginary parts from clipping.
    np.testing.assert_allclose(result.lambda_ci, expected, rtol=3e-13, atol=1e-26)
    check(result.lambda_ci_squared,expected**2,2e-14,record_property,'lambda_ci_squared')


def test_nonlinear_projection_order(record_property):
    """Opposite rotations cancel after tensor projection but scalar strengths add."""
    data,labels = box_mesh(edges=((0.,1.,2.),(0.,1.),(0.,1.)))
    tensor = np.zeros((2,3,3,3,3,3))
    tensor[0,...,1,2],tensor[0,...,2,1] = -2.,2.
    tensor[1,...,1,2],tensor[1,...,2,1] = 2.,-2.
    for element,sign in zip(data.elem,[1.,-1.]):
        x,y,z=element.pos
        element.vel=np.stack((0*x,-2*sign*z,2*sign*y))
    expected,_,groups=explicit_assembly(data,labels,tensor)
    fields=compute_projected_vortex_fields(data,periodic_axes=())
    check(fields.gradients,expected,3e-12,record_property,'projected_rotation_tensor')
    projected_strength=compute_swirling_strength(fields.gradients).lambda_ci
    scalar_first=project_scalar_field(build_l2_projection_operator(data,periodic_axes=()),np.full(tensor.shape[:4],2.))
    for contributors in groups.values():
        if len(contributors)==2:
            for local in contributors:
                check(projected_strength[local],0.,3e-12,record_property,'cancelled_rotation')
                assert scalar_first[local] == pytest.approx(2.)


@pytest.mark.parametrize('count',[8,10,12])
@pytest.mark.parametrize('reverse',[False,True])
def test_p7_reconstruction_analytic_tensor(count,reverse,monkeypatch,record_property):
    """Exact degree-seven velocity and analytic block eigenvalues at output xyz."""
    import nek_post.vortex_3d_reconstruction as module
    data,_=box_mesh((0,2,1),reverse,n=8,edges=((0.,.4,1.),(0.,1.),(0.,1.)))
    for element in data.elem:
        x,y,z=element.pos
        element.vel=np.stack((x**7/7,y**7/7-z,(1+x*x)*y+z**7/7))
    fields=compute_projected_vortex_fields(data,periodic_axes=())
    captured=[]
    original=module.compute_swirling_strength
    def observe(tensor,**kwargs):
        captured.append(tensor.copy())
        return original(tensor,**kwargs)
    monkeypatch.setattr(module,'compute_swirling_strength',observe)
    grid,_=build_reconstructed_vortex_grid(data,fields,sampling_nodes=count)
    # The grid stores original local ordering while captured tensors are canonical.
    for ei,element in enumerate(data.elem):
        low,high=element.pos[0].min(),element.pos[0].max()
        q=reference_rule(count)[0]
        z,y,x=np.meshgrid((q+1)/2,(q+1)/2,low+(high-low)*(q+1)/2,indexing='ij')
        expected=np.zeros((*x.shape,3,3))
        expected[...,0,0]=x**6
        expected[...,1,1],expected[...,1,2]=y**6,-1.
        expected[...,2,0],expected[...,2,1],expected[...,2,2]=2*x*y,1+x*x,z**6
        check(captured[ei],expected,4e-12,record_property,'reconstructed_tensor')
    x,y,z=grid.points.T
    check(grid['omega_x'],2+x*x,4e-12,record_property,'reconstructed_curl')
    check(grid['lambda_ci'],np.sqrt(1+x*x-((y**6-z**6)/2)**2),4e-12,record_property,'reconstructed_lambda')
    # All nine polynomial components, including degree seven in each input axis.
    t,s,r=np.meshgrid(*([reference_rule(8)[0]]*3),indexing='ij')
    target=np.meshgrid(*([reference_rule(count)[0]]*3),indexing='ij')
    for i,j in product(range(3),repeat=2):
        def polynomial(t,s,r):
            return (i+1)*t**7*s**(j+1)+(j+1)*r**7+t*s*r+i-j
        check(refine_gll_tensor3(polynomial(t,s,r),target_node_count=count),polynomial(*target),4e-12,record_property,'P7_component')


@pytest.mark.parametrize('strain,rotation', [(0.,0.),(3.,0.),(.5,2.),(2.,.5),(1.,1.)])
def test_rotation_plus_extension(strain,rotation,record_property):
    """Characteristic quadratic is lambda^2 = strain^2 - rotation^2."""
    tensor = np.array([[strain,-rotation,0.],[rotation,-strain,0.],[0.,0.,.3]])
    expected = np.sqrt(max(0.,rotation**2-strain**2))
    check(compute_swirling_strength(tensor).lambda_ci,expected,3e-14,record_property,'strain_rotation')


def test_dense_tensor_before_eigenvalues(record_property):
    """Analytic sqrt(1+x^2) exposes eigenvalues after tensor reconstruction."""
    data,_=box_mesh(n=8,edges=((0.,1.),)*3)
    element=data.elem[0]
    x,y,z=element.pos
    element.vel=np.stack((0*x,-z,(1+x*x)*y))
    fields=compute_projected_vortex_fields(data,periodic_axes=())
    grid,_=build_reconstructed_vortex_grid(data,fields,sampling_nodes=12)
    check(grid['lambda_ci'],np.sqrt(1+grid.points[:,0]**2),3e-12,record_property,'dense_nonlinear_order')
    scalar_first=refine_gll_tensor3(np.sqrt(1+x*x),target_node_count=12)
    q=(reference_rule(12)[0]+1)/2
    difference=float(np.max(np.abs(scalar_first[0,0,:]-np.sqrt(1+q*q))))
    record_property('scalar_interpolation_discrepancy',difference)
    assert difference > 1e-9


@pytest.mark.parametrize('periodic',[(),('x','y')])
def test_p7_independent_mass_moments(periodic,record_property):
    """P7 unequal volumes with moment-solved weights and explicit entity labels."""
    data,labels=box_mesh((1,0,2),True,periodic,n=8)
    values=np.arange(8*8**3,dtype=float).reshape((8,8,8,8))/1000
    weights_1d=reference_rule(8)[1]
    masses=np.empty_like(values)
    groups={}
    for ei,element in enumerate(data.elem):
        for node,label in labels[ei].items():
            local=(ei,*node)
            masses[local]=element.volume/8*np.prod(weights_1d[list(node)])
            groups.setdefault(label,[]).append(local)
    expected=np.empty_like(values)
    for contributors in groups.values():
        mass=sum(masses[p] for p in contributors)
        mean=sum(masses[p]*values[p] for p in contributors)/mass
        for local in contributors:
            expected[local]=mean
    operator=build_l2_projection_operator(data,periodic_axes=periodic)
    check(operator.weights,masses,5e-13,record_property,'P7_local_mass')
    check(project_scalar_field(operator,values),expected,5e-12,record_property,'P7_recovery')
