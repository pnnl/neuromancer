from hypothesis import given, settings, strategies as st
import torch
import torch.nn as nn
import slim
import numpy as np
from neuromancer import dynamics, estimators, integrators, blocks
from neuromancer.constraint import variable
import neuromancer.simulator as sim
import psl
psl_nonauto_systems = [psl.nonautonomous.systems['TwoTank'],
                       psl.nonautonomous.systems['CSTR'],
                       psl.nonautonomous.systems['HindmarshRose'],
                       psl.nonautonomous.systems['LorenzControl']]

psl_nonauto_time_systems = [psl.autonomous.systems['Duffing'],
                            psl.autonomous.systems['UniversalOscillator']]

psl_auto_systems = [psl.autonomous.systems['ThomasAttractor'],
                     psl.autonomous.systems['LorenzSystem'],
                     psl.autonomous.systems['VanDerPol']]

psl_ssm_systems_names = ['Reno_full',
                         'HollandschHuys_full',
                         'SimpleSingleZone']

"""
Test functions for neuromancer simulator classes 
testing shapes, datatypes, and keys
"""

@given(
    st.integers(1, 30),
    st.integers(1, 30),
    st.integers(1, 30),
)
@settings(max_examples=200, deadline=None)
def test_DynamicsLinSSM_shapes_types(nx, nu, ny):
    if ny > nx:
        ny = nx
    A = np.random.randn(nx, nx)
    B = np.random.randn(nx, nu)
    C = np.random.randn(ny, nx)
    sys = sim.DynamicsLinSSM(A, B, C)
    input_dict = {'x': np.random.randn(nx),
                  'u': np.random.randn(nu)}
    output_dict = sys.step(input_dict)
    assert isinstance(output_dict['x'], np.ndarray)
    assert isinstance(output_dict['y'], np.ndarray)
    assert output_dict['x'].shape == (nx,)
    assert output_dict['y'].shape == (ny,)


@given(
    st.sampled_from(psl_ssm_systems_names),
)
@settings(max_examples=5, deadline=None)
def test_DynamicsPSL_ssm_shapes_types(system_name):
    psl_model = psl.ssm.systems[system_name](system=system_name)
    sys = sim.DynamicsPSL(psl_model, input_key_map={'u': 'u'})
    input_dict = {'x': np.random.randn(psl_model.nx),
                  'u': np.random.randn(psl_model.nu)}
    output_dict = sys.step(input_dict)
    assert isinstance(output_dict['x'], np.ndarray)
    assert isinstance(output_dict['y'], np.ndarray)
    assert output_dict['x'].shape == (psl_model.nx,)
    assert output_dict['y'].shape == (psl_model.ny,)


@given(
    st.sampled_from(psl_nonauto_systems),
)
@settings(max_examples=5, deadline=None)
def test_DynamicsPSL_nonauto_shapes_types(system):
    psl_model = system()
    sys = sim.DynamicsPSL(psl_model, input_key_map={'u': 'u'})
    input_dict = {'x': abs(np.random.randn(psl_model.nx)),
                  'u': abs(np.random.randn(psl_model.nu))}
    output_dict = sys.step(input_dict)
    assert isinstance(output_dict['x'], np.ndarray)
    assert isinstance(output_dict['y'], np.ndarray)
    assert output_dict['x'].shape == (psl_model.nx,)
    assert output_dict['y'].shape == (psl_model.nx * 2,)


@given(
    st.sampled_from(psl_nonauto_time_systems),
)
@settings(max_examples=5, deadline=None)
def test_DynamicsPSL_nonauto_time_shapes_types(system):
    psl_model = system()
    sys = sim.DynamicsPSL(psl_model, input_key_map={'Time': 'Time'})
    input_dict = {'x': np.random.randn(psl_model.nx),
                  'Time': np.arange(0, 1) * psl_model.ts}
    output_dict = sys.step(input_dict)
    assert isinstance(output_dict['x'], np.ndarray)
    assert isinstance(output_dict['y'], np.ndarray)
    assert output_dict['x'].shape == (psl_model.nx,)
    assert output_dict['y'].shape == (psl_model.nx,)


@given(
    st.sampled_from(psl_auto_systems),
)
@settings(max_examples=5, deadline=None)
def test_DynamicsPSL_nauto_shapes_types(system):
    psl_model = system()
    sys = sim.DynamicsPSL(psl_model)
    input_dict = {'x': np.random.randn(psl_model.nx)}
    output_dict = sys.step(input_dict)
    assert isinstance(output_dict['x'], np.ndarray)
    assert isinstance(output_dict['y'], np.ndarray)
    assert output_dict['x'].shape == (psl_model.nx,)
    assert output_dict['y'].shape == (psl_model.nx,)


@given(
    st.integers(1, 30),
    st.integers(1, 30),
    st.integers(1, 30),
    st.integers(1, 40),
    st.integers(1, 10),
)
@settings(max_examples=400, deadline=None)
def test_DynamicsNeuromancer_SMM_shapes_types(nx, nu, ny, hsize, nlayers):
    if ny > nx:
        ny = nx
    hsizes = [hsize for k in range(nlayers)]
    fx, fu = [blocks.MLP(insize, nx, hsizes=hsizes) for insize in [nx, nu]]
    fy = blocks.MLP(nx, ny, hsizes=hsizes)
    model = dynamics.BlockSSM(fx, fy, fu=fu, name='block_ssm')
    sys = sim.DynamicsNeuromancer(model, input_key_map={'u': 'u'})
    input_dict = {'x': np.random.randn(nx),
                  'u': np.random.randn(nu)}
    output_dict = sys.step(input_dict)
    assert isinstance(output_dict['x'], np.ndarray)
    assert isinstance(output_dict['y'], np.ndarray)
    assert output_dict['x'].shape == (nx,)
    assert output_dict['y'].shape == (ny,)


@given(
    st.integers(1, 30),
    st.integers(1, 30),
    st.integers(1, 30),
    st.integers(1, 40),
    st.integers(1, 10),
)
@settings(max_examples=400, deadline=None)
def test_DynamicsNeuromancer_ODE_shapes_types(nx, nu, ny, hsize, nlayers):
    if ny > nx:
        ny = nx
    black_box_ode = blocks.MLP(insize=nx + nu, outsize=nx, hsizes=nlayers*[hsize])
    interp_u = lambda tq, t, u: u
    fx_int = integrators.RK4(black_box_ode, interp_u=interp_u, h=1.)
    fy = slim.maps['identity'](nx, ny)
    model = dynamics.ODENonAuto(fx_int, fy, extra_inputs=['Uf'],
                                    input_key_map={"x0": "x1", 'Yf': 'Yf', 'Uf': 'Uf1'},
                                    name='dynamics', online_flag=False)
    sys = sim.DynamicsNeuromancer(model, input_key_map={'x': 'x', 'u': 'u'})
    input_dict = {'x': np.random.randn(nx),
                  'u': np.random.randn(nu)}
    output_dict = sys.step(input_dict)
    assert isinstance(output_dict['x'], np.ndarray)
    assert isinstance(output_dict['y'], np.ndarray)
    assert output_dict['x'].shape == (nx,)
    assert output_dict['y'].shape == (ny,)


@given(
    st.integers(1, 30),
    st.integers(1, 30),
    st.integers(1, 30),
)
@settings(max_examples=200, deadline=None)
def test_ControllerLinear_shapes_types(nx, nu, nd):
    K = np.random.rand(nu, nx+nd)
    sys = sim.ControllerLinear(policy=K, input_keys=['x', 'd'])
    input_dict = {'x': np.random.randn(nx), 'd': np.random.randn(nd)}
    output_dict = sys.step(input_dict)
    assert isinstance(output_dict['u'], np.ndarray)
    assert output_dict['u'].shape == (nu,)


@given(
    st.integers(1, 30),
    st.integers(1, 30),
)
@settings(max_examples=200, deadline=None)
def test_ControllerCallable_shapes_types(nx, nd):
    func = lambda x: x - x ** 2
    sys = sim.ControllerCallable(policy=func, input_keys=['x', 'd'])
    input_dict = {'x': np.random.randn(nx), 'd': np.random.randn(nd)}
    output_dict = sys.step(input_dict)
    assert isinstance(output_dict['u'], np.ndarray)
    assert output_dict['u'].shape == (nx+nd,)


@given(
    st.integers(1, 30),
    st.integers(1, 30),
    st.integers(1, 30),
    st.integers(1, 30),
    st.integers(1, 10),
)
@settings(max_examples=400, deadline=None)
def test_ControllerPytorch1_shapes_types(nx, nd, nu, n_hidden, nlayers):
    block = blocks.MLP(nx + nd, nu, hsizes=nlayers*[n_hidden])
    sys = sim.ControllerPytorch(policy=block, input_keys=['x', 'd'])
    input_dict = {'x': np.random.randn(nx), 'd': np.random.randn(nd)}
    output_dict = sys.step(input_dict)
    assert isinstance(output_dict['u'], np.ndarray)
    assert output_dict['u'].shape == (nu,)


@given(
    st.integers(1, 30),
    st.integers(1, 30),
    st.integers(1, 30),
    st.integers(1, 30),
)
@settings(max_examples=400, deadline=None)
def test_ControllerPytorch2_shapes_types(nx, nd, nu, n_hidden):
    controller = nn.Sequential(nn.Linear(nx + nd, n_hidden),
                               nn.ReLU(),
                               nn.Linear(n_hidden, nu),
                               nn.Sigmoid())
    sys = sim.ControllerPytorch(policy=controller, input_keys=['x', 'd'])
    input_dict = {'x': np.random.randn(nx), 'd': np.random.randn(nd)}
    output_dict = sys.step(input_dict)
    assert isinstance(output_dict['u'], np.ndarray)
    assert output_dict['u'].shape == (nu,)


@given(
    st.integers(1, 30),
    st.integers(1, 30),
    st.integers(1, 30),
)
@settings(max_examples=200, deadline=None)
def test_EstimatorLinear_shapes_types(ny, nx, nd):
    if ny > nx:
        ny = nx
    K = np.random.rand(nx, ny + nd)
    sys = sim.EstimatorLinear(estimator=K, input_keys=['y', 'd'])
    input_dict = {'y': np.random.randn(ny), 'd': np.random.randn(nd)}
    output_dict = sys.step(input_dict)
    assert isinstance(output_dict['x'], np.ndarray)
    assert output_dict['x'].shape == (nx,)


@given(
    st.integers(1, 30),
    st.integers(1, 30),
)
@settings(max_examples=200, deadline=None)
def test_EstimatorCallable_shapes_types(ny, nd):
    func = lambda x: x - x ** 2
    sys = sim.EstimatorCallable(estimator=func, input_keys=['y', 'd'])
    input_dict = {'y': np.random.randn(ny), 'd': np.random.randn(nd)}
    output_dict = sys.step(input_dict)
    assert isinstance(output_dict['x'], np.ndarray)
    assert output_dict['x'].shape == (ny+nd,)


@given(
    st.integers(1, 30),
    st.integers(1, 30),
    st.integers(1, 30),
    st.integers(1, 30),
    st.integers(1, 10),
)
@settings(max_examples=400, deadline=None)
def test_EstimatorPytorch1_shapes_types(nx, ny, nd, n_hidden, nlayers):
    if ny > nx:
        ny = nx
    block = blocks.MLP(ny + nd, nx, hsizes=nlayers*[n_hidden])
    sys = sim.EstimatorPytorch(estimator=block, input_keys=['y', 'd'])
    input_dict = {'y': np.random.randn(ny), 'd': np.random.randn(nd)}
    output_dict = sys.step(input_dict)
    assert isinstance(output_dict['x'], np.ndarray)
    assert output_dict['x'].shape == (nx,)


@given(
    st.integers(1, 30),
    st.integers(1, 30),
    st.integers(1, 30),
    st.integers(1, 30),
)
@settings(max_examples=300, deadline=None)
def test_EstimatorPytorch2_shapes_types(nx, ny, nd, n_hidden):
    if ny > nx:
        ny = nx
    block = nn.Sequential(nn.Linear(ny + nd, n_hidden),
                              nn.ReLU(),
                              nn.Linear(n_hidden, nx),
                              nn.Sigmoid())
    sys = sim.EstimatorPytorch(estimator=block, input_keys=['y', 'd'])
    input_dict = {'y': np.random.randn(ny), 'd': np.random.randn(nd)}
    output_dict = sys.step(input_dict)
    assert isinstance(output_dict['x'], np.ndarray)
    assert output_dict['x'].shape == (nx,)


@given(
    st.integers(1, 100),
)
@settings(max_examples=20, deadline=None)
def test_SystemConstraints_con_shapes_types(nx):
    x = variable("x")
    xmin = variable("xmin")
    xmax = variable("xmax")
    con1 = (x > xmin)
    con2 = (x < xmax)
    con3 = (x == 1)
    constraints = [con1, con2, con3]
    sys = sim.SystemConstraints(constraints)
    input_dict = {'x': np.random.randn(nx),
                  'xmin': -np.ones(nx), 'xmax': np.ones(nx)}
    output_dict = sys.step(input_dict)
    for con in constraints:
        for key in con.output_keys:
            assert isinstance(output_dict[key], np.ndarray)
        assert output_dict[con.output_keys[0]].shape == ()
        assert output_dict[con.output_keys[1]].shape == (nx,)
        assert output_dict[con.output_keys[2]].shape == (nx,)


@given(
    st.integers(1, 100),
)
@settings(max_examples=20, deadline=None)
def test_SystemConstraints_obj_shapes_types(nx):
    x = variable("x")
    y = variable("y")
    f1 = (1 - x) ** 2 + (y - x ** 2) ** 2
    obj1 = f1.minimize(weight=1., name='obj1')
    f2 = x ** 2 + y ** 2
    obj2 = f2.minimize(weight=1., name='obj2')
    objectives = [obj1, obj2]
    sys = sim.SystemConstraints(objectives)
    input_dict = {'x': np.random.randn(nx),
                  'y': np.random.randn(nx)}
    output_dict = sys.step(input_dict)
    for obj in objectives:
        assert isinstance(output_dict[obj.output_keys[0]], np.ndarray)
        assert output_dict[obj.output_keys[0]].shape == ()


@given(
    st.integers(1, 100),
    st.integers(1, 100),
    st.integers(1, 50),
)
@settings(max_examples=200, deadline=None)
def test_MovingHorizon_shapes_types(nx, nu, nsteps):
    sys = sim.MovingHorizon(input_keys=['x', 'u'], nsteps=nsteps)
    input_dict = {'x': np.random.randn(nx),
                  'u': np.random.randn(nu)}
    output_dict = sys.step(input_dict)
    assert isinstance(output_dict['x_p'], np.ndarray)
    assert output_dict['x_p'].shape == (nsteps, nx)
    assert isinstance(output_dict['u_p'], np.ndarray)
    assert output_dict['u_p'].shape == (nsteps, nu)


@given(
    st.integers(1, 100),
    st.integers(1, 100),
    st.integers(1, 50),
)
@settings(max_examples=10, deadline=None)
def test_MovingHorizon_name(nx, nu, nsteps):
    sys = sim.MovingHorizon(input_keys=['x', 'u'], nsteps=nsteps, name='mh')
    input_dict = {'x': np.random.randn(nx),
                  'u': np.random.randn(nu)}
    output_dict = sys.step(input_dict)
    assert isinstance(output_dict['x_mh'], np.ndarray)
    assert output_dict['x_mh'].shape == (nsteps, nx)
    assert isinstance(output_dict['u_mh'], np.ndarray)
    assert output_dict['u_mh'].shape == (nsteps, nu)


@given(
    st.sampled_from(psl_nonauto_systems),
    st.integers(1, 50),
    st.integers(1, 10),
    st.integers(1, 20),
)
@settings(max_examples=20, deadline=None)
def test_SystemSimulator_systemID_shapes_types(system, sim_steps, nx, hsize):
    psl_model = system()
    ny = psl_model.nx
    nu = psl_model.nu
    sys_psl = sim.DynamicsPSL(psl_model, input_key_map={'x': 'x_psl', 'u': 'u'}, name='psl')
    fx, fu = [blocks.MLP(insize, nx, hsizes=2*[hsize]) for insize in [nx, nu]]
    fy = blocks.MLP(nx, ny, hsizes=[hsize])
    model = dynamics.BlockSSM(fx, fy, fu=fu, name='block_ssm')
    sys_nm = sim.DynamicsNeuromancer(model, input_key_map={'x': 'x_nm', 'u': 'u'}, name='nm')
    components = [sys_psl, sys_nm]
    sys = sim.SystemSimulator(components)
    data_init = {'x_psl': np.asarray(psl_model.x0),
                 'x_nm': np.random.randn(nx)}
    raw = psl_model.simulate()
    data_traj = {'u': raw['U'][:sim_steps + 1, :]}
    trajectories = sys.simulate(nsim=sim_steps, data_init=data_init,
                                      data_traj=data_traj)
    assert isinstance(trajectories['x_psl'], np.ndarray)
    assert isinstance(trajectories['y_psl'], np.ndarray)
    assert isinstance(trajectories['x_nm'], np.ndarray)
    assert isinstance(trajectories['y_nm'], np.ndarray)
    assert isinstance(trajectories['u'], np.ndarray)
    assert trajectories['x_psl'].shape == (sim_steps+1, ny)
    assert trajectories['y_psl'].shape == (sim_steps, ny)
    assert trajectories['x_nm'].shape == (sim_steps+1, nx)
    assert trajectories['y_nm'].shape == (sim_steps, ny)
    assert trajectories['u'].shape == (sim_steps+1, nu)


@given(
    st.integers(1, 100),
    st.integers(1, 100),
)
@settings(max_examples=200, deadline=None)
def test_SystemSimulator_control_shapes_types(nx, nu):
    pass


