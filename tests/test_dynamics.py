import torch
from neuromancer import dynamics
from neuromancer import blocks
from neuromancer import integrators
from neuromancer import interpolation
from hypothesis import given, settings, strategies as st
import inspect
from neuromancer.blocks import MLP
import neuromancer.slim as slim


blks = [v for k, v in blocks.blocks.items() if 'hsizes' in inspect.signature(v).parameters]


@given(st.integers(1, 10),
       st.integers(1, 3),
       st.integers(1, 5),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 5),
       st.integers(1, 3),
       st.sampled_from(blks))
@settings(max_examples=1000, deadline=None)
def test_block_ssm_shape(samples, nsteps, nx, ny, nu, nd, hsize, nlayers, blk):
    x = torch.rand(samples, nx)
    U = torch.rand(samples, nsteps, nu)
    D = torch.rand(samples, nsteps, nd)
    Y = torch.rand(samples, nsteps, ny)
    data = {'x0': x, 'Up': U, 'Uf': U, 'Dp': D, 'Df': D, 'Yf': Y}
    hsizes = [hsize for k in range(nlayers)]
    fx, fu, fd = [blk(insize, nx, hsizes=hsizes) for insize in [nx, nu, nd]]
    fy = blk(nx, ny, hsizes=hsizes)
    model = dynamics.BlockSSM(fx, fy, fu=fu, fd=fd, name='block_ssm')
    output = model(data)

    assert output['X_pred_block_ssm'].shape[1] == nsteps
    assert output['X_pred_block_ssm'].shape[0] == samples
    assert output['X_pred_block_ssm'].shape[2] == nx
    assert output['Y_pred_block_ssm'].shape[1] == nsteps
    assert output['Y_pred_block_ssm'].shape[0] == samples
    assert output['Y_pred_block_ssm'].shape[2] == ny
    assert output['fU_block_ssm'].shape[1] == nsteps
    assert output['fU_block_ssm'].shape[0] == samples
    assert output['fU_block_ssm'].shape[2] == nx
    assert output['fD_block_ssm'].shape[1] == nsteps
    assert output['fD_block_ssm'].shape[0] == samples
    assert output['fD_block_ssm'].shape[2] == nx


@given(st.integers(1, 10),
       st.integers(1, 3),
       st.integers(1, 5),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 5),
       st.integers(1, 3),
       st.sampled_from(blks))
@settings(max_examples=1000, deadline=None)
def test_black_ssm_shape(samples, nsteps, nx, ny, nu, nd, hsize, nlayers, blk):
    x = torch.rand(samples, nx)
    U = torch.rand(samples, nsteps, nu)
    D = torch.rand(samples, nsteps, nd)
    Y = torch.rand(samples, nsteps, ny)
    data = {'x0': x, 'Up': U, 'Uf': U, 'Dp':D, 'Df': D, 'Yf': Y}
    hsizes = [hsize for k in range(nlayers)]
    fxud = blk(nx + nu + nd, nx, hsizes=hsizes)
    fy = blk(nx, ny, hsizes=hsizes)
    model = dynamics.BlackSSM(fxud, fy, extra_inputs=['Uf', 'Df'])
    output = model(data)

    assert output['X_pred_black_ssm'].shape[1] == nsteps
    assert output['X_pred_black_ssm'].shape[0] == samples
    assert output['X_pred_black_ssm'].shape[2] == nx
    assert output['Y_pred_black_ssm'].shape[1] == nsteps
    assert output['Y_pred_black_ssm'].shape[0] == samples
    assert output['Y_pred_black_ssm'].shape[2] == ny


integrators_generic = [v for v in integrators.integrators.values()]


@given(st.integers(1, 10),
       st.integers(1, 3),
       st.integers(1, 5),
       st.integers(1, 5),
       st.integers(1, 3),
       st.sampled_from(integrators_generic),
       )
@settings(max_examples=1000, deadline=None)
def test_ode_auto_shape(samples, nsteps, nx, hsize, nlayers, integrator):
    x = torch.rand(samples, nx)
    Y = torch.rand(samples, nsteps, nx)
    data = {'x0': x, 'Yf': Y}
    hsizes = [hsize for k in range(nlayers)]
    fx = MLP(nx, nx, hsizes=hsizes)
    fxRK4 = integrator(fx, h=1.0)
    fy = slim.maps['identity'](nx, nx)
    model = dynamics.ODEAuto(fxRK4, fy, name='neural_ode')
    output = model(data)

    assert output[f'X_pred_{model.name}'].shape[1] == nsteps
    assert output[f'X_pred_{model.name}'].shape[0] == samples
    assert output[f'X_pred_{model.name}'].shape[2] == nx
    assert output[f'Y_pred_{model.name}'].shape[1] == nsteps
    assert output[f'Y_pred_{model.name}'].shape[0] == samples
    assert output[f'Y_pred_{model.name}'].shape[2] == nx


@given(st.integers(1, 10),
       st.integers(2, 10),
       st.integers(1, 5),
       st.integers(1, 5),
       st.integers(1, 3),
       st.sampled_from(integrators_generic),
       st.sampled_from([True, False])
       )
@settings(max_examples=200, deadline=None)
def test_ode_nonauto_time_shape(samples, nsteps, nx, hsize, nlayers, integrator, online_flag):
    # create input interpolator
    if online_flag:
        interp_u = interpolation.LinInterp_Online()
    else:
        t = (torch.arange(nsteps) * 10).unsqueeze(-1)
        interp_u = interpolation.LinInterp_Offline(t, t)

    # create NODE model
    hsizes = [hsize for k in range(nlayers)]
    fx = MLP(nx+1, nx, bias=True, hsizes=hsizes, linear_map=slim.maps['linear'])
    fxRK4 = integrator(fx, interp_u=interp_u, h=0.1)
    fy = slim.maps['identity'](nx, nx)
    model = dynamics.ODENonAuto(fxRK4, fy, name='neural_ode', online_flag = online_flag)

    # test model
    x = torch.rand(samples, nx)
    Y = torch.rand(samples, nsteps, nx)
    Time = torch.rand(samples, nsteps, 1)
    data = {'x0': x, 'Yf': Y, 'Time': Time}
    output = model(data)

    assert output[f'X_pred_{model.name}'].shape[1] == nsteps
    assert output[f'X_pred_{model.name}'].shape[0] == samples
    assert output[f'X_pred_{model.name}'].shape[2] == nx
    assert output[f'Y_pred_{model.name}'].shape[1] == nsteps
    assert output[f'Y_pred_{model.name}'].shape[0] == samples
    assert output[f'Y_pred_{model.name}'].shape[2] == nx


@given(st.integers(1, 10),
       st.integers(2, 10),
       st.integers(1, 5),
       st.integers(1, 5),
       st.integers(1, 5),
       st.integers(1, 3),
       st.sampled_from(integrators_generic),
       st.sampled_from([True, False])
)
@settings(max_examples=200, deadline=None)
def test_ode_nonauto_inputs_shape(samples, nsteps, nx, nu, hsize, nlayers, integrator, online_flag):
    # create input interpolator
    if online_flag:
        interp_u = interpolation.LinInterp_Online()
    else:
        t = (torch.arange(nsteps) * 10).unsqueeze(-1)
        u = torch.sin(t).repeat(1, nu)
        interp_u = interpolation.LinInterp_Offline(t, u)

    # create NODE model
    hsizes = [hsize for k in range(nlayers)]
    fx = MLP(nx+nu, nx, bias=True, hsizes=hsizes, linear_map=slim.maps['linear'])
    fxRK4 = integrator(fx, interp_u=interp_u, h=0.1)
    fy = slim.maps['identity'](nx, nx)
    model = dynamics.ODENonAuto(fxRK4, fy, extra_inputs=['Uf'], name='neural_ode', online_flag = online_flag)

    # test model
    x = torch.rand(samples, nx)
    U = torch.rand(samples, nsteps, nu)
    Y = torch.rand(samples, nsteps, nx)
    Time = torch.rand(samples, nsteps, 1)
    data = {'x0': x, 'Yf': Y, 'Uf': U, 'Time': Time}
    output = model(data)

    assert output[f'X_pred_{model.name}'].shape[1] == nsteps
    assert output[f'X_pred_{model.name}'].shape[0] == samples
    assert output[f'X_pred_{model.name}'].shape[2] == nx
    assert output[f'Y_pred_{model.name}'].shape[1] == nsteps
    assert output[f'Y_pred_{model.name}'].shape[0] == samples
    assert output[f'Y_pred_{model.name}'].shape[2] == nx


