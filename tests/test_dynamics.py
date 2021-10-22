import torch
from neuromancer import dynamics
from neuromancer import blocks
from hypothesis import given, settings, strategies as st
import math
import inspect

blks = [v for k, v in blocks.blocks.items() if 'hsizes' in inspect.signature(v).parameters]


@given(st.integers(1, 10),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 5),
       st.integers(1, 3),
       st.sampled_from(blks))
@settings(max_examples=1000, deadline=None)
def test_block_ssm_shape(samples, nsteps, nx, ny, nu, nd, hsize, nlayers, blk):
    x = torch.rand(samples, nx)
    U = torch.rand(nsteps, samples, nu)
    D = torch.rand(nsteps, samples, nd)
    Y = torch.rand(nsteps, samples, ny)
    data = {'x0': x, 'Up': U, 'Uf': U, 'Dp': D, 'Df': D, 'Yf': Y}
    hsizes = [hsize for k in range(nlayers)]
    fx, fu, fd = [blk(insize, nx, hsizes=hsizes) for insize in [nx, nu, nd]]
    fy = blk(nx, ny, hsizes=hsizes)
    model = dynamics.BlockSSM(fx, fy, fu=fu, fd=fd, name='block_ssm')
    output = model(data)

    assert output['X_pred_block_ssm'].shape[0] == nsteps
    assert output['X_pred_block_ssm'].shape[1] == samples
    assert output['X_pred_block_ssm'].shape[2] == nx

    assert output['Y_pred_block_ssm'].shape[0] == nsteps
    assert output['Y_pred_block_ssm'].shape[1] == samples
    assert output['Y_pred_block_ssm'].shape[2] == ny

    assert output['fU_block_ssm'].shape[0] == nsteps
    assert output['fU_block_ssm'].shape[1] == samples
    assert output['fU_block_ssm'].shape[2] == nx

    assert output['fD_block_ssm'].shape[0] == nsteps
    assert output['fD_block_ssm'].shape[1] == samples
    assert output['fD_block_ssm'].shape[2] == nx


@given(st.integers(1, 10),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 5),
       st.integers(1, 3),
       st.sampled_from(blks))
@settings(max_examples=1000, deadline=None)
def test_black_ssm_shape(samples, nsteps, nx, ny, nu, nd, hsize, nlayers, blk):
    x = torch.rand(samples, nx)
    U = torch.rand(nsteps, samples, nu)
    D = torch.rand(nsteps, samples, nd)
    Y = torch.rand(nsteps, samples, ny)
    data = {'x0': x, 'Up': U, 'Uf': U, 'Dp':D, 'Df': D, 'Yf': Y}
    hsizes = [hsize for k in range(nlayers)]
    fxud = blk(nx + nu + nd, nx, hsizes=hsizes)
    fy = blk(nx, ny, hsizes=hsizes)
    model = dynamics.BlackSSM(fxud, fy, extra_inputs=['Uf', 'Df'])
    output = model(data)

    assert output['X_pred_black_ssm'].shape[0] == nsteps
    assert output['X_pred_black_ssm'].shape[1] == samples
    assert output['X_pred_black_ssm'].shape[2] == nx

    assert output['Y_pred_black_ssm'].shape[0] == nsteps
    assert output['Y_pred_black_ssm'].shape[1] == samples
    assert output['Y_pred_black_ssm'].shape[2] == ny


@given(st.integers(1, 10),
       st.integers(1, 5),
       st.floats(0.0, 1),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 5),
       st.integers(1, 3),
       st.sampled_from(blks))
@settings(max_examples=1000, deadline=None)
def test_tdblock_ssm_shape(samples, nsteps, time_delay, nx, ny, nu, nd, hsize, nlayers, blk):
    td = math.ceil((nsteps-1) * time_delay)
    nx_td = (td + 1) * nx
    nu_td = (td + 1) * nu
    nd_td = (td + 1) * nd
    x = torch.rand(td + 1, samples, nx)
    U = torch.rand(nsteps, samples, nu)
    D = torch.rand(nsteps, samples, nd)
    Y = torch.rand(nsteps, samples, ny)
    data = {'Xtd': x, 'Up': U, 'Uf': U, 'Dp': D, 'Df': D, 'Yf': Y}
    hsizes = [hsize for k in range(nlayers)]
    fx, fu, fd = [blk(insize, nx, hsizes=hsizes) for insize in [nx_td, nu_td, nd_td]]
    fy = blk(nx_td, ny, hsizes=hsizes)
    model = dynamics.TimeDelayBlockSSM(fx, fy, fu, fd, timedelay=td)
    output = model(data)

    assert output['X_pred_block_ssm'].shape[0] == nsteps
    assert output['X_pred_block_ssm'].shape[1] == samples
    assert output['X_pred_block_ssm'].shape[2] == nx

    assert output['Y_pred_block_ssm'].shape[0] == nsteps
    assert output['Y_pred_block_ssm'].shape[1] == samples
    assert output['Y_pred_block_ssm'].shape[2] == ny

    assert output['fU_block_ssm'].shape[0] == nsteps
    assert output['fU_block_ssm'].shape[1] == samples
    assert output['fU_block_ssm'].shape[2] == nx

    assert output['fD_block_ssm'].shape[0] == nsteps
    assert output['fD_block_ssm'].shape[1] == samples
    assert output['fD_block_ssm'].shape[2] == nx


@given(st.integers(1, 10),
       st.integers(1, 5),
       st.floats(0.0, 1),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 5),
       st.integers(1, 3),
       st.sampled_from(blks))
@settings(max_examples=1000, deadline=None)
def test_tdblack_ssm_shape(samples, nsteps, time_delay, nx, ny, nu, nd, hsize, nlayers, blk):
    td = math.ceil((nsteps-1) * time_delay)
    insize = (td + 1) * (nx + nu + nd)
    nx_td = (td + 1) * nx
    x = torch.rand(td + 1, samples, nx)
    U = torch.rand(nsteps, samples, nu)
    D = torch.rand(nsteps, samples, nd)
    Y = torch.rand(nsteps, samples, ny)
    data = {'Xtd': x, 'Up': U, 'Uf': U, 'Dp': D, 'Df': D, 'Yf': Y}
    hsizes = [hsize for k in range(nlayers)]
    fxud = blk(insize, nx, hsizes=hsizes)
    fy = blk(nx_td, ny, hsizes=hsizes)
    model = dynamics.TimeDelayBlackSSM(fxud, fy, timedelay=td, extra_inputs=['Up', 'Uf', 'Dp', 'Df'])
    output = model(data)

    assert output['X_pred_black_ssm'].shape[0] == nsteps
    assert output['X_pred_black_ssm'].shape[1] == samples
    assert output['X_pred_black_ssm'].shape[2] == nx

    assert output['Y_pred_black_ssm'].shape[0] == nsteps
    assert output['Y_pred_black_ssm'].shape[1] == samples
    assert output['Y_pred_black_ssm'].shape[2] == ny

