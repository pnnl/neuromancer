import torch
from hypothesis import given, settings, strategies as st
from neuromancer.policies import policies, ConvolutionalForecastPolicy
from neuromancer.slim.linear import square_maps, maps, TrivialNullSpaceLinear
from neuromancer.activations import activations

rect_maps = [v for k, v in maps.items() if v not in square_maps and v is not TrivialNullSpaceLinear]
activations = [v for k, v in activations.items()]


@given(st.integers(1, 10),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 5),
       st.integers(1, 3),
       st.booleans(),
       st.sampled_from(activations),
       st.sampled_from(policies),
       st.sampled_from(rect_maps))
@settings(max_examples=1000, deadline=None)
def test_policies_shape(samples, nsteps, nx, ny, nu, nd,
                          hsize, nlayers, bias, act, pol, lin):

    x = torch.rand(samples, nx)
    D = torch.rand(samples, nsteps, nd)
    R = torch.rand(samples, nsteps, ny)

    data_dims = {'x0': (nx,), 'D': (nsteps, nd,), 'R': (nsteps, ny), 'U': (nsteps, nu)}
    data = {'x0': x, 'D': D, 'R': R}
    input_keys = ['x0', 'D', 'R']
    hsizes = [hsize for k in range(nlayers)]

    model = pol(data_dims, nsteps=nsteps, bias=bias, linear_map=lin, nonlin=act, hsizes=hsizes, input_keys=input_keys)
    output = model(data)
    upred = output[f'U_pred_{model.name}']
    assert upred.shape[1] == nsteps, 'Samples not equal'
    assert upred.shape[0] == samples, 'Steps not equal'
    assert upred.shape[2] == nu, 'Dimension not equal'


@given(st.integers(1, 10),
       st.integers(6, 15),
       st.integers(1, 5),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(2, 4))
@settings(max_examples=1000, deadline=None)
def test_cnv_forecast_policy_shape(samples, nsteps, nx, ny, nu, nd, kernel_size):
    x = torch.rand(samples, nx)
    U = torch.rand(samples, nsteps, nu)
    D = torch.rand(samples, nsteps, nd)
    Y = torch.rand(samples, nsteps, ny)
    data = {'x0': x, 'Up': U, 'Uf': U, 'Dp':D, 'Df': D, 'Yf': Y}
    data_dims = {'x0': (nx,), 'Df': (nsteps, nd,), 'R': (nsteps, ny), 'U': (nsteps, nu)}
    model = ConvolutionalForecastPolicy(data_dims,nsteps= nsteps,kernel_size= kernel_size)
    output = model(data)
    assert output['U_pred_CNV_policy'].shape[1] == nsteps
    assert output['U_pred_CNV_policy'].shape[0] == samples
    assert output['U_pred_CNV_policy'].shape[2] == nu
