import torch
from hypothesis import given, settings, strategies as st

from neuromancer.policies import policies
from slim.linear import square_maps, maps
from neuromancer.activations import activations

rect_maps = [v for k, v in maps.items() if v not in square_maps]
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
    D = torch.rand(nsteps, samples, nd)
    R = torch.rand(nsteps, samples, ny)

    data_dims = {'x0': (nx,), 'D': (nsteps, nd,), 'R': (nsteps, ny), 'U': (nsteps, nu)}
    data = {'x0': x, 'D': D, 'R': R}
    input_keys = ['x0', 'D', 'R']
    hsizes = [hsize for k in range(nlayers)]

    model = pol(data_dims, nsteps=nsteps, bias=bias, linear_map=lin, nonlin=act, hsizes=hsizes, input_keys=input_keys)
    output = model(data)
    upred = output[f'U_pred_{model.name}']
    assert upred.shape[0] == nsteps, 'Samples not equal'
    assert upred.shape[1] == samples, 'Steps not equal'
    assert upred.shape[2] == nu, 'Dimension not equal'
