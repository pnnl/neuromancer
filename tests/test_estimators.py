import torch
from hypothesis import given, settings, strategies as st
import math

from neuromancer.estimators import estimators, seq2seq_estimators
from slim.linear import square_maps, maps
from neuromancer.activations import activations
import neuromancer.estimators as estim

rect_maps = [v for k, v in maps.items() if v not in square_maps]
activations = [v for k, v in activations.items()]
estimators = [v for k, v in estimators.items() if k != 'fullobservable']
seq2seq_estimators = [v for k, v in seq2seq_estimators.items()]


@given(st.integers(1, 10),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 5),
       st.integers(1, 3),
       st.booleans(),
       st.floats(0.1, 1.0),
       st.sampled_from(activations),
       st.sampled_from(estimators),
       st.sampled_from(rect_maps))
@settings(max_examples=1000, deadline=None)
def test_estimators_shape(samples, nsteps, nx, ny, nu, nd,
                          hsize, nlayers, bias, window, act, est, lin):
    data_dims = {'x0': (nx,), 'Yp': (samples, ny), 'Up': (samples, nu), 'Dp': (samples, nd)}
    input_keys = ['Yp', 'Up', 'Dp']
    U = torch.rand(nsteps, samples, nu)
    D = torch.rand(nsteps, samples, nd)
    Y = torch.rand(nsteps, samples, ny)
    data = {'Yp': Y, 'Up': U, 'Dp': D}
    hsizes = [hsize for k in range(nlayers)]

    model = est(data_dims, input_keys=input_keys, bias=bias, hsizes=hsizes, nsteps=nsteps,
                linear_map=lin, nonlin=act, window_size=math.ceil(window*nsteps))
    output = model(data)
    x0 = [v for k, v in output.items() if len(v.shape) == 2][0]
    assert x0.shape[0] == samples
    assert x0.shape[1] == nx


@given(st.integers(1, 10),
       st.integers(1, 3),
       st.integers(1, 3))
@settings(max_examples=1000, deadline=None)
def test_fully_observable_shape(samples, nsteps, nx):
    data_dims = {'x0': (nx,), 'Yp': (samples, nx)}
    Y = torch.rand(nsteps, samples, nx)
    data = {'Yp': Y}

    model = estim.FullyObservable(data_dims)
    output = model(data)
    x0 = [v for k, v in output.items() if len(v.shape) == 2][0]
    assert x0.shape[0] == samples
    assert x0.shape[1] == nx


@given(st.integers(1, 10),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 3),
       st.integers(1, 5),
       st.integers(1, 3),
       st.booleans(),
       st.floats(0.1, 1.0),
       st.sampled_from(activations),
       st.sampled_from(seq2seq_estimators),
       st.sampled_from(rect_maps),
       st.floats(0.1, 1.0))
@settings(max_examples=1000, deadline=None)
def test_seq2seq_estimators_shape(samples, nsteps, nx, ny, nu, nd,
                                  hsize, nlayers, bias, window, act, est, lin, td):
    data_dims = {'x0': (nx,), 'Yp': (samples, ny), 'Up': (samples, nu), 'Dp': (samples, nd)}
    input_keys = ['Yp', 'Up', 'Dp']
    U = torch.rand(nsteps, samples, nu)
    D = torch.rand(nsteps, samples, nd)
    Y = torch.rand(nsteps, samples, ny)
    data = {'Yp': Y, 'Up': U, 'Dp': D}
    hsizes = [hsize for k in range(nlayers)]
    time_delay = math.ceil(td*(nsteps-1))
    model = est(data_dims, input_keys=input_keys, bias=bias, hsizes=hsizes, nsteps=nsteps,
                linear_map=lin, nonlin=act, window_size=math.ceil(window*nsteps),
                timedelay=time_delay)
    output = model(data)
    print(samples, nsteps, nx, ny, nu, nd, hsize, nlayers, est)
    print({k: v.shape for k, v in output.items()})
    x0 = [v for k, v in output.items() if len(v.shape) == 3][0]
    print(time_delay)
    assert x0.shape[0] == time_delay + 1
    assert x0.shape[1] == samples
    assert x0.shape[2] == nx


# if __name__ == '__main__':
#     nx, ny, nu, nd = 15, 7, 5, 2
#     N = 40
#
#     samples = 1
#     # Data format: (N,samples,dim)
#     Y = torch.rand(N, samples, ny)
#     U = torch.rand(N, samples, nu)
#     D = torch.rand(N, samples, nd)
#     data = {'Yp': Y, 'Up': U, 'Dp': D}
#     data_dims = {'x0': (nx,), 'Yp': (N, ny), 'Up': (N, nu), 'Dp': (N, nd)}
#     input_keys = ['Yp']
#
#     for bias in [True, False]:
#         for name, est in estimators.items():
#             print(name)
#             e = est(data_dims, input_keys=input_keys)
#             e_out = e(data)
#             for k, v in e_out.items():
#                 print(f'{k}: {v.shape}')
#             for lin in set(slim.maps.values()) - slim.square_maps:
#                 print(lin)
#                 e = est(data_dims, input_keys=input_keys, bias=bias, linear_map=lin)
#                 e_out = e(data)
#                 for k, v in e_out.items():
#                     print(f'{k}: {v.shape}')
#
#     for bias in [True, False]:
#         for name, est in estimators.items():
#             print(name)
#             e = est(data_dims, nsteps=N, window_size=N, input_keys=input_keys)
#             e_out = e(data)
#             for k, v in e_out.items():
#                 print(f'{k}: {v.shape}')
#             for lin in set(slim.maps.values()) - slim.square_maps:
#                 print(lin)
#                 e = est(data_dims, nsteps=N, window_size=N, input_keys=input_keys, bias=bias, linear_map=lin)
#                 e_out = e(data)
#                 for k, v in e_out.items():
#                     print(f'{k}: {v.shape}')
#
#     for bias in [True, False]:
#         for name, est in estimators.items():
#             print(name)
#             e = est(data_dims, nsteps=N, window_size=N-1, input_keys=input_keys)
#             e_out = e(data)
#             for k, v in e_out.items():
#                 print(f'{k}: {v.shape}')
#             for lin in set(slim.maps.values()) - slim.square_maps:
#                 print(lin)
#                 e = est(data_dims, nsteps=N, window_size=N-1, input_keys=input_keys, bias=bias, linear_map=lin)
#                 e_out = e(data)
#                 for k, v in e_out.items():
#                     print(f'{k}: {v.shape}')
#
#     for bias in [True, False]:
#         for name, est in seq2seq_estimators.items():
#             print(name)
#             e = est(data_dims, nsteps=N, window_size=N, timedelay=N-1, input_keys=input_keys)
#             e_out = e(data)
#             for k, v in e_out.items():
#                 print(f'{k}: {v.shape}')
#             for lin in set(slim.maps.values()) - slim.square_maps:
#                 print(lin)
#                 e = est(data_dims, nsteps=N, window_size=N, timedelay=N-1, input_keys=input_keys, bias=bias, linear_map=lin)
#                 e_out = e(data)
#                 for k, v in e_out.items():
#                     print(f'{k}: {v.shape}')
#
#     for bias in [True, False]:
#         for name, est in seq2seq_estimators.items():
#             print(name)
#             e = est(data_dims, nsteps=N, window_size=N-1, timedelay=0, input_keys=input_keys)
#             e_out = e(data)
#             for k, v in e_out.items():
#                 print(f'{k}: {v.shape}')
#             for lin in set(slim.maps.values()) - slim.square_maps:
#                 print(lin)
#                 e = est(data_dims, nsteps=N, window_size=N-1, timedelay=0, input_keys=input_keys, bias=bias, linear_map=lin)
#                 e_out = e(data)
#                 for k, v in e_out.items():
#                     print(f'{k}: {v.shape}')
#
#
#     print('Kalman filter')
#     fx, fu, fd = [slim.Linear(insize, nx) for insize in [nx, nu, nd]]
#     fy = slim.Linear(nx, ny)
#     model = BlockSSM(fx, fy, fu, fd)
#     est = LinearKalmanFilter(model=model)
#     est_out = est(data)
#     for k, v in est_out.items():
#         print(f'{k}: {v.shape}')