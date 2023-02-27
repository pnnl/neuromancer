from neuromancer.blocks import Poly2, MLP, ResMLP, RNN, BilinearTorch, PytorchRNN, Linear
import torch
from hypothesis import given, settings, strategies as st
import neuromancer.slim as slim
from neuromancer.slim.linear import square_maps, maps
from neuromancer.activations import activations

rect_maps = [v for k, v in maps.items() if v not in square_maps and v is not slim.linear.TrivialNullSpaceLinear]
activations = [v for k, v in activations.items()]


@given(st.integers(1, 500),
       st.integers(1, 500),
       st.integers(1, 500),
       st.booleans(),
       st.lists(st.integers(1, 100),
                min_size=0, max_size=3),
       st.sampled_from(rect_maps),
       st.sampled_from(activations))
@settings(max_examples=100, deadline=None)
def test_linear_shape(batchsize, insize, outsize, bias, hsizes, lin, act):
    model = Linear(insize, outsize, bias=bias, hsizes=hsizes, linear_map=lin, nonlin=act)
    x = torch.randn([batchsize, insize])
    y = model(x)
    assert y.shape[0] == batchsize and y.shape[1] == outsize


@given(st.integers(1, 500),
       st.integers(1, 500),
       st.integers(1, 500),
       st.booleans(),
       st.lists(st.integers(1, 100),
                min_size=0, max_size=3),
       st.sampled_from(rect_maps),
       st.sampled_from(activations))
@settings(max_examples=100, deadline=None)
def test_mlp_shape(batchsize, insize, outsize, bias, hsizes, lin, act):
    model = MLP(insize, outsize, bias=bias, hsizes=hsizes, linear_map=lin, nonlin=act)
    x = torch.randn([batchsize, insize])
    y = model(x)
    assert y.shape[0] == batchsize and y.shape[1] == outsize


@given(st.integers(1, 500),
       st.integers(1, 500),
       st.integers(1, 500),
       st.booleans(),
       st.integers(1, 100),
       st.integers(1, 10),
       st.sampled_from(rect_maps),
       st.sampled_from(activations))
@settings(max_examples=100, deadline=None)
def test_res_mlp_shape(batchsize, insize, outsize, bias, hsize, nlayers, lin, act):
    model = ResMLP(insize, outsize, bias=bias, hsizes=[hsize for k in range(nlayers)], linear_map=lin, nonlin=act)
    x = torch.randn([batchsize, insize])
    y = model(x)
    assert y.shape[0] == batchsize and y.shape[1] == outsize


@given(st.integers(1, 11),
       st.integers(1, 500),
       st.integers(1, 500),
       st.integers(1, 500),
       st.booleans(),
       st.integers(1, 100),
       st.sampled_from([1]),
       st.sampled_from(rect_maps),
       st.sampled_from(activations),
       st.sampled_from([RNN, PytorchRNN]))
@settings(max_examples=100, deadline=None)
def test_rnns_shape(nsteps, batchsize, insize, outsize, bias, hsize, nlayers, lin, act, implementation):
    model = implementation(insize, outsize, bias=bias, linear_map=lin, nonlin=act, hsizes=[hsize for k in range(nlayers)])
    x = torch.randn([batchsize, nsteps, insize])
    y = model(x)
    assert y.shape[0] == batchsize and y.shape[1] == outsize


@given(st.integers(1, 500),
       st.integers(1, 500),
       st.integers(1, 500),
       st.sampled_from(rect_maps))
@settings(max_examples=100, deadline=None)
def test_bilinear_shape(batchsize, insize, outsize, lin):
    model = BilinearTorch(insize, outsize, linear_map=lin)
    x = torch.randn([batchsize, insize])
    y = model(x)
    assert y.shape[0] == batchsize and y.shape[1] == outsize


@given(st.integers(1, 500),
       st.integers(1, 500))
@settings(max_examples=100, deadline=None)
def test_poly2_shape(batchsize, insize):
    model = Poly2()
    x = torch.randn([batchsize, insize])
    y = model(x)
    assert y.shape[0] == batchsize and y.shape[1] == (insize*(insize + 1))/2 + insize


@given(st.integers(1, 500),
       st.integers(1, 500),
       st.integers(1, 500),
       st.booleans(),
       st.sampled_from(rect_maps))
@settings(max_examples=100, deadline=None)
def test_basis_linear_shape(batchsize, insize, outsize, bias, lin):
    model = Linear(insize, outsize, bias=bias, linear_map=lin)
    x = torch.randn([batchsize, insize])
    y = model(x)
    assert y.shape[0] == batchsize and y.shape[1] == outsize
