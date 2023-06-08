import torch
from hypothesis import given, settings, strategies as st
from neuromancer.modules.activations import activations

activations = [v for k, v in activations.items()]


@given(st.integers(1, 500),
       st.integers(1, 500),
       st.sampled_from(activations))
@settings(max_examples=1000, deadline=None)
def test_act_shape(batchsize, insize, act):
    model = act()
    x = torch.randn([batchsize, insize])
    y = model(x)
    assert y.shape[0] == batchsize and y.shape[1] == insize


@given(st.integers(1, 500),
       st.integers(1, 500),
       st.sampled_from(activations))
@settings(max_examples=1000, deadline=None)
def test_NaN_init(batchsize, insize, act):
    model = act()
    x = torch.randn([batchsize, insize])
    y = model(x)
    assert not y.isnan().any()