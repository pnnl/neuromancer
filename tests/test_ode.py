import torch
from neuromancer import ode
from hypothesis import given, settings, strategies as st
from neuromancer.blocks import MLP

ode_param_systems_auto = [v for v in ode.ode_param_systems_auto.values()]
ode_param_systems_nonauto = [v for v in ode.ode_param_systems_nonauto.values()]
ode_hybrid_systems_auto = [v for v in ode.ode_hybrid_systems_auto.values()]


@given(st.integers(1, 500),
       st.sampled_from(ode_param_systems_auto))
@settings(max_examples=200, deadline=None)
def test_ode_auto_param_shape(batchsize, ode):
    model = ode()
    x = torch.randn([batchsize, model.in_features])
    y = model(x)
    assert y.shape[0] == batchsize and y.shape[1] == model.out_features


@given(st.integers(1, 500),
       st.sampled_from(ode_param_systems_nonauto))
@settings(max_examples=200, deadline=None)
def test_ode_nonauto_param_shape(batchsize, ode):
    model = ode()
    x = torch.randn([batchsize, model.in_features])
    y = model(x)
    assert y.shape[0] == batchsize and y.shape[1] == model.out_features


@given(st.integers(1, 500),
       st.sampled_from(ode_hybrid_systems_auto))
@settings(max_examples=200, deadline=None)
def test_ode_auto_hybrid_shape(batchsize, ode):
    # this test is intented only for hybrid ode's whose black box parts map R^2 to R
    block = MLP(2, 1, bias=True, hsizes=[20, 20])
    model = ode(block)
    x = torch.randn([batchsize, model.in_features])
    y = model(x)
    assert y.shape[0] == batchsize and y.shape[1] == model.out_features
