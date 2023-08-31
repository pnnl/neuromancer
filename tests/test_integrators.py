import torch
from neuromancer.dynamics import ode, integrators
from hypothesis import given, settings, strategies as st
from neuromancer.modules.blocks import MLP
import neuromancer.slim as slim


integrators_generic = [v for v in integrators.integrators.values()]
ode_param_systems_auto = [v for v in ode.ode_param_systems_auto.values()]
ode_param_systems_nonauto = [v for v in ode.ode_param_systems_nonauto.values()]
ode_hybrid_systems_auto = [v for v in ode.ode_hybrid_systems_auto.values()]


@given(st.integers(1, 300),
       st.integers(1, 10),
       st.sampled_from(integrators_generic))
@settings(max_examples=200, deadline=None)
def test_integrator_black_ode_auto_shape(batchsize, nx, integrator):
    fx = MLP(nx, nx, bias=True, hsizes=[20, 20], linear_map=slim.maps['linear'])
    model = integrator(fx)
    x = torch.randn([batchsize, nx])
    y = model(x)
    assert y.shape[0] == batchsize and y.shape[1] == nx


@given(st.integers(1, 300),
       st.integers(1, 10),
       st.integers(1, 10),
       st.sampled_from(integrators_generic))
@settings(max_examples=200, deadline=None)
def test_integrator_black_ode_non_auto_shape(batchsize, nx, nu, integrator):
    # instantiate mlp and integrator
    fx = MLP(nx+nu, nx, bias=True, hsizes=[20, 20], linear_map=slim.maps['linear'])
    model = integrator(fx)
    # test model
    x = torch.randn([batchsize, nx])
    u = torch.randn([batchsize, nu])
    y = model(x, u)
    assert y.shape[0] == batchsize and y.shape[1] == nx


@given(st.integers(1, 300),
       st.sampled_from(integrators_generic),
       st.sampled_from(ode_param_systems_auto))
@settings(max_examples=200, deadline=None)
def test_integrator_white_ode_auto_params_shape(batchsize, integrator, ode):
    fx = ode()
    model = integrator(fx)
    x = torch.randn([batchsize, fx.in_features])
    y = model(x)
    assert y.shape[0] == batchsize and y.shape[1] == fx.out_features


@given(st.integers(1, 300),
       st.sampled_from(integrators_generic),
       st.sampled_from(ode_param_systems_nonauto))
@settings(max_examples=200, deadline=None)
def test_integrator_white_ode_nonauto_params_shape(batchsize, integrator, ode):
    # crate ode
    fx = ode()
    nx = fx.out_features
    nu = fx.in_features - nx
    # create integrator
    model = integrator(fx)
    # test model
    x = torch.randn([batchsize, nx])
    u = torch.randn([batchsize, nu])
    y = model(x, u)
    assert y.shape[0] == batchsize and y.shape[1] == fx.out_features


@given(st.integers(1, 300),
       st.sampled_from(integrators_generic),
       st.sampled_from(ode_hybrid_systems_auto))
@settings(max_examples=200, deadline=None)
def test_integrator_white_ode_auto_hybrid_shape(batchsize, integrator, ode):
    # this test is intented only for hybrid ode's whose black box parts map R^2 to R
    block = MLP(2, 1, bias=True, hsizes=[20, 20])
    fx = ode(block)
    model = integrator(fx)
    x = torch.randn([batchsize, fx.in_features])
    y = model(x)
    assert y.shape[0] == batchsize and y.shape[1] == fx.out_features


