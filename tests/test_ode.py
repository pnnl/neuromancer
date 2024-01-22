import torch
from neuromancer.dynamics import ode, physics
from hypothesis import given, settings, strategies as st
import random
from neuromancer.modules.blocks import MLP

ode_param_systems_auto = [v for v in ode.ode_param_systems_auto.values()]
ode_param_systems_nonauto = [v for v in ode.ode_param_systems_nonauto.values()]
ode_hybrid_systems_auto = [v for v in ode.ode_hybrid_systems_auto.values()]

ode_networked_systems = [v for v in ode.ode_networked_systems.values()]
agent_list = [v for v in physics.agents.values()]
coupling_list = [v for v in physics.couplings.values()]
bias = ['additive','compositional']

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
    nx = model.out_features
    nu = model.in_features - model.out_features
    x = torch.randn([batchsize, nx])
    u = torch.randn([batchsize, nu])
    y = model(x, u)
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


@given(st.integers(1, 100),
       st.integers(0, 1000),
       st.integers(0, 100),
       st.integers(1, 500),
       st.sampled_from(bias),
       st.sampled_from(ode_networked_systems))
@settings(max_examples=200, deadline=None)
def test_random_network(nAgents, nCouplings, nu, batchsize, bias, system):

    # Number of agents in total:
    insize = nAgents + nu
   
    # Instantiate the agents:
    agents = [random.choice(agent_list)(state_names=["T"]) for _ in range(insize)]
    map = physics.map_from_agents(agents)

    # Define the graph and interactions:
    adjacency = list(torch.randint(insize, (nCouplings, 2)))
    couplings = [random.choice(coupling_list)(feature_name="T", pins=[pair]) for pair in adjacency]

    ode = system(
        map=map,
        agents=agents,
        couplings=couplings,
        insize=insize,
        outsize=nAgents,
        inductive_bias=bias)

    x = torch.randn([batchsize, insize])
    y = ode(x)

    assert y.shape[0] == batchsize and y.shape[1] == nAgents