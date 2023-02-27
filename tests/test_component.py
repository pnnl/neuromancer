from hypothesis import given, settings, strategies as st

import torch
import neuromancer.slim as slim
from neuromancer.component import Component
from neuromancer import (
    estimators,
    blocks,
    problem,
    policies,
    dynamics,
)

_ = torch.set_grad_enabled(False)


def get_test_data(dims, batch_size, nsteps):
    data = {
        k: torch.rand(batch_size, nsteps, *v)
        if k != "x0"
        else torch.rand(batch_size, *v)
        for k, v in dims.items()
    }
    data["name"] = "test"
    return data

@given(
    st.sampled_from(list(estimators.estimators)),
    st.integers(1, 50),
    st.integers(1, 50),
    st.integers(0, 1),
    st.integers(2, 100),
    st.integers(2, 50),
)
@settings(max_examples=1000, deadline=None)
def test_estimators(kind, x0_dim, y_dim, u_dim, batch_size, nsteps):
    dims = {k: (s,) for k, s in zip(["x0", "Yp", "Up"], [x0_dim, y_dim, u_dim]) if s != 0}
    dims["x0"] = (dims["Yp"][-1] * 5,) if kind != "fullobservable" else (5,)
    test_data = get_test_data(dims, batch_size, nsteps)
    dims = {k: (nsteps, v.shape[-1]) for k, v in test_data.items() if k != "name"}
    constructor = estimators.estimators[kind]
    estim = constructor(
        dims,
        nsteps=nsteps,
        window_size=nsteps,
        hsizes=[dims["x0"][-1]] * 2,
        input_keys=[k for k, v in dims.items() if v[0] != 0 and k != "x0"],
        name=kind,
    )

    output = estim(test_data)

    assert all([k in output for k in estim.output_keys])


# dynamics models
## standard block SSMs
@given(
    st.sampled_from(list(dynamics._bssm_kinds)),
    st.integers(1, 50),
    st.integers(1, 50),
    st.integers(0, 1),
    st.integers(0, 1),
    st.integers(1, 100),
    st.integers(1, 50),
)
@settings(max_examples=1000, deadline=None)
def test_block_ssm(kind, x0_dim, y_dim, u_dim, d_dim, batch_size, nsteps):
    dims = {k: (s,) for k, s in zip(["x0", "Yf", "Uf", "Df"],
                                    [x0_dim, y_dim, u_dim, d_dim]) if s != 0}
    ssm = dynamics.block_model(kind, dims, slim.Linear, blocks.MLP,
                               bias=False, n_layers=1, name=kind)

    test_data = get_test_data(dims, batch_size, nsteps)
    output = ssm(test_data)

    assert all([k in output for k in ssm.output_keys])


# blackbox SSM
@given(
    st.integers(1, 50),
    st.integers(1, 50),
    st.integers(0, 1),
    st.integers(0, 1),
    st.integers(1, 100),
    st.integers(1, 50),
)
@settings(max_examples=1000, deadline=None)
def test_black_ssm(x0_dim, y_dim, u_dim, d_dim, batch_size, nsteps):
    dims = {k: (s,) for k, s in zip(["x0", "Yf", "Uf", "Df"], [x0_dim, y_dim, u_dim, d_dim]) if s != 0}
    extra_inputs = []
    if u_dim != 0:
        extra_inputs += ['Uf']
    if d_dim != 0:
        extra_inputs += ['Df']
    ssm = dynamics.blackbox_model(
        dims,
        slim.Linear,
        blocks.MLP,
        bias=False,
        n_layers=1,
        extra_inputs=extra_inputs,
        name="blackbox",
    )

    test_data = get_test_data(dims, batch_size, nsteps)
    output = ssm(test_data)

    assert all([k in output for k in ssm.output_keys])


@given(
    st.sampled_from(policies.policies),
    st.integers(1, 50),
    st.integers(0, 1),
    st.integers(1, 5),
    st.integers(0, 1),
    st.integers(0, 1),
    st.integers(1, 100),
    st.integers(1, 50),
)
@settings(max_examples=1000, deadline=None)
def test_policies(constructor, x0_dim, r_dim, u_dim, past_u, d_dim, batch_size, nsteps):
    test_data = {"x0": torch.rand(batch_size, x0_dim)}
    dims = {"x0": (x0_dim,), "U": (nsteps, u_dim)}

    if d_dim != 0:
        test_data["D"] = torch.rand(batch_size, nsteps, d_dim)
        dims["D"] = (nsteps, d_dim)
    if r_dim != 0:
        test_data["R"] = torch.rand(batch_size, nsteps, r_dim)
        dims["R"] = (nsteps, r_dim)
    if past_u != 0:
        test_data["Up"] = torch.rand(batch_size, nsteps, u_dim)
        dims["Up"] = (nsteps, u_dim)

    policy = constructor(dims, nsteps=nsteps, input_keys=[x for x in dims if x != "U"])
    output = policy(test_data)

    assert all([k in output for k in policy.output_keys])


class DummyComponent(Component):
    def __init__(
        self,
        input_keys,
        output_keys,
        name="dummy",
    ):
        super().__init__(input_keys=input_keys, output_keys=output_keys, name=name)

    def forward(self, data):
        return {k: v for k, v in zip(self.output_keys, data.values())}


def test_dummy_component():
    input_keys = ["X", "Y"]
    output_keys = ["X_pred", "Y_pred"]
    component = DummyComponent(
        input_keys=input_keys, output_keys=output_keys,
        name="dummy_test")

    data = {"X": 123, "Y": 321}
    out = component(data)

    assert all([k in output_keys for k in out.keys()])
