from hypothesis import given, settings, strategies as st

import torch
import slim
from neuromancer.component import Component
from neuromancer import (
    blocks,
    dynamics,
    estimators,
    policies,
    problem,
)

_ = torch.set_grad_enabled(False)


def get_test_data(dims, batch_size, nsteps):
    data = {
        k: torch.rand(nsteps, batch_size, *v)
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
    dims = {k: (s,) for k, s in zip(["x0", "Yf", "Uf", "Df"], [x0_dim, y_dim, u_dim, d_dim]) if s != 0}
    ssm = dynamics.block_model(kind, dims, slim.Linear, blocks.MLP, bias=False, n_layers=1, name=kind)

    test_data = get_test_data(dims, batch_size, nsteps)
    output = ssm(test_data)

    assert all([k in output for k in ssm.output_keys])


## blackbox SSM
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
    ssm = dynamics.blackbox_model(
        dims,
        slim.Linear,
        blocks.MLP,
        bias=False,
        n_layers=1,
        input_keys=dims.keys(),
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
        test_data["D"] = torch.rand(nsteps, batch_size, d_dim)
        dims["D"] = (nsteps, d_dim)
    if r_dim != 0:
        test_data["R"] = torch.rand(nsteps, batch_size, r_dim)
        dims["R"] = (nsteps, r_dim)
    if past_u != 0:
        test_data["Up"] = torch.rand(nsteps, batch_size, u_dim)
        dims["Up"] = (nsteps, u_dim)

    policy = constructor(dims, nsteps=nsteps, input_keys=[x for x in dims if x != "U"])
    output = policy(test_data)

    assert all([k in output for k in policy.output_keys])


class DummyComponent(Component):
    DEFAULT_INPUT_KEYS = ["X", "Y"]
    DEFAULT_OUTPUT_KEYS = ["X_pred", "Y_pred", "reg_error"]

    def __init__(
        self,
        input_keys=None,
        output_keys=None,
        name="dummy",
    ):
        super().__init__(
            input_keys=input_keys or DummyComponent.DEFAULT_INPUT_KEYS,
            output_keys=output_keys or DummyComponent.DEFAULT_OUTPUT_KEYS,
            name=name,
        )

    def forward(self, data):
        return {
            "X_pred": data["X"],
            "Y_pred": data["Y"],
            "reg_error": 13.0,
        }


def test_input_key_list():
    input_keys = ["X", "Y", "Z"]
    component = DummyComponent(
        input_keys=input_keys,
        name="input_list_test",
    )

    data = {"X": 123, "Y": 321}

    out = component(data)


def test_input_remap():
    remapped_keys = {"X_renamed": "X", "Y_renamed": "Y"}
    component = DummyComponent(
        input_keys=remapped_keys,
        name="input_remap_test",
    )

    data = {"X_renamed": 123, "Y_renamed": 321}
    out = component._remap_input(component, (data,))

    assert all([k in remapped_keys.values() for k in out.keys()])


def test_output_remap():
    remapped_keys = {"X_pred": "X_renamed", "Y_pred": "Y_renamed"}
    component = DummyComponent(
        output_keys=remapped_keys,
        name="output_remap_test",
    )

    data = {"X_pred": 123, "Y_pred": 321}
    out = component._remap_output(component, {}, data)

    print(out)

    assert all([k in component.output_keys for k in out.keys()])


def test_forward_remap():
    remapped_input = {"X_renamed": "X", "Y_renamed": "Y"}
    remapped_output = {"X_pred": "X_renamed", "Y_pred": "Y_renamed"}
    component = DummyComponent(
        input_keys=remapped_input,
        output_keys=remapped_output,
        name="full_remap_test",
    )

    data = {"X_renamed": 123, "Y_renamed": 321}
    out = component(data)

    assert all([k in component.output_keys for k in out.keys()])
