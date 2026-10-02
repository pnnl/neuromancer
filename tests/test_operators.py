import pytest
import torch
import torch.nn as nn

from neuromancer.modules.operators import (
    DeepONetCartesianProd,
    DeepXDEIntegratorWrapper,
    DeepXDEWrapper,
    _StripMetadataMixin,
)


class _DummyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        self.received_state_keys = None

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state["_metadata"] = {"dummy": True}
        return state

    def load_state_dict(self, state, **kwargs):
        self.received_state_keys = set(state.keys())
        return super().load_state_dict(state, **kwargs)


class _DummyStripModel(_StripMetadataMixin, _DummyBase):
    pass


class _DummyDeepONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_inputs = None

    def forward(self, inputs):
        self.last_inputs = inputs
        branch, trunk = inputs
        return branch.mean() + trunk.mean()


class _DummyMIONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_inputs = None

    def forward(self, inputs):
        self.last_inputs = inputs
        branch1, branch2, trunk = inputs
        return branch1.mean() + branch2.mean() + trunk.mean()


class _DummyMIONetPositional(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_inputs = None

    def forward(self, branch1, branch2, trunk):
        self.last_inputs = (branch1, branch2, trunk)
        return branch1.mean() + branch2.mean() + trunk.mean()


class _DummyBranchTrunkNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_inputs = None

    def forward(self, branch, trunk):
        self.last_inputs = (branch, trunk)
        return branch.mean() + trunk.mean()


def _build_deepxde_call(call_style, branches, trunk, outputs):
    if call_style == "dict":
        batch = {
            "branch_inputs": branches[0],
            "trunk_inputs": trunk,
            "outputs": outputs,
        }
        return (batch,)
    if call_style == "positional":
        return (*branches, trunk)
    if call_style == "list":
        return ([branches[0]], trunk)
    if call_style == "tuple":
        return ((branches[0], branches[1]), trunk)
    raise ValueError(f"Unknown call_style: {call_style}")


def test_strip_state():
    """Ensure _metadata is stripped from the state dict."""
    model = _DummyStripModel()
    state = model.state_dict()
    assert "_metadata" not in state
    assert "linear.weight" in state


def test_strip_load():
    """Ensure _metadata is ignored on load_state_dict and input is unchanged."""
    model = _DummyStripModel()
    base = _DummyBase()
    state = base.state_dict()
    state_keys = set(state.keys())

    model.load_state_dict(state, strict=False)

    assert "_metadata" in state_keys
    assert set(state.keys()) == state_keys
    assert "_metadata" not in model.received_state_keys


@pytest.mark.parametrize(
    "is_cartesian,trunk_shape,use_first_grid,expect_error",
    [
        pytest.param(True, (2, 4, 5), True, False, id="cartesian_batched_grid"),
        pytest.param(True, (4, 5), False, False, id="cartesian_grid"),
        pytest.param(False, (2, 5), False, False, id="pointwise"),
        pytest.param(False, (2, 4, 5), False, True, id="pointwise_bad_rank"),
    ],
)
def test_deepxde_trunk_routing(is_cartesian, trunk_shape, use_first_grid, expect_error):
    """Check trunk routing for cartesian and pointwise cases."""
    model = _DummyDeepONet()
    wrapper = DeepXDEWrapper(model, is_cartesian=is_cartesian)
    branch = torch.randn(2, 3)
    trunk = torch.randn(*trunk_shape)

    if expect_error:
        with pytest.raises(ValueError):
            _ = wrapper(branch, trunk)
        return

    _ = wrapper(branch, trunk)

    expected_trunk = trunk[0] if use_first_grid else trunk
    assert torch.allclose(model.last_inputs[1], expected_trunk)


@pytest.mark.parametrize(
    (
        "model_cls,is_cartesian,branch_keys,branch_shapes,trunk_shape,"
        "call_style,use_first_grid,expect_targets"
    ),
    [
        pytest.param(
            _DummyDeepONet,
            False,
            None,
            [(2, 3)],
            (2, 5),
            "positional",
            False,
            False,
            id="positional_single_branch",
        ),
        pytest.param(
            _DummyDeepONet,
            False,
            ["b1"],
            [(2, 3)],
            (2, 5),
            "list",
            False,
            False,
            id="list_single_branch",
        ),
        pytest.param(
            _DummyMIONet,
            False,
            ("b1", "b2"),
            [(2, 3), (2, 4)],
            (2, 5),
            "tuple",
            False,
            False,
            id="tuple_two_branch",
        ),
        pytest.param(
            _DummyMIONet,
            True,
            ["b1", "b2"],
            [(2, 3), (2, 4)],
            (2, 5, 6),
            "positional",
            True,
            False,
            id="positional_two_branch_cartesian",
        ),
        pytest.param(
            _DummyDeepONet,
            False,
            None,
            [(2, 3)],
            (2, 5),
            "dict",
            False,
            True,
            id="dict_with_targets",
        ),
    ],
)
def test_deepxde_input_formats(
    model_cls,
    is_cartesian,
    branch_keys,
    branch_shapes,
    trunk_shape,
    call_style,
    use_first_grid,
    expect_targets,
):
    """Check supported DeepXDEWrapper input formats."""
    model = model_cls()
    wrapper_kwargs = {"model": model, "is_cartesian": is_cartesian}
    if branch_keys is not None:
        wrapper_kwargs["branch_keys"] = branch_keys
    wrapper = DeepXDEWrapper(**wrapper_kwargs)

    branches = [torch.randn(*shape) for shape in branch_shapes]
    trunk = torch.randn(*trunk_shape)
    outputs = torch.randn(branches[0].shape[0], 1) if expect_targets else None

    call_args = _build_deepxde_call(call_style, branches, trunk, outputs)
    result = wrapper(*call_args)

    if expect_targets:
        preds, targets = result
        assert torch.is_tensor(preds)
        assert torch.allclose(targets, outputs)
    else:
        assert torch.is_tensor(result)

    expected_trunk = trunk[0] if use_first_grid else trunk
    if len(branches) == 1:
        assert torch.allclose(model.last_inputs[0], branches[0])
        assert torch.allclose(model.last_inputs[1], expected_trunk)
    else:
        assert torch.allclose(model.last_inputs[0], branches[0])
        assert torch.allclose(model.last_inputs[1], branches[1])
        assert torch.allclose(model.last_inputs[2], expected_trunk)


def test_deepxde_branch_count_mismatch():
    """Check mismatch between expected and received branches."""
    model = _DummyDeepONet()
    wrapper = DeepXDEWrapper(model, is_cartesian=False, branch_keys=["b1", "b2"])
    branch = torch.randn(2, 3)
    trunk = torch.randn(2, 5)

    with pytest.raises(ValueError, match=r"Expected 2 branch inputs, received 1"):
        _ = wrapper(branch, trunk)


def test_deepxde_bad_branch_keys_type():
    """Check invalid branch_keys types raise TypeError."""
    model = _DummyDeepONet()
    with pytest.raises(TypeError, match="branch_keys must be a string or list/tuple"):
        _ = DeepXDEWrapper(model, is_cartesian=False, branch_keys=1)


def test_cartprod_shared_grid():
    """Check cartesian prod uses the first trunk grid."""
    branch_net = nn.Identity()
    trunk_net = nn.Identity()
    model = DeepONetCartesianProd(branch_net, trunk_net, bias=False)
    branch = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.0, -1.0]])
    grid0 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    grid1 = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    trunk = torch.stack([grid0, grid1], dim=0)

    out = model(branch, trunk)

    expected = torch.einsum("bp,mp->bm", branch, grid0)
    assert torch.allclose(out, expected)


def test_cartprod_transpose_bias():
    """Check transpose output and bias addition."""
    branch_net = nn.Identity()
    trunk_net = nn.Identity()
    model = DeepONetCartesianProd(
        branch_net, trunk_net, bias=True, return_transposed=True
    )
    model.bias.data.fill_(0.5)
    branch = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.0, -1.0]])
    grid = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    trunk = torch.stack([grid, grid], dim=0)

    out = model(branch, trunk)

    expected = torch.einsum("bp,mp->bm", branch, grid) + 0.5
    assert out.shape == (grid.shape[0], branch.shape[0])
    assert torch.allclose(out, expected.transpose(0, 1))


@pytest.mark.parametrize(
    "model_cls,trunk_shape,branch_shapes,manual_features,feat_axis",
    [
        pytest.param(
            _DummyBranchTrunkNet,
            (4, 5),
            [(2, 3)],
            None,
            0,
            id="single_branch_trunk_2d",
        ),
        pytest.param(
            _DummyMIONetPositional,
            (2, 6, 5),
            [(2, 3), (2, 4)],
            None,
            1,
            id="two_branch_trunk_3d",
        ),
        pytest.param(
            _DummyBranchTrunkNet,
            (4, 5),
            [(2, 3)],
            (7, 9),
            None,
            id="manual_features",
        ),
    ],
)
def test_integrator_wrapper_feature_inference(
    model_cls, trunk_shape, branch_shapes, manual_features, feat_axis
):
    """Check integrator wrapper feature inference and trunk binding."""
    model = model_cls()
    trunk = torch.randn(*trunk_shape)
    wrapper_kwargs = {"model": model, "trunk_inputs": trunk}
    if manual_features is not None:
        wrapper_kwargs["in_features"] = manual_features[0]
        wrapper_kwargs["out_features"] = manual_features[1]
    wrapper = DeepXDEIntegratorWrapper(**wrapper_kwargs)

    branches = [torch.randn(*shape) for shape in branch_shapes]
    _ = wrapper(*branches)

    if manual_features is not None:
        expected_in, expected_out = manual_features
    else:
        expected_in = trunk.shape[feat_axis]
        expected_out = trunk.shape[feat_axis]

    assert wrapper.in_features == expected_in
    assert wrapper.out_features == expected_out

    if len(branches) == 1:
        assert torch.allclose(model.last_inputs[0], branches[0])
        assert torch.allclose(model.last_inputs[1], trunk)
    else:
        assert torch.allclose(model.last_inputs[0], branches[0])
        assert torch.allclose(model.last_inputs[1], branches[1])
        assert torch.allclose(model.last_inputs[2], trunk)


def test_integrator_wrapper_bad_trunk_shape():
    """Check invalid trunk shape raises ValueError."""
    model = _DummyBranchTrunkNet()
    trunk = torch.randn(2, 3, 4, 5)

    with pytest.raises(ValueError):
        _ = DeepXDEIntegratorWrapper(model, trunk_inputs=trunk)


def test_integrator_wrapper_trunk_is_buffer():
    """Check trunk inputs are registered as a buffer."""
    model = _DummyBranchTrunkNet()
    trunk = torch.randn(4, 5)
    wrapper = DeepXDEIntegratorWrapper(model, trunk_inputs=trunk)

    assert "trunk_inputs" in dict(wrapper.named_buffers())
