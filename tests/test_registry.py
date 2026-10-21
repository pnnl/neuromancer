import inspect

import pytest

from neuromancer.dynamics import integrators, ode
from neuromancer.modules import blocks
from neuromancer.modules.activations import activations
from neuromancer.registry import registry
from neuromancer.slim.linear import maps

described_classes = [
    cls
    for category in ("blocks", "integrators", "dynamics")
    for cls in registry[category].values()
    if hasattr(cls, "describe")
]


def test_registry_aggregates_module_registries():
    assert registry["blocks"] == blocks.blocks
    assert registry["dynamics"] == ode.odes
    assert registry["slim"] == maps
    assert registry["activations"] == activations
    for name, cls in integrators.integrators.items():
        assert registry["integrators"][name] is cls


def test_describe_is_declared_on_first_set_of_classes():
    for cls in (blocks.MLP, blocks.MLP_bounds, integrators.Integrator, ode.TwoTankParam):
        assert "describe" in cls.__dict__, cls.__name__


@pytest.mark.parametrize("cls", described_classes, ids=lambda cls: cls.__name__)
def test_described_arguments_are_constructor_parameters(cls):
    spec = cls.describe()
    assert isinstance(spec["description"], str) and spec["description"]
    parameters = inspect.signature(cls.__init__).parameters
    for argument in spec["arguments"]:
        assert argument["name"] in parameters, argument["name"]
        if "default" not in argument:
            assert argument.get("required"), argument["name"]


def test_mlp_bounds_describe_extends_mlp():
    mlp_names = [a["name"] for a in blocks.MLP.describe()["arguments"]]
    bounds_names = [a["name"] for a in blocks.MLP_bounds.describe()["arguments"]]
    assert bounds_names == mlp_names + ["min", "max", "method"]
    method = blocks.MLP_bounds.describe()["arguments"][-1]
    assert method["choices"] == sorted(blocks.MLP_bounds.bound_methods)
