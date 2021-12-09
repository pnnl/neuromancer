from typing import List

from torch import nn
import torch


def _validate_key_params(keys):
    return keys is None or isinstance(keys, dict) or isinstance(keys, list)


def check_key_subset(k1, k2):
    """
    Check that keys in k1 are a subset of keys in k2

    :param k1: iterable of str
    :param k2: iterable of str
    """
    common_keys = set(k1) - set(k2)

    assert len(common_keys) == 0, \
            f'Missing values in dataset: {common_keys}\n' \
            f'  input_keys: {set(k1)}\n  data_keys: {set(k2)}'


class Component(nn.Module):
    DEFAULT_INPUT_KEYS: List[str]
    DEFAULT_OUTPUT_KEYS: List[str]

    def __init__(self, input_key_map={}, name=None):
        """
        The NeuroMANCER component base class.

        This class is used to manage naming of component input and output variables as they flow
        through the computational graph, as well as handle potential remapping of input and output keys
        to different names. It additionally provides a useful reference for users to see how components
        can be connected together in the overall computational graph.

        Components that inherit from this class should specify the class attributes DEFAULT_INPUT_KEYS
        and DEFAULT_OUTPUT_KEYS; these are used as the "canonical" names for input and output variables,
        respectively. These can be used to compare different components' output and input keys to see
        whether one component can accept another's output by default.
        
        By default, components have a `name` argument which is used to tag the output variables they
        generate; for instance, for a component called "estim", the canonical output "x0" is renamed to
        "x0_estim".

        >>> estim = LinearEstimator(..., name="estim")  # output "x0" remapped to "x0_estim"
        >>> ssm = BlockSSM(..., input_key_map={f"x0_{estim.name}": "x0"})  # input_keys used to remap to canonical name

        :param input_keys: (dict {str: str}) dictionary mapping arbitrary variable names to canonical
            input keys.
        :param name: (str) The name of the component, used to tag output variable keys with the name
            of the component that produced them.
        """
        super().__init__()

        self.name = name
        self.update_input_keys(input_key_map=input_key_map)
        self.register_forward_pre_hook(self._check_inputs)
        self.output_keys = [f"{k}_{name}" if self.name is not None else k for k in self.DEFAULT_OUTPUT_KEYS]
        self.register_forward_hook(self._remap_output)

    def update_input_keys(self, input_key_map={}):
        assert isinstance(input_key_map, dict), \
            f"{type(self).__name__} input_key_map must be dict for remapping input variable names; "
        self.input_key_map = {
            **{k: k for k in self.DEFAULT_INPUT_KEYS if k not in input_key_map.keys()},
            **input_key_map
        }
        self.input_keys = list(self.input_key_map.values())
        assert len(self.input_keys) == len(self.DEFAULT_INPUT_KEYS), \
            "Length of given input keys must equal the length of default input keys"

    def _check_inputs(self, module, input_data):

        input_data = input_data[0]
        set_diff = set(self.input_keys) - set(input_data)
        assert len(set_diff) == 0, \
            f" Missing inputs {set_diff}"

    def _remap_output(self, module, input_data, output_data):
        assert set(output_data.keys()) == set(self.DEFAULT_OUTPUT_KEYS), \
            f' Key mismatch' \
            f' Forward pass keys: {set(output_data.keys())}  ' \
            f' Default output keys: {set(self.DEFAULT_OUTPUT_KEYS)}'
        if self.name is not None:
            output_data = {
                f"{k}_{self.name}": output_data[k] for k in self.DEFAULT_OUTPUT_KEYS
            }

        return output_data

    def __repr__(self):
        return f"{self.name}({', '.join(self.input_keys)}) -> {', '.join(self.output_keys)}"


class Function(Component):
    def __init__(
        self,
        func,
        input_keys,
        output_keys,
        name=None,
    ):
        """

        :param func:
        :param input_keys:
        :param output_keys:
        :param name:
        """
        self.DEFAULT_INPUT_KEYS = input_keys if input_keys is not None else []
        self.DEFAULT_OUTPUT_KEYS = output_keys if isinstance(output_keys, list) else [output_keys]
        super().__init__(name=name)
        self.func = func

    def forward(self, data):
        x = [data[k] for k in self.input_keys]
        out = self.func(*x)

        out_d = {
            k: v for k, v in zip(
                self.DEFAULT_OUTPUT_KEYS,
                out if isinstance(out, tuple) else (out,)
            )
        }

        return out_d
