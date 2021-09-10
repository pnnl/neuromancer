from typing import List

import torch
from torch import nn


def _validate_key_params(keys):
    return keys is None or isinstance(keys, dict) or isinstance(keys, list)


class Component(nn.Module):
    DEFAULT_INPUT_KEYS: List[str]
    DEFAULT_OUTPUT_KEYS: List[str]

    def __init__(self, input_keys, output_keys, name=None):
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
        "x0_estim". If you wish to connect two components, a dictionary mapping should be specified in
        the `input_keys` argument to remap the renamed outputs of a component to the canonical keys of
        the receiving component:

        >>> estim = LinearEstimator(..., name="estim")  # output "x0" remapped to "x0_estim"
        >>> ssm = BlockSSM(..., input_keys={f"x0_{estim.name}": "x0"})  # input_keys used to remap to canonical name

        Remapping of output variable names can also be done by passing another dictionary to the
        `output_keys` argument, though note that renamed variables will also be tagged with the
        component's name.

        :param input_keys: (dict {str: str}) dictionary mapping arbitrary variable names to canonical
            input keys.
        :param output_keys: (dict {str: str}) dictionary mapping canonical output keys to arbitrary
            variable names.
        :param name: (str) The name of the component, used to tag output variable keys with the name
            of the component that produced them.

        .. todo:: Handle input and output key validation here rather than in component implementations.
        .. todo:: Validate that any remapped keys actually exist in a components' defaults.
        .. todo:: Figure out how to handle potentially optional inputs/outputs; e.g. autonomous dynamics
            won't use `Uf` for input or return `fU` as output.

        .. todo:: Remove `output_keys` as an argument, just tag canonical output variable names by default?
            This may not be totally doable for components whose outputs are defined by how they are
            instantiated, e.g. `BlockSSM`.
        """
        super().__init__()

        self.name = name or type(self).__name__

        assert _validate_key_params(input_keys), \
            f"{type(self).__name__} input_keys must be None, list, or dict if remapping; " \
            f"type is {type(input_keys)}"

        if input_keys is None:
            input_keys = self.DEFAULT_INPUT_KEYS

        self._do_input_remap = isinstance(input_keys, dict)

        if self._do_input_remap:
            self.input_keys = {
                **{k: k for k in self.DEFAULT_INPUT_KEYS if k not in input_keys.values()},
                **input_keys
            }
            assert len(self.input_keys) == len(self.DEFAULT_INPUT_KEYS), \
                " Length of given input keys must equal the length of default input keys"
            self.register_forward_pre_hook(self._check_inputs)
        else:
            self.input_keys = self.DEFAULT_INPUT_KEYS

        self.output_keys = [f"{k}_{name}" if self.name is not None else k for k in output_keys]
        self.register_forward_hook(self._remap_output)

    def _check_inputs(self, module, input_data):
        input_data = input_data[0]
        set_diff = set(self.input_keys) - set(input_data)
        assert len(set_diff) == 0, \
            f" Missing input keys {set_diff}"

    def _remap_output(self, module, input_data, output_data):
        assert set(output_data.keys()) == set(self.DEFAULT_OUTPUT_KEYS), \
            f' Key mismatch\n' \
            f' Forward pass keys: {set(output_data.keys())}\n  ' \
            f' Default output keys: {set(self.DEFAULT_OUTPUT_KEYS)}'
        if self.name is not None:
            output_data = {
                f"{k}_{self.name}": output_data[k] for k in self.DEFAULT_OUTPUT_KEYS
            }
        return output_data


# TODO: just use output_keys list
class Function(Component):
    def __init__(
        self,
        func,
        input_keys,
        output_keys,
        name,
    ):
        # TODO: continue
        self.DEFAULT_INPUT_KEYS = input_keys if input_keys is not None else []
        self.DEFAULT_OUTPUT_KEYS = output_keys if isinstance(output_keys, list) else [output_keys]
        super().__init__(self.DEFAULT_INPUT_KEYS, output_keys, name)
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

        # TODO: testing this stuff
        print(x[0].shape)
        print(out.shape)
        print(out_d.keys())
        print(out_d[list(out_d.keys())[0]].shape)
        # TODO: something gets overwritten here
        # TODO: returns empty dict instead of out_d
        return out_d

        # return {
        #     k: v for k, v in zip(
        #         self.output_keys,
        #         out if isinstance(out, tuple) else (out,)
        #     )
        # }


class RecurrentFunction(Function):
    def __init__(
        self,
        func,
        input_keys,
        iter_key,
        output_keys,
        name,
    ):
        super().__init__(func, input_keys, output_keys, name)
        self.iter_key = iter_key
        self._retain_keys = set(input_keys) & set(output_keys)

    def forward(self, data):
        steps = data[self.iter_key].shape[0]
