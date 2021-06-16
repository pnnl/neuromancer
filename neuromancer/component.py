from typing import List

import torch
from torch import nn


def check_keys(k1, k2):
    """
    Check that some elements in k1 are contained in k2

    :param k1: iterable of str
    :param k2: iterable of str
    """
    common_keys = set(k1) - set(k2)

    assert len(common_keys) == 0, \
            f'Missing values in dataset: {common_keys}\n' \
            f'  input_keys: {set(k1)}\n  data_keys: {set(k2)}'


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
            input_keys = {
                **{k: k for k in self.DEFAULT_INPUT_KEYS if k not in input_keys.values()},
                **input_keys
            }
            # conversion of dict to list of tuples used because torch handles dicts weirdly in module hooks
            self._input_keys = [(k, v) for k, v in input_keys.items()]
            self.register_forward_pre_hook(self._remap_input)
        else:
            self._input_keys = [(k, k) for k in input_keys]

        self._output_keys = [(k, f"{k}_{name}") for k in output_keys]
        self.register_forward_hook(self._remap_output)

    def _remap_input(self, module, input_data):
        input_data = input_data[0]
        check_keys({x[0] for x in self._input_keys}, input_data.keys())
        return {
            k2: input_data[k1] for k1, k2 in self._input_keys
            if k1 in input_data
        }

    def _remap_output(self, module, input_data, output_data):
        # check_keys(output_data.keys(), {x[0] for x in self._output_keys})
        return {
            k2: output_data[k1] for k1, k2 in self._output_keys
            if k1 in output_data
        }

    @property
    def input_keys(self):
        """
        Retrieve component's input variable names. This returns remapped names if `input_keys` was
        given a mapping; to see a component's canonical input keys, see class attribute `DEFAULT_INPUT_KEYS`.
        """
        # reverse mapping
        rvalues, rkeys = zip(*self._input_keys)
        #rkeys = dict(zip(rkeys, rvalues))
        return rkeys

    @property
    def output_keys(self):
        """
        Retrieve component's output variable names. This returns remapped names if `output_keys` was
        given a mapping; to see a component's canonical output keys, see class attribute `DEFAULT_OUTPUT_KEYS`.
        """
        rkeys = dict(self._output_keys)
        return [rkeys.get(k, k) for k in self.DEFAULT_OUTPUT_KEYS]

    # NOTE: the following two methods may become unnecessary if certain components are broken up into
    # separate components and composed in a Problem.
    @classmethod
    def add_optional_inputs(cls, new_keys, remapping=[]):
        # assert isinstance(remapping, list) or isinstance(remapping, dict)
        if isinstance(remapping, dict) and len(remapping) > 0:
            reverse_map = dict(zip(remapping.values(), remapping.keys()))
            keys = {
                **{k: k for k in cls.DEFAULT_INPUT_KEYS},
                **{k: v for k, v in remapping.items() if v not in new_keys},
                **{reverse_map[k] if k in reverse_map else k: k for k in new_keys},
            }
        else:
            keys = cls.DEFAULT_INPUT_KEYS + new_keys
        return keys

    @classmethod
    def add_optional_outputs(cls, new_keys):
        return cls.DEFAULT_OUTPUT_KEYS + new_keys

    def __repr__(self):
        return f"{self.name}({', '.join(self.input_keys)}) -> {', '.join(self.output_keys)}"