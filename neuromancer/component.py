import torch
from torch import nn


def check_keys(k1, k2):
    """
    Check that all elements in k1 are contained in k2

    :param k1: iterable of str
    :param k2: iterable of str
    """
    assert set(k1) - set(k2) == set(), \
        f'Missing values in dataset. Input_keys: {set(k1)}, data_keys: {set(k2)}'


class Component(nn.Module):
    DEFAULT_INPUT_KEYS = []
    DEFAULT_OUTPUT_KEYS = []
    def __init__(self, input_keys=None, output_keys=None, name=None):
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
        """
        super().__init__()

        self.name = name or type(self).__name__

        if input_keys is None:
            input_keys = self.DEFAULT_INPUT_KEYS

        self._do_input_remap = isinstance(input_keys, dict)

        if self._do_input_remap:
            input_keys = {**{k: k for k in self.DEFAULT_INPUT_KEYS}, **input_keys}

        self._input_keys = (
            [(k, v) for k, v in input_keys.items()]
            if self._do_input_remap
            else input_keys
        )

        if self._do_input_remap:
            self.register_forward_pre_hook(self._remap_input)

        if output_keys is None:
            output_keys = self.DEFAULT_OUTPUT_KEYS

        if isinstance(output_keys, dict):
            output_keys = {**{k: k for k in self.DEFAULT_OUTPUT_KEYS}, **output_keys}

        self._output_keys = (
            [(k, f"{v}_{name}") for k, v in output_keys.items()]
            if isinstance(output_keys, dict)
            else [(k, f"{k}_{name}") for k in output_keys]
        )

        self.register_forward_hook(self._remap_output)

    def _remap_input(self, module, input_data):
        input_data = input_data[0]
        # check_keys({x[1] for x in self._input_keys}, input_data.keys())
        return {
            k1: input_data[k2] for k1, k2 in self._input_keys
            if k2 in input_data
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
        return (
            [x for x, _ in self._input_keys]
            if self._do_input_remap
            else self._input_keys
        )

    @property
    def output_keys(self):
        """
        Retrieve component's output variable names. This returns remapped names if `output_keys` was
        given a mapping; to see a component's canonical output keys, see class attribute `DEFAULT_OUTPUT_KEYS`.
        """
        return [x for _, x in self._output_keys]

    def __repr__(self):
        return f"{self.name}({', '.join(self.input_keys)}) -> {', '.join(self.output_keys)}"
