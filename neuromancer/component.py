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
    def __init__(self, input_keys=None, output_keys=None, name=None):
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
        # print(input_data.keys())
        # print(self._input_keys)
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
        return (
            [x for x, _ in self._input_keys]
            if self._do_input_remap
            else self._input_keys
        )

    @property
    def output_keys(self):
        return [x for _, x in self._output_keys]

    def __repr__(self):
        return f"{self.name}({', '.join(self.input_keys)}) -> {', '.join(self.output_keys)}"
