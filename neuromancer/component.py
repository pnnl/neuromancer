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
    def __init__(self, input_keys, output_keys, name):
        super().__init__()
        self._input_keys = (
            [(k, v) for k, v in input_keys.items()]
            if isinstance(input_keys, dict)
            else input_keys
        )
        self._output_keys = (
            [(k, v) for k, v in output_keys.items()]
            if isinstance(output_keys, dict)
            else [(x, x) for x in output_keys]
        )
        self.name = name

        self._do_input_remap = isinstance(input_keys, dict)

        if self._do_input_remap:
            def _remap_input(module, input_data):
                d = {
                    k1: input_data[0][k2] for k1, k2 in self._input_keys
                    if k2 in input_data[0]
                }
                return d
            self.register_forward_pre_hook(_remap_input)

        def _remap_output(module, input_data, output_data):
            d1 = {
                f"{k2}_{self.name}": output_data[k1] for k1, k2 in self._output_keys
                if k1 in output_data
            }
            return d1
        self.register_forward_hook(_remap_output)

    @property
    def input_keys(self):
        return (
            [x for x, _ in self._input_keys]
            if self._do_input_remap
            else self._input_keys
        )

    @property
    def output_keys(self):
        return [f"{x}_{self.name}" for x, _ in self.output_keys]
