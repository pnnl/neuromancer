import torch
from torch import nn


class Component(nn.Module):
    def __init__(self, input_keys, output_keys, name):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = [f"{x}_{name}" for x in output_keys]
        self.name = name
"""
    def forward(self, data):
        output = self.module(*[data[k] for k in self.input_keys])
        #if not isinstance(output, collections.abc.Sequence):
        #    output = (output, )
        output = {k: v for k, v in zip(self.output_keys, output)}
        return output
"""
