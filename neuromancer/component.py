import torch
from torch import nn


class Component(nn.Module):
    def __init__(self, input_keys, output_keys, name):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = [f"{x}_{name}" for x in output_keys]
        self.name = name

    def __repr__(self):
        return f"{self.name}({', '.join(self.input_keys)}) -> {', '.join(self.output_keys)}"
