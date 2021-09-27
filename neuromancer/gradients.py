"""
Support functions and objects for differentiating neuromancer objects
Computing gradients, jacobians, and PWA forms for components, variables, and constraints

"""

import torch
from neuromancer.component import Component


def gradient(y, x, grad_outputs=None):
    """
    Compute gradients dy/dx
    :param y: [tensors] outputs
    :param x: [tensors] inputs
    :param grad_outputs:
    :return:
    """
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


class Gradient(Component):

    DEFAULT_INPUT_KEYS = ["y", "x"]
    DEFAULT_OUTPUT_KEYS = ["dy/dx"]

    def __init__(self, input_key_map={}, name=None):
        """
        Gradient component class for computing gradients of neuromancer objects given the
        generated dictionary dataset with keys corresponding to variables to be differentiated
        :param input_key_map:
        :param name:
        """
        super().__init__(input_key_map, name)

    def forward(self, data):
        """

        :param data: (dict: {str: Tensor})
        :return: output (dict: {str: Tensor})
        """

        output = {}
        output[self.DEFAULT_OUTPUT_KEYS[0]] = gradient(data[self.input_keys[0]], data[self.input_keys[1]])
        return output