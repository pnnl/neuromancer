"""
Support functions and objects for differentiating neuromancer objects
Computing gradients, jacobians, and PWA forms for components, variables, and constraints

"""

import torch
from neuromancer.component import Component


def gradient(y, x, grad_outputs=None, create_graph=True):
    """
    Compute gradients dy/dx
    :param y: [tensors] outputs
    :param x: [tensors] inputs
    :param grad_outputs:
    :return:
    """
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs,
                               create_graph=create_graph)[0]
    return grad


def jacobian(y, x):
    """
    Compute J = [dy_1/dx_1, ..., dy_1/dx_n, \\ dy_m/dx_1, ..., dy_m/dx_n]
    computes gradients dy/dx at grad_outputs in [1, 0, ..., 0], [0, 1, 0, ..., 0], ...., [0, ..., 0, 1]
    :param y: [tensor] outputs
    :param x: tensor] inputs
    :return:
    """
    jac = torch.zeros(y.shape[0], x.shape[0])
    for i in range(y.shape[0]):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[i] = 1
        jac[i] = gradient(y, x, grad_outputs=grad_outputs, create_graph=True)
    return jac


class Gradient(Component):
    DEFAULT_INPUT_KEYS = ["y", "x"]
    DEFAULT_OUTPUT_KEYS = ["dy/dx"]
    """
    Gradient component class for computing gradients of neuromancer objects given the
    generated dictionary dataset with keys corresponding to variables to be differentiated.
    """
    def __init__(self, input_key_map={}, name=None):
        """
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

