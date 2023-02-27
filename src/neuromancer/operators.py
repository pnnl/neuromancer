"""
An operator :math:`\odot` is a shape preserving function of two arbitrarilly shaped tensors with the same shapes.
For example matrix addition or the Hadamard product (elementwise multiplication).

"""
import torch
import torch.nn as nn

from neuromancer.activations import soft_exp


class InterpolateAddMultiply(nn.Module):
    """
    Implementation of smooth interpolation between addition and multiplication
    using soft exponential activation: https://arxiv.org/pdf/1602.01321.pdf
    """
    def __init__(self, alpha=0.0, tune_alpha=True):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=tune_alpha)

    def forward(self, p, q):
        return soft_exp(self.alpha, soft_exp(-self.alpha, p) + soft_exp(-self.alpha, q))


operators = {'add': torch.add, 'mul': torch.mul, 'addmul': InterpolateAddMultiply()}