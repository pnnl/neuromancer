import math

import torch
from torch import nn

from .permutation_multiply import permutation_mult, permutation_mult_single


class Permutation(nn.Module):
    """Product of log N permutation factors.

    Parameters:
        size: size of input (and of output)
        share_logit: whether the logits in the permutation factors are shared.
            If True, will have 4N parameters, else will have 2 N log N parameters (not counting bias)
        increasing_stride: whether to multiply from smaller stride to larger stride, or in the reverse order.
    """

    def __init__(self, size, share_logit=False, increasing_stride=False):
        super().__init__()
        self.size = size
        m = int(math.ceil(math.log2(size)))
        assert size == 1 << m, "size must be a power of 2"
        self.share_logit = share_logit
        self.increasing_stride = increasing_stride
        self.logit = nn.Parameter(torch.randn(3)) if share_logit else nn.Parameter(torch.randn(m - 1, 3))

    def forward(self, input):
        """
        Parameters:
            input: (batch, size) if real or (batch, size, 2) if complex
        Return:
            output: (batch, size) if real or (batch, size, 2) if complex
        """
        prob = torch.sigmoid(self.logit)
        if self.share_logit:
            m = int(math.ceil(math.log2(self.size)))
            prob = prob.unsqueeze(0).expand(m - 1, 3)
        return permutation_mult(prob, input, increasing_stride=self.increasing_stride)

    def argmax(self):
        """
        Return:
            p: (self.size, ) array of int, the most probable permutation.
        """
        prob = torch.sigmoid(self.logit).round()
        if self.share_logit:
            m = int(math.ceil(math.log2(self.size)))
            prob = prob.unsqueeze(0).expand(m - 1, 3)
        input = torch.arange(self.size, dtype=prob.dtype, device=self.logit.device).unsqueeze(0)
        return permutation_mult(prob, input, increasing_stride=self.increasing_stride).squeeze(0).round().long()

    def extra_repr(self):
        return 'size={}, share_logit={}, increasing_stride={}'.format(
            self.size, self.share_logit, self.increasing_stride
        )


class FixedPermutation(nn.Module):

    def __init__(self, permutation):
        """Fixed permutation. Used to store argmax of Permutation.
        Parameter:
            permutation: (n, ) tensor of ints
        """
        super().__init__()
        self.register_buffer('permutation', permutation)

    def forward(self, input):
        """
        Parameters:
            input: (batch, size) if real or (batch, size, 2) if complex
        Return:
            output: (batch, size) if real or (batch, size, 2) if complex
        """
        return input[:, self.permutation]


class PermutationFactor(nn.Module):
    """A single permutation factor.

    Parameters:
        size: size of input (and of output)
    """

    def __init__(self, size):
        super().__init__()
        self.size = size
        m = int(math.ceil(math.log2(size)))
        assert size == 1 << m, "size must be a power of 2"
        self.logit = nn.Parameter(torch.randn(3))

    def forward(self, input):
        """
        Parameters:
            input: (batch, size) if real or (batch, size, 2) if complex
        Return:
            output: (batch, size) if real or (batch, size, 2) if complex
        """
        prob = torch.sigmoid(self.logit)
        return permutation_mult_single(prob, input)

    def argmax(self):
        """
        Return:
            p: (self.size, ) array of int, the most probable permutation.
        """
        prob = torch.sigmoid(self.logit).round()
        input = torch.arange(self.size, dtype=prob.dtype, device=self.logit.device).unsqueeze(0)
        return permutation_mult_single(prob, input).squeeze(0).round().long()

    def extra_repr(self):
        return 'size={}'.format(self.size)
