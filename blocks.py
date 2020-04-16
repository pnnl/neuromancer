"""
Function approximators of various degrees of generality.
Sparsity inducing prior can be gotten from the LassoLinear in linear module
"""
import linear
import torch
import torch.nn as nn
import scipy.misc


def expand(x):
    """
    square expansion of tensor x
    """
    expansion = torch.matmul(x.unsqueeze(-1), x.unsqueeze(1)).view(-1, x.shape[1]**2)
    return torch.cat([x, expansion], dim=1)


class Bilinear(nn.Module):
    """
    bilinear term: why expansion and not nn.Bilinear?
    """
    def init(self, insize, outsize, bias=False, lin_cls=linear.Linear):
        self.insize, self.outsize, = insize, outsize
        self.linear = lin_cls(insize**2, outsize, bias=bias)

    def regularization(self):
        return self.linear.regularization

    def forward(self, x):
        return self.linear(expand(x))


class Fourier(nn.Module):
    pass


class Chebyshev(nn.Module):
    pass


class Polynomial(nn.Module):
    pass


class Multinomial(nn.Module):
    
    def init(self, inputsize, outputsize, p=2, bias=False, lin_cls=linear.Linear):
        self.p = p
        for i in range(p-1):
            inputsize += inputsize**2
        self.linear = lin_cls(scipy.misc.comb(inputsize + p, p + 1), outputsize, bias=bias)

    def regularization(self):
        return self.linear.regularization

    def forward(self, x):
        for i in range(self.p):
            x = expand(x)
        return self.linear(x)


class SINDy():
    pass


class MLP():
    pass


class ResMLP():
    pass


class RNN():
    pass

