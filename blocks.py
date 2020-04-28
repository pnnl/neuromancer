"""
Function approximators of various degrees of generality.
Sparsity inducing prior can be gotten from the LassoLinear in linear module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.misc
import linear


def get_modules(model):
    return {name: module for name, module in model.named_modules() if len(list(module.named_children())) == 0}


def expand(x):
    """
    square expansion of tensor x
    """
    expansion = torch.matmul(x.unsqueeze(-1), x.unsqueeze(1)).view(-1, x.shape[1]**2)
    return torch.cat([x, expansion], dim=1)


class FunctionBasis(nn.Module):
    def __init__(self, insize, outsize, basis_functions, bias=False):
        self.basis_functions = basis_functions


class DeepBasisNetwork(nn.Module):
    pass


class Bilinear(nn.Module):
    def init(self, insize, outsize, bias=False, Linear=linear.Linear):
        """
        bilinear term: why expansion and not nn.Bilinear?
        """
        self.insize, self.outsize, = insize, outsize
        self.linear = Linear(insize**2, outsize, bias=bias)
        self.bias = nn.Parameter(torch.zeros(1, outsize), requires_grad=not bias)

    def regularization(self):
        return self.linear.regularization

    def forward(self, x):
        return self.linear(expand(x))


class SoftExponential(nn.Module):
    pass
# https://arxiv.org/abs/1602.01321

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


class MLP(nn.Module):
    def __init__(self,  insize, outsize, bias=True,
                 Linear=linear.Linear, nonlin=F.relu, hsizes=[64], **linargs):
        """

        :param layers: list of ints (insize, h1size, h2size, ..., hnsize, outsize)
        :param nonlin: Activation function
        :param bias: Whether to use bias
        """
        super().__init__()
        self.in_features, self.out_features = insize, outsize
        self.nhidden = len(hsizes)
        sizes = [insize] + hsizes + [outsize]
        self.nonlin = [nonlin]*self.nhidden + [nn.Identity()]
        self.linear = nn.ModuleList([Linear(sizes[k],
                                            sizes[k+1],
                                            bias=bias,
                                            **linargs) for k in range(self.nhidden+1)])

    def reg_error(self):
        return sum([k.reg_error()
                    for k in self.linear
                    if hasattr(k, 'reg_error')])

    def forward(self, x):
        for lin, nlin in zip(self.linear, self.nonlin):
            x = lin(nlin(x))
        return x


class ResMLP(MLP):
    def __init__(self,  insize, outsize, bias=True,
                 Linear=linear.Linear, nonlin=F.relu, hsizes=[64], skip=1, **linargs):

        super().__init__(insize, outsize, bias=bias,
                         Linear=Linear, nonlin=nonlin, hsizes=hsizes, **linargs)
        assert len(set(hsizes)) == 1, 'All hidden sizes should be equal for residual network'
        self.skip = skip
        self.inmap = Linear(insize, hsizes[0], bias=False, **linargs)
        self.outmap = Linear(hsizes[0], outsize, bias=False, **linargs)

    def forward(self, x):
        px = self.inmap(x)
        for layer, (lin, nlin) in enumerate(zip(self.linear[:-1], self.nonlin[:-1])):
            x = nlin(lin(x))
            if layer % self.skip == 0:
                x = x + px
                px = x
        return self.linear[-1](x) + self.outmap(px)


class RNN():
    pass


# Some other options for activation functions
# https://towardsdatascience.com/extending-pytorch-with-custom-activation-functions-2d8b065ef2fa
# https://github.com/Lexie88rus/Activation-functions-examples-pytorch/blob/master/custom_activations_example.py

if __name__ == '__main__':

    block = MLP(5, 7, bias=True, hsizes=[5, 10, 2, 7])
    y = torch.randn([25, 5])
    print(block(y).shape)

    block = ResMLP(5, 7, hsizes=[64, 64, 64, 64, 64, 64], skip=3, bias=True)
    y = torch.randn([25, 5])
    print(block(y).shape)



