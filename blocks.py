"""
Function approximators of various degrees of generality.
Sparsity inducing prior can be gotten from the LassoLinear in linear module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.misc

import linear


def expand(x):
    expansion = torch.matmul(x.unsqueeze(-1), x.unsqueeze(1)).view(-1, x.shape[1]**2)
    return torch.cat([x, expansion], dim=1)


class FunctionBasis(nn.Module):
    def __init__(self, insize, outsize, basis_functions, bias=False):
        self.basis_functions = basis_functions


class DeepBasisNetwork(nn.Module):
    pass


class Bilinear(nn.Module):
    def init(self, insize, outsize, bias=False, Linear=linear.Linear):
        self.insize, self.outsize, = insize, outsize
        self.linear = Linear(insize**2, outsize, bias=bias)
        self.bias = nn.Parameter(torch.zeros(1, outsize), requires_grad=not bias)

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


class MLP(nn.Module):
    def __init__(self,  insize, outsize, bias=True,
                 Linear=linear.Linear, nonlin=F.relu, hsizes=[64], **linargs):
        """

        :param layers: list of ints (insize, h1size, h2size, ..., hnsize, outsize)
        :param nonlin: Activation function
        :param bias: Whether to use bias
        """
        super().__init__()
        self.nhidden = len(hsizes)
        sizes = [insize] + hsizes + [outsize]
        self.nonlin = [nonlin]*self.nhidden + [nn.Identity()]
        self.linear = nn.ModuleList([Linear(sizes[k],
                                            sizes[k+1],
                                            bias=bias,
                                            **linargs) for k in range(self.nhidden+1)])

    def reg_error(self):
        return torch.mean([l.regularization_error() for l in self.linear])

    def forward(self, x):
        for lin, nlin in zip(self.linear, self.nonlin):
            x = lin(nlin(x))
        return x


class ResMLP(MLP):
    def __init__(self, layers, Linear=linear.Linear, nonlin=F.relu, bias=True, skip=2, **linargs):
        """

        :param layers: list of ints (insize, h1size, h2size, ..., hnsize, outsize)
        :param nonlin: Activation function
        :param bias: Whether to use bias
        """
        super().__init__(layers, Linear=Linear, nonlin=nonlin, bias=bias, **linargs)
        self.skip = skip
        self.residual = [nn.Identity()]*(self.nlayers+1)
        for k in range(self.nlayers, skip-1, -1):
            self.residual[k] = Linear(layers[k-skip], layers[k], bias=False)
        for k in range(skip-1, 0, -1):
            self.residual[k] = Linear(layers[0], layers[k], bias=False)

        # self.residual[0] = nn.Identity()

        # pshape = layers[0]
        # for k in range(self.nlayers-skip+1):
        #     print(k)
        #     if k % skip == 0 or k + skip >= self.nlayers:
        #         self.residual.append(Linear(pshape, layers[k + skip], bias=False, **linargs))
        #         pshape = layers[k + skip]
        #     else:
        #         self.residual.append(nn.Identity())
        print(self.residual)

    def regularization_error(self):
        return super.regularization_error + torch.mean([l.regularization_error()
                                                        for l in self.residual.values()
                                                        if type(l) is linear.LinearBase])

    def forward(self, x):
        px = x
        for layer, (lin, nlin) in enumerate(zip(self.linear, self.nonlin)):
            if (layer % self.skip == 0 or layer == self.nlayers-1) and layer != 0:
                self.residual[layer]
                x = x + self.residual[layer](px)
                px = x
            x = nlin(lin(x))
        return x



class RNN():
    pass

if __name__ == '__main__':

    block = MLP([5, 10, 2, 7], bias=True)
    y = torch.randn([25, 5])
    print(block(y).shape)

    block = ResMLP([5, 10, 2, 7, 8, 11], skip=3, bias=True)
    y = torch.randn([25, 5])
    print(block(y).shape)



