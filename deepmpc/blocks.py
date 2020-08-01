"""
TODO: have blocs corresponding to matlab style function basis creation
TODO: debug bilinear
TODO: custom activation functions
TODO: SoftExponential activation from https://arxiv.org/abs/1602.01321
TODO: Fourier, Chebyshev, Polynomial, basis expansions
TODO: Finish and test Multinomial basis expansion
TODO: Implement SINDy block
TODO: wrapper for pytorch modules e.g.: RNN, LSTM for benchmarking OR check if reg_error in module in components

Function approximators of various degrees of generality.
Sparsity inducing prior can be gotten from the LassoLinear in linear module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
#local imports
import linear
import rnn
import scipy


def get_modules(model):
    return {name: module for name, module in model.named_modules() if len(list(module.named_children())) == 0}


def expand(x):
    """
    square expansion of tensor x
    """
    expansion = torch.matmul(x.unsqueeze(-1), x.unsqueeze(1)).view(-1, x.shape[1]**2)
    return torch.cat([x, expansion], dim=1)


class MLP(nn.Module):
    def __init__(self,  insize, outsize, bias=True,
                 Linear=linear.Linear, nonlin=F.gelu, hsizes=[64], linargs=dict()):
        """

        :param insize:
        :param outsize:
        :param bias:
        :param Linear:
        :param nonlin:
        :param hsizes:
        :param linargs:
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
        """

        :return:
        """
        return sum([k.reg_error()
                    for k in self.linear
                    if hasattr(k, 'reg_error')])

    def forward(self, x):
        """

        :param x:
        :return:
        """
        for lin, nlin in zip(self.linear, self.nonlin):
            x = lin(nlin(x))
        return x


class ResMLP(MLP):
    def __init__(self, insize, outsize, bias=True,
                 Linear=linear.Linear, nonlin=F.gelu, hsizes=[64], skip=1, linargs=dict()):
        """

        :param insize:
        :param outsize:
        :param bias:
        :param Linear:
        :param nonlin:
        :param hsizes:
        :param skip:
        :param linargs:
        """

        super().__init__(insize, outsize, bias=bias,
                         Linear=Linear, nonlin=nonlin, hsizes=hsizes, **linargs)
        assert len(set(hsizes)) == 1, 'All hidden sizes should be equal for residual network'
        self.skip = skip
        self.inmap = Linear(insize, hsizes[0], bias=bias, **linargs)
        self.outmap = Linear(hsizes[0], outsize, bias=bias, **linargs)
        self.in_features, self.out_features = insize, outsize

    def forward(self, x):
        """

        :param x:
        :return:
        """
        px = self.inmap(x)
        for layer, (lin, nlin) in enumerate(zip(self.linear[:-1], self.nonlin[:-1])):
            x = nlin(lin(x))
            if layer % self.skip == 0:
                x = x + px
                px = x
        return self.linear[-1](x) + self.outmap(px)


class RNN(nn.Module):
    """
    This wraps the rnn.RNN class for to give output which is a linear map from final hidden state.
    """
    def __init__(self, insize, outsize, bias=False,
                 Linear=linear.Linear, nonlin=F.gelu, hsizes=[1], linargs=dict()):
        """

        :param insize:
        :param outsize:
        :param bias:
        :param Linear:
        :param nonlin:
        :param hsizes:
        :param linargs:
        """
        super().__init__()
        assert len(set(hsizes)) == 1
        self.in_features, self.out_features = insize, outsize
        self.rnn = rnn.RNN(insize, hsizes=hsizes,
                           bias=bias, nonlin=nonlin, Linear=Linear, linargs=linargs)
        self.output = Linear(hsizes[-1], outsize, bias=bias, **linargs)
        self.init_states = list(self.rnn.init_states)

    def reg_error(self):
        return self.rnn.reg_error()

    def reset(self):
        self.init_states = None

    def forward(self, x):
        """
        There is some logic here so that the RNN will still get context from state in open loop simulation.

        :param x: (torch.Tensor, shape=(nsteps, nsamples, dim)) Input sequence is expanded for order 2 tensors
        :return: (torch.Tensor, shape=(nsamples, outsize)
        """
        if len(x.shape) == 2:
            x = x.reshape(1, *x.shape)
        if self.init_states[0].shape[0] == x.shape[1] and not self.training:
            _, hiddens = self.rnn(x, init_states=self.init_states)
        else:
            _, hiddens = self.rnn(x)
        self.init_states = hiddens
        return self.output(hiddens[-1])


class Bilinear(nn.Module):
    def __init__(self, insize, outsize, bias=False, Linear=linear.Linear, linargs=dict()):
        """
        bilinear term: why expansion and not nn.Bilinear?
        """
        super().__init__()
        # self.insize, self.outsize, = insize, outsize
        self.in_features, self.out_features = insize, outsize
        self.linear = Linear(insize**2, outsize, bias=bias, **linargs)
        self.bias = nn.Parameter(torch.zeros(1, outsize), requires_grad=not bias)

    def reg_error(self):
        return self.linear.reg_error()

    def forward(self, x):
        return self.linear(expand(x))


class BilinearTorch(nn.Module):
    def __init__(self, insize, outsize, bias=False, Linear=linear.Linear, linargs=dict()):
        """
        bilinear term from Torch
        """
        super().__init__()
        self.in_features, self.out_features = insize, outsize
        self.f = nn.Bilinear(self.in_features, self.in_features, self.out_features, bias=bias)
        self.error_matrix = nn.Parameter(torch.zeros(1), requires_grad=False)

    def reg_error(self):
        return self.error_matrix

    def forward(self, x):
        return self.f(x, x)


class Multinomial(nn.Module):
    def __init__(self, insize, outsize, p=2, bias=False, lin_cls=linear.Linear, linargs=dict()):
        super().__init__()
        self.p = p
        self.in_features, self.out_features = insize, outsize
        for i in range(p-1):
            insize += insize**2
        self.linear = lin_cls(scipy.misc.comb(insize + p, p + 1), outsize, bias=bias, **linargs)

    def reg_error(self):
        return self.linear.regularization

    def forward(self, x):
        for i in range(self.p):
            x = expand(x)
        return self.linear(x)


if __name__ == '__main__':

    block = MLP(5, 7, bias=True, hsizes=[5, 10, 2, 7])
    y = torch.randn([25, 5])
    print(block(y).shape)

    block = ResMLP(5, 7, bias=True, hsizes=[64, 64, 64, 64, 64, 64], skip=3)
    y = torch.randn([25, 5])
    print(block(y).shape)

    block = RNN(5, 7, bias=True, hsizes=[64, 64, 64, 64, 64, 64])
    y = torch.randn([25, 32, 5])
    print(block(y).shape)
    print(block(y).shape)
    print(block(y).shape)



