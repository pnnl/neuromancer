"""
Function approximators of various degrees of generality.
Sparsity inducing prior can be gotten from the LassoLinear in linear module
"""
# machine learning/data science imports
import numpy as np
import torch
import torch.nn as nn

# eosystem imports
import slim

# local imports
import neuromancer.rnn as rnn
from neuromancer.activations import SoftExponential


def get_modules(model):
    return {name: module for name, module in model.named_modules() if len(list(module.named_children())) == 0}


class Linear(nn.Module):
    def __init__(self,  insize, outsize, bias=True,
                 Linear=slim.Linear, nonlin=SoftExponential, hsizes=[64], linargs=dict()):
        super().__init__()
        self.linear = Linear(insize, outsize, bias=bias, **linargs)

    def reg_error(self):
        return self.linear.reg_error()

    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    def __init__(self,  insize, outsize, bias=True,
                 Linear=slim.Linear, nonlin=SoftExponential, hsizes=[64], linargs=dict()):
        """

        :param insize:
        :param outsize:
        :param bias:
        :param Linear:
        :param nonlin:
        :param hsizes:
        :param linargs: (dict) arguments for
        """
        super().__init__()
        self.in_features, self.out_features = insize, outsize
        self.nhidden = len(hsizes)
        sizes = [insize] + hsizes + [outsize]
        self.nonlin = [nonlin() for k in range(self.nhidden)] + [nn.Identity()]
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
                 Linear=slim.Linear, nonlin=SoftExponential, hsizes=[64], linargs=dict(), skip=1):
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
                         Linear=Linear, nonlin=nonlin, hsizes=hsizes, linargs=linargs)
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


class PytorchRNN(nn.Module):

    """
    This wraps the torch.nn.RNN class to give output which is a linear map from final hidden state.
    """

    def __init__(self, insize, outsize, bias=True,
                 Linear=slim.Linear, nonlin=SoftExponential, hsizes=[10], linargs=dict()):
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
        self.rnn = nn.RNN(insize, hsizes[0],
                          bias=bias, nonlinearity='relu')
        self.output = Linear(hsizes[-1], outsize, bias=bias, **linargs)

    def reg_error(self):
        return self.output.reg_error()

    def forward(self, x):
        """
        There is some logic here so that the RNN will still get context from state in open loop simulation.

        :param x: (torch.Tensor, shape=(nsteps, nsamples, dim)) Input sequence is expanded for order 2 tensors
        :return: (torch.Tensor, shape=(nsamples, outsize)
        """
        if len(x.shape) == 2:
            x = x.reshape(1, *x.shape)
        _, hiddens = self.rnn(x)
        return self.output(hiddens[-1])


class RNN(nn.Module):
    """
    This wraps the rnn.RNN class to give output which is a linear map from final hidden state.
    """
    def __init__(self, insize, outsize, bias=True,
                 Linear=slim.Linear, nonlin=SoftExponential, hsizes=[1], linargs=dict()):
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
        return self.rnn.reg_error() + self.output.reg_error()

    def reset(self):
        self.init_states = None

    def forward(self, x, hx=None):
        """
        There is some logic here so that the RNN will still get context from state in open loop simulation.

        :param x: (torch.Tensor, shape=(nsteps, nsamples, dim)) Input sequence is expanded for order 2 tensors
        :return: (torch.Tensor, shape=(nsamples, outsize)
        """
        if len(x.shape) == 2:
            x = x.reshape(1, *x.shape)
        _, hiddens = self.rnn(x, init_states=hx)
        self.init_states = hiddens
        return self.output(hiddens[-1])


class BilinearTorch(nn.Module):
    def __init__(self, insize, outsize, bias=True,
                 Linear=slim.Linear, nonlin=SoftExponential, hsizes=[64], linargs=dict()):
        """
        bilinear term from Torch
        """
        super().__init__()
        self.in_features, self.out_features = insize, outsize
        self.f = nn.Bilinear(self.in_features, self.in_features, self.out_features, bias=bias)

    def reg_error(self):
        return torch.tensor(0.0).to(self.f.weight)

    def forward(self, x):
        return self.f(x, x)


class Poly2(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        row_idxs, col_idxs = np.triu_indices(x.shape[-1])
        expansion = torch.matmul(x.unsqueeze(-1), x.unsqueeze(1))  # outer product
        expansion = expansion[:, row_idxs, col_idxs]  # upper triangular
        return torch.cat([x, expansion], dim=-1)  # concatenate


class BasisLinear(nn.Module):
    def __init__(self, insize, outsize, bias=True,
                 Linear=slim.Linear, nonlin=SoftExponential, hsizes=[64], linargs=dict(),
                 expand=Poly2()):
        """
        For mapping inputs to functional basis feature expansion.
        """
        super().__init__()
        self.in_features, self.out_features = insize, outsize
        self.expand = expand
        inlin = self.expand(torch.zeros(1, insize)).shape[-1]
        self.linear = Linear(inlin, outsize, bias=bias, **linargs)
        self.bias = nn.Parameter(torch.zeros(1, outsize), requires_grad=not bias)

    def reg_error(self):
        return self.linear.reg_error()

    def forward(self, x):
        return self.linear(self.expand(x))


blocks = {'mlp': MLP, 'rnn': RNN, 'pytorch_rnn': PytorchRNN,
          'linear': Linear, 'residual_mlp': ResMLP,
          'basislinear': BasisLinear}


if __name__ == '__main__':
    y = torch.randn([25, 5])
    for name, block in blocks.items():
        block = block(5, 7, bias=True, hsizes=[64, 64, 64, 64, 64, 64])
        print(name)
        print(block(y).shape)

    expand = Poly2()
    print(expand(torch.tensor([[2, 3]])))

    expand = Poly2()
    print(expand(torch.tensor([[2, 5]])))

    block = MLP(5, 7, bias=True, hsizes=[5, 10, 2, 7])
    y = torch.randn([25, 5])
    print(block(y).shape)

    block = ResMLP(5, 7, bias=True, hsizes=[64, 64, 64, 64, 64, 64], skip=3)
    y = torch.randn([25, 5])
    print(block(y).shape)

    block = RNN(5, 7, bias=True, hsizes=[64, 64, 64, 64, 64, 64])
    y = torch.randn([32, 25, 5])
    print(block(y).shape)

    block = PytorchRNN(5, 7, bias=True, hsizes=[64, 64, 64, 64, 64, 64])
    y = torch.randn([32, 25, 5])
    print(block(y).shape)

    block = BasisLinear(5, 7, bias=True)
    y = torch.randn([25, 5])
    print(block(y).shape)





