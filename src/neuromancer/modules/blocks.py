"""
Function approximators of various degrees of generality which implement a consistent block interface.
Neural network module building blocks for neural state space models, state estimators and control policies.
"""

import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

import neuromancer.slim as slim
import neuromancer.modules.rnn as rnn
from neuromancer.modules.activations import soft_exp, SoftExponential, SmoothedReLU


class Block(nn.Module, ABC):
    """
    Canonical abstract class of the block function approximator
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def block_eval(self, x):
        pass

    def forward(self, *inputs):
        """
        Handling varying number of tensor inputs

        :param inputs: (list(torch.Tensor, shape=[batchsize, insize]) or torch.Tensor, shape=[batchsize, insize])
        :return: (torch.Tensor, shape=[batchsize, outsize])
        """
        if len(inputs) > 1:
            x = torch.cat(inputs, dim=-1)
        else:
            x = inputs[0]
        return self.block_eval(x)


class Linear(Block):
    """
    Linear map consistent with block interface
    """
    def __init__(
        self,
        insize,
        outsize,
        bias=True,
        linear_map=slim.Linear,
        nonlin=None,
        hsizes=None,
        linargs=dict(),
    ):
        """

        :param insize: (int) dimensionality of input
        :param outsize: (int) dimensionality of output
        :param bias: (bool) Whether to use bias
        :param linear_map: (class) Linear map class from neuromancer.slim.linear
        :param nonlin: (callable) Not used in this module but included as argument for consistent interface
        :param hsizes: (list of ints) Not used in this module but included as argument for consistent interface
        :param linargs: (dict) Arguments for instantiating linear layer
        """

        super().__init__()
        self.in_features, self.out_features = insize, outsize
        self.linear = linear_map(insize, outsize, bias=bias, **linargs)

    def reg_error(self):
        return self.linear.reg_error()

    def block_eval(self, x):
        """

        :param x: (torch.Tensor, shape=[batchsize, insize])
        :return: (torch.Tensor, shape=[batchsize, outsize])
        """
        return self.linear(x)


class Dropout(Block):
    def __init__(self, p=0.0, at_train=False, at_test=True):
        """Wrapper for standard dropout that allows its use at test time.
        By default, dropout is disabled during training as it appears difficult
        to train models with it enabled.

        :param p: probability that an input component will be set to zero
        :param at_train: enable dropout during training
        :param at_test: enable dropout during testing
        """
        super().__init__()
        self.p = p
        self.at_train = at_train
        self.at_test = at_test

    def block_eval(self, x):
        use_dropout = (self.training and self.at_train) or (not self.training and self.at_test)
        return torch.nn.functional.dropout(x, p=self.p, training=use_dropout)


def set_model_dropout_mode(model, at_train=None, at_test=None):
    """Change dropout mode, useful for enabling MC sampling during inference time.
    """
    def _apply_fn(x):
        if type(x) == Dropout:
            if at_test is not None: x.at_test = at_test
            if at_train is not None: x.at_train = at_train
    model.apply(_apply_fn)


class MLP(Block):
    """
    Multi-Layer Perceptron consistent with blocks interface
    """
    def __init__(
        self,
        insize,
        outsize,
        bias=True,
        linear_map=slim.Linear,
        nonlin=SoftExponential,
        hsizes=[64],
        linargs=dict(),
    ):
        """

        :param insize: (int) dimensionality of input
        :param outsize: (int) dimensionality of output
        :param bias: (bool) Whether to use bias
        :param linear_map: (class) Linear map class from neuromancer.slim.linear
        :param nonlin: (callable) Elementwise nonlinearity which takes as input torch.Tensor and outputs torch.Tensor of same shape
        :param hsizes: (list of ints) List of hidden layer sizes
        :param linargs: (dict) Arguments for instantiating linear layer
        :param dropout: (float) Dropout probability
        """
        super().__init__()
        self.in_features, self.out_features = insize, outsize
        self.nhidden = len(hsizes)
        sizes = [insize] + hsizes + [outsize]
        self.nonlin = nn.ModuleList([nonlin() for k in range(self.nhidden)] + [nn.Identity()])
        self.linear = nn.ModuleList(
            [
                linear_map(sizes[k], sizes[k + 1], bias=bias, **linargs)
                for k in range(self.nhidden + 1)
            ]
        )

    def reg_error(self):
        return sum([k.reg_error() for k in self.linear if hasattr(k, "reg_error")])

    def block_eval(self, x):
        """

        :param x: (torch.Tensor, shape=[batchsize, insize])
        :return: (torch.Tensor, shape=[batchsize, outsize])
        """
        for lin, nlin in zip(self.linear, self.nonlin):
            x = nlin(lin(x))
        return x



def sigmoid_scale(x, min, max):
    return (max - min) * torch.sigmoid(x) + min

def relu_clamp(x, min, max):
    x = x + torch.relu(-x + min)
    x = x - torch.relu(x - max)
    return x


class MLP_bounds(MLP):
    """
    Multi-Layer Perceptron consistent with blocks interface
    """
    bound_methods = {'sigmoid_scale': sigmoid_scale,
                    'relu_clamp': relu_clamp}

    def __init__(
        self,
        insize,
        outsize,
        bias=True,
        linear_map=slim.Linear,
        nonlin=SoftExponential,
        hsizes=[64],
        linargs=dict(),
        min=0.0,
        max=1.0,
        method='sigmoid_scale',
    ):
        """

        :param insize: (int) dimensionality of input
        :param outsize: (int) dimensionality of output
        :param bias: (bool) Whether to use bias
        :param linear_map: (class) Linear map class from neuromancer.slim.linear
        :param nonlin: (callable) Elementwise nonlinearity which takes as input torch.Tensor and outputs torch.Tensor of same shape
        :param hsizes: (list of ints) List of hidden layer sizes
        :param linargs: (dict) Arguments for instantiating linear layer
        :param dropout: (float) Dropout probability
        """
        super().__init__(insize=insize, outsize=outsize, bias=bias,
                         linear_map=linear_map, nonlin=nonlin,
                         hsizes=hsizes, linargs=linargs)
        self.min = min
        self.max = max
        self.method = self._set_method(method)

    def _set_method(self, method):
        if method in self.bound_methods.keys():
            return self.bound_methods[method]
        else:
            assert callable(method), \
                f'Method, {method} must be a key in {self.bound_methods} ' \
                f'or a differentiable callable.'
            return method

    def block_eval(self, x):
        """

        :param x: (torch.Tensor, shape=[batchsize, insize])
        :return: (torch.Tensor, shape=[batchsize, outsize])
        """
        for lin, nlin in zip(self.linear, self.nonlin):
            x = nlin(lin(x))
        return self.method(x, self.min, self.max)


class InteractionEmbeddingMLP(nn.Module):
    """
    Multi-Layer Perceptron which is a hypernetwork hidden state embeddings decided by interaction type and concatenated
    to hidden state.
    """
    def __init__(
        self,
        insize,
        outsize,
        bias=True,
        linear_map=slim.Linear,
        nonlin=SoftExponential,
        hsizes=[64],
        linargs=dict(),
        n_interactors=9,
    ):
        """

        :param insize: (int) dimensionality of input
        :param outsize: (int) dimensionality of output
        :param bias: (bool) Whether to use bias
        :param linear_map: (class) Linear map class from neuromancer.slim.linear
        :param nonlin: (callable) Elementwise nonlinearity which takes as input torch.Tensor and outputs torch.Tensor of same shape
        :param hsizes: (list of ints) List of hidden layer sizes
        :param linargs: (dict) Arguments for instantiating linear layer
        :param dropout: (float) Dropout probability
        :param n_interactors: (int) Number of interacting entity types number of interactions is n_interactors squared.
        """
        super().__init__()
        self.in_features, self.out_features = insize, outsize
        self.nhidden = len(hsizes)
        self.n_interactors = n_interactors
        em_sizes = [hsizes[0], *hsizes]
        sizes = [insize] + hsizes
        sizes = [size + em_size for size, em_size in zip(sizes, em_sizes)]
        sizes += [outsize]
        self.nonlin = nn.ModuleList([nonlin() for k in range(self.nhidden)] + [nn.Identity()])
        self.embeddings = [nn.Embedding(int(n_interactors**2), n_embed) for n_embed in em_sizes]
        self.linear = nn.ModuleList(
            [
                linear_map(sizes[k], sizes[k + 1], bias=bias, **linargs)
                for k in range(self.nhidden + 1)
            ]
        )

    def reg_error(self):
        return sum([k.reg_error() for k in self.linear if hasattr(k, "reg_error")])

    def forward(self, x, i, j):
        """

        :param x: (torch.Tensor, shape=[batchsize, insize])
        :return: (torch.Tensor, shape=[batchsize, outsize])
        """
        for lin, nlin, embedder in zip(self.linear, self.nonlin, self.embeddings):
            x = torch.cat([x, embedder(self.n_interactors*i + j)])
            x = nlin(lin(x))
        return x


class MLPDropout(Block):
    """
    Multi-Layer Perceptron with dropout consistent with blocks interface
    """
    def __init__(
        self,
        insize,
        outsize,
        bias=True,
        linear_map=slim.Linear,
        nonlin=SoftExponential,
        hsizes=[64],
        linargs=dict(),
        dropout=0.0
    ):
        """

        :param insize: (int) dimensionality of input
        :param outsize: (int) dimensionality of output
        :param bias: (bool) Whether to use bias
        :param linear_map: (class) Linear map class from neuromancer.slim.linear
        :param nonlin: (callable) Elementwise nonlinearity which takes as input torch.Tensor and outputs torch.Tensor of same shape
        :param hsizes: (list of ints) List of hidden layer sizes
        :param linargs: (dict) Arguments for instantiating linear layer
        :param dropout: (float) Dropout probability
        """
        super().__init__()
        self.in_features, self.out_features = insize, outsize
        self.nhidden = len(hsizes)
        sizes = [insize] + hsizes + [outsize]
        self.nonlin = nn.ModuleList([nonlin() for k in range(self.nhidden)] + [nn.Identity()])
        self.linear = nn.ModuleList(
            [
                linear_map(sizes[k], sizes[k + 1], bias=bias, **linargs)
                for k in range(self.nhidden + 1)
            ]
        )
        self.dropout = nn.ModuleList(
            [Dropout(p=dropout) if dropout > 0.0 else nn.Identity() for _ in range(self.nhidden)]
            + [nn.Identity()]
        )

    def reg_error(self):
        return sum([k.reg_error() for k in self.linear if hasattr(k, "reg_error")])

    def block_eval(self, x):
        """

        :param x: (torch.Tensor, shape=[batchsize, insize])
        :return: (torch.Tensor, shape=[batchsize, outsize])
        """
        for lin, nlin, drop in zip(self.linear, self.nonlin, self.dropout):
            x = drop(nlin(lin(x)))
        return x


class ResMLP(MLP):
    """
    Residual MLP consistent with the block interface.
    """
    def __init__(
        self,
        insize,
        outsize,
        bias=True,
        linear_map=slim.Linear,
        nonlin=SoftExponential,
        hsizes=[64],
        linargs=dict(),
        skip=1,
    ):
        """

        :param insize: (int) dimensionality of input
        :param outsize: (int) dimensionality of output
        :param bias: (bool) Whether to use bias
        :param linear_map: (class) Linear map class from neuromancer.slim.linear
        :param nonlin: (callable) Elementwise nonlinearity which takes as input torch.Tensor and outputs torch.Tensor of same shape
        :param hsizes: (list of ints) List of hidden layer sizes
        :param linargs: (dict) Arguments for instantiating linear layer
        """

        super().__init__(
            insize,
            outsize,
            bias=bias,
            linear_map=linear_map,
            nonlin=nonlin,
            hsizes=hsizes,
            linargs=linargs,
        )
        assert (
            len(set(hsizes)) == 1
        ), "All hidden sizes should be equal for residual network"
        self.skip = skip
        self.inmap = linear_map(insize, hsizes[0], bias=bias, **linargs)
        self.outmap = linear_map(hsizes[0], outsize, bias=bias, **linargs)
        self.in_features, self.out_features = insize, outsize

    def block_eval(self, x):
        """

        :param x: (torch.Tensor, shape=[batchsize, insize])
        :return: (torch.Tensor, shape=[batchsize, outsize])
        """
        px = self.inmap(x)
        for layer, (lin, nlin) in enumerate(zip(self.linear[:-1], self.nonlin[:-1])):
            x = nlin(lin(x))
            if layer % self.skip == 0:
                x = x + px
                px = x
        return self.linear[-1](x) + self.outmap(px)


class InputConvexNN(MLP):
    """
    Input convex neural network
    z1 =  sig(W0(x) + b0)
    z_i+1 = sig_i(Ui(zi) + Wi(x) + bi),  i = 1, ..., k-1
    V = g(x) = zk

    Equation 11 from https://arxiv.org/abs/2001.06116
    """

    def __init__(self,
                 insize,
                 outsize,
                 bias=True,
                 linear_map=slim.Linear,
                 nonlin=nn.ReLU,
                 hsizes=[64],
                 linargs=dict()
                 ):
        super().__init__(
            insize,
            outsize,
            bias=bias,
            linear_map=linear_map,
            nonlin=nonlin,
            hsizes=hsizes,
            linargs=linargs,

        )
        assert (
                len(set(hsizes)) == 1
        ), "All hidden sizes should be equal for residual network"

        sizes = hsizes + [outsize]
        self.linear = nn.ModuleList(
            [
                linear_map(insize, sizes[k + 1], bias=bias, **linargs)
                for k in range(self.nhidden)
            ]
        )
        self.poslinear = nn.ModuleList(
            [
                slim.NonNegativeLinear(sizes[k], sizes[k + 1], bias=False, **linargs)
                for k in range(self.nhidden)
            ]
        )
        self.nonlin = nn.ModuleList([nonlin() for k in range(self.nhidden + 1)])

        self.inmap = linear_map(insize, hsizes[0], bias=bias, **linargs)
        self.in_features, self.out_features = insize, outsize

    def block_eval(self, x):
        xi = x
        px = self.inmap(xi)
        x = self.nonlin[0](px)
        for layer, (linU, nlin, linW) in enumerate(zip(self.poslinear, self.nonlin[1:], self.linear)):
            px = linW(xi)
            ux = linU(x)
            x = nlin(ux + px)
        return x


class PosDef(Block):
    """
    Enforce positive-definiteness of lyapunov function ICNN, V = g(x)
    Equation 12 from https://arxiv.org/abs/2001.06116
    """
    def __init__(self, g, max=None, eps=0.01, d=1.0, *args):
        """

        :param g: (nn.Module) An ICNN network
        :param eps: (float) quadratic form regularization weight
        :param d: (float) d parameter for ReLU with a quadratic region in [0,d]
        :param max: (float) max value of the output function
        """
        super().__init__()
        self.g = g
        self.in_features = self.g.in_features
        self.out_features = self.g.out_features
        self.zero = torch.nn.Parameter(torch.zeros(1, self.g.in_features), requires_grad=False)
        self.eps = eps
        self.d = d
        self.smReLU = SmoothedReLU(self.d)
        self.max = max

    def block_eval(self, x):
        shift_to_zero = self.smReLU(self.g(x) - self.g(self.zero))
        quad_psd = self.eps*(x**2).sum(1, keepdim=True)
        z = shift_to_zero + quad_psd
        if self.max is not None:
            z = z - torch.relu(z - self.max)
        return z


class PytorchRNN(Block):

    """
    This wraps the torch.nn.RNN class consistent with the blocks interface
    to give output which is a linear map from final hidden state.
    """

    def __init__(
        self,
        insize,
        outsize,
        bias=True,
        linear_map=slim.Linear,
        nonlin=SoftExponential,
        hsizes=[10],
        linargs=dict(),
    ):
        """

        :param insize: (int) dimensionality of input
        :param outsize: (int) dimensionality of output
        :param bias: (bool) Whether to use bias
        :param linear_map: (class) Linear map class from neuromancer.slim.linear
        :param nonlin: (callable) Elementwise nonlinearity which takes as input torch.Tensor and outputs torch.Tensor of same shape
        :param hsizes: (list of ints) List of hidden layer sizes
        :param linargs: (dict) Arguments for instantiating linear layer
        """
        super().__init__()
        assert len(set(hsizes)) == 1
        self.in_features, self.out_features = insize, outsize
        self.rnn = nn.RNN(insize, hsizes[0], bias=bias, nonlinearity="relu")
        self.output = linear_map(hsizes[-1], outsize, bias=bias, **linargs)

    def reg_error(self):
        return self.output.reg_error()

    def block_eval(self, x):
        """

        :param x: (torch.Tensor, shape=[nsteps, batchsize, dim]) Input sequence is expanded for order 2 tensors
        :return: (torch.Tensor, shape=[batchsize, outsize]) Returns linear transform of final hidden state of RNN.
        """
        if len(x.shape) == 2:
            x = x.reshape(1, *x.shape)
        elif len(x.shape) == 3:
            x = x.permute(1, 0, 2)
        _, hiddens = self.rnn(x)
        return self.output(hiddens[-1])


class RNN(Block):
    """
    This wraps the rnn.RNN class consistent with blocks interface to give output which is a linear map from final hidden state.
    """

    def __init__(
        self,
        insize,
        outsize,
        bias=True,
        linear_map=slim.Linear,
        nonlin=SoftExponential,
        hsizes=[1],
        linargs=dict(),
    ):
        """

        :param insize: (int) dimensionality of input
        :param outsize: (int) dimensionality of output
        :param bias: (bool) Whether to use bias
        :param linear_map: (class) Linear map class from neuromancer.slim.linear
        :param nonlin: (callable) Elementwise nonlinearity which takes as input torch.Tensor and outputs torch.Tensor of same shape
        :param hsizes: (list of ints) List of hidden layer sizes
        :param linargs: (dict) Arguments for instantiating linear layer
        """
        super().__init__()
        assert len(set(hsizes)) == 1
        self.in_features, self.out_features = insize, outsize
        self.rnn = rnn.RNN(
            insize,
            hsizes=hsizes,
            bias=bias,
            nonlin=nonlin,
            linear_map=linear_map,
            linargs=linargs,
        )
        self.output = linear_map(hsizes[-1], outsize, bias=bias, **linargs)
        self.init_states = list(self.rnn.init_states)

    def reg_error(self):
        return self.rnn.reg_error() + self.output.reg_error()

    def reset(self):
        self.init_states = None

    def block_eval(self, x, hx=None):
        """
        There is some logic here so that the RNN will still get context from state in open loop simulation.

        :param x: (torch.Tensor, shape=[nsteps, batchsize, dim]) Input sequence is expanded for order 2 tensors
        :return: (torch.Tensor, shape=[batchsize, outsize]) Returns linear transform of final hidden state of RNN.
        """
        if len(x.shape) == 2:
            x = x.reshape(1, *x.shape)
        elif len(x.shape) == 3:
            x = x.permute(1, 0, 2)
        _, hiddens = self.rnn(x, init_states=hx)
        self.init_states = hiddens
        return self.output(hiddens[-1])


class BilinearTorch(Block):
    """
    Wraps torch.nn.Bilinear to be consistent with the blocks interface
    """
    def __init__(
        self,
        insize,
        outsize,
        bias=True,
        linear_map=slim.Linear,
        nonlin=SoftExponential,
        hsizes=[64],
        linargs=dict(),
    ):
        """

        :param insize: (int) dimensionality of input
        :param outsize: (int) dimensionality of output
        :param bias: (bool) Whether to use bias
        :param linear_map: (class) Not used in this module
        :param nonlin: (callable) Not used in this module
        :param hsizes: (list of ints) Not used in this module
        :param linargs: (dict) Not used in this module
        """
        super().__init__()
        self.in_features, self.out_features = insize, outsize
        self.f = nn.Bilinear(
            self.in_features, self.in_features, self.out_features, bias=bias
        )

    def reg_error(self):
        return torch.tensor(0.0).to(self.f.weight)

    def block_eval(self, x):
        return self.f(x, x)


class Poly2(Block):
    """
    Feature expansion of network to include pairwise multiplications of features.
    """
    def __init__(self, *args):
        super().__init__()

    def block_eval(self, x):
        """

        :param x: (torch.Tensor, shape=[batchsize, N]) Input tensor
        :return: (torch.Tensor, shape=[batchsize, :math:`\frac{N(N+1)}{2} + N`]) Feature expanded tensor
        """
        row_idxs, col_idxs = np.triu_indices(x.shape[-1])
        expansion = torch.matmul(x.unsqueeze(-1), x.unsqueeze(1))  # outer product
        expansion = expansion[:, row_idxs, col_idxs]  # upper triangular
        return torch.cat([x, expansion], dim=-1)  # concatenate


class BasisLinear(Block):
    """
    For mapping inputs to functional basis feature expansion. This could implement a dictionary of lifting functions.
    Takes a linear combination of the expanded features.

    """
    def __init__(
        self,
        insize,
        outsize,
        bias=True,
        linear_map=slim.Linear,
        nonlin=SoftExponential,
        hsizes=[64],
        linargs=dict(),
        expand=Poly2(),
    ):
        """

        :param insize: (int) dimensionality of input
        :param outsize: (int) dimensionality of output
        :param bias: (bool) Whether to use bias
        :param linear_map: (class) Not used in this module
        :param nonlin: (callable) Not used in this module
        :param hsizes: (list of ints) Not used in this module
        :param linargs: (dict) Not used in this module
        """
        super().__init__()
        self.in_features, self.out_features = insize, outsize
        self.expand = expand
        inlin = self.expand(torch.zeros(1, insize)).shape[-1]
        self.linear = linear_map(inlin, outsize, bias=bias, **linargs)
        self.bias = nn.Parameter(torch.zeros(1, outsize), requires_grad=not bias)

    def reg_error(self):
        return self.linear.reg_error()

    def block_eval(self, x):
        """

        :param x: (torch.Tensor, shape=[batchsize, insize])
        :return: (torch.Tensor, shape=[batchsize, outsize])
        """
        return self.linear(self.expand(x))


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


blocks = {
    "mlp": MLP,
    "mlp_dropout": MLPDropout,
    "mlp_bounds": MLP_bounds,
    "rnn": RNN,
    "pytorch_rnn": PytorchRNN,
    "linear": Linear,
    "residual_mlp": ResMLP,
    "basislinear": BasisLinear,
    "poly2": Poly2,
    "bilinear": BilinearTorch,
    "icnn": InputConvexNN,
    "pos_def": PosDef
}
