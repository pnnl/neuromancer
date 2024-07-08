"""
Function approximators of various degrees of generality which implement a consistent block interface.
Neural network module building blocks for neural state space models, state estimators and control policies.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        use_dropout = (self.training and self.at_train) or (
            not self.training and self.at_test
        )
        return torch.nn.functional.dropout(x, p=self.p, training=use_dropout)


def set_model_dropout_mode(model, at_train=None, at_test=None):
    """Change dropout mode, useful for enabling MC sampling during inference time."""

    def _apply_fn(x):
        if isinstance(x, Dropout):
            if at_test is not None:
                x.at_test = at_test
            if at_train is not None:
                x.at_train = at_train

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
        self.nonlin = nn.ModuleList(
            [nonlin() for k in range(self.nhidden)] + [nn.Identity()]
        )
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


class KANLinear(torch.nn.Module):
    """
    KANLinear module based on the efficient implementation of Kolmogorov-Arnold Network.
    * Reference: https://github.com/Blealtan/efficient-kan.
    """

    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler, a=np.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Approximate, memory-efficient implementation of the regularization loss.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    """
    KAN module based on the efficient implementation of Kolmogorov-Arnold Network.
    *Reference: https://github.com/Blealtan/efficient-kan.
    """

    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


class KANBlock(Block):
    def __init__(
        self,
        insize,
        outsize,
        num_layers=1,
        hidden_size=None,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super().__init__()
        self.in_features = insize
        self.out_features = outsize
        self.kan_layers = nn.ModuleList()

        if hidden_size is None:
            hidden_size = outsize

        layer_sizes = [insize] + [hidden_size] * (num_layers - 1) + [outsize]
        for in_features, out_features in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.kan_layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    enable_standalone_scale_spline=enable_standalone_scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def block_eval(self, x):
        for layer in self.kan_layers:
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.kan_layers
        )

    def update_grid(self, x: torch.Tensor, margin=0.01):
        for layer in self.kan_layers:
            layer.update_grid(x, margin=margin)


class MLP_bounds(MLP):
    """
    Multi-Layer Perceptron consistent with blocks interface
    """

    bound_methods = {"sigmoid_scale": sigmoid_scale, "relu_clamp": relu_clamp}

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
        method="sigmoid_scale",
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

        super().__init__(
            insize=insize,
            outsize=outsize,
            bias=bias,
            linear_map=linear_map,
            nonlin=nonlin,
            hsizes=hsizes,
            linargs=linargs,
        )

        self.min = min
        self.max = max
        self.method = self._set_method(method)

    def _set_method(self, method):
        if method in self.bound_methods.keys():
            return self.bound_methods[method]
        else:
            assert callable(method), (
                f"Method, {method} must be a key in {self.bound_methods} "
                f"or a differentiable callable."
            )
            return method

    def block_eval(self, x):
        """

        :param x: (torch.Tensor, shape=[batchsize, insize])
        :return: (torch.Tensor, shape=[batchsize, outsize])
        """
        for lin, nlin in zip(self.linear, self.nonlin):
            x = nlin(lin(x))
        return self.method(x, self.min, self.max)
        

class StackedMLP(Block):
    """
    Stacked Multi-Layer Perceptron (MFMLP) designed for multi-fidelity learning where multiple layers are
    stacked to refine the prediction progressively. Each layer is a blend of linear and nonlinear transformations
    controlled by an adaptive parameter alpha, influencing the trade-off between the two.

    Attributes:
        insize (int): Input feature dimension.
        outsize (int): Output feature dimension.
        bias (bool): If True, bias is used in linear transformations.
        linear_map (class): Linear map class used for layers, by default set to slim.Linear.
        nonlin (callable): Nonlinear activation function applied after linear transformations.
        h_sf_size (list of int): Sizes of hidden layers in the single-fidelity MLP.
        n_stacked_mf_layers (int): Number of stacked multi-fidelity layers.
        h_linear_sizes (list of int): Sizes of hidden layers in each linear sub-network within the multi-fidelity layers.
        h_nonlinear_sizes (list of int): Sizes of hidden layers in each nonlinear sub-network within the multi-fidelity layers.
        linargs (dict): Additional arguments for the linear layer instantiation.
        alpha_init (float): Initial value of alpha parameter controlling linear-nonlinear blend.
        verbose (bool): If True, print messages about network progress and actions.
    """

    def __init__(
        self,
        insize,
        outsize,
        bias=True,
        linear_map=Linear,
        nonlin=nn.Tanh,
        h_sf_size=[20, 20],
        n_stacked_mf_layers=3,
        h_linear_sizes=[10, 10],
        h_nonlinear_sizes=[20, 20],
        linargs=dict(), 
        alpha_init=0.1,
        verbose=False
    ):
        super().__init__()
        self.in_features, self.out_features = insize, outsize
        self.num_layers = n_stacked_mf_layers
        self.current_block = 0
        self.current_epoch = 0
        self.alpha = nn.ParameterList([nn.Parameter(torch.tensor(alpha_init), requires_grad=True) for _ in range(n_stacked_mf_layers)])
        self.alpha_loss = 0.0
        self.verbose = verbose
        
        # Initialize the first layer (single-fidelity MLP)
        self.first_layer = MLP(
            insize, outsize, bias=bias, linear_map=linear_map, nonlin=nonlin, hsizes=h_sf_size, linargs=linargs
        )
    
        # Initialize subsequent layers (multi-fidelity)
        self.layers = nn.ModuleList()
        for i in range(n_stacked_mf_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "linear": MLP(outsize, outsize, bias=True, linear_map=linear_map, nonlin=nn.Identity, hsizes=h_linear_sizes, linargs=linargs),
                        "nonlinear": MLP(
                            insize + outsize,
                            outsize,
                            bias=bias,
                            linear_map=linear_map,
                            nonlin=nonlin,
                            hsizes=h_nonlinear_sizes,
                            linargs=linargs,
                        ),
                    }
                )
            )

    def block_eval(self, x):
        """
        Process input through the multi-fidelity network blocks up to the current block, combining the outputs
        of linear and nonlinear transformations weighted by alpha.

        :param x: Input tensor.
        :return: Output tensor from the last activated block.
        """
        out = self.first_layer(x)
        alpha_loss = 0.0
        # for i in range(self.current_block):
        for i in range(self.num_layers):
            layer = self.layers[i] # Pick the corresponding stacked net
            alpha = self.alpha[i]  # Pick the corresponding alpha for each stacked net
            linear_out = layer["linear"](out)
            nonlinear_out = layer["nonlinear"](torch.cat([x, out], dim=1))
            out = torch.abs(alpha) * nonlinear_out + (1 - torch.abs(alpha)) * linear_out
            alpha_loss += torch.pow(alpha, 4)
        self.alpha_loss = alpha_loss
        return out

    def get_alpha_loss(self):
        """
        Retrieve the accumulated loss from alpha parameters used for regularization purposes.

        :return: Alpha loss as a torch scalar.
        """
        return self.alpha_loss
        

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
        self.nonlin = nn.ModuleList(
            [nonlin() for k in range(self.nhidden)] + [nn.Identity()]
        )
        self.embeddings = [
            nn.Embedding(int(n_interactors**2), n_embed) for n_embed in em_sizes
        ]
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
            x = torch.cat([x, embedder(self.n_interactors * i + j)])
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
        dropout=0.0,
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
        self.nonlin = nn.ModuleList(
            [nonlin() for k in range(self.nhidden)] + [nn.Identity()]
        )
        self.linear = nn.ModuleList(
            [
                linear_map(sizes[k], sizes[k + 1], bias=bias, **linargs)
                for k in range(self.nhidden + 1)
            ]
        )
        self.dropout = nn.ModuleList(
            [
                Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
                for _ in range(self.nhidden)
            ]
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

    def __init__(
        self,
        insize,
        outsize,
        bias=True,
        linear_map=slim.Linear,
        nonlin=nn.ReLU,
        hsizes=[64],
        linargs=dict(),
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
        for layer, (linU, nlin, linW) in enumerate(
            zip(self.poslinear, self.nonlin[1:], self.linear)
        ):
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
        self.zero = torch.nn.Parameter(
            torch.zeros(1, self.g.in_features), requires_grad=False
        )
        self.eps = eps
        self.d = d
        self.smReLU = SmoothedReLU(self.d)
        self.max = max

    def block_eval(self, x):
        shift_to_zero = self.smReLU(self.g(x) - self.g(self.zero))
        quad_psd = self.eps * (x**2).sum(1, keepdim=True)
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
    "pos_def": PosDef,
    "kan": KANBlock,
    "stacked_mlp": StackedMLP
}

