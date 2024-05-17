

import abc
import torch
from torch import nn
import torchsde
import abc

from torch.distributions import Normal
from typing import Sequence


class BaseSDESystem(abc.ABC, nn.Module):
    """
    Base class for SDEs for integration with TorchSDE library
    """
    def __init__(self):
        super().__init__()
        self.noise_type = "diagonal" #only supports diagonal diffusion right now
        self.sde_type = "ito" #only supports Ito integrals right now
        self.in_features = 0 #for compatibility with Neuromancer integrators; unused
        self.out_features = 0

    @abc.abstractmethod
    def f(self, t, y):
        """
        Define the ordinary differential equations (ODEs) for the system.

        Args:
            t (Tensor): The current time (often unused)
            y (Tensor): The current state variables of the system.

        Returns:
            Tensor: The derivatives of the state variables with respect to time.
                    The output should be of shape [batch size x state size]
        """
        pass

    @abc.abstractmethod
    def g(self, t,y):
        """
        Define the diffusion equations for the system.

        Args:
            t (Tensor): The current time (often unused)
            y (Tensor): The current state variables of the system.

        Returns:
            Tensor: The diffusion coefficients per batch item (output is of size 
                    [batch size x state size]) for noise_type 'diagonal'
        """
        pass

class Encoder(nn.Module):
    """
    Encoder module to handle time-series data (as in the case of stochastic data and SDE)
    GRU is used to handle mapping to latent space in this case
    This class is used only in LatentSDE_Encoder
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        out, _ = self.gru(inp)
        out = self.lin(out)
        return out
    
class LatentSDE_Encoder(BaseSDESystem):
    def __init__(self, data_size, latent_size, context_size, hidden_size, ts, adjoint=False):
        """
        LatentSDE_Encoder is a neural network module designed for encoding time-series data into a latent space representation,
        which is then used to model the system dynamics using Stochastic Differential Equations (SDEs).

        The primary purpose of this class is to transform high-dimensional time-series data into a lower-dimensional latent space
        while capturing the underlying stochastic dynamics. This transformation facilitates efficient modeling, prediction, and
        inference of complex temporal processes. 

        Taken from https://github.com/google-research/torchsde/blob/master/examples/latent_sde_lorenz.py and modified to support 
        NeuroMANCER library

        :param data_size: (int) state size of the data 
        :param latent_size: (int) input latent size for the encoder 
        :param context_size: (int) size of context vector (output of encoder)
        :param hidden_size: (int) size of the hidden layer of encoder 
        :param ts: (tensor) tensor of timesteps over which data should be predicted

        """
        super().__init__()

        self.adjoint = adjoint

        # Encoder.
        self.encoder = Encoder(input_size=data_size, hidden_size=hidden_size, output_size=context_size)
        self.qz0_net = nn.Linear(context_size, latent_size + latent_size) #Layer to return mean and variance of the parameterized latent space

        # Decoder.
        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        self.h_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        # This needs to be an element-wise function for the SDE to satisfy diagonal noise.
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.Softplus(),
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid()
                )
                for _ in range(latent_size)
            ]
        )
        self.projector = nn.Linear(latent_size, data_size)

        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))

        self._ctx = None
        self.ts = ts

    def contextualize(self, ctx):
        self._ctx = ctx  # A tuple of tensors of sizes (T,), (T, batch_size, d).

    def f(self, t, y):
        ts, ctx = self._ctx

        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)

        return self.f_net(torch.cat((y, ctx[i]), dim=1))

    def h(self, t, y):
        return self.h_net(y)

    def g(self, t, y):  # Diagonal diffusion.
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    def forward(self, xs):
        # Contextualization is only needed for posterior inference.
        ctx = self.encoder(torch.flip(xs, dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((self.ts, ctx))

        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)
        if not self.adjoint:
            return z0, xs, self.ts, qz0_mean, qz0_logstd
        else:
            adjoint_params = (
                    (ctx,) +
                    tuple(self.f_net.parameters()) + tuple(self.g_nets.parameters()) + tuple(self.h_net.parameters())
            )
            return z0, xs, self.ts, qz0_mean, qz0_logstd, adjoint_params

class LatentSDE_Decoder(BaseSDESystem):
    """
    Second part of Wrapper for torchsde's Latent SDE class to integrate with Neuromancer. This takes in output of
    LatentSDEIntegrator and decodes it back into the "real" data space and also outputs associated Gaussian distributions
    to be used in the final loss function.
    Please see https://github.com/google-research/torchsde/blob/master/examples/latent_sde_lorenz.py

    :param data_size: (int) state size of the data 
    :param latent_size: (int) input latent size for the encoder 
    :param noise_std: (float) standard deviation of the Gaussian noise applied during decoding
    """
    def __init__(self, data_size, latent_size, noise_std):
        super().__init__()
        self.noise_std = noise_std
        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))
        self.projector = nn.Linear(latent_size, data_size)
    
    def f(self, t, y): 
        pass #unused 
    
    def g(self, t, y): 
        pass #unused

    def forward(self, xs, zs, log_ratio, qz0_mean, qz0_logstd):
        _xs = self.projector(zs)
        xs_dist = Normal(loc=_xs, scale=self.noise_std)
        log_pxs = xs_dist.log_prob(xs).sum(dim=(0, 2)).mean(dim=0)

        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        logqp_path = log_ratio.sum(dim=0).mean(dim=0)
        return _xs, log_pxs, logqp0 + logqp_path, log_ratio
    
"""
---------------------------------- Data Generation Classes, for forward pass only -------------------------------------------
"""
class StochasticLorenzAttractor(BaseSDESystem):
    def __init__(self, a: Sequence = (10., 28., 8 / 3), b: Sequence = (.1, .28, .3)):
        super().__init__()
        self.a = a
        self.b = b

    def f(self, t, y):
        x1, x2, x3 = torch.split(y, split_size_or_sections=(1, 1, 1), dim=1)
        a1, a2, a3 = self.a

        f1 = a1 * (x2 - x1)
        f2 = a2 * x1 - x2 - x1 * x3
        f3 = x1 * x2 - a3 * x3
        return torch.cat([f1, f2, f3], dim=1)

    def g(self, t, y):
        x1, x2, x3 = torch.split(y, split_size_or_sections=(1, 1, 1), dim=1)
        b1, b2, b3 = self.b

        g1 = x1 * b1
        g2 = x2 * b2
        g3 = x3 * b3
        return torch.cat([g1, g2, g3], dim=1)

    @torch.no_grad()
    def sample(self, x0, ts, noise_std, normalize):
        """Sample data for training. Store data normalization constants if necessary."""
        xs = torchsde.sdeint(self, x0, ts)
        if normalize:
            mean, std = torch.mean(xs, dim=(0, 1)), torch.std(xs, dim=(0, 1))
            xs.sub_(mean).div_(std).add_(torch.randn_like(xs) * noise_std)
        return xs
    

class SDECoxIngersollRand(BaseSDESystem):
    def __init__(self, alpha: float=0.1,
                        beta: float=0.05,
                        sigma: float=0.02):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma

    def f(self, t, y):
        r = y
        return self.alpha * (self.beta - r)

    def g(self, t, y):
        r = y
        return self.sigma * torch.sqrt(torch.abs(r))


class SDEOrnsteinUhlenbeck(BaseSDESystem):
    def __init__(self, theta: float = 0.1, sigma: float = 0.2):
        super(BaseSDESystem).__init__()
        self.theta = theta
        self.sigma = sigma

    def f(self, t, y):
        return -self.theta * y

    def g(self, t, y):
        return self.sigma


class LotkaVolterraSDE(BaseSDESystem):
    def __init__(self, a, b, c, d, g_params):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.g_params = g_params

    def f(self, t, x):
        x1 = x[:,[0]]
        x2 = x[:,[1]]
        dx1 = self.a * x1 - self.b * x1*x2
        dx2 = self.c * x1*x2 - self.d * x2
        return torch.cat([dx1, dx2], dim=-1)

    def g(self, t, x):
        return self.g_params