import os
import logging
from typing import Sequence

import torch
import torch.nn as nn
import torch.optim as optim
import torchsde
from torch.distributions import Normal
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from neuromancer.modules import blocks
from neuromancer.dynamics import integrators, ode
from neuromancer.trainer import Trainer, LitTrainer
from neuromancer.problem import Problem
from neuromancer.loggers import BasicLogger
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.system import Node

from neuromancer.psl import plot

# Import specific modules from neuromancer
from neuromancer.modules.blocks import BasicSDE, LatentSDE_Encoder
from neuromancer.dynamics.integrators import BasicSDEIntegrator

import pytest
torch.manual_seed(0)



class TestBasicSDEBlockAndIntegrator:
    """ Testing class for BasicSDE and BasicSDEIntegrator """

    @pytest.fixture(params=[
        lambda t, y: torch.sin(t) + 0.1 * y,
        lambda t, y: 1.0 * t**2 + 0.1 * y**2 #quadratic drift
    ])
    def f_function(self, request):
        return request.param
    
    @pytest.fixture(params=[
        lambda t, y: 0.3 * torch.sigmoid(torch.cos(t) * torch.exp(-y)), 
        lambda t, y: torch.exp(-1. * t) * torch.sqrt(t) + torch.exp(-1. * y) #exponential diffusion
    ])
    def g_function(self, request):
        return request.param

    @pytest.fixture(params=[1])  # state size = 1
    def state_size(self, request):
        return request.param
    
    @pytest.fixture(params=[1,100])  # Different time sizes
    def time_size(self, request):
        return request.param
    
    @pytest.fixture(params=[1,5])  # Different batch sizes
    def batch_size(self, request):
        return request.param

    @pytest.fixture
    def basic_sde(self, f_function, g_function):
        t = variable('t')
        y = variable('y')
        return BasicSDE(f_function, g_function, t, y)

    
    def test_g_output_shape(self, basic_sde, batch_size, state_size, time_size):
        ts = torch.linspace(0, 1, time_size)
        y0 = torch.full(size=(batch_size, state_size), fill_value=0.1)
        output = basic_sde.g(ts, y0)
       
        assert output.shape[0] == y0.shape[0], "Dimension 0 of g output not equal to state size"
        assert output.shape[1] == time_size, "Dimension 1 of g output not equal to time size"
    
    def test_basic_sde_initialization(self, basic_sde): 
        assert hasattr(basic_sde, 'noise_type'), "BasicSDE does not have a noise_type attribute"
        assert basic_sde.noise_type == "diagonal", "noise_type attribute does not equal 'diagonal'"
        assert hasattr(basic_sde, 'sde_type'), "BasicSDE does not have a noise_type attribute"
        assert basic_sde.sde_type == "ito", "sde_type attribute does not equal 'ito'"
        assert basic_sde.in_features == 0
        assert basic_sde.out_features == 0


    def test_f_output_shape(self, basic_sde, batch_size, state_size, time_size):
        ts = torch.linspace(0, 1, time_size)
        y0 = torch.full(size=(batch_size, state_size), fill_value=0.1)
        output = basic_sde.f(ts, y0)
        assert output.shape[0] == y0.shape[0], "Dimension 0 of f output not equal to state size"
        assert output.shape[1] == time_size, "Dimension 1 of f output not equal to time size"

    
    def test_integrate_shape(self, basic_sde,batch_size, state_size, time_size): 
        integrator = BasicSDEIntegrator(basic_sde)
        model = Node(integrator, input_keys=['y','t'], output_keys=['ys'])
        batch_size, state_size, t_size = 5, 1, 100

        ts = torch.linspace(0, 1, time_size)
        y0 = torch.full(size=(batch_size, state_size), fill_value=0.1)
        my_data = {'y': y0, 't': ts}
        output = model(my_data)['ys']
        assert output.shape == torch.Size([time_size, batch_size, state_size])


class StochasticLorenz(object):
    """Stochastic Lorenz attractor.

    Used for simulating ground truth and obtaining noisy data.
    Details described in Section 7.2 https://arxiv.org/pdf/2001.01328.pdf
    Default a, b from https://openreview.net/pdf?id=HkzRQhR9YX
    """
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, a: Sequence = (10., 28., 8 / 3), b: Sequence = (.1, .28, .3)):
        super(StochasticLorenz, self).__init__()
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


class TestLatentSDEBlockAndIntegrator:
    """ Testing class for LatentSDE_Encoder and LatentSDEIntegrator """

    def setup_method(self): 
        torch.manual_seed(0)
        batch_size=1
        self.latent_size=4
        self.context_size=16
        self.hidden_size=16
        t0=0.
        t1=2.
        self.noise_std=0.01
        _y0 = torch.randn(batch_size, 3)
        self.ts = torch.linspace(t0, t1, steps=5)
        xs = StochasticLorenz().sample(_y0, self.ts, self.noise_std, normalize=True)
        train_data = DictDataset({'xs':xs},name='train')
        self.train_data_loader = DataLoader(train_data, batch_size=1024, collate_fn=train_data.collate_fn, shuffle=False)
    
    def test_latent_sde_initialization(self): 
        sde_block_encoder = blocks.LatentSDE_Encoder(3, self.latent_size, self.context_size, self.hidden_size, ts=self.ts, adjoint=False) 
        assert torch.allclose(sde_block_encoder.ts, self.ts)
        assert sde_block_encoder.adjoint == False 
    

    def test_latent_sde_forward(self): 
        sde_block_encoder = blocks.LatentSDE_Encoder(3, self.latent_size, self.context_size, self.hidden_size, ts=self.ts, adjoint=False) 
        integrator = integrators.LatentSDEIntegrator(sde_block_encoder, adjoint=False)
        model_1 = Node(integrator, input_keys=['xs'], output_keys=['zs', 'z0', 'log_ratio',  'xs', 'qz0_mean', 'qz0_logstd'], name='m1')

        sample = next(iter(self.train_data_loader))
        output_1 = model_1(sample)
        assert isinstance(output_1, dict), "Output of LatentSDE_Encoder should be a dictionary"
        assert sorted(list(output_1.keys())) == ['log_ratio', 'qz0_logstd', 'qz0_mean', 'xs', 'z0', 'zs'], "Keys of output of LatentSDE_Encoder are incorrect"
        assert output_1['z0'].shape == torch.Size([1,4])
        assert output_1['zs'].shape == torch.Size([5,1,4])
        

