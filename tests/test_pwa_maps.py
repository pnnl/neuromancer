"""
Test PWA map representations of feedforward neural nets from pwa_maps.py

"""

from torch import nn
from neuromancer.modules import blocks
from neuromancer.analysis.pwa_maps import pwa_batched
import torch
from hypothesis import given, settings, strategies as st

activations = [nn.LeakyReLU, nn.ReLU, nn.PReLU, nn.GELU, nn.CELU, nn.ELU,
                nn.LogSigmoid, nn.Sigmoid, nn.Tanh]


@given(st.integers(1, 5),
       st.lists(st.integers(1, 20), min_size=1, max_size=4),
       st.sampled_from(activations))
@settings(max_examples=1000, deadline=None)
def test_pwa_maps_sample_bias(nx, hsizes, nonlin):
    # random feature point
    x_z = torch.randn(1, nx)
    # define square neural net
    fx = blocks.MLP(nx, nx, nonlin=nonlin, hsizes=hsizes, bias=True)
    Astar, bstar, *_ = pwa_batched(fx, x_z)

    mlp_out = fx(x_z)
    print(f'MLP: {mlp_out}')
    # pwa_out = torch.matmul(x_z, Astar) + bstar
    pwa_out = (torch.matmul(x_z, Astar)).squeeze(1) + bstar
    print(f'pwa_batched: {pwa_out}')

    difference = torch.norm(mlp_out - pwa_out, p=2)
    print(difference < 1e-6)
    assert difference < 1e-6


@given(st.integers(1, 5),
       st.lists(st.integers(1, 20), min_size=1, max_size=4),
       st.sampled_from(activations))
@settings(max_examples=1000, deadline=None)
def test_pwa_maps_sample_nobias(nx, hsizes, nonlin):
    # random feature point
    x_z = torch.randn(1, nx)
    # define square neural net
    fx = blocks.MLP(nx, nx, nonlin=nonlin, hsizes=hsizes, bias=False)
    Astar, bstar, *_ = pwa_batched(fx, x_z)

    mlp_out = fx(x_z)
    print(f'MLP: {mlp_out}')
    # pwa_out = torch.matmul(x_z, Astar) + bstar
    pwa_out = (torch.matmul(x_z, Astar)).squeeze(1) + bstar
    print(f'pwa_batched: {pwa_out}')

    difference = torch.norm(mlp_out - pwa_out, p=2)
    print(difference < 1e-6)
    assert difference < 1e-6


@given(st.integers(1, 5),
       st.lists(st.integers(1, 20), min_size=1, max_size=4),
       st.sampled_from(activations))
@settings(max_examples=1000, deadline=None)
def test_pwa_maps_sample_nobias_batched(nx, hsizes, nonlin):
    # random feature point
    x_z = torch.randn(5, nx)
    # define square neural net
    fx = blocks.MLP(nx, nx, nonlin=nonlin, hsizes=hsizes, bias=False)
    Astar, bstar, *_ = pwa_batched(fx, x_z)

    mlp_out = fx(x_z)
    print(f'MLP: {mlp_out}')
    # pwa_out = torch.matmul(x_z, Astar) + bstar
    pwa_out = torch.bmm(Astar.transpose(1, 2), x_z.unsqueeze(2)).squeeze(2) + bstar
    print(f'pwa_batched: {pwa_out}')

    difference = torch.norm(mlp_out - pwa_out, p=2)
    print(difference < 1e-6)
    assert difference < 1e-6


@given(st.integers(1, 5),
       st.lists(st.integers(1, 20), min_size=1, max_size=4),
       st.sampled_from(activations))
@settings(max_examples=1000, deadline=None)
def test_pwa_maps_sample_bias_batched(nx, hsizes, nonlin):
    # random feature point
    x_z = torch.randn(5, nx)
    # define square neural net
    fx = blocks.MLP(nx, nx, nonlin=nonlin, hsizes=hsizes, bias=True)
    Astar, bstar, *_ = pwa_batched(fx, x_z)

    mlp_out = fx(x_z)
    print(f'MLP: {mlp_out}')
    # pwa_out = torch.matmul(x_z, Astar) + bstar
    pwa_out = torch.bmm(Astar.transpose(1, 2), x_z.unsqueeze(2)).squeeze(2) + bstar
    print(f'pwa_batched: {pwa_out}')

    difference = torch.norm(mlp_out - pwa_out, p=2)
    print(difference < 1e-6)
    assert difference < 1e-6
