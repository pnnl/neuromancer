import numpy as np
import scipy as sp
import torch
from torch import nn
import matplotlib.pyplot as plt
import slim
from neuromancer import blocks
from neuromancer.pwa_maps import lpv, lpv_batched
import torch
from hypothesis import given, settings, strategies as st
import math
from slim.linear import square_maps, maps, TrivialNullSpaceLinear
from neuromancer.activations import activations

# activations = [v for k, v in activations.items()]
activations = [nn.ReLU6, nn.ReLU, nn.PReLU, nn.GELU, nn.CELU, nn.ELU, nn.Tanh]


@given(st.integers(1, 5),
       st.lists(st.integers(1, 20), min_size=1, max_size=4),
       st.sampled_from(activations))
@settings(max_examples=1000, deadline=None)
def test_pwa_maps_sample_bias(nx, hsizes, nonlin):
    # random feature point
    x_z = torch.randn(1, nx)
    # define square neural net
    fx = blocks.MLP(nx, nx, nonlin=nonlin, hsizes=hsizes, bias=True)
    Astar, Astar_b, bstar, *_ = lpv(fx, x_z)

    mlp_out = fx(x_z)
    print(mlp_out)
    pwa_out = torch.matmul(x_z, Astar_b)
    print(pwa_out)
    assert torch.equal(mlp_out, pwa_out)


@given(st.integers(1, 5),
       st.lists(st.integers(1, 20), min_size=1, max_size=4),
       st.sampled_from(activations))
@settings(max_examples=1000, deadline=None)
def test_pwa_maps_sample_nobias(nx, hsizes, nonlin):
    # random feature point
    x_z = torch.randn(1, nx)
    # define square neural net
    fx = blocks.MLP(nx, nx, nonlin=nonlin, hsizes=hsizes, bias=False)
    Astar, Astar_b, bstar, *_ = lpv(fx, x_z)

    mlp_out = fx(x_z)
    print(mlp_out)
    pwa_out = torch.matmul(x_z, Astar)
    print(pwa_out)
    assert torch.equal(mlp_out, pwa_out)


@given(st.integers(1, 10),
       st.integers(1, 5),
       st.lists(st.integers(1, 20), min_size=1, max_size=4),
       st.sampled_from(activations))
@settings(max_examples=1000, deadline=None)
def test_pwa_maps_batched_bias(samples, nx, hsizes, nonlin):
    # random feature point
    x_z = torch.randn(samples, nx)
    # define square neural net
    fx = blocks.MLP(nx, nx, nonlin=nonlin, hsizes=hsizes, bias=True)

    Astar, Astar_b, bstar, *_ = lpv_batched(fx, x_z)
    mlp_out = fx(x_z)
    print(mlp_out)
    pwa_out = torch.matmul(x_z, Astar_b)
    print(pwa_out)
    assert torch.equal(mlp_out, pwa_out)


@given(st.integers(1, 10),
       st.integers(1, 5),
       st.lists(st.integers(1, 20), min_size=1, max_size=4),
       st.sampled_from(activations))
@settings(max_examples=1000, deadline=None)
def test_pwa_maps_batched_bias(samples, nx, hsizes, nonlin):
    # random feature point
    x_z = torch.randn(samples, nx)
    # define square neural net
    fx = blocks.MLP(nx, nx, nonlin=nonlin, hsizes=hsizes, bias=False)

    Astar, Astar_b, bstar, *_ = lpv_batched(fx, x_z)
    mlp_out = fx(x_z)
    print(mlp_out)
    pwa_out = torch.matmul(x_z, Astar)
    print(pwa_out)
    assert torch.equal(mlp_out, pwa_out)

