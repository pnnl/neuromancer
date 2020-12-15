import numpy as np
import scipy as sp
import torch
from torch import nn
import matplotlib.pyplot as plt

import slim
from neuromancer import blocks

def lpv_batched(fx, x):
    x_layer = x

    Aprime_mats = []
    activation_mats = []
    bprimes = []

    for nlin, lin in zip(fx.nonlin, fx.linear):
        A = lin.effective_W()  # layer weight

        b = lin.bias if lin.bias is not None else torch.zeros(A.shape[-1])
        Ax = torch.matmul(x_layer, A) + b  # affine transform

        zeros = Ax == 0
        lambda_h = nlin(Ax) / Ax  # activation scaling
        lambda_h[zeros] = 0.  # TODO: use activation gradients instead of zeros

        lambda_h_mats = [torch.diag(v) for v in lambda_h]
        activation_mats += lambda_h_mats
        lambda_h_mats = torch.stack(lambda_h_mats)

        x_layer = Ax * lambda_h

        Aprime = torch.matmul(A, lambda_h_mats)
        # Aprime = A * lambda_h
        Aprime_mats += [Aprime]

        bprime = lambda_h * b
        bprimes += [bprime]

    # network-wise parameter varying linear map:  A* = A'_L ... A'_1
    Astar = Aprime_mats[0]
    bstar = bprimes[0] # b x nx
    for Aprime, bprime in zip(Aprime_mats[1:], bprimes[1:]):
        Astar = torch.bmm(Astar, Aprime)
        bstar = torch.bmm(bstar.unsqueeze(-2), Aprime).squeeze(-2) + bprime

    return Astar, bstar, Aprime_mats, bprimes, activation_mats


def lpv(fx, x):
    """pared-down version of LPV_net"""

    x_layer = x
    x_layer_b = x
    x_layer_orig = x
    Aprime_mats = []
    Aprime_b_mats = []
    activation_mats = []
    activation_bias_mats = []
    bprimes = []

    for i, (nlin, lin) in enumerate(zip(fx.nonlin, fx.linear)):
        x_layer_orig = nlin(lin(x_layer_orig))

        A = lin.effective_W()  # layer weight
        Ax = torch.matmul(x_layer, A)  # linear transform

        # TODO: if *any* are zero, this will break
        # need to compute derivatives of activations where Ax == 0
        if sum(Ax.squeeze()) == 0:
            lambda_h = torch.zeros(Ax.shape)
        else:
            lambda_h = nlin(Ax) / Ax  # activation scaling

        lambda_h_matrix = torch.diag(lambda_h.squeeze())
        activation_mats += [lambda_h_matrix]

        x_layer = torch.matmul(Ax, lambda_h_matrix)

        # compute layer-wise parameter-varying linear map
        Aprime = torch.matmul(A, lambda_h_matrix)
        # x_layer_Aprime = torch.matmul(x_layer_Aprime, Aprime)

        Aprime_mats += [Aprime]

        b = lin.bias if lin.bias is not None else torch.zeros(x_layer_b.shape)
        Ax_b = torch.matmul(x_layer_b, A) + b  # affine transform

        if sum(Ax_b.squeeze()) == 0:
            lambda_h_b = torch.zeros(Ax_b.shape)
        else:
            lambda_h_b = nlin(Ax_b) / Ax_b     # activation scaling

        lambda_h_b_mat = torch.diag(lambda_h_b.squeeze())
        activation_bias_mats += [lambda_h_b_mat]

        x_layer_b = torch.matmul(Ax_b, lambda_h_b_mat)

        Aprime_b = torch.matmul(A, lambda_h_b_mat)
        Aprime_b_mats += [Aprime_b]

        bprime = lambda_h_b * b
        bprimes += [bprime]
        # x_layer_Aprime_b = torch.matmul(x_layer_Aprime_b, Aprime_b) + bprime

    bstar = bprimes[0]
    for Aprime_b, bprime in zip(Aprime_b_mats[1:], bprimes[1:]):
        bstar = torch.matmul(bstar, Aprime_b) + bprime

    # network-wise parameter varying linear map:  A* = A'_L ... A'_1
    Astar = torch.chain_matmul(*Aprime_mats)
    Astar_b = torch.chain_matmul(*Aprime_b_mats)

    return Astar, Astar_b, bstar, Aprime_mats, Aprime_b_mats, bprimes

if __name__ == "__main__":
    import time
    torch.manual_seed(2)
    np.random.seed(2)
    nx = 3
    test_bias = True

    # random feature point
    x_z = torch.randn(1, nx)

    # define single layer square neural net
    fx_layer = blocks.MLP(nx, nx, nonlin=nn.ReLU, hsizes=[], bias=test_bias)
    # verify linear operations on MLP layers
    fx_layer.linear[0](x_z)
    torch.matmul(x_z, fx_layer.linear[0].effective_W()) + fx_layer.linear[0].bias
    # verify single layer linear parameter varying form
    lpv(fx_layer, torch.randn(1, nx))

    # define square neural net
    fx = blocks.MLP(nx, nx, nonlin=nn.ReLU, hsizes=[nx, nx, nx], bias=test_bias)
    if test_bias:
        for i in range(nx):
            fx.linear[i].bias.data = torch.randn(1, nx)
    # verify multi-layer linear parameter varying form
    lpv(fx, torch.randn(1, nx))

    # verify different activations
    activations = [nn.ReLU6, nn.ReLU, nn.PReLU, nn.GELU, nn.CELU, nn.ELU,
                nn.LogSigmoid, nn.Sigmoid, nn.Tanh]
    for act in activations:
        print(f'current activation {act}')
        fx_a = blocks.MLP(nx, nx, nonlin=act, hsizes=[nx, nx, nx], bias=test_bias)
        lpv(fx_a, torch.randn(1, nx))

    # perf testing for batched vs. sequential LPV implementations
    for act in activations:
        print(f'current activation {act}')
        batch = torch.randn(2, nx)
        fx_a = blocks.MLP(nx, nx, nonlin=act, hsizes=[nx]*8, linear_map=slim.PerronFrobeniusLinear, bias=test_bias)

        t = time.time()
        Astar, _, _, _, _ = lpv_batched(fx_a, batch)
        print("batched:   ", time.time() - t)
        t = time.time()
        for x in batch:
            Astar, Astar_b, _, _, _, _ = lpv(fx_a, x)
        print("sequential:", time.time() - t)