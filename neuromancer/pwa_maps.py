import numpy as np
import scipy as sp
import torch
from torch import nn
import matplotlib.pyplot as plt

import slim
from neuromancer import blocks

# TODO: debug PWA maps: they break with bias, they also break on activations without trivial nullspace
# TODO: LPV maps break with bias

def pwa_batched(fx, x, use_bias=True):
    x_layer = x

    Aprime_mats = []
    Lambdas = []
    bprimes = []
    weights = []
    biases = []

    for nlin, lin in zip(fx.nonlin, fx.linear):

        # z = A*x + b
        A = lin.effective_W()  # layer weight
        b = lin.bias if use_bias and lin.bias is not None else torch.zeros(A.shape[-1])
        z = torch.matmul(x_layer, A) + b  # affine transform
        weights.append(A)
        biases.append(b)

        # sigma(z) = Lambda*z + sigma(0)
        sigma_null_space = nlin(torch.zeros(z.shape[-1]))     # sigma(0)
        lambda_vec = (nlin(z) - sigma_null_space) / z  # activation scaling vector
        lambda_vec[z == 0] = 0.                     # fixing division by zero
        Lambda = torch.stack([torch.diag(v) for v in lambda_vec])  # activation scaling matrix Lambda
        Lambdas.append(Lambda)

        # print(nlin(lin(x_layer)))
        # print(nlin(z))

        # layer transform:  Lambda*(A*x + b) + sigma(0)
        x_layer = z * lambda_vec + sigma_null_space
        # print(x_layer)

        # A' = Lambda*A
        Aprime = torch.matmul(A, Lambda)
        Aprime_mats += [Aprime]

        # b' = Lambda*b + sigma(0)
        bprime = torch.matmul(b, Lambda) + sigma_null_space
        bprimes += [bprime]

    # network-wise local linear map
    # A* = A'_L ... A'_1
    Astar = Aprime_mats[0]
    for Aprime in Aprime_mats[1:]:
        Astar = torch.bmm(Astar, Aprime)

    # TODO: correct until this point - there are small errors in bias
    # network-wise local bias
    # b* = b*_L
    # b*_0 = Lambda_0*b_0 + sigma(0)_0    ->    b*_0 = b'_0
    bstar = bprime[0]
    for Aprime, bprime in zip(Aprime_mats[1:], bprimes[1:]):
        # b*_l = Lambda_l+1 * A_l+1 * (Lambda_l * b_l + sigma(0)_l) + Lambda_l+1 * b_l+1 + sigma(0)_l+1
        # b*_l = A'_l * b*_l-1 + b'_l
        bstar = torch.matmul(bstar, Aprime) + bprime

    return Astar, bstar, Aprime_mats, bprimes, Lambdas


def lpv_batched(fx, x, use_bias=False):
    x_layer = x

    Aprime_mats = []
    activation_mats = []
    bprimes = []

    for nlin, lin in zip(fx.nonlin, fx.linear):
        A = lin.effective_W()  # layer weight

        b = lin.bias if use_bias and lin.bias is not None else torch.zeros(A.shape[-1])
        Ax = torch.matmul(x_layer, A) + b  # affine transform

        zeros = Ax == 0
        lambda_h = nlin(Ax) / Ax  # activation scaling
        lambda_h[zeros] = 0.

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

    for nlin, lin in zip(fx.nonlin, fx.linear):
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
    z = torch.matmul(x_z, fx_layer.linear[0].effective_W())
    if test_bias:
        z += fx_layer.linear[0].bias
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
    activations = [nn.ReLU6, nn.LeakyReLU, nn.ReLU, nn.PReLU, nn.GELU, nn.CELU, nn.ELU,
                nn.LogSigmoid, nn.Sigmoid, nn.Tanh]
    for act in activations:
        print(f'\n current activation {act()}')

        fx_a = blocks.MLP(nx, nx, nonlin=act, hsizes=[nx, nx, nx], linear_map=slim.Linear, bias=test_bias)
        print(f'MLP: {fx_a(x_z)}')

        Astar, bstar, *_ = pwa_batched(fx_a, x_z)
        # x_pwa_batched = torch.matmul(x_z, Astar) + bstar if test_bias else torch.matmul(x_z, Astar)
        x_pwa_batched = torch.matmul(x_z, Astar) + bstar
        print(f'pwa_batched: {x_pwa_batched}')

        Astar, Astar_b, bstar, *_ = lpv(fx_a, x_z)
        x_lpv = torch.matmul(x_z, Astar) + bstar if test_bias else torch.matmul(x_z, Astar)
        print(f'lpv: {x_lpv}')

        Astar, bstar, *_ = lpv_batched(fx_a, x_z)
        x_lpv_batched = torch.matmul(x_z, Astar) + bstar if test_bias else torch.matmul(x_z, Astar)
        print(f'lpv_batched: {x_lpv_batched}')

    # # perf testing for batched vs. sequential LPV implementations
    # for act in activations:
    #     print(f'current activation {act}')
    #     batch = torch.randn(2, nx)
    #     fx_a = blocks.MLP(nx, nx, nonlin=act, hsizes=[nx]*8, linear_map=slim.Linear, bias=test_bias)
    #
    #     t = time.time()
    #     Astar, _, _, _, _ = lpv_batched(fx_a, batch)
    #     print("batched:   ", time.time() - t)
    #
    #     t = time.time()
    #     for x in batch:
    #         Astar, Astar_b, _, _, _, _ = lpv(fx_a, x)
    #     print("sequential:", time.time() - t)
