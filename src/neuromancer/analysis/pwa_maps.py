"""
Functions for obtaining piecewise affine maps (PWA) of fully connected neural networks


"""


import numpy as np
import torch
from torch import nn

import neuromancer.slim as slim
from neuromancer.modules import blocks


def pwa_batched(fx, x, use_bias=True):
    """
    function returning parameters of a local affine map of a fully connected neural network
    for a given input x, where the following holds:
    y = fx(x) = Astar*x + bstar

    :param fx: (nn.Module) fully connected neural network
    :param x: (torch.Tensor, shape=[batchsize, in_features])
    :param use_bias: flag for turning bias on and off
    :return: Astar, bstar of a local affine map of the network fx: y = Astar*x + bstar
    """
    x_layer = x

    Aprime_mats = []
    Lambdas = []
    bprimes = []
    weights = []
    biases = []
    iter = 0

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

        # layer transform:  Lambda*(A*x + b) + sigma(0)
        x_layer = z * lambda_vec + sigma_null_space

        # A' = Lambda*A
        Aprime = torch.matmul(A, Lambda)
        Aprime_mats += [Aprime]

        # b' = Lambda*b + sigma(0)
        bprime = torch.matmul(b, Lambda) + sigma_null_space
        bprimes += [bprime]

        if iter == 0:
            bstar = bprime
            Astar = Aprime
        else:
            # network-wise local bias
            # b*_l+1 = Lambda_l+1 * A_l+1 * (Lambda_l * b_l + sigma(0)_l) + Lambda_l+1 * b_l+1 + sigma(0)_l+1
            # b*_l+1 = A'_l+1 * b'_l + b'_l+1
            Aprime_bstar = torch.bmm(Aprime.transpose(1,2), bstar.unsqueeze(2))
            bstar = Aprime_bstar.squeeze(2) + bprime

            # network-wise local linear map
            # A* = A'_L ... A'_1
            Astar = torch.bmm(Astar, Aprime)
        iter += 1

    return Astar, bstar, Aprime_mats, bprimes, Lambdas


def lpv_batched(fx, x, use_bias=True):
    """
    TODO Broken implementation

    :param fx:
    :param x:
    :param use_bias:
    :return:
    """
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


if __name__ == "__main__":
    torch.manual_seed(2)
    np.random.seed(2)
    nx = 3
    test_bias = True
    batches = 2

    # random feature point
    x_z = torch.randn(batches, nx)

    # verify different activations
    activations = [nn.ReLU6, nn.LeakyReLU, nn.ReLU, nn.PReLU, nn.GELU, nn.CELU, nn.ELU,
                nn.LogSigmoid, nn.Sigmoid, nn.Tanh]
    for act in activations:
        print(f'\n current activation {act()}')

        fx_a = blocks.MLP(nx, nx, nonlin=act, hsizes=[nx, nx, nx], linear_map=slim.Linear, bias=test_bias)
        mlp_out = fx_a(x_z)
        print(f'MLP: {mlp_out}')

        Astar, bstar, *_ = pwa_batched(fx_a, x_z)
        # unbatched and batched evaluation
        if batches == 1:
            x_pwa_batched = (torch.matmul(x_z, Astar)).squeeze(1) + bstar
        else:
            x_pwa_batched = torch.bmm(Astar.transpose(1, 2), x_z.unsqueeze(2)).squeeze(2) + bstar
        print(f'pwa_batched: {x_pwa_batched}')

        difference = torch.norm(mlp_out - x_pwa_batched, p=2)
        print(difference)
        print(difference < 1e-6)

        # lpv_batched is depreciated
        Astar, bstar, *_ = lpv_batched(fx_a, x_z)
        x_lpv_batched = torch.matmul(x_z, Astar) + bstar if test_bias else torch.matmul(x_z, Astar)
        print(f'lpv_batched: {x_lpv_batched}')

        difference = torch.norm(mlp_out - x_lpv_batched, p=2)
        print(difference)
        print(difference < 1e-6)

