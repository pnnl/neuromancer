import numpy as np
import scipy as sp
import torch
from torch import nn
import matplotlib.pyplot as plt

import slim
from neuromancer import blocks


def lpv(fx, x):
    """pared-down version of LPV_net"""
    # nonlinearities = fx.nonlin
    x_layer = x
    x_layer_b = x
    x_layer_orig = x
    x_layer_Aprime = x
    x_layer_Aprime_b = x
    A_mats = []
    Aprime_mats = []
    Aprime_b_mats = []
    activation_mats = []
    activation_bias_mats = []
    bprimes = []

    for nlin, lin in zip(fx.nonlin, fx.linear):
        A = lin.effective_W()  # layer weight
        A_mats += [A]
        Ax = torch.matmul(x_layer, A)  # linear transform

        # TODO: if *any* are zero, this will break
        if sum(Ax.squeeze()) == 0:
            lambda_h = torch.zeros(Ax.shape)
        else:
            lambda_h = nlin(Ax) / Ax  # activation scaling

        lambda_h = torch.diag(lambda_h.squeeze())
        activation_mats += [lambda_h]

        x_layer = torch.matmul(Ax, lambda_h)
        x_layer_orig = nlin(lin(x_layer_orig))

        # compute layer-wise parameter-varying linear map
        Aprime = torch.matmul(A, lambda_h)
        x_layer_Aprime = torch.matmul(x_layer_Aprime, Aprime)

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
        x_layer_Aprime_b = torch.matmul(x_layer_Aprime_b, Aprime_b) + bprime

    bstar = bprimes[0]
    for Aprime_b, bprime in zip(Aprime_b_mats[1:], bprimes[1:]):
        bstar = torch.matmul(bstar, Aprime_b) + bprime

    # network-wise parameter varying linear map:  A* = A'_L ... A'_1
    Astar = torch.chain_matmul(*Aprime_mats)
    Astar_b = torch.chain_matmul(*Aprime_b_mats)
    return Astar, Astar_b, bstar, Aprime_mats, A_mats, Aprime_b_mats, bprimes