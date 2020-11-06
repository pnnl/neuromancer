import numpy as np
import scipy as sp
import torch
from torch import nn


def lpv(fx, x):
    """pared-down version of LPV_net"""
    # nonlinearities = fx.nonlin
    x_layer = x
    x_layer_orig = x
    x_layer_Aprime = x

    A_mats = []
    Aprime_mats = []
    activation_mats = []

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

    # network-wise parameter varying linear map:  A* = A'_L ... A'_1
    Astar = torch.chain_matmul(*Aprime_mats)

    return Astar, Aprime_mats, A_mats