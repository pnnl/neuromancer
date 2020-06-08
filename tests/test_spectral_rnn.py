"""
Here is the numpy version from the original tensorflow implementation.
https://github.com/zhangjiong724/spectral-RNN/blob/master/code/spectral_rnn.py
This script was used to test our pytorch implementation against the original.
"""
import numpy as np
from deepmpc.linear import SpectralLinear
import torch


def hprod(H, u, k):
    alpha = 2 * np.dot(H[:, -k:], u[-k:]) / np.dot(u[-k:],u[-k:]) # alpha.shape = (batch,)
    H_out = H.copy()
    H_out[:, -k:] -= np.outer(alpha, u[-k:])
    return H_out


def Hprod(x, u, k):
    """

    :param x: bs X dim
    :param u: dim
    :param k: int
    :return: bs X dim
    """
    alpha = 2 * torch.matmul(x[:, -k:], u[-k:]) / (u[-k:] * u[-k:]).sum()
    x[:, -k:] -= torch.matmul(alpha.view(-1, 1), u[-k:].view(1, -1))  # Subtract outer product
    return x


def np_svdProd(H,U):
    U_shape = U.shape
    n_r = U_shape[0]; n_h = U_shape[1]
    assert(H.shape[1] == n_h)
    H_copy = H.copy()
    for i in range(0, n_r):
        H_copy = hprod(H_copy, U[i], n_h-i)
    return H_copy


def vmultiply(H,U):
    U_shape = U.shape
    n_r = U_shape[0]; n_h = U_shape[1]
    assert( H.shape[1] == n_h)
    H_copy = H.copy()
    for i in range(n_r-1,-1,-1):
        H_copy = hprod(H_copy, U[i], n_h-i)
    return H_copy


class np_svd_linear():
    def __init__(self, U, V, Sigma):
        self.U = U
        self.V = V
        self.Sigma = Sigma

    def __call__(self, x):
        x = np_svdProd(x, self.V)
        x = x*Sigma
        x = vmultiply(x, self.U)
        return x


def Vmultiply(x, V):
    """

    :param x: BS X dim
    :return: BS X dim
    """
    n_r, n_h = V.shape
    assert x.shape[1] == n_h
    for i in range(0, n_r):
        x = Hprod(x, V[i], n_h - i)
    return x


def Umultiply(x, U):
    """
    :param x: bs X dim
    :return:
    """
    n_r, n_h = U.shape
    assert x.shape[1] == n_h
    for i in range(n_r - 1, -1, -1):
        x = Hprod(x, U[i], n_h - i)
    return x


if __name__ == '__main__':
    tlin = SpectralLinear(4, 3, bias=False, n_U_reflectors=3, n_V_reflectors=3, sigma_min=.9, sigma_max=1.1)

    tx = torch.randn(6, 4)
    Sigma = np.ones((1, 3))

    # U = tlin.U.detach().numpy()
    # V = tlin.V.detach().numpy()
    #
    # nlin = np_svd_linear(V, U, Sigma)
    # print(nlin(tx.detach().numpy()))
    print(tlin(tx))