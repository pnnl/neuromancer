"""
Pytorch weight initializations

# TODO: Confirm sparse parametrizations
torch.nn.init.xavier_normal_(tensor, gain=1.0)
torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
torch.nn.init.orthogonal_(tensor, gain=1)
torch.nn.init.sparse_(tensor, sparsity, std=0.01)
"""

from abc import ABC, abstractmethod
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBase(nn.Module, ABC):
    """
    """

    def __init__(self, insize, outsize, bias=False):
        super().__init__()
        self.in_features, self.out_features = insize, outsize
        self.weight = nn.Parameter(torch.Tensor(insize, outsize))
        self.bias = nn.Parameter(torch.zeros(1, outsize), requires_grad=not bias)
        # initialize default weights to be standard pytorch init
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def reg_error(self):
        return torch.tensor(0.0)

    @abstractmethod
    def effective_W(self):
        pass

    def forward(self, x):
        return torch.matmul(x, self.effective_W()) + self.bias


class SquareLinear(LinearBase, ABC):
    """
    """
    def __init__(self, insize, outsize, bias=False, **kwargs):
        assert insize == outsize, f'Map must be square. insize={insize} and outsize={outsize}'
        super().__init__(insize, outsize, bias=bias)

    @abstractmethod
    def effective_W(self):
        pass


class Linear(LinearBase):
    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize, bias=bias)
        self.linear = nn.Linear(insize, outsize, bias=bias)

    def effective_W(self):
        return self.linear.weight.T

    def forward(self, x):
        return self.linear(x)


class NonNegativeLinear(LinearBase):
    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize, bias=bias)
        self.weight = nn.Parameter(torch.abs(self.weight)*0.1)

    def effective_W(self):
        return F.relu(self.weight)


class PSDLinear(SquareLinear):

    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize, bias=bias)

    def effective_W(self):
        return torch.matmul(self.weight, self.weight.T)


class LassoLinear(LinearBase):
    """
    From https://leon.bottou.org/publications/pdf/compstat-2010.pdf
    """

    def __init__(self, insize, outsize, bias=False, gamma=1.0, **kwargs):
        super().__init__(insize, outsize, bias=bias)
        u = torch.empty(insize, outsize)
        torch.nn.init.kaiming_normal_(u)
        self.u_param = nn.Parameter(torch.abs(u) / 2.0)
        v = torch.empty(insize, outsize)
        torch.nn.init.kaiming_normal_(v)
        self.v_param = nn.Parameter(torch.abs(v) / 2.0)
        self.gamma = gamma

    def effective_W(self):
        # Thresholding for sparsity
        return F.relu(self.u_param) - F.relu(self.v_param)

    def reg_error(self):
        # shrinkage
        return self.gamma * self.effective_W().norm(p=1)


class RightStochasticLinear(LinearBase):
    """
    A right stochastic matrix is a real square matrix, with each row summing to 1.
    https://en.wikipedia.org/wiki/Stochastic_matrix
    """

    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize, bias=bias)

    def effective_W(self):
        return F.softmax(self.weight, dim=1)


class LeftStochasticLinear(LinearBase):
    """
    A left stochastic matrix is a real square matrix, with each column summing to 1.
    https://en.wikipedia.org/wiki/Stochastic_matrix
    """

    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize, bias=bias)

    def effective_W(self):
        return F.softmax(self.weight, dim=0)


class PerronFrobeniusLinear(LinearBase):

    def __init__(self, insize, outsize, bias=False, sigma_min=0.8, sigma_max=1.0, **kwargs):
        """
        Perron-Frobenius theorem based regularization of matrix

        :param insize: (int) Dimension of input vectors
        :param outsize: (int) Dimension of output vectors
        :param bias: (bool) Whether to add bias to linear transform
        :param sigma_min: (float) maximum allowed value of dominant eigenvalue
        :param sigma_max: (float)  minimum allowed value of dominant eigenvalue
        """
        super().__init__(insize, outsize, bias=bias)
        # matrix scaling to allow for different row sums
        self.scaling = nn.Parameter(torch.rand(insize, outsize))
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def effective_W(self):
        s_clamped = self.sigma_max - (self.sigma_max - self.sigma_min) * torch.sigmoid(self.scaling)
        w_sofmax = s_clamped * F.softmax(self.weight, dim=1)
        return w_sofmax


class SymmetricLinear(SquareLinear):
    """
    symmetric matrix A (effective_W) is a square matrix that is equal to its transpose.
    A = A^T
    https://en.wikipedia.org/wiki/Symmetric_matrix
    """

    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize, bias=bias)

    def effective_W(self):
        return (self.weight + torch.t(self.weight)) / 2


class SkewSymmetricLinear(SquareLinear):
    """
    skew-symmetric (or antisymmetric) matrix A (effective_W) is a square matrix whose transpose equals its negative.
    A = -A^T
    https://en.wikipedia.org/wiki/Skew-symmetric_matrix
    """

    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize, bias=bias)

    def effective_W(self):
        return self.weight.triu() - self.weight.triu().T


class DampedSkewSymmetricLinear(SkewSymmetricLinear):
    """
    skew-symmetric (or antisymmetric) matrix A (effective_W) is a square matrix whose transpose equals its negative.
    A = -A^T
    https://en.wikipedia.org/wiki/Skew-symmetric_matrix
    """

    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize, bias=bias)
        self.eye = nn.Parameter(torch.eye(insize, outsize), requires_grad=False)
        self.gamma = nn.Parameter(0.01 * torch.randn(1, 1))

    def effective_W(self):
        return super().effective_W() - self.gamma * self.gamma * self.eye


class SplitLinear(LinearBase):
    """
    A = B − C, with B ≥ 0 and C ≥ 0.
    https://en.wikipedia.org/wiki/Matrix_splitting
    """

    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize, bias=bias)
        self.B = NonNegativeLinear(insize, outsize, bias)
        self.C = NonNegativeLinear(insize, outsize, bias)

    def effective_W(self):
        A = self.B.effective_W() - self.C.effective_W()
        return A


class StableSplitLinear(LinearBase):
    """
    A = B − C, with stable B and stable C
    https://en.wikipedia.org/wiki/Matrix_splitting
    """

    def __init__(self, insize, outsize, bias=False, sigma_min=0.1, sigma_max=1.0, **kwargs):
        super().__init__(insize, outsize, bias=bias)
        self.B = PerronFrobeniusLinear(insize, outsize, bias, sigma_max, sigma_max)
        self.C = PerronFrobeniusLinear(insize, outsize, bias, 0, sigma_max - sigma_min)

    def effective_W(self):
        A = self.B.effective_W() - self.C.effective_W()
        return A


class SVDLinear(LinearBase):
    def __init__(self, insize, outsize, bias=False, sigma_min=0.1, sigma_max=1, **kwargs):
        """

        SVD based regularization of matrix A
        A = U*Sigma*V
        U,V = unitary matrices (orthogonal for real matrices A)
        Sigma = diagonal matrix of singular values (square roots of eigenvalues)
        nu = number of columns
        nx = number of rows
        sigma_min = minum allowed value of  eigenvalues
        sigma_max = maximum allowed value of eigenvalues
        """
        super().__init__(insize, outsize, bias=bias)
        u = torch.empty(insize, insize)
        torch.nn.init.orthogonal_(u)
        self.U = nn.Parameter(u)
        v = torch.empty(outsize, outsize)
        torch.nn.init.orthogonal_(v)
        self.V = nn.Parameter(v)
        self.sigma = nn.Parameter(torch.rand(insize, 1))  # scaling of singular values
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def orthogonal_error(self, weight):
        size = weight.shape[0]
        return torch.norm(torch.norm(torch.eye(size).to(weight.device) -
                              torch.mm(weight, torch.t(weight)), 2) +
                           torch.norm(torch.eye(size).to(weight.device) -
                                      torch.mm(torch.t(weight), weight), 2), 2)

    def reg_error(self):
        return self.orthogonal_error(self.U) + self.orthogonal_error(self.V)

    def effective_W(self):
        """

        :return: Matrix for linear transformation with dominant eigenvalue between sigma_max and sigma_min
        """
        sigma_clapmed = self.sigma_max - (self.sigma_max - self.sigma_min) * torch.sigmoid(self.sigma)
        Sigma_bounded = torch.eye(self.in_features, self.out_features).to(self.sigma.device) * sigma_clapmed
        w_svd = torch.mm(self.U, torch.mm(Sigma_bounded, self.V))
        return w_svd


def Hprod(x, u, k):
    """

    :param x: bs X dim
    :param u: dim
    :param k: int
    :return: bs X dim
    """
    alpha = 2 * torch.matmul(x[:, -k:], u[-k:]) / (u[-k:] * u[-k:]).sum()
    if k < x.shape[1]:
        return torch.cat([x[:, :-k], x[:, -k:] - torch.matmul(alpha.view(-1, 1), u[-k:].view(1, -1))],
                         dim=1)  # Subtract outer product
    else:
        return x[:, -k:] - torch.matmul(alpha.view(-1, 1), u[-k:].view(1, -1))


class OrthogonalLinear(SquareLinear):

    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize, bias=bias)
        self.U = nn.Parameter(torch.triu(torch.randn(insize, insize)))

    def effective_W(self):
        return self.forward(torch.eye(self.in_features).to(self.U.device))

    def forward(self, x):
        """

        :param x: BS X dim
        :return: BS X dim
        """
        for i in range(0, self.in_features):
            x = Hprod(x, self.U[i], self.in_features - i)
        return x + self.bias


class SpectralLinear(LinearBase):
    """
    Translated from tensorflow code: https://github.com/zhangjiong724/spectral-RNN/blob/master/code/spectral_rnn.py
    SVD paramaterized linear map of form U\SigmaV. Singular values can be constrained to a range
    """

    def __init__(self, insize, outsize, bias=False,
                 n_U_reflectors=None, n_V_reflectors=None,
                 sigma_min=0.1, sigma_max=1.0, **kwargs):
        """

        :param insize: (int) Dimension of input vectors
        :param outsize: (int) Dimension of output vectors
        :param n_U_reflectors: (int) It looks like this should effectively constrain the rank of the matrix
        :param n_V_reflectors: (int) It looks like this should effectively constrain the rank of the matrix
        :param bias: (bool) whether to add a bias term.
        :param sig_min: min value of singular values
        :param sig_max: max value of singular values
        """
        super().__init__(insize, outsize, bias=bias)
        if n_U_reflectors is not None and n_U_reflectors is not None:
            assert n_U_reflectors <= insize, 'Too many reflectors'
            assert n_V_reflectors <= outsize, 'Too may reflectors'
            self.n_U_reflectors, self.n_V_reflectors = n_U_reflectors, n_V_reflectors
        else:
            self.n_U_reflectors, self.n_V_reflectors = min(insize, outsize), min(insize, outsize)

        self.r = (sigma_max - sigma_min) / 2
        self.sigma_mean = sigma_min + self.r
        nsigma = min(insize, outsize)
        self.p = nn.Parameter(torch.zeros(nsigma) + 0.001 * torch.randn(nsigma))
        self.V = nn.Parameter(torch.triu(torch.randn(outsize, outsize)))
        self.U = nn.Parameter(torch.triu(torch.randn(insize, insize)))

    def Sigma(self):
        sigmas = 2 * self.r * (torch.sigmoid(self.p) - 0.5) + self.sigma_mean
        square_matrix = torch.diag(torch.cat([sigmas, torch.zeros(abs(self.in_features - self.out_features)).to(sigmas.device)]))
        return square_matrix[:self.in_features, :self.out_features]

    def Umultiply(self, x):
        """

        :param x: BS X
        :return: BS X dim
        """
        assert x.shape[1] == self.in_features, f'x.shape: {x.shape}, in_features: {self.in_features}'
        for i in range(0, self.n_U_reflectors):
            x = Hprod(x, self.U[i], self.in_features - i)
        return x

    def Vmultiply(self, x):
        """
        :param x: bs X dim
        :return:
        """
        assert x.shape[1] == self.out_features
        for i in range(self.n_V_reflectors - 1, -1, -1):
            x = Hprod(x, self.V[i], self.out_features - i)
        return x

    def effective_W(self):
        return self.forward(torch.eye(self.in_features).to(self.p.device))

    def forward(self, x):
        """
        args: a list of 2D, batch x n, Tensors.

        :param args:
        :return:
        """
        x = self.Umultiply(x)
        x = torch.matmul(x, self.Sigma())
        x = self.Vmultiply(x)
        return x + self.bias


class SymplecticLinear(SquareLinear):
    """
    https://en.wikipedia.org/wiki/Symplectic_matrix
    https://arxiv.org/abs/1705.03341
    """

    def __init__(self, insize, outsize, bias=False, **kwargs):
        assert insize % 2 == 0, 'Symplectic Matrix must have even dimensions'
        super().__init__(insize, outsize, bias=bias)
        self.weight = torch.nn.Parameter(torch.empty(int(insize/2), int(outsize/2)))
        torch.nn.init.kaiming_normal_(self.weight)
        self.weight = nn.Parameter(self.weight)

    def effective_W(self):
        return torch.cat([torch.cat([torch.zeros(self.in_features // 2, self.in_features // 2), self.weight], dim=1),
                          torch.cat([-1 * self.weight.T, torch.zeros(self.in_features // 2, self.in_features // 2)], dim=1)])


# class DoublyStochasticLinear(LinearBase):
#     """
#     A doubly stochastic matrix is a real square matrix, with each column and each row summing to 1.
#     https://en.wikipedia.org/wiki/Doubly_stochastic_matrix
#     https://github.com/btaba/sinkhorn_knopp
#     https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
#     """
#
#     def __init__(self, insize, outsize, bias, **kwargs):
#         super().__init__(insize, outsize, bias=bias)


# class HamiltonianLinear(LinearBase):
#     """
#     https://en.wikipedia.org/wiki/Hamiltonian_matrix
#     """
#
#     def __init__(self, insize, outsize, bias, **kwargs):
#         super().__init__(insize, outsize)
#
#
# class DiagDominantLinear(LinearBase):
#     """
#     A strictly diagonally dominant matrix is non-singular.
#     https://en.wikipedia.org/wiki/Diagonally_dominant_matrix
#     """
#
#     def __init__(self, insize, outsize, bias, **kwargs):
#         super().__init__(insize, outsize, bias=bias)
#
#
# class RegularSplitLinear(LinearBase):
#     """
#     Definition: A = B − C is a regular splitting of A if B^−1 ≥ 0 and C ≥ 0.
#     https://en.wikipedia.org/wiki/Matrix_splitting
#     """
#
#     def __init__(self, insize, outsize, bias, **kwargs):
#         super().__init__(insize, outsize, bias=bias)
#
#
# class DiagSplitLinear(LinearBase):
#     """
#     https://en.wikipedia.org/wiki/Matrix_splitting
#     https://en.wikipedia.org/wiki/Jacobi_method
#     """
#
#     def __init__(self, insize, outsize, bias, **kwargs):
#         super().__init__(insize, outsize, bias=bias)


square_maps = {SymmetricLinear, SkewSymmetricLinear, DampedSkewSymmetricLinear, PSDLinear,
               OrthogonalLinear, SymplecticLinear}

maps = {'linear': Linear,
        'nneg': NonNegativeLinear,
        'lasso': LassoLinear,
        'lstochastic': LeftStochasticLinear,
        'rstochastic': RightStochasticLinear,
        'pf': PerronFrobeniusLinear,
        'symmetric': SymmetricLinear,
        'skew_symetric': SkewSymmetricLinear,
        'damp_skew_symmetric': DampedSkewSymmetricLinear,
        'split': SplitLinear,
        'stable_split': StableSplitLinear,
        'spectral': SpectralLinear,
        'softSVD': SVDLinear,
        'orthogonal': OrthogonalLinear,
        'psd': PSDLinear,
        'symplectic': SymplecticLinear}


if __name__ == '__main__':
    """
    Tests
    """
    square = torch.rand(8, 8)
    long = torch.rand(3, 8)
    tall = torch.rand(8, 3)

    for linear in set(list(maps.values())) - square_maps:
        print(linear)
        map = linear(3, 5)
        x = map(tall)
        assert (x.shape[0], x.shape[1]) == (8, 5)
        map = linear(8, 3)
        x = map(long)
        assert (x.shape[0], x.shape[1]) == (3, 3)

    for linear in square_maps:
        print(linear)
        map = linear(8, 8)
        x = map(square)
        assert (x.shape[0], x.shape[1]) == (8, 8)
        x = map(long)
        assert (x.shape[0], x.shape[1]) == (3, 8)







