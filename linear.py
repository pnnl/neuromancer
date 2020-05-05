from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from matrix import SoftOrthogonal, SoftInvertible


class LinearBase(nn.Module, ABC):
    """
    """
    def __init__(self, insize, outsize):
        super().__init__()
        self.in_features, self.out_features = insize, outsize
        self.error_matrix = nn.Parameter(torch.zeros(1), requires_grad=False)

    def reg_error(self):
        return self.error_matrix

    @abstractmethod
    def effective_W(self):
        pass


class Linear(LinearBase):
    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize)
        self.linear = nn.Linear(insize, outsize, bias=bias)

    def effective_W(self):
        return self.linear.weight.T

    def forward(self, x):
        return self.linear(x)


class NonnegativeLinear(LinearBase):
    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize)
        self.weight = nn.Parameter(torch.rand(insize, outsize))
        self.bias = nn.Parameter(torch.zeros(1, outsize), requires_grad=not bias)

    def effective_W(self):
        w_LB = F.relu(self.weight)
        return w_LB

    def forward(self, x):
        return torch.matmul(x, self.effective_W()) + self.bias


class LassoLinear(LinearBase):
    """
    From https://leon.bottou.org/publications/pdf/compstat-2010.pdf
    """
    def __init__(self, insize, outsize, bias=False, gamma=1.0, **kwargs):
        super().__init__(insize, outsize)
        self.u_param = nn.Parameter(torch.rand(insize, outsize))
        self.v_param = nn.Parameter(torch.rand(insize, outsize))
        self.bias = nn.Parameter(torch.zeros(1, outsize), requires_grad=not bias)
        self.gamma = gamma

    def effective_W(self):
        # Thresholding for sparsity
        return F.relu(self.u_param) - F.relu(self.v_param)

    def reg_error(self):
        # shrinkage
        return self.gamma*self.effective_W().norm(p=1)

    def forward(self, x):
        return torch.matmul(x, self.effective_W()) + self.bias


class RightStochasticLinear(LinearBase):
    """
    A right stochastic matrix is a real square matrix, with each row summing to 1.
    https://en.wikipedia.org/wiki/Stochastic_matrix
    """
    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize)
        self.weight = nn.Parameter(torch.rand(insize, outsize))
        self.bias = nn.Parameter(torch.zeros(1, outsize), requires_grad=not bias)
        
    def effective_W(self):
        w_rStochastic = F.softmax(self.weight, dim=1)
        return w_rStochastic      
        
    def forward(self, x):
        return torch.matmul(x, self.effective_W()) + self.bias


class LeftStochasticLinear(LinearBase):
    """
    A left stochastic matrix is a real square matrix, with each column summing to 1.
    https://en.wikipedia.org/wiki/Stochastic_matrix
    """
    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize)
        self.weight = nn.Parameter(torch.rand(insize, outsize)) 
        self.do_bias = bias
        self.bias = nn.Parameter(torch.zeros(1, outsize), requires_grad=not bias)
        
    def effective_W(self):
        w_rStochastic = F.softmax(self.weight.T, dim=1)
        return w_rStochastic.T      
        
    def forward(self, x):
        return torch.matmul(x, self.effective_W()) + self.bias


class PerronFrobeniusLinear(LinearBase):

    def __init__(self, insize, outsize, bias=False, sigma_min=0.9, sigma_max=1.0,
                 init='basic', **kwargs):
        """
        Perron-Frobenius theorem based regularization of matrix

        :param insize: (int) Dimension of input vectors
        :param outsize: (int) Dimension of output vectors
        :param bias: (bool) Whether to add bias to linear transform
        :param sigma_min: (float) maximum allowed value of dominant eigenvalue
        :param sigma_max: (float)  minimum allowed value of dominant eigenvalue
        :param init: (str) 'init' or 'basic'. Whether to use identity initialization for hidden transition
        """
        super().__init__(insize, outsize)
        self.weight = nn.Parameter(torch.rand(insize, outsize))
        self.scaling = nn.Parameter(torch.rand(insize, outsize))  # matrix scaling to allow for different row sums
        if init == 'basic':
            self.weight = nn.Parameter(torch.rand(insize, outsize))
            self.scaling = nn.Parameter(torch.rand(insize, outsize))  # matrix scaling to allow for different row sums
        elif init == 'identity':
            self.weight = nn.Parameter(-1000*torch.ones(insize, outsize) + torch.eye(insize, outsize)*1001)
            self.scaling = nn.Parameter(-100*torch.ones(insize, outsize))
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.do_bias = bias
        self.bias = nn.Parameter(torch.zeros(1, outsize), requires_grad=not bias)

    def effective_W(self):
        s_clamped = self.sigma_max - (self.sigma_max - self.sigma_min) * torch.sigmoid(self.scaling)
        w_sofmax = s_clamped * F.softmax(self.weight, dim=1)
        return w_sofmax

    def forward(self, x):
        return torch.matmul(x, self.effective_W()) + self.bias


class SymmetricLinear(LinearBase):
    """
    symmetric matrix A (effective_W) is a square matrix that is equal to its transpose.
    A = A^T
    https://en.wikipedia.org/wiki/Symmetric_matrix
    """
    def __init__(self, insize, outsize, bias=False, **kwargs):
        assert insize == outsize, 'skew-symmetric matrices must be square'
        super().__init__(insize, outsize)
        self.weight = nn.Parameter(torch.rand(insize, insize)) # identity matrix with small noise
        self.do_bias = bias
        self.bias = nn.Parameter(torch.zeros(1, insize), requires_grad=not bias)

    def effective_W(self):
         sym_weight = (self.weight + torch.t(self.weight))/2
         return sym_weight
        
    def forward(self, x):  
        return torch.matmul(x, self.effective_W()) + self.bias


class SkewSymmetricLinear(LinearBase):
    """
    skew-symmetric (or antisymmetric) matrix A (effective_W) is a square matrix whose transpose equals its negative. 
    A = -A^T
    https://en.wikipedia.org/wiki/Skew-symmetric_matrix
    """
    def __init__(self, insize, outsize, bias=False, **kwargs):
        assert insize == outsize, 'skew-symmetric matrices must be square'
        super().__init__(insize, outsize)
        self.weight = nn.Parameter(torch.randn(insize, outsize))
        self.do_bias = bias
        self.bias = nn.Parameter(torch.zeros(1, insize), requires_grad=not bias)

    def effective_W(self):
         skewsym_weight = self.weight.triu() - self.weight.triu().T
         return skewsym_weight
        
    def forward(self, x):  
        return torch.matmul(x, self.effective_W()) + self.bias


class SplitLinear(LinearBase):
    """
    A = B − C, with B ≥ 0 and C ≥ 0.
    https://en.wikipedia.org/wiki/Matrix_splitting
    """
    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize)
        self.B = NonnegativeLinear(insize, outsize, bias)
        self.C = NonnegativeLinear(insize, outsize, bias)
        self.bias = nn.Parameter(torch.zeros(1, outsize), requires_grad=not bias)

    def effective_W(self):
         A = self.B.effective_W() - self.C.effective_W() 
         return A
        
    def forward(self, x):
        return torch.matmul(x, self.effective_W()) + self.bias


class StableSplitLinear(LinearBase):
    """
    A = B − C, with stable B and stable C
    https://en.wikipedia.org/wiki/Matrix_splitting
    """
    def __init__(self, insize, outsize, bias=False, sigma_min=0.1, sigma_max=1.0, **kwargs):
        super().__init__(insize, outsize)
        self.B = PerronFrobeniusLinear(insize, outsize, bias, sigma_max, sigma_max)
        self.C = PerronFrobeniusLinear(insize, outsize, bias, 0, sigma_max-sigma_min)
        self.bias = nn.Parameter(torch.zeros(1, outsize), requires_grad=not bias)

    def effective_W(self):
         A = self.B.effective_W() - self.C.effective_W()
         return A
        
    def forward(self, x):
        return torch.matmul(x, self.effective_W()) + self.bias
    

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
        super().__init__(insize, outsize)
        self.U = SoftOrthogonal(insize)
        self.V = SoftOrthogonal(outsize)
        self.sigma = nn.Parameter(torch.rand(insize, 1))  # scaling of singular values
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.insize, self.outsize = insize, outsize
        self.bias = nn.Parameter(torch.zeros(1, outsize), requires_grad=not bias)

    def effective_W(self):
        """

        :return: Matrix for linear transformation with dominant eigenvalue between sigma_max and sigma_min
        """
        sigma_clapmed = self.sigma_max - (self.sigma_max - self.sigma_min) * torch.sigmoid(self.sigma)
        # Sigma_bounded = torch.eye(self.insize, self.outsize).to(self.sigma.device) * sigma_clapmed
        Sigma_bounded = torch.eye(self.insize, self.outsize)* sigma_clapmed
        w_svd = torch.mm(self.U.weight, torch.mm(Sigma_bounded, self.V.weight))
        return w_svd

    def reg_error(self):
        return self.U() + self.V()  # error of spectral regularization

    def forward(self, x):
        return torch.matmul(x, self.effective_W()) + self.bias


class SpectralLinear(LinearBase):
    """
    Translated from tensorflow code: https://github.com/zhangjiong724/spectral-RNN/blob/master/code/spectral_rnn.py
    SVD paramaterized linear map of form U\SigmaV. Singular values can be constrained to a range
    """

    def __init__(self, insize, outsize, bias=False,
                 n_U_reflectors=None, n_V_reflectors=None, sigma_min=0.1, sigma_max=1.0, **kwargs):
        """

        :param insize: (int) Dimension of input vectors
        :param outsize: (int) Dimension of output vectors
        :param reflector_size: (int) It looks like this should effectively constrain the rank of the matrix (bonus!)
        :param bias: (bool) whether to add a bias term.
        :param sig_mean: initial and "mean" value of singular values, usually set to 1.0
        :param r: singular margin, the allowed margin for singular values
        """
        super().__init__(insize, outsize)
        if n_U_reflectors is not None and n_U_reflectors is not None:
            assert n_U_reflectors <= insize, 'Too many reflectors'
            assert n_V_reflectors <= outsize, 'Too may reflectors'
            self.n_U_reflectors, self.n_V_reflectors = n_U_reflectors, n_V_reflectors
        else:
            self.n_U_reflectors, self.n_V_reflectors = min(insize, outsize), min(insize, outsize)

        self.insize, self.outsize = insize, outsize
        self.r = (sigma_max - sigma_min)/2
        self.sigma_mean = sigma_min + self.r
        self.U = nn.Parameter(torch.triu(torch.randn(insize, insize)))
        nsigma = min(insize, outsize)
        self.p = nn.Parameter(torch.zeros(nsigma) + 0.001*torch.randn(nsigma))
        self.V = nn.Parameter(torch.triu(torch.randn(outsize, outsize)))
        self.bias = nn.Parameter(torch.zeros(outsize), requires_grad=not bias)

    def Sigma(self):
        sigmas = 2 * self.r * (torch.sigmoid(self.p) - 0.5) + self.sigma_mean
        square_matrix = torch.diag(torch.cat([sigmas, torch.zeros(abs(self.insize - self.outsize))]))
        return square_matrix[:self.insize, :self.outsize]

    def Hprod(self, x, u, k):
        """

        :param x: bs X dim
        :param u: dim
        :param k: int
        :return: bs X dim
        """
        alpha = 2 * torch.matmul(x[:, -k:], u[-k:]) / (u[-k:] * u[-k:]).sum()
        if k < x.shape[1]:
            return torch.cat([x[:, :-k], x[:, -k:] - torch.matmul(alpha.view(-1, 1), u[-k:].view(1, -1))], dim=1)  # Subtract outer product
        else:
            return x[:, -k:] - torch.matmul(alpha.view(-1, 1), u[-k:].view(1, -1))

    def Umultiply(self, x):
        """

        :param x: BS X
        :return: BS X dim
        """
        assert x.shape[1] == self.insize
        for i in range(0, self.n_U_reflectors):
            x = self.Hprod(x, self.U[i], self.insize - i)
        return x

    def Vmultiply(self, x):
        """
        :param x: bs X dim
        :return:
        """
        assert x.shape[1] == self.outsize
        for i in range(self.n_V_reflectors - 1, -1, -1):
            x = self.Hprod(x, self.V[i], self.outsize - i)
        return x

    def effective_W(self):
        return self.forward(torch.eye(self.insize))
        # return self.forward(torch.eye(self.insize).to(self.p.device))

    def forward(self, x):
        """
        args: a list of 2D, batch x n, Tensors.

        :param args:
        :return:
        """
        x = self.Umultiply(x)
        x = torch.matmul(x, self.Sigma())
        x = self.Vmultiply(x)
        if self.bias is not None:
            x += self.bias
        return x


class DoublyStochasticLinear(LinearBase):
    """
    A doubly stochastic matrix is a real square matrix, with each column and each row summing to 1.
    https://en.wikipedia.org/wiki/Doubly_stochastic_matrix
    https://github.com/btaba/sinkhorn_knopp
    https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """
    def __init__(self, insize, outsize, bias, **kwargs):
        super().__init__(insize, outsize)


class HamiltonianLinear(LinearBase):
    """
    https://en.wikipedia.org/wiki/Hamiltonian_matrix
    """

    def __init__(self, insize, outsize, bias, **kwargs):
        super().__init__(insize, outsize)


class SympleticLinear(LinearBase):
    """
    https://en.wikipedia.org/wiki/Symplectic_matrix
    """

    def __init__(self, insize, outsize, bias, **kwargs):
        super().__init__(insize, outsize)


class DiagDominantLinear(LinearBase):
    """
    A strictly diagonally dominant matrix is non-singular.
    https://en.wikipedia.org/wiki/Diagonally_dominant_matrix
    """

    def __init__(self, insize, outsize, bias, **kwargs):
        super().__init__(insize, outsize)


class RegularSplitLinear(LinearBase):
    """
    Definition: A = B − C is a regular splitting of A if B^−1 ≥ 0 and C ≥ 0.
    https://en.wikipedia.org/wiki/Matrix_splitting
    """

    def __init__(self, insize, outsize, bias, **kwargs):
        super().__init__(insize, outsize)


class DiagSplitLinear(LinearBase):
    """
    https://en.wikipedia.org/wiki/Matrix_splitting
    https://en.wikipedia.org/wiki/Jacobi_method
    """

    def __init__(self, insize, outsize, bias, **kwargs):
        super().__init__(insize, outsize)


maps = [Linear, NonnegativeLinear, LassoLinear,
        LeftStochasticLinear, RightStochasticLinear, PerronFrobeniusLinear,
        SymmetricLinear, SkewSymmetricLinear, SplitLinear, StableSplitLinear,
        SpectralLinear, SVDLinear]
square_maps = {SymmetricLinear, SkewSymmetricLinear, DoublyStochasticLinear}

if __name__ == '__main__':
    """
    Tests
    """
    square = torch.rand(7, 7)
    long = torch.rand(3, 7)
    tall = torch.rand(7, 3)

    for linear in maps:
        print(linear)
        map = linear(7, 7)
        x = map(square)
        assert (x.shape[0], x.shape[1]) == (7, 7)
        x = map(long)
        assert (x.shape[0], x.shape[1]) == (3, 7)

    for linear in set(maps) - square_maps:
        print(linear)
        map = linear(3, 5)
        x = map(tall)
        assert (x.shape[0], x.shape[1]) == (7, 5)
        map = linear(7, 3)
        x = map(long)
        assert (x.shape[0], x.shape[1]) == (3, 3)









