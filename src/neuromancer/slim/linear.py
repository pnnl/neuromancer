"""
Structured linear maps which are drop in replacements for torch.nn.Linear


.. todo::

    + Generalize to batch matrix multiplication for arbitrary N-dimensional tensors
    + Additional linear parametrizations:

        - Strictly diagonally dominant matrix is non-singular:
            + https://en.wikipedia.org/wiki/Diagonally_dominant_matrix
        - Doubly stochastic matrix:
            + https://en.wikipedia.org/wiki/Doubly_stochastic_matrix
            + https://github.com/btaba/sinkhorn_knopp
            + https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
        - Hamiltonian matrix:
            + https://en.wikipedia.org/wiki/Hamiltonian_matrix
        - Regular split: :math:`A = B − C` is a regular splitting of :math:`A` if :math:`B^{−1} ≥ 0` and :math:`C ≥ 0`:
            + https://en.wikipedia.org/wiki/Matrix_splitting

Pytorch weight initializations used in this module:

+ torch.nn.init.xavier_normal_(tensor, gain=1.0)
+ torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
+ torch.nn.init.orthogonal_(tensor, gain=1)
+ torch.nn.init.sparse_(tensor, sparsity, std=0.01)
"""

from abc import ABC, abstractmethod
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuromancer.slim.butterfly import Butterfly


class LinearBase(nn.Module, ABC):
    """
    Base class defining linear map interface.
    """

    def __init__(self, insize, outsize, bias=False, provide_weights=True):
        """

        :param insize: (int) Input dimensionality
        :param outsize: (int) Output dimensionality
        :param bias: (bool) Whether to use affine or linear map
        """
        super().__init__()
        self.in_features, self.out_features = insize, outsize
        self.use_bias = bias 
        if bias: 
            bound = 1 / math.sqrt(insize)
            self.bias = nn.Parameter(torch.zeros(1, outsize), requires_grad=bias)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else: 
            self.register_parameter('bias', None)

        if provide_weights:
            self.weight = nn.Parameter(torch.Tensor(insize, outsize))
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @property
    def device(self):
        return next(self.parameters()).device

    def reg_error(self):
        """
        Regularization error associated with linear map parametrization.

        :return: (torch.float)

        """
        return torch.tensor(0.0).to(self.device)

    def eig(self):
        """
        Returns the eigenvalues (optionally eigenvectors) of the linear map used in matrix multiplication.

        :return: (torch.Tensor) Vector of eigenvalues, optionally a tuple including a matrix of eigenvectors.
        """
        return torch.linalg.eig(self.effective_W())

    @abstractmethod
    def effective_W(self):
        """
        The matrix used in the equivalent matrix multiplication for the parametrization

        :return: (torch.Tensor, shape=[insize, outsize]) Matrix used in matrix multiply

        """
        pass

    def forward(self, x):
        """
        :param x: (torch.Tensor, shape=[batchsize, in_features])
        :return: (torch.Tensor, shape=[batchsize, out_features])
        """
        if self.use_bias: 
            return torch.matmul(x, self.effective_W()) + self.bias
        else: 
            return torch.matmul(x, self.effective_W())

class Linear(LinearBase):
    """
    Wrapper for torch.nn.Linear with additional slim methods returning matrix,
    eigenvectors, eigenvalues and regularization error.
    """
    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize, bias=bias, provide_weights=False)
        self.linear = nn.Linear(insize, outsize, bias=bias)
        self.weight = self.linear.weight
        self.bias = self.linear.bias

    def effective_W(self):
        return self.linear.weight.T

    def forward(self, x):
        return self.linear(x)


class L0Linear(LinearBase):
    """
    Implementation of L0 regularization for the input units of a fully connected layer

    + Reference implementation: https://github.com/AMLab-Amsterdam/L0_regularization/blob/master/l0_layers.py
    + Paper: https://arxiv.org/pdf/1712.01312.pdf

    .. note::
        This implementation may need to be adjusted as there is the same sampling for each input
        in the minibatch which may inhibit convergence. Also, there will be a different sampling
        for each call during training so it may cause issues included in a layer for a recurrent
        computation (fx in state space model).

    """
    def __init__(self, insize, outsize, bias=True, weight_decay=1.0,
                 droprate_init=0.5, temperature=2./3., lamda=1.0):
        """
        :param weight_decay: Strength of the L2 penalty
        :param droprate_init: Dropout rate that the L0 gates will be initialized to
        :param temperature: Temperature of the concrete distribution
        :param lamba: Strength of the L0 penalty
        """
        super().__init__(insize, outsize, bias=bias, provide_weights=True)
        self.in_features = insize
        self.out_features = outsize
        self.use_bias = bias
        self.register_buffer('prior_prec', torch.tensor(weight_decay))
        self.qz_loga = nn.Parameter(torch.Tensor(insize, outsize))
        self.register_buffer('temperature', torch.tensor(temperature))
        self.register_buffer('droprate_init', torch.tensor(droprate_init) if droprate_init != 0. else torch.tensor(0.5))
        self.register_buffer('lamda', torch.tensor(lamda))
        self.register_buffer('limit_a', torch.tensor(-.1))
        self.register_buffer('limit_b', torch.tensor(1.1))
        self.register_buffer('epsilon', torch.tensor(1e-6))
        self.qz_loga.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - self.limit_a) / (self.limit_b - self.limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - self.qz_loga).clamp(min=self.epsilon, max=1 - self.epsilon)

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = torch.sigmoid((torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature)
        return y * (self.limit_b - self.limit_a) + self.limit_a

    def reg_error(self):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        logpw_col = torch.sum(- (.5 * self.prior_prec * self.weight.pow(2)) - self.lamda, 1)
        logpw = torch.sum((1 - self.cdf_qz(0)) * logpw_col.unsqueeze(1))
        logpb = 0 if not self.use_bias else - torch.sum(.5 * self.prior_prec * self.bias.pow(2))
        return logpw + logpb

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = torch.FloatTensor(*size).uniform_(self.epsilon, 1-self.epsilon).to(self.weight.device)
        eps = torch.autograd.Variable(eps)
        return eps

    def effective_W(self):
        if self.training:
            z = self.quantile_concrete(self.get_eps([self.in_features, self.out_features]))
            mask = F.hardtanh(z, min_val=0, max_val=1)
        else:
            pi = F.sigmoid(self.qz_loga)
            mask = F.hardtanh(pi * (self.limit_b - self.limit_a) + self.limit_a, min_val=0, max_val=1)
        return mask * self.weight


class ButterflyLinear(LinearBase):
    """
    Sparse structured linear maps from: https://github.com/HazyResearch/learning-circuits
    """
    def __init__(self, insize, outsize, bias=False,
                 complex=False, tied_weight=True, increasing_stride=True, ortho_init=False,
                 **kwargs):
        super().__init__(insize, outsize, bias=bias, provide_weights=False)
        self.linmap = Butterfly(insize, outsize, bias=bias, complex=complex,
                                tied_weight=tied_weight, increasing_stride=increasing_stride,
                                ortho_init=ortho_init)

    def effective_W(self):
        return self.linmap(torch.eye(self.in_features).to(self.linmap.twiddle.device))

    def forward(self, x):
        return self.linmap(x)


class SquareLinear(LinearBase, ABC):
    """
    Base class for linear map parametrizations that assume a square matrix.
    """
    def __init__(self, insize, outsize, bias=False, provide_weights=True, **kwargs):
        assert insize == outsize, f'Map must be square. insize={insize} and outsize={outsize}'
        super().__init__(insize, outsize, bias=bias, provide_weights=provide_weights)

    @abstractmethod
    def effective_W(self):
        pass


class IdentityInitLinear(Linear):
    """
    Linear map initialized to Identity matrix.
    """
    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize, bias=bias)
        self.linear = nn.Linear(insize, outsize, bias=bias)
        self.weight = self.linear.weight
        self.bias = self.linear.bias
        torch.nn.init.eye_(self.weight)
        if bias:
            torch.nn.init.zeros_(self.bias)


class IdentityLinear(IdentityInitLinear):
    """
    Identity operation compatible with all LinearBase functionality.
    """
    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize, bias=bias)
        self.linear.requires_grad_(False)


class NonNegativeLinear(LinearBase):
    """
    Positive parametrization of linear map via Relu.
    """
    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize, bias=bias, provide_weights=True)
        self.weight = nn.Parameter(torch.abs(self.weight)*0.1)

    def effective_W(self):
        return F.relu(self.weight)


class PSDLinear(SquareLinear):
    """
    Symmetric Positive semi-definite matrix.
    """
    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize, bias=bias, provide_weights=True)

    def effective_W(self):
        return torch.matmul(self.weight.T, self.weight)


class IdentityGradReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input. Here we are just passing through the previous gradient since we want
        the gradient for this max operation to be gradient of identity.
        """
        return grad_output


class LassoLinearRELU(LinearBase):
    """
    From https://leon.bottou.org/publications/pdf/compstat-2010.pdf
    """

    def __init__(self, insize, outsize, bias=False, gamma=1.0, **kwargs):
        super().__init__(insize, outsize, bias=bias, provide_weights=False)
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


class LassoLinear(LinearBase):
    """
    From https://leon.bottou.org/publications/pdf/compstat-2010.pdf
    """

    def __init__(self, insize, outsize, bias=False, gamma=1.0, **kwargs):
        super().__init__(insize, outsize, bias=bias, provide_weights=False)
        u = torch.empty(insize, outsize)
        torch.nn.init.kaiming_normal_(u)
        self.u_param = nn.Parameter(torch.abs(u) / 2.0)
        v = torch.empty(insize, outsize)
        torch.nn.init.kaiming_normal_(v)
        self.v_param = nn.Parameter(torch.abs(v) / 2.0)
        self.gamma = gamma

    def effective_W(self):
        # Thresholding for sparsity
        return self.u_param - self.v_param

    def reg_error(self):
        # shrinkage
        return self.gamma * self.effective_W().norm(p=1)

    def forward(self, x):
        self.v_param.data = F.relu(self.v_param.data)
        self.u_param.data = F.relu(self.u_param.data)
        return super().forward(x)


class RightStochasticLinear(LinearBase):
    """
    A right stochastic matrix is a real square matrix, with each row summing to 1.

    + https://en.wikipedia.org/wiki/Stochastic_matrix
    """

    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize, bias=bias, provide_weights=True)

    def effective_W(self):
        return F.softmax(self.weight, dim=1)


class LeftStochasticLinear(LinearBase):
    """
    A left stochastic matrix is a real square matrix, with each column summing to 1.

    + https://en.wikipedia.org/wiki/Stochastic_matrix
    """

    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize, bias=bias, provide_weights=True)

    def effective_W(self):
        return F.softmax(self.weight, dim=0)


class PerronFrobeniusLinear(LinearBase):

    def __init__(self, insize, outsize, bias=False, sigma_min=0.8, sigma_max=1.0, **kwargs):
        """
        Perron-Frobenius theorem based regularization of matrix rows sum to in between sigma_min and sigma max.

        + See https://arxiv.org/abs/2004.10883 for extensive description.

        :param insize: (int) Dimension of input vectors
        :param outsize: (int) Dimension of output vectors
        :param bias: (bool) Whether to add bias to linear transform
        :param sigma_min: (float) maximum allowed value of dominant eigenvalue
        :param sigma_max: (float)  minimum allowed value of dominant eigenvalue
        """
        super().__init__(insize, outsize, bias=bias, provide_weights=True)
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
    Symmetric matrix :math:`A` (effective_W) is a square matrix that is equal to its transpose. :math:`A = A^T`

    + https://en.wikipedia.org/wiki/Symmetric_matrix
    """

    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize, bias=bias, provide_weights=True)

    def effective_W(self):
        return (self.weight + torch.t(self.weight)) / 2


class SkewSymmetricLinear(SquareLinear):
    """
    Skew-symmetric (or antisymmetric) matrix :math:`A` (effective_W) is a square matrix whose transpose equals its negative.
    :math:`A = -A^T`

    + https://en.wikipedia.org/wiki/Skew-symmetric_matrix
    """

    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize, bias=bias, provide_weights=True)

    def effective_W(self):
        return self.weight.triu() - self.weight.triu().T


class DampedSkewSymmetricLinear(SkewSymmetricLinear):
    """
    Skew-symmetric linear map with damping.

    + https://en.wikipedia.org/wiki/Skew-symmetric_matrix
    """

    def __init__(self, insize, outsize, bias=False, sigma_min=0.1, sigma_max=0.5, **kwargs):
        super().__init__(insize, outsize, bias=bias, provide_weights=True)
        self.eye = nn.Parameter(torch.eye(insize, outsize), requires_grad=False)
        self.gamma = nn.Parameter(sigma_min + (sigma_max-sigma_min) * torch.rand(1, 1))

    def effective_W(self):
        return super().effective_W() + self.gamma * self.eye


class SplitLinear(LinearBase):
    """
    :math:`A = B − C`, with :math:`B ≥ 0` and :math:`C ≥ 0`.

    + https://en.wikipedia.org/wiki/Matrix_splitting
    """

    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize, bias=bias, provide_weights=False)
        self.B = NonNegativeLinear(insize, outsize, bias)
        self.C = NonNegativeLinear(insize, outsize, bias)

    def effective_W(self):
        A = self.B.effective_W() - self.C.effective_W()
        return A


class StableSplitLinear(LinearBase):
    """
    :math:`A = B − C`, with stable `B` and stable `C`

    + https://en.wikipedia.org/wiki/Matrix_splitting
    """

    def __init__(self, insize, outsize, bias=False, sigma_min=0.1, sigma_max=1.0, **kwargs):
        super().__init__(insize, outsize, bias=bias, provide_weights=False)
        self.B = PerronFrobeniusLinear(insize, outsize, bias, sigma_max, sigma_max)
        self.C = PerronFrobeniusLinear(insize, outsize, bias, 0, sigma_max - sigma_min)

    def effective_W(self):
        A = self.B.effective_W() - self.C.effective_W()
        return A


class SVDLinear(LinearBase):
    """
    Linear map with constrained eigenvalues via approximate SVD factorization.
    Soft SVD based regularization of matrix :math:`A`.
    :math:`A = U \Sigma V`.
    :math:`U,V` are unitary matrices (orthogonal for real matrices :math:`A`).
    :math:`\Sigma` is a diagonal matrix of singular values (square roots of eigenvalues).

    + https://arxiv.org/abs/2101.01864

    This below paper uses the same factorization and orthogonality constraint as implemented here
    but enforces a low rank prior on the map by introducing a sparse prior on the singular values:

    + https://openaccess.thecvf.com/content_CVPRW_2020/papers/w40/Yang_Learning_Low-Rank_Deep_Neural_Networks_via_Singular_Vector_Orthogonality_Regularization_CVPRW_2020_paper.pdf

    Also a similar regularization on the factors as to our implementation:

    + https://pdfs.semanticscholar.org/78b2/9eba4d6c836483c0aa67d637205e95223ae4.pdf
    """
    def __init__(self, insize, outsize, bias=False, sigma_min=0.1, sigma_max=1.0, **kwargs):
        """
        :param sigma_min: (int) Minimum singular value.
        :param sigma_max: (int) Maximum singular value.
        """
        super().__init__(insize, outsize, bias=bias, provide_weights=False)
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
        """
        Regularization error enforces orthogonality constraint for matrix factors
        """
        return self.orthogonal_error(self.U) + self.orthogonal_error(self.V)

    def effective_W(self):
        """

        :return: Matrix for linear transformation with dominant eigenvalue between sigma_max and sigma_min
        """
        sigma_clapmed = self.sigma_max - (self.sigma_max - self.sigma_min) * torch.sigmoid(self.sigma)
        Sigma_bounded = torch.eye(self.in_features, self.out_features).to(self.sigma.device) * sigma_clapmed
        w_svd = torch.mm(self.U, torch.mm(Sigma_bounded, self.V))
        return w_svd


class SVDLinearLearnBounds(SVDLinear):
    def __init__(self, insize, outsize, bias=False, sigma_min=0.1, sigma_max=1.0, **kwargs):
        """
        Parametrizes bounds on singular value which are learned with other parameters via gradient descent.
        """
        super().__init__(insize, outsize, bias=bias, sigma_min=sigma_min, sigma_max=sigma_max)
        self.sigma_min = nn.Parameter(torch.tensor(sigma_min))
        self.sigma_max = nn.Parameter(torch.tensor(sigma_max))


class SymmetricSVDLinear(SVDLinear):
    """
    :math:`U = V`
    """
    def __init__(self, insize, outsize, bias=False, sigma_min=0.1, sigma_max=1.0, **kwargs):
        super().__init__(insize, outsize, bias=bias, sigma_min=sigma_min, sigma_max=sigma_max)
        self.U = self.V


def Hprod(x, u, k):
    """
    Helper function for computing matrix multiply via householder reflection representation.
    :param x: (torch.Tensor shape=[batchsize, dimension])
    :param u: (torch.Tensor shape=[dimension])
    :param k: (int)
    :return: (torch.Tensor shape=[batchsize, dimension])
    """
    alpha = 2 * torch.matmul(x[:, -k:], u[-k:]) / (u[-k:] * u[-k:]).sum()
    if k < x.shape[1]:
        return torch.cat([x[:, :-k], x[:, -k:] - torch.matmul(alpha.view(-1, 1), u[-k:].view(1, -1))],
                         dim=1)  # Subtract outer product
    else:
        return x[:, -k:] - torch.matmul(alpha.view(-1, 1), u[-k:].view(1, -1))


class OrthogonalLinear(SquareLinear):
    """
    Orthogonal parametrization via householder reflection

    + https://arxiv.org/abs/1612.00188
    """
    def __init__(self, insize, outsize, bias=False, **kwargs):
        super().__init__(insize, outsize, bias=bias, provide_weights=False)
        self.U = nn.Parameter(torch.triu(torch.randn(insize, insize)))

    def effective_W(self):
        return self.forward(torch.eye(self.in_features).to(self.U.device))

    def forward(self, x):
        for i in range(0, self.in_features):
            x = Hprod(x, self.U[i], self.in_features - i)
        return x + self.bias


class SchurDecompositionLinear(SquareLinear):
    """
    + https://papers.nips.cc/paper/9513-non-normal-recurrent-neural-network-nnrnn-learning-long-time-dependencies-while-improving-expressivity-with-transient-dynamics.pdf
    """
    def __init__(self, insize, outsize, bias=False, l2=1e-2, **kwargs):
        super().__init__(insize, outsize, bias=bias, provide_weights=False)
        assert insize % 2 == 0, 'Insize must be divisible by 2.'
        self.P = OrthogonalLinear(insize, insize)
        self.theta = nn.Parameter(2*math.pi*torch.rand([insize//2]))
        self.gamma = nn.Parameter(torch.ones([insize//2]))
        self.T = self.build_T(torch.zeros(insize, insize))
        self.l2 = l2

    def build_T(self, T):
        for k, (theta, gamma) in enumerate(zip(self.theta, self.gamma)):
            rk = gamma * torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                       [torch.sin(theta), torch.cos(theta)]])
            T[2*k:2*k+2, 2*k:2*k+2] = rk
        return T

    def reg_error(self):
        return self.l2*F.mse_loss(torch.ones(self.insize/2), self.gamma)

    def effective_W(self):
        return self.P(self.T) @ self.P.effective_W().T


class SpectralLinear(LinearBase):
    """
    SVD paramaterized linear map of form :math:`U \Sigma V` via Householder reflection.
    Singular values can be constrained to a range.
    Translated from tensorflow code:

    + https://github.com/zhangjiong724/spectral-RNN/blob/master/code/spectral_rnn.py
    """

    def __init__(self, insize, outsize, bias=False,
                 n_U_reflectors=None, n_V_reflectors=None,
                 sigma_min=0.1, sigma_max=1.0, **kwargs):
        """

        :param n_U_reflectors: (int) It looks like this should effectively constrain the rank of the matrix
        :param n_V_reflectors: (int) It looks like this should effectively constrain the rank of the matrix
        :param sigma_min: min value of singular values
        :param sigma_max: max value of singular values
        """
        super().__init__(insize, outsize, bias=bias, provide_weights=False)
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
        assert x.shape[1] == self.in_features, f'x.shape: {x.shape}, in_features: {self.in_features}'
        for i in range(0, self.n_U_reflectors):
            x = Hprod(x, self.U[i], self.in_features - i)
        return x

    def Vmultiply(self, x):
        assert x.shape[1] == self.out_features
        for i in range(self.n_V_reflectors - 1, -1, -1):
            x = Hprod(x, self.V[i], self.out_features - i)
        return x

    def effective_W(self):
        return self.forward(torch.eye(self.in_features).to(self.p.device))

    def forward(self, x):
        x = self.Umultiply(x)
        x = torch.matmul(x, self.Sigma())
        x = self.Vmultiply(x)
        return x + self.bias


class SymmetricSpectralLinear(SpectralLinear):
    """
    :math:`U = V`
    """

    def __init__(self, insize, outsize, bias=False, n_reflectors=None, sigma_min=0.1, sigma_max=1.0, **kwargs):
        super().__init__(insize, outsize, bias=bias,
                         n_U_reflectors=n_reflectors, n_V_reflectors=n_reflectors,
                         sigma_min=sigma_min, sigma_max=sigma_max, **kwargs)
        self.U = self.V


class SymplecticLinear(LinearBase):
    """
    + https://en.wikipedia.org/wiki/Symplectic_matrix
    + https://arxiv.org/abs/1705.03341
    """

    def __init__(self, insize, outsize, bias=False, **kwargs):
        assert insize % 2 == 0, 'Symplectic Matrix must have even dimensions'
        super().__init__(insize, outsize, bias=bias, provide_weights=False)
        self.weight = torch.nn.Parameter(torch.empty(int(insize/2), int(outsize/2)))
        torch.nn.init.kaiming_normal_(self.weight)
        self.weight = nn.Parameter(self.weight)

    def effective_W(self):
        return torch.cat([torch.cat([torch.zeros(self.in_features // 2, self.in_features // 2), self.weight], dim=1),
                          torch.cat([-1 * self.weight.T, torch.zeros(self.in_features // 2, self.in_features // 2)], dim=1)])


class GershgorinLinear(SquareLinear):
    """
    Uses Gershgorin Disc parametrization to constrain eigenvalues of the matrix. See:

    + https://arxiv.org/abs/2011.13492
    """
    def __init__(self, insize, outsize, bias=False, sigma_min=0.0, sigma_max=1.0, real=True, **kwargs):
        super().__init__(insize, outsize, bias=bias, provide_weights=False)
        self.real = real

        self.mean = (sigma_min + sigma_max)/2.0
        self.radius = (sigma_min - sigma_max)/2.0

        self.nonident = ~(torch.eye(insize, insize).type(torch.bool))
        self.w = nn.Parameter(torch.rand(insize, insize))*self.nonident
        self.diag = torch.randn(insize)

    def effective_W(self):
        if self.real:
            # make weights symmetric
            w = self.w + self.w.T
        else:
            w = self.w
        # Set diagonals to be centered in eigenvalue range with offset bounded (0, .5*radius)
        eW = torch.diag(self.radius * (torch.sigmoid(self.diag) - 0.5) + self.mean)
        # Perform softmax on off diagonal elements, then scale so sum is equal to .5 radius
        w = F.softmax(w[self.nonident], dim=-1).view(self.in_features, self.in_features-1)*(self.radius/2.0)
        # Get normalized upper triangular elements and put them in effective weights
        idxs = torch.triu(torch.ones(self.in_features, self.in_features - 1)) == 1
        uppervec = w[idxs]
        eW[torch.triu(torch.ones(self.in_features, self.in_features), diagonal=1) == 1] = uppervec
        # Get normalized lower triangular elements and put them in effective weights
        lowervec = w[~idxs]
        eW[torch.tril(torch.ones(self.in_features, self.in_features), diagonal=-1) == 1] = lowervec
        return eW


class BoundedNormLinear(Linear):
    """
    sigma_min <= ||A||_p <= sigma_max
    p = type of the matrix norm
    sigma_min = minimum allowed value of  eigenvalues
    sigma_max = maximum allowed value of eigenvalues
    """
    def __init__(self, insize, outsize, bias=False,
                 sigma_min=0.1, sigma_max=1.0, p=2, **kwargs):
        super().__init__(insize, outsize, bias=bias, provide_weights=True)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p = p

    def reg_error(self):
        return torch.max(torch.norm(self.weight, self.p) - self.sigma_max, torch.zeros(1)) + \
               torch.max(self.sigma_min - torch.norm(self.weight, self.p), torch.zeros(1))


class TrivialNullSpaceLinear(LinearBase):
    """
    Matrix with trivial null space
    as defined via eq. 2 in https://arxiv.org/abs/1808.00924
    """

    def __init__(self, insize, outsize, bias=False, rank=None, epsilon=0.1, **kwargs):
        super().__init__(insize, outsize, bias=bias, provide_weights=False)
        assert bias == False, f'Map must have zero bias to have trivial null space'
        assert insize <= outsize, f'Map must not decrease the dimension of its input. ' \
                                  f'insize={insize} and outsize={outsize}'
        self.rank = rank if rank is not None else int((insize+1)/2)
        self.Gl1 = nn.Parameter(torch.rand(self.rank, insize))
        self.Gl2 = nn.Parameter(torch.rand(outsize-insize, insize))
        self.epsilon = epsilon
        self.insize = insize
        self.outsize = outsize

    def effective_W(self):
        W_upper = torch.matmul(self.Gl1.T, self.Gl1) + self.epsilon*torch.eye(self.insize)
        return torch.cat([W_upper, self.Gl2]).T











class PowerBoundLinear(LinearBase):
    """
    Linear map with constrained spectral radius via the power method.
    """
    def __init__(self, insize, outsize, max_p = 1,pwr_iters = 200, bias=False, **kwargs):
        """

        :param max_p: Upper bound on the spectral radius.
        :param pwr_iters: Number of power method iterations to use in estimating the spectral radius.

        """
        assert insize == outsize, f'Map must be square. insize={insize} and outsize={outsize}'
        super().__init__(insize, outsize, bias=bias, provide_weights=False)
        self.insize = insize
        self.outsize = outsize
        
        self.W = nn.Parameter(torch.rand(self.insize, self.insize))
        self.max_p = max_p
        self.pwr_iters = pwr_iters


    def eig_v_estimate(self):
        n_iterates = self.pwr_iters
        
        a = torch.normal(0,1,(self.in_features,1))
        a = a.to(self.device)
        b = torch.normal(0,1,(self.in_features,1))
        b = b.to(self.device)

         
        with torch.no_grad():
            for i in range(n_iterates):
                a = torch.mm(self.W,a)
                b = torch.mm(self.W,b)
                a_ib_nrm = torch.sqrt( torch.mm(torch.t(a),a) + torch.mm(torch.t(b),b)   )
                a = (1/a_ib_nrm)*a
                b = (1/a_ib_nrm)*b

        return [a,b]
        

    def reg_error(self):
        """
        Regularization error enforces upper bound on spectral radius
        """
        [a,b] = self.eig_v_estimate()    
        a_ib_nsq = torch.mm(torch.t(a),a) + torch.mm(torch.t(b),b)
        v = torch.mm(torch.t(torch.mm(self.W,a)),torch.mm(self.W,a)) + torch.mm(torch.t(torch.mm(self.W,b)),torch.mm(self.W,b))
        v = v/a_ib_nsq
        v = v[0][0]
        return torch.nn.functional.relu( v - self.max_p  )
        
    

    def effective_W(self):
        return self.W












square_maps = {SymmetricLinear, SkewSymmetricLinear, DampedSkewSymmetricLinear, PSDLinear,
               OrthogonalLinear, SymplecticLinear, SchurDecompositionLinear, SymmetricSpectralLinear,
               SymmetricSVDLinear, GershgorinLinear, PowerBoundLinear}

tall_maps = {TrivialNullSpaceLinear}

maps = {'l0': L0Linear,
        'linear': Linear,
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
        'learnSVD': SVDLinearLearnBounds,
        'orthogonal': OrthogonalLinear,
        'psd': PSDLinear,
        'symplectic': SymplecticLinear,
        'butterfly': ButterflyLinear,
        'schur': SchurDecompositionLinear,
        'identity': IdentityLinear,
        'gershgorin': GershgorinLinear,
        'bounded_Lp_norm': BoundedNormLinear,
        'trivial_nullspace': TrivialNullSpaceLinear,
        'Power_bound': PowerBoundLinear}




















if __name__ == '__main__':
    import sys
    import inspect
    """
    Tests
    """
    print(inspect.getmembers(sys.modules[__name__],
                       lambda member: inspect.isclass(member) and member.__module__ == __name__))

    square = torch.rand(8, 8)
    long = torch.rand(3, 8)
    tall = torch.rand(8, 3)

    for linear in set(list(maps.values())) - square_maps - tall_maps:
        print(linear)
        map = linear(3, 5)
        print(map.reg_error())
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

    for linear in tall_maps:
        print(linear)
        map = linear(3, 5)
        x = map(tall)
        assert (x.shape[0], x.shape[1]) == (8, 5)









