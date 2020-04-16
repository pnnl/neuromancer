import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, insize, outsize, bias=False):
        super().__init__()
        self.linear = nn.Linear(insize, outsize, bias=bias)

    def effective_W(self):
        return self.linear.weight.T

    def forward(self, x):
        return self.linear(x)


class NonnegativeLinear(nn.Module):
    def __init__(self, insize, outsize, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(insize, outsize))
        self.bias = nn.Parameter(torch.zeros(1, outsize), requires_grad=not bias)

    def effective_W(self):
        w_LB = F.relu(self.weight)
        return w_LB

    def forward(self, x):
        return torch.matmul(x, self.effective_W()) + self.bias


class LassoLinear(nn.Module):
    """
    Use this for sparse id of non-linear dynamics (SINDy)
    """
    def __init__(self, insize, outsize, bias=False, gamma=1.0):
        self.u_param = nn.Parameter(torch.rand(insize, outsize))
        self.v_param = nn.Parameter(torch.rand(insize, outsize))
        self.bias = nn.Parameter(torch.zeros(1, outsize), requires_grad=not bias)
        self.gamma = 1.0

    def effective_W(self):
        return F.relu(self.u_param) - F.relu(self.v_param)

    def regularization_error(self):
        return self.gamma*(self.u_param.norm(p=1) + self.v_param.norm(p=1))

    def forward(self, x):
        return torch.matmul(x, self.effective_W) + self.bias

class RightStochasticLinear(nn.Module):
    """
    A right stochastic matrix is a real square matrix, with each row summing to 1.
    https://en.wikipedia.org/wiki/Stochastic_matrix
    """
    def __init__(self, insize, outsize, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(insize, outsize)) 
        self.do_bias = bias
        self.bias = nn.Parameter(torch.zeros(1, outsize), requires_grad=not bias)
        
    def effective_W(self):
        w_rStochastic = F.softmax(self.weight, dim=1)
        return w_rStochastic      
        
    def forward(self, x):
        return torch.matmul(x, self.effective_W) + self.bias

class LeftStochasticLinear(nn.Module):
    """
    A left stochastic matrix is a real square matrix, with each column summing to 1.
    https://en.wikipedia.org/wiki/Stochastic_matrix
    """
    def __init__(self, insize, outsize, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(insize, outsize)) 
        self.do_bias = bias
        self.bias = nn.Parameter(torch.zeros(1, outsize), requires_grad=not bias)
        
    def effective_W(self):
        w_rStochastic = F.softmax(self.weight.T, dim=1)
        return w_rStochastic.T      
        
    def forward(self, x):
        return torch.matmul(x, self.effective_W) + self.bias


class DoublyStochasticLinear(nn.Module):
    """
    A left stochastic matrix is a real square matrix, with each column summing to 1.
    https://en.wikipedia.org/wiki/Doubly_stochastic_matrix
    """
    pass


class PerronFrobeniusLinear(nn.Module):

    def __init__(self, insize, outsize, bias=False, sigma_min=0.95, sigma_max=1.0, init='basic'):
        """
        Perron-Frobenius theorem based regularization of matrix

        :param insize: (int) Dimension of input vectors
        :param outsize: (int) Dimension of output vectors
        :param bias: (bool) Whether to add bias to linear transform
        :param sigma_min: (float) maximum allowed value of dominant eigenvalue
        :param sigma_max: (float)  minimum allowed value of dominant eigenvalue
        :param init: (str) 'init' or 'basic'. Whether to use identity initialization for hidden transition
        """
        super().__init__()
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


class SymmetricLinear(nn.Module):
    """
    symmetric matrix A (effective_W) is a square matrix that is equal to its transpose.
    A = A^T
    https://en.wikipedia.org/wiki/Symmetric_matrix
    """
    def __init__(self, insize, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(insize, insize)) # identity matrix with small noise    
        self.do_bias = bias
        self.bias = nn.Parameter(torch.zeros(1, insize), requires_grad=not bias)

    def effective_W(self):
         sym_weight = (self.weight + torch.t(self.weight))/2
         return sym_weight
        
    def forward(self, x):  
        return torch.matmul(x, self.effective_W()) + self.bias
 
class SkewSymmetricLinear(nn.Module):
    """
    skew-symmetric (or antisymmetric) matrix A (effective_W) is a square matrix whose transpose equals its negative. 
    A = -A^T
    https://en.wikipedia.org/wiki/Skew-symmetric_matrix
    """
    def __init__(self, insize, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(insize, insize)) # identity matrix with small noise    
        self.do_bias = bias
        self.bias = nn.Parameter(torch.zeros(1, insize), requires_grad=not bias)

    def effective_W(self):
         skewsym_weight = self.weight.triu() - self.weight.triu().T
         return skewsym_weight
        
    def forward(self, x):  
        return torch.matmul(x, self.effective_W()) + self.bias
 
class HamiltonianLinear(nn.Module):  
    """
    https://en.wikipedia.org/wiki/Hamiltonian_matrix
    """
    pass     

class SympleticLinear(nn.Module):  
    """
    https://en.wikipedia.org/wiki/Symplectic_matrix
    """
    pass     
    
    

class InvertibleLinear(nn.Module):
    """
    an invertible matrix is a square matrix A (weight), for which following holds:
    A*B = B*A = I
    return transformation: A*x
    invertibility error to be penalized in the loss: err = ||I - A*B||^2 + ||I - B*A ||^2
    https://en.wikipedia.org/wiki/Invertible_matrix
    """

    def __init__(self, insize):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(insize, insize) + 0.01 * torch.randn(insize, insize))  # identity matrix with small noise
        self.B = nn.Parameter(torch.eye(insize, insize) + 0.01 * torch.randn(insize, insize))  # identity matrix with small noise
        self.insize = insize

    def forward(self):
        OrthoError = torch.norm(torch.norm(torch.eye(self.insize).to(self.weight.device) - torch.mm(self.weight, self.B), 2) + torch.norm(
            torch.eye(self.nx).to(self.weight.device) - torch.mm(self.B, self.weight), 2), 2)
        return OrthoError


class OrthogonalLinear(nn.Module):
    """
    an orthogonal matrix Q (weight) is a square matrix whose columns and rows are orthogonal unit vectors (orthonormal vectors).
    Q*Q^T = Q^T*Q = I
    return transformation: Q*x
    orthogonality error to be penalized in the loss: err = ||I - Q*Q^T||^2 + ||I - Q^T*Q||^2
    https://en.wikipedia.org/wiki/Orthogonal_matrix
    """

    def __init__(self, nx):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(nx, nx) + 0.01 * torch.randn(nx, nx))  # identity matrix with small noise
        self.nx = nx

    def forward(self):
        OrthoError = torch.norm(torch.norm(torch.eye(self.nx).to(self.weight.device) - torch.mm(self.weight, torch.t(self.weight)), 2) + torch.norm(
            torch.eye(self.nx).to(self.weight.device) - torch.mm(torch.t(self.weight), self.weight), 2), 2)
        return OrthoError


class SVDLinear(nn.Module):
    def __init__(self, insize, outsize, bias=False, sigma_min=0.1, sigma_max=1):
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
        super().__init__()
        self.U = OrthogonalLinear(insize)
        self.V = OrthogonalLinear(outsize)
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
        Sigma_bounded = torch.eye(self.insize, self.outsize).to(self.sigma.device) * sigma_clapmed
        w_svd = torch.mm(self.U.weight, torch.mm(Sigma_bounded, self.V.weight))
        return w_svd

    @property
    def spectral_error(self):
        return self.U() + self.V()  # error of spectral regularization

    def forward(self, x):
        return torch.matmul(x, self.effective_W()) + self.bias


class SpectralLinear(nn.Module):
    """
    Translated from tensorflow code: https://github.com/zhangjiong724/spectral-RNN/blob/master/code/spectral_rnn.py
    SVD paramaterized linear map of form U\SigmaV. Singular values can be constrained to a range
    """

    def __init__(self, insize, outsize, bias=False,
                 n_U_reflectors=20, n_V_reflectors=20, sigma_min=0.6, sigma_max=1.0):
        """

        :param insize: (int) Dimension of input vectors
        :param outsize: (int) Dimension of output vectors
        :param reflector_size: (int) It looks like this should effectively constrain the rank of the matrix (bonus!)
        :param bias: (bool) whether to add a bias term.
        :param sig_mean: initial and "mean" value of singular values, usually set to 1.0
        :param r: singular margin, the allowed margin for singular values
        """
        super().__init__()
        self.n_U_reflectors, self.n_V_reflectors = n_U_reflectors, n_V_reflectors
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
        return self.forward(torch.eye(self.insize).to(self.p.device))

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


