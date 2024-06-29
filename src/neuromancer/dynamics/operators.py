from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import torch
from torch import nn

if TYPE_CHECKING:
    from neuromancer.modules.blocks import Block

TDeepONet = TypeVar("TDeepONet", bound="DeepONet")


class DeepONet(nn.Module):
    """Deep Operator Network."""

    def __init__(
            self: TDeepONet,
            branch_net: Block,
            trunk_net: Block,
            bias: bool = True,
    ) -> None:
        """Deep Operator Network.

        :param branch_net: (Block) Branch network
        :param trunk_net: (Block) Trunk network
        :param bias: (bool) Whether to use bias or not
        """
        super().__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=not bias)

    @staticmethod
    def transpose_branch_inputs(branch_inputs: torch.Tensor) -> torch.Tensor:
        """Transpose branch inputs.

        :param branch_inputs: (torch.Tensor, shape=[Nu, Nsamples])
        :return: (torch.Tensor, shape=[Nsamples, Nu])
        """
        transposed_branch_inputs = torch.transpose(branch_inputs, 0, 1)
        return transposed_branch_inputs

    def forward(self: TDeepONet, branch_inputs: torch.Tensor,
                trunk_inputs: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward propagation.

        :param branch_inputs: (torch.Tensor, shape=[Nu, Nsamples])
        :param trunk_inputs: (torch.Tensor, shape=[Nu, in_size_trunk])
        :return:
            output: (torch.Tensor, shape=[Nsamples, Nu]),
            branch_output: (torch.Tensor, shape=[Nsamples, interact_size]),
            trunk_output: (torch.Tensor, shape=[Nu, interact_size])
        """
        branch_output = self.branch_net(self.transpose_branch_inputs(branch_inputs))
        trunk_output = self.trunk_net(trunk_inputs)
        output = torch.matmul(branch_output, trunk_output.T) + self.bias
        # return branch_output and trunk_output as well for control use cases
        return output, branch_output, trunk_output


import abc

import numpy as np
from scipy import interpolate
from sklearn import gaussian_process as gp


class FunctionSpace(abc.ABC):
    """Function space base class.

    Example:
    -------
        .. code-block:: python

            space = dde.data.GRF()
            feats = space.random(10)
            xs = np.linspace(0, 1, num=100)[:, None]
            y = space.eval_batch(feats, xs)

    """

    @abc.abstractmethod
    def random(self, size):
        """Generate feature vectors of random functions.

        Args:
        ----
            size (int): The number of random functions to generate.

        Returns:
        -------
            A NumPy array of shape (`size`, n_features).

        """

    @abc.abstractmethod
    def eval_one(self, feature, x):
        """Evaluate the function at one point.

        Args:
        ----
            feature: The feature vector of the function to be evaluated.
            x: The point to be evaluated.

        Returns:
        -------
            float: The function value at `x`.

        """

    @abc.abstractmethod
    def eval_batch(self, features, xs):
        """Evaluate a list of functions at a list of points.

        Args:
        ----
            features: A NumPy array of shape (n_functions, n_features). A list of the
                feature vectors of the functions to be evaluated.
            xs: A NumPy array of shape (n_points, dim). A list of points to be
                evaluated.

        Returns:
        -------
            A NumPy array of shape (n_functions, n_points). The values of
            different functions at different points.

        """

class GRF(FunctionSpace):
    """Gaussian random field (Gaussian process) in 1D.

    The random sampling algorithm is based on Cholesky decomposition of the covariance
    matrix.

    Args:
    ----
        T (float): `T` > 0. The domain is [0, `T`].
        kernel (str): Name of the kernel function. "RBF" (radial-basis function kernel,
            squared-exponential kernel, Gaussian kernel), "AE"
            (absolute exponential kernel), or "ExpSineSquared" (Exp-Sine-Squared kernel,
            periodic kernel).
        length_scale (float): The length scale of the kernel.
        N (int): The size of the covariance matrix.
        interp (str): The interpolation to interpolate the random function. "linear",
            "quadratic", or "cubic".

    """

    def __init__(self, T=1, kernel="RBF", length_scale=1, N=1000, interp="cubic"):
        self.N = N
        self.interp = interp
        self.x = np.linspace(0, T, num=N)[:, None]
        if kernel == "RBF":
            K = gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            K = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        elif kernel == "ExpSineSquared":
            K = gp.kernels.ExpSineSquared(length_scale=length_scale, periodicity=T)
        self.K = K(self.x)
        self.L = np.linalg.cholesky(self.K + 1e-13 * np.eye(self.N))

    def random(self, size):
        u = np.random.randn(self.N, size)
        return np.dot(self.L, u).T

    def eval_one(self, feature, x):
        if self.interp == "linear":
            return np.interp(x, np.ravel(self.x), feature)
        f = interpolate.interp1d(
            np.ravel(self.x), feature, kind=self.interp, copy=False, assume_sorted=True,
        )
        return f(x)

    def eval_batch(self, features, xs):
        if self.interp == "linear":
            return np.vstack([np.interp(xs, np.ravel(self.x), y).T for y in features])
        res = map(
            lambda y: interpolate.interp1d(
                np.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True,
            )(xs).T,
            features,
        )
        return np.vstack(list(res)).astype(config.real(np))

grf = GRF(N=100)
data_maybe = grf.random(size=150)
