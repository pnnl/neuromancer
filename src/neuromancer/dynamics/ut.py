"""
Continuous-time Unscented Transform for stochastic dynamics.

Learn a neural SDE: System identification of stochastic differential equations. 

Propagates mean and covariance of uncertain states through nonlinear
stochastic ODEs of the form:

    dx = g1(x) dt + sqrt(2 * g2(x)) dW

where g1 is the drift and g2 is the (state-dependent) diffusion coefficient.

Three use cases are supported via factory constructors on :class:`ContinuousUT`:

+-----------+-----------------------------+-----------+----------+------------+
| Use Case  | Description                 | g1        | g2       | Training   |
+===========+=============================+===========+==========+============+
| 1         | Propagate IC uncertainty    | Known     | Zero     | None       |
+-----------+-----------------------------+-----------+----------+------------+
| 2         | Learn diffusion effects     | Known     | Learned  | g2 only    |
+-----------+-----------------------------+-----------+----------+------------+
| 3         | Full system identification  | Learned   | Learned  | g1 + g2    |
+-----------+-----------------------------+-----------+----------+------------+

Example
-------
>>> from neuromancer.dynamics.continuous_ut import ContinuousUT
>>> model = ContinuousUT.from_known_dynamics(my_drift, nx=2, dt=0.01)
>>> result = model.propagate([2.0, 0.0], [0.01, 0.01], n_steps=500)

References
----------
.. [1] S. J. Julier and J. K. Uhlmann, "Unscented filtering and nonlinear
       estimation," Proceedings of the IEEE, 2004.
"""

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neuromancer.constraint import variable
from neuromancer.dataset import DictDataset
from neuromancer.dynamics import integrators
from neuromancer.loggers import BasicLogger
from neuromancer.loss import PenaltyLoss
from neuromancer.modules.blocks import MLP, Block
from neuromancer.problem import Problem
from neuromancer.system import Node, System
from neuromancer.trainer import Trainer

__all__ = [
    # Main API
    "ContinuousUT",
    # Config
    "UseCase",
    "TrainingConfig",
    "UTConfig",
    "NetworkConfig",
    # Result
    "PropagationResult",
    # Blocks
    "ContinuousUTDynamics",
    "ContinuousUTIntegrator",
    "ZeroG2",
    "ConstantG2",
    "LearnableG2",
    "SwitchableG2",
    "DriftWrapper",
    # Data
    "simulate_sde",
    "generate_moment_data",
    "estimate_sigma",
    # Utils
    "ut_weights",
    "cholesky_safe",
    "pack_moments",
    "unpack_moments",
    "moments_to_ode_state",
    "ode_state_to_moments",
]


# ============================================================================
# Enums and configuration dataclasses
# ============================================================================


class UseCase(Enum):
    """Enumeration of supported use cases."""

    PROPAGATE_IC = auto()
    LEARN_DIFFUSION = auto()
    FULL_SYSTEM_ID = auto()


@dataclass
class TrainingConfig:
    """Training hyper-parameters for :class:`ContinuousUT`.

    Attributes
    ----------
    lr : float
        Learning rate for Adam optimiser.
    patience : int
        Early-stopping patience (epochs with no improvement).
    warmup : int
        Number of warm-up epochs before early-stopping activates.
    batch_size : int
        Default batch size when generating data loaders internally.
    mu_loss_weight : float
        Weight for the mean-tracking component of the loss.
    var_loss_weight : float
        Weight for the variance-tracking component of the loss.
    g1_warmup_epochs : int
        Epochs for Phase 1 (drift warm-up with constant diffusion). Only
        applies to :attr:`UseCase.FULL_SYSTEM_ID`.
    joint_epochs : int
        Epochs for Phase 2 (joint or g2-only training).
    verbosity : int
        Print training metrics every *verbosity* epochs.
    log_precision : int
        Number of decimal digits in scientific-notation logging.
    savedir : str
        Directory for logger output.
    """

    lr: float = 1e-3
    patience: int = 30
    warmup: int = 10
    batch_size: int = 64
    mu_loss_weight: float = 1.0
    var_loss_weight: float = 0.1
    g1_warmup_epochs: int = 40
    joint_epochs: int = 110
    verbosity: int = 1
    log_precision: int = 10
    savedir: str = "./logs"


@dataclass
class UTConfig:
    """Unscented-transform scaling parameters.

    Attributes
    ----------
    alpha : float
        Spread of sigma points around the mean (typically 1e-3 to 1).
    beta : float
        Incorporates prior knowledge of the distribution (2 is optimal
        for Gaussian).
    kappa : float
        Secondary scaling parameter (usually 0 or 3 - n).
    """

    alpha: float = 1.0
    beta: float = 0.0
    kappa: float = 0.0


@dataclass
class NetworkConfig:
    """Architecture hyper-parameters for drift and diffusion networks.

    Attributes
    ----------
    g1_hsizes : list of int
        Hidden-layer sizes for the drift MLP (Use Case 3 only).
    g2_hsizes : list of int
        Hidden-layer sizes for the learnable diffusion MLP.
    g2_min : float
        Floor value for diffusion coefficient (numerical safety).
    """

    g1_hsizes: List[int] = field(default_factory=lambda: [64, 64, 64])
    g2_hsizes: List[int] = field(default_factory=lambda: [64, 64])
    g2_min: float = 1e-6


# ============================================================================
# Result container
# ============================================================================


@dataclass
class PropagationResult:
    """Container for moment-propagation output.

    Attributes
    ----------
    time : np.ndarray
        Time grid of shape ``(n_steps + 1,)``.
    mu : np.ndarray
        Mean trajectory of shape ``(n_steps + 1, nx)``.
    var : np.ndarray
        Marginal-variance trajectory of shape ``(n_steps + 1, nx)``.
    """

    time: np.ndarray
    mu: np.ndarray
    var: np.ndarray

    def std(self) -> np.ndarray:
        """Element-wise standard deviation."""
        return np.sqrt(self.var)

    def confidence_band(
        self, n_sigma: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)`` bounds at ±*n_sigma* standard deviations."""
        s = n_sigma * self.std()
        return self.mu - s, self.mu + s


# ============================================================================
# UT math helpers
# ============================================================================


def ut_weights(
    n: int,
    alpha: float = 1.0,
    beta: float = 0.0,
    kappa: float = 0.0,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Compute mean weights, covariance weights, and spread factor.

    Parameters
    ----------
    n : int
        Dimensionality of the augmented state.
    alpha, beta, kappa : float
        Standard UT scaling parameters.

    Returns
    -------
    Wm : torch.Tensor
        Mean weights, shape ``(2n + 1,)``.
    Wc : torch.Tensor
        Covariance weights, shape ``(2n + 1,)``.
    gamma : float
        Sigma-point spread factor ``sqrt(n + lambda)``.
    """
    lam = alpha**2 * (n + kappa) - n
    gamma = math.sqrt(n + lam)
    Wm = torch.zeros(2 * n + 1, device=device, dtype=dtype)
    Wc = torch.zeros(2 * n + 1, device=device, dtype=dtype)
    Wm[0] = lam / (n + lam)
    Wc[0] = lam / (n + lam) + (1 - alpha**2 + beta)
    w = 1.0 / (2 * (n + lam))
    Wm[1:] = w
    Wc[1:] = w
    return Wm, Wc, gamma


def cholesky_safe(
    A: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Cholesky decomposition with a small diagonal jitter for stability."""
    n = A.shape[-1]
    return torch.linalg.cholesky(
        A + eps * torch.eye(n, device=A.device, dtype=A.dtype)
    )


def pack_moments(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Concatenate mean and marginal variance into a single vector."""
    return torch.cat([mu, var], dim=-1)


def unpack_moments(
    xn: torch.Tensor, nx: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a packed moment vector into mean and variance."""
    return xn[..., :nx], xn[..., nx:]


def moments_to_ode_state(
    mu: torch.Tensor, var: torch.Tensor
) -> torch.Tensor:
    """Convert (mean, diagonal-variance) to the flat ODE state ``[mu, vec(P)]``."""
    P = torch.diag_embed(var)
    return torch.cat([mu, P.reshape(*mu.shape[:-1], -1)], dim=-1)


def ode_state_to_moments(
    state: torch.Tensor, nx: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract (mean, diagonal-variance) from the flat ODE state."""
    mu = state[..., :nx]
    P = state[..., nx:].reshape(*state.shape[:-1], nx, nx)
    var = torch.clamp(torch.diagonal(P, dim1=-2, dim2=-1), min=1e-8)
    return mu, var


# ============================================================================
# Block classes  (all extend neuromancer Block)
# ============================================================================


class ContinuousUTDynamics(Block):
    """ODE right-hand side for continuous-time UT moment propagation.

    Computes ``d[m, vec(P)] / dt`` by:

    1. Forming sigma points from the current mean *m* and covariance *P*,
    2. Propagating them through drift ``g1`` and diffusion ``g2``,
    3. Recovering moment rates via the UT weighted statistics.

    Parameters
    ----------
    g1_net : Block or nn.Module
        Drift network  ``x -> g1(x)``.
    g2_net : Block or nn.Module
        Diffusion network  ``x -> g2(x)`` (half the diffusion coefficient).
    nx : int
        State dimensionality.
    alpha, beta, kappa : float
        UT scaling parameters.
    g2_min : float
        Floor for diffusion output.
    """

    def __init__(
        self,
        g1_net: nn.Module,
        g2_net: nn.Module,
        nx: int,
        alpha: float = 1.0,
        beta: float = 0.0,
        kappa: float = 0.0,
        g2_min: float = 1e-8,
    ):
        super().__init__()
        self.g1_net = g1_net
        self.g2_net = g2_net
        self.nx = nx
        self.g2_min = g2_min
        self.nw = nx  # noise dimension = state dimension
        self.n_aug = nx + self.nw
        self.K = 2 * self.n_aug + 1

        Wm, Wc, gamma = ut_weights(self.n_aug, alpha, beta, kappa)
        self.register_buffer("Wm", Wm)
        self.register_buffer("Wc", Wc)
        self.gamma = gamma

        self.in_features = nx + nx * nx
        self.out_features = nx + nx * nx

    def _sigma_points(
        self, m: torch.Tensor, P: torch.Tensor
    ) -> torch.Tensor:
        """Generate 2*n_aug + 1 augmented sigma points."""
        B, device, dtype = m.shape[0], m.device, m.dtype
        nx, nw, n_aug, K = self.nx, self.nw, self.n_aug, self.K

        P_aug = torch.zeros(B, n_aug, n_aug, device=device, dtype=dtype)
        P_aug[:, :nx, :nx] = P
        for i in range(nw):
            P_aug[:, nx + i, nx + i] = 1.0

        L = cholesky_safe(self.gamma**2 * P_aug)

        m_aug = torch.zeros(B, n_aug, device=device, dtype=dtype)
        m_aug[:, :nx] = m

        Z = torch.zeros(B, K, n_aug, device=device, dtype=dtype)
        Z[:, 0, :] = m_aug
        for i in range(n_aug):
            Z[:, 1 + i, :] = m_aug + L[:, :, i]
            Z[:, 1 + n_aug + i, :] = m_aug - L[:, :, i]
        return Z

    def _dynamics(self, Z: torch.Tensor) -> torch.Tensor:
        """Evaluate augmented dynamics at sigma points."""
        B, K, _ = Z.shape
        Z_x, Z_w = Z[:, :, : self.nx], Z[:, :, self.nx :]
        Z_x_flat = Z_x.reshape(B * K, self.nx)

        g1 = self.g1_net(Z_x_flat).view(B, K, self.nx)
        g2 = torch.clamp(
            self.g2_net(Z_x_flat).view(B, K, self.nx), min=self.g2_min
        )

        F_x = g1 + torch.sqrt(2 * g2) * Z_w
        F_w = torch.zeros(B, K, self.nw, device=Z.device, dtype=Z.dtype)
        return torch.cat([F_x, F_w], dim=-1)

    def block_eval(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the moment ODE right-hand side.

        Parameters
        ----------
        x : torch.Tensor
            Flat ODE state ``[m, vec(P)]`` of shape ``(B, nx + nx*nx)``.

        Returns
        -------
        torch.Tensor
            Time derivative ``[dm/dt, d vec(P)/dt]``, same shape as *x*.
        """
        B, nx = x.shape[0], self.nx
        m = x[:, :nx]
        P = x[:, nx:].view(B, nx, nx)
        P = 0.5 * (P + P.transpose(-1, -2))  # symmetrise

        Z = self._sigma_points(m, P)
        F_Z = self._dynamics(Z)

        Z_x = Z[:, :, :nx]
        F_x = F_Z[:, :, :nx]

        Wm = self.Wm.view(1, -1, 1)
        dm_dt = (Wm * F_x).sum(dim=1)

        m_x = (Wm * Z_x).sum(dim=1, keepdim=True)
        m_F = (Wm * F_x).sum(dim=1, keepdim=True)

        Wc = self.Wc.view(1, -1)
        P_ZF = torch.einsum(
            "bk,bki,bkj->bij",
            Wc.expand(B, -1),
            Z_x - m_x,
            F_x - m_F,
        )
        dP_dt = P_ZF + P_ZF.transpose(-1, -2)

        g2_m = torch.clamp(self.g2_net(m), min=self.g2_min)
        dP_dt = dP_dt + 2 * torch.diag_embed(g2_m)

        return torch.cat([dm_dt, dP_dt.view(B, nx * nx)], dim=-1)


class ContinuousUTIntegrator(Block):
    """Integrator wrapper with moment-space I/O.

    Converts between the packed ``(mu, var)`` representation used by the
    rest of the pipeline and the flat ``[mu, vec(P)]`` ODE state expected
    by :class:`ContinuousUTDynamics`.

    Parameters
    ----------
    integrator : nn.Module
        A NeuroMANCER integrator (e.g. ``integrators.RK4``).
    nx : int
        State dimensionality.
    """

    def __init__(self, integrator: nn.Module, nx: int):
        super().__init__()
        self.integrator = integrator
        self.nx = nx
        self.in_features = 2 * nx
        self.out_features = 2 * nx

    def block_eval(self, xn: torch.Tensor) -> torch.Tensor:
        """Unpack moments, integrate, re-pack."""
        nx = self.nx
        mu, var = unpack_moments(xn, nx)
        ode_state = moments_to_ode_state(mu, var)
        ode_out = self.integrator(ode_state)
        mu_next, var_next = ode_state_to_moments(ode_out, nx)
        return pack_moments(mu_next, var_next)


# ============================================================================
# Diffusion blocks
# ============================================================================


class ZeroG2(Block):
    """Zero diffusion for deterministic propagation (Use Case 1)."""

    def __init__(self, nx: int):
        super().__init__()
        self.in_features = nx
        self.out_features = nx

    def block_eval(self, x: torch.Tensor) -> torch.Tensor:
        """Return a zero tensor with the same shape as *x*."""
        return torch.zeros_like(x)


class ConstantG2(Block):
    """Constant diffusion coefficient ``g2 = sigma^2 / 2``.

    Parameters
    ----------
    nx : int
        State dimensionality.
    sigma : float
        Scalar diffusion strength.
    """

    def __init__(self, nx: int, sigma: float = 0.1):
        super().__init__()
        self.in_features = nx
        self.out_features = nx
        self.register_buffer("g2_const", torch.ones(nx) * (sigma**2 / 2))

    def block_eval(self, x: torch.Tensor) -> torch.Tensor:
        """Return the constant diffusion value broadcast to batch shape."""
        return self.g2_const.expand(
            x.shape[:-1] + self.g2_const.shape
        ).clone()


class LearnableG2(Block):
    """Learnable state-dependent diffusion with softplus positivity.

    Parameters
    ----------
    nx : int
        State dimensionality.
    hsizes : list of int
        Hidden-layer widths.
    min_val : float
        Minimum output floor for numerical stability.
    """

    def __init__(
        self,
        nx: int,
        hsizes: Optional[List[int]] = None,
        min_val: float = 1e-6,
    ):
        super().__init__()
        if hsizes is None:
            hsizes = [64, 64]
        self.in_features = nx
        self.out_features = nx
        self.min_val = min_val

        layers: list = []
        in_dim = nx
        for h in hsizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.SiLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, nx))
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    def block_eval(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP with softplus activation."""
        return self.softplus(self.net(x)) + self.min_val


class SwitchableG2(Block):
    """Switches between a constant warm-up g2 and a learnable g2.

    Used internally during the two-phase training schedule of
    :class:`ContinuousUT`.
    """

    def __init__(self, g2_learnable: Block, g2_constant: Block):
        super().__init__()
        self.g2_learnable = g2_learnable
        self.g2_constant = g2_constant
        self.use_constant = True
        self.in_features = g2_learnable.in_features
        self.out_features = g2_learnable.out_features

    def block_eval(self, x: torch.Tensor) -> torch.Tensor:
        """Dispatch to constant or learned g2 depending on current mode."""
        if self.use_constant:
            return self.g2_constant(x)
        return self.g2_learnable(x)

    def enable_learning(self) -> None:
        """Switch from the constant warm-up value to the learned network."""
        self.use_constant = False


# ============================================================================
# Drift wrapper
# ============================================================================


class _FnModule(nn.Module):
    """Thin wrapper turning a plain callable into an ``nn.Module``."""

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def forward(self, x):
        """Evaluate the wrapped callable."""
        return self._fn(x)


class DriftWrapper(Block):
    """Wraps a user-provided ``nn.Module`` or callable as a Block.

    Parameters
    ----------
    fn : nn.Module or callable
        The drift function ``x -> g1(x)``.
    nx : int
        State dimensionality.
    """

    def __init__(self, fn, nx: int):
        super().__init__()
        self.in_features = nx
        self.out_features = nx
        if isinstance(fn, nn.Module):
            self.fn = fn
        else:
            self.fn = _FnModule(fn)

    def block_eval(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the wrapped drift function."""
        return self.fn(x)





# ============================================================================
# Data utilities
# ============================================================================


def simulate_sde(
    g1_func,
    sigmas,
    x0_mean,
    x0_var,
    t_final: float,
    dt: float,
    n_particles: int = 1000,
    seed: Optional[int] = None,
) -> PropagationResult:
    """Euler-Maruyama SDE simulation returning Monte-Carlo moment statistics.

    Parameters
    ----------
    g1_func : nn.Module or callable
        Drift function ``x -> g1(x)``.
    sigmas : array-like
        Per-dimension diffusion strengths.
    x0_mean, x0_var : array-like
        Mean and variance of the initial-condition distribution.
    t_final : float
        Simulation horizon.
    dt : float
        Integration step size.
    n_particles : int
        Number of Monte-Carlo particles.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    PropagationResult
        Time, mean, and variance trajectories.
    """
    if seed is not None:
        np.random.seed(seed)
    nx = len(x0_mean)
    x0_mean = np.asarray(x0_mean, dtype=np.float64)
    x0_var = np.asarray(x0_var, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    n_steps = int(t_final / dt) + 1

    x = x0_mean + np.sqrt(x0_var) * np.random.randn(n_particles, nx)
    mu_traj, var_traj = [x.mean(0)], [x.var(0)]

    is_mod = isinstance(g1_func, nn.Module)
    if is_mod:
        g1_func.eval()
    with torch.no_grad():
        for _ in range(n_steps - 1):
            drift = g1_func(
                torch.tensor(x, dtype=torch.float32)
            ).numpy()
            x = (
                x
                + drift * dt
                + sigmas * np.random.randn(n_particles, nx) * np.sqrt(dt)
            )
            mu_traj.append(x.mean(0))
            var_traj.append(x.var(0))

    return PropagationResult(
        np.linspace(0, t_final, n_steps),
        np.array(mu_traj),
        np.array(var_traj),
    )


def generate_moment_data(
    drift_fn,
    sigma: float,
    nx: int,
    base_x0: np.ndarray,
    x0_std: float = 0.5,
    n_ensembles: int = 512,
    n_particles: int = 512,
    t_per_ensemble: float = 2.0,
    dt: float = 0.01,
    batch_size: int = 64,
    seed: Optional[int] = None,
):
    """Generate moment-transition training data from SDE ensembles.

    Each ensemble starts from a perturbed initial condition and is
    simulated forward with Euler-Maruyama.  The empirical mean and
    variance at consecutive time-steps form one training pair.

    Parameters
    ----------
    drift_fn : nn.Module or callable
        Drift function.
    sigma : float
        Scalar diffusion strength.
    nx : int
        State dimensionality.
    base_x0 : np.ndarray
        Nominal initial condition.
    x0_std : float
        Standard deviation of the initial-condition perturbation.
    n_ensembles : int
        Number of independent ensembles.
    n_particles : int
        Particles per ensemble.
    t_per_ensemble : float
        Simulation horizon per ensemble.
    dt : float
        Integration step.
    batch_size : int
        Data-loader batch size.
    seed : int, optional
        Random seed.

    Returns
    -------
    train_loader : DataLoader
    dev_loader : DataLoader
    dev_data : dict
    (xn_all, xn_next_all) : tuple of Tensor
        Raw moment pairs for sigma estimation.
    """
    if seed is not None:
        np.random.seed(seed)
    sigmas = [sigma] * nx
    base_x0 = np.asarray(base_x0)

    all_xn, all_xn_next = [], []
    for i in range(n_ensembles):
        x0 = base_x0 + x0_std * np.random.randn(nx)
        r = simulate_sde(
            drift_fn,
            sigmas,
            x0,
            np.ones(nx) * 0.01,
            t_per_ensemble,
            dt,
            n_particles,
        )
        for t in range(len(r.mu) - 1):
            all_xn.append(np.concatenate([r.mu[t], r.var[t]]))
            all_xn_next.append(np.concatenate([r.mu[t + 1], r.var[t + 1]]))
        if (i + 1) % max(1, n_ensembles // 5) == 0:
            print(f"  Ensembles: {i + 1}/{n_ensembles}")

    xn = torch.tensor(np.array(all_xn), dtype=torch.float32)
    xn_next = torch.tensor(np.array(all_xn_next), dtype=torch.float32)

    n = xn.shape[0]
    n_train = int(0.8 * n)
    idx = torch.randperm(n)
    train_data = {
        "xn": xn[idx[:n_train]].unsqueeze(1),
        "xn_next_true": xn_next[idx[:n_train]].unsqueeze(1),
    }
    dev_data = {
        "xn": xn[idx[n_train:]].unsqueeze(1),
        "xn_next_true": xn_next[idx[n_train:]].unsqueeze(1),
    }

    train_ds = DictDataset(train_data, name="train")
    dev_ds = DictDataset(dev_data, name="dev")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_ds.collate_fn,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dev_ds.collate_fn,
    )
    print(f"  Data: {len(train_ds)} train, {len(dev_ds)} dev")
    return train_loader, dev_loader, dev_data, (xn, xn_next)


def estimate_sigma(
    xn: torch.Tensor,
    xn_next: torch.Tensor,
    nx: int,
    dt: float,
) -> Tuple[float, np.ndarray]:
    """Estimate diffusion sigma from moment-transition data.

    Uses the median positive variance increment to robustly infer the
    diffusion strength.

    Parameters
    ----------
    xn, xn_next : torch.Tensor
        Packed moment vectors ``[mu, var]``.
    nx : int
        State dimensionality.
    dt : float
        Time step between consecutive moments.

    Returns
    -------
    sigma_mean : float
        Scalar average diffusion estimate.
    sigma_per_dim : np.ndarray
        Per-dimension estimates of shape ``(nx,)``.
    """
    var_c = xn[:, nx:].numpy()
    var_n = xn_next[:, nx:].numpy()
    dv = var_n - var_c

    s2 = []
    for d in range(nx):
        pos = dv[:, d][dv[:, d] > 0]
        s2.append(
            max(
                np.median(pos) / dt if len(pos) > 0
                else np.median(np.abs(dv[:, d])) / dt,
                1e-8,
            )
        )
    per_dim = np.sqrt(np.array(s2))
    return float(np.mean(per_dim)), per_dim


# ============================================================================
# Main API
# ============================================================================


class ContinuousUT:
    """Unified interface for continuous-time UT uncertainty propagation.

    Use the factory constructors rather than ``__init__`` directly:

    * :meth:`from_known_dynamics` — Use Case 1 (no training)
    * :meth:`from_known_drift` — Use Case 2 (learn g2 only)
    * :meth:`from_data` — Use Case 3 (learn g1 + g2)

    Internally builds a NeuroMANCER ``Node`` / ``System`` / ``Problem`` /
    ``Trainer`` pipeline for training.

    Parameters
    ----------
    use_case : UseCase
        Which scenario this model represents.
    nx : int
        State dimensionality.
    dt : float
        Integration step size.
    g1_net, g2_net : Block
        Drift and diffusion networks.
    g1_fixed, g2_fixed : bool
        Whether the respective networks are frozen during training.
    ut_cfg : UTConfig, optional
    train_cfg : TrainingConfig, optional
    net_cfg : NetworkConfig, optional
    sigma_init : float, optional
        Initial diffusion estimate (bypasses automatic estimation).
    """

    def __init__(
        self,
        use_case: UseCase,
        nx: int,
        dt: float,
        g1_net: nn.Module,
        g2_net: nn.Module,
        g1_fixed: bool,
        g2_fixed: bool,
        ut_cfg: Optional[UTConfig] = None,
        train_cfg: Optional[TrainingConfig] = None,
        net_cfg: Optional[NetworkConfig] = None,
        sigma_init: Optional[float] = None,
    ):
        self.use_case = use_case
        self.nx = nx
        self.dt = dt
        self.g1_net = g1_net
        self.g2_net = g2_net
        self.g1_fixed = g1_fixed
        self.g2_fixed = g2_fixed
        self.ut_cfg = ut_cfg or UTConfig()
        self.train_cfg = train_cfg or TrainingConfig()
        self.net_cfg = net_cfg or NetworkConfig()
        self.sigma_init = sigma_init
        self._eval_block: Optional[ContinuousUTIntegrator] = None
        self._trained = use_case == UseCase.PROPAGATE_IC

    # -----------------------------------------------------------------
    # Factory constructors
    # -----------------------------------------------------------------

    @classmethod
    def from_known_dynamics(
        cls,
        drift_fn,
        nx: int,
        dt: float,
        ut_cfg: Optional[UTConfig] = None,
    ) -> "ContinuousUT":
        """Use Case 1: known deterministic dynamics, uncertain IC.

        No training required — call :meth:`propagate` directly.

        Parameters
        ----------
        drift_fn : nn.Module or callable
            Known drift  ``x -> g1(x)``.
        nx : int
            State dimensionality.
        dt : float
            Integration step.
        ut_cfg : UTConfig, optional
            UT scaling parameters.
        """
        g1 = DriftWrapper(drift_fn, nx)
        g2 = ZeroG2(nx)
        return cls(
            UseCase.PROPAGATE_IC, nx, dt, g1, g2,
            g1_fixed=True, g2_fixed=True, ut_cfg=ut_cfg,
        )

    @classmethod
    def from_known_drift(
        cls,
        drift_fn,
        nx: int,
        dt: float,
        sigma_init: Optional[float] = None,
        ut_cfg: Optional[UTConfig] = None,
        train_cfg: Optional[TrainingConfig] = None,
        net_cfg: Optional[NetworkConfig] = None,
    ) -> "ContinuousUT":
        """Use Case 2: known drift, learn diffusion from data.

        Parameters
        ----------
        drift_fn : nn.Module or callable
            Known (or pre-trained) drift.  Will be frozen during training.
        nx : int
            State dimensionality.
        dt : float
            Integration step.
        sigma_init : float, optional
            Initial diffusion estimate.
        ut_cfg : UTConfig, optional
        train_cfg : TrainingConfig, optional
        net_cfg : NetworkConfig, optional
        """
        ncfg = net_cfg or NetworkConfig()
        g1 = DriftWrapper(drift_fn, nx)
        g2 = LearnableG2(nx, hsizes=ncfg.g2_hsizes, min_val=ncfg.g2_min)
        return cls(
            UseCase.LEARN_DIFFUSION, nx, dt, g1, g2,
            g1_fixed=True, g2_fixed=False,
            ut_cfg=ut_cfg, train_cfg=train_cfg, net_cfg=ncfg,
            sigma_init=sigma_init,
        )

    @classmethod
    def from_data(
        cls,
        nx: int,
        dt: float,
        sigma_init: Optional[float] = None,
        g1_hsizes: Optional[List[int]] = None,
        ut_cfg: Optional[UTConfig] = None,
        train_cfg: Optional[TrainingConfig] = None,
        net_cfg: Optional[NetworkConfig] = None,
    ) -> "ContinuousUT":
        """Use Case 3: full system identification — learn both g1 and g2.

        Parameters
        ----------
        nx : int
            State dimensionality.
        dt : float
            Integration step.
        sigma_init : float, optional
            Initial diffusion estimate.
        g1_hsizes : list of int, optional
            Override hidden sizes for the drift MLP.
        ut_cfg : UTConfig, optional
        train_cfg : TrainingConfig, optional
        net_cfg : NetworkConfig, optional
        """
        ncfg = net_cfg or NetworkConfig()
        if g1_hsizes:
            ncfg.g1_hsizes = g1_hsizes
        g1 = MLP(
            insize=nx,
            outsize=nx,
            bias=True,
            linear_map=nn.Linear,
            nonlin=nn.SiLU,
            hsizes=ncfg.g1_hsizes,
        )
        g2 = LearnableG2(nx, hsizes=ncfg.g2_hsizes, min_val=ncfg.g2_min)
        return cls(
            UseCase.FULL_SYSTEM_ID, nx, dt, g1, g2,
            g1_fixed=False, g2_fixed=False,
            ut_cfg=ut_cfg, train_cfg=train_cfg, net_cfg=ncfg,
            sigma_init=sigma_init,
        )

    # -----------------------------------------------------------------
    # Internal pipeline builders
    # -----------------------------------------------------------------

    def _build_loss(self) -> PenaltyLoss:
        """Construct the mean + variance tracking loss."""
        nx = self.nx
        cfg = self.train_cfg
        xn_pred = variable("xn")[:, 1:, :]
        xn_true = variable("xn_next_true")
        mu_loss = cfg.mu_loss_weight * (
            (xn_pred[:, :, :nx] == xn_true[:, :, :nx]) ^ 2
        )
        mu_loss.name = "mu_loss"
        var_loss = cfg.var_loss_weight * (
            (xn_pred[:, :, nx:] == xn_true[:, :, nx:]) ^ 2
        )
        var_loss.name = "var_loss"
        return PenaltyLoss([mu_loss, var_loss], [])

    def _build_system(
        self, g1: nn.Module, g2: nn.Module, name: str = "UT"
    ) -> System:
        """Build the Node -> System pipeline for a given (g1, g2) pair."""
        ut = self.ut_cfg
        dynamics = ContinuousUTDynamics(
            g1, g2, self.nx,
            alpha=ut.alpha, beta=ut.beta, kappa=ut.kappa,
        )
        integ = integrators.RK4(dynamics, h=self.dt)
        block = ContinuousUTIntegrator(integ, self.nx)
        node = Node(block, ["xn"], ["xn"], name=name)
        return System([node], name=f"{name}_sys", nsteps=1)

    def _make_logger(self, phase: str) -> PreciseLogger:
        """Create a logger for a training phase."""
        cfg = self.train_cfg
        return PreciseLogger(
            args=None,
            savedir=f"{cfg.savedir}/{phase}",
            verbosity=cfg.verbosity,
            stdout=["dev_loss", "train_loss"],
            precision=cfg.log_precision,
        )

    def _run_trainer(
        self, problem, train_loader, dev_loader, dev_data,
        optimizer, epochs, label,
    ):
        """Run the NeuroMANCER Trainer for one phase."""
        cfg = self.train_cfg
        trainer = Trainer(
            problem,
            train_loader,
            dev_loader,
            dev_data,
            optimizer,
            patience=cfg.patience,
            warmup=cfg.warmup,
            epochs=epochs,
            eval_metric="dev_loss",
            train_metric="train_loss",
            dev_metric="dev_loss",
            test_metric="dev_loss",
            logger=self._make_logger(label),
        )
        best = trainer.train()
        problem.load_state_dict(best)
        return best

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        dev_data: Optional[dict] = None,
        sigma_from_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        """Train the model using the NeuroMANCER ``Trainer``.

        Parameters
        ----------
        train_loader, dev_loader : DataLoader
            Training and validation data loaders (typically from
            :func:`generate_moment_data`).
        dev_data : dict, optional
            Raw validation dictionary required by the NeuroMANCER Trainer.
        sigma_from_data : tuple of Tensor, optional
            ``(xn_all, xn_next_all)`` for automatic sigma estimation.
        """
        if self.use_case == UseCase.PROPAGATE_IC:
            print("Use Case 1 -- no training needed.")
            return

        cfg = self.train_cfg
        nx = self.nx
        loss_fn = self._build_loss()

        # Estimate sigma for warm-up constant g2
        sigma_w = self.sigma_init
        if sigma_w is None and sigma_from_data is not None:
            xn_all, xn_next_all = sigma_from_data
            sigma_w, per_dim = estimate_sigma(xn_all, xn_next_all, nx, self.dt)
            print(f"Estimated sigma: {sigma_w:.4f} (per dim: {per_dim})")
        sigma_w = sigma_w or 0.1
        g2_const = ConstantG2(nx, sigma=sigma_w)

        # Phase 1: warm-up g1 with constant g2 (full sys ID only)
        if (
            self.use_case == UseCase.FULL_SYSTEM_ID
            and cfg.g1_warmup_epochs > 0
        ):
            print(f"\n{'=' * 60}")
            print("Phase 1: Warmup -- training g1 with constant g2")
            print(f"{'=' * 60}")
            for p in self.g2_net.parameters():
                p.requires_grad = False
            system = self._build_system(self.g1_net, g2_const, name="warmup")
            problem = Problem([system], loss_fn)
            optimizer = torch.optim.Adam(self.g1_net.parameters(), lr=cfg.lr)
            self._run_trainer(
                problem, train_loader, dev_loader, dev_data,
                optimizer, cfg.g1_warmup_epochs, "warmup",
            )

        # Phase 2: learn g2 (and continue g1 for full sys ID)
        for p in self.g2_net.parameters():
            p.requires_grad = True
        switchable = SwitchableG2(self.g2_net, g2_const)
        switchable.enable_learning()

        phase = (
            "g2 only (drift fixed)"
            if self.use_case == UseCase.LEARN_DIFFUSION
            else "joint g1 + g2"
        )
        n_ep = (
            cfg.joint_epochs
            if self.use_case == UseCase.FULL_SYSTEM_ID
            else cfg.g1_warmup_epochs + cfg.joint_epochs
        )
        print(f"\n{'=' * 60}")
        print(f"Phase 2: Training {phase} ({n_ep} epochs)")
        print(f"{'=' * 60}")

        system = self._build_system(self.g1_net, switchable, name="joint")
        problem = Problem([system], loss_fn)
        params = list(self.g2_net.parameters())
        if not self.g1_fixed:
            params += list(self.g1_net.parameters())
        optimizer = torch.optim.Adam(params, lr=cfg.lr)
        self._run_trainer(
            problem, train_loader, dev_loader, dev_data,
            optimizer, n_ep, "joint",
        )

        self._trained = True
        self._eval_block = None
        print("Training complete.")

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------

    def _build_eval_block(self) -> None:
        """Construct the evaluation-time integrator."""
        ut = self.ut_cfg
        dynamics = ContinuousUTDynamics(
            self.g1_net, self.g2_net, self.nx,
            alpha=ut.alpha, beta=ut.beta, kappa=ut.kappa,
        )
        integ = integrators.RK4(dynamics, h=self.dt)
        self._eval_block = ContinuousUTIntegrator(integ, self.nx)
        self._eval_block.eval()

    def propagate(
        self,
        mu0,
        var0,
        n_steps: int = 500,
    ) -> PropagationResult:
        """Propagate moments forward in time via RK4 integration.

        Parameters
        ----------
        mu0 : array-like
            Initial mean, shape ``(nx,)`` or ``(1, nx)``.
        var0 : array-like
            Initial marginal variance, same shape as *mu0*.
        n_steps : int
            Number of integration steps.

        Returns
        -------
        PropagationResult
            Time, mean, and variance trajectories.

        Raises
        ------
        RuntimeError
            If the model has not been trained (Use Cases 2 and 3).
        """
        if not self._trained:
            raise RuntimeError("Call fit() first.")
        if self._eval_block is None:
            self._build_eval_block()

        mu0 = torch.as_tensor(mu0, dtype=torch.float32)
        var0 = torch.as_tensor(var0, dtype=torch.float32)
        if mu0.dim() == 1:
            mu0, var0 = mu0.unsqueeze(0), var0.unsqueeze(0)

        xn = pack_moments(mu0, var0)
        mus, vs = [mu0[0].numpy()], [var0[0].numpy()]
        with torch.no_grad():
            for _ in range(n_steps):
                xn = self._eval_block(xn)
                m, v = unpack_moments(xn, self.nx)
                mus.append(m[0].numpy())
                vs.append(v[0].numpy())

        return PropagationResult(
            np.linspace(0, n_steps * self.dt, n_steps + 1),
            np.array(mus),
            np.array(vs),
        )

    # -----------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------

    def plot(
        self,
        result: PropagationResult,
        reference: Optional[PropagationResult] = None,
        title: str = "Moment Propagation",
        dim_names: Optional[List[str]] = None,
        n_sigma: float = 2.0,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """Plot mean and variance trajectories.

        Parameters
        ----------
        result : PropagationResult
            Output of :meth:`propagate`.
        reference : PropagationResult, optional
            Monte-Carlo reference for overlay comparison.
        title : str
            Figure suptitle.
        dim_names : list of str, optional
            LaTeX-friendly dimension labels.
        n_sigma : float
            Width of the confidence band in standard deviations.
        save_path : str, optional
            Save figure to this path.
        show : bool
            Whether to call ``plt.show()``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        nx = result.mu.shape[1]
        dim_names = dim_names or [f"$x_{{{i + 1}}}$" for i in range(nx)]
        colors = ["tab:blue", "tab:red", "tab:green", "tab:orange"][:nx]

        fig, axes = plt.subplots(2, nx, figsize=(7 * nx, 10))
        if nx == 1:
            axes = axes.reshape(-1, 1)

        for d in range(nx):
            c = colors[d]
            ax = axes[0, d]
            if reference is not None:
                ax.plot(
                    reference.time, reference.mu[:, d],
                    "k-", lw=2, label="MC reference",
                )
            ax.plot(
                result.time, result.mu[:, d],
                "--", color=c, lw=2, label="UT model",
            )
            lo, hi = result.confidence_band(n_sigma)
            ax.fill_between(
                result.time, lo[:, d], hi[:, d], alpha=0.25, color=c,
            )
            ax.set_xlabel("Time")
            ax.set_ylabel(f"$\\mu_{{{d + 1}}}$")
            ax.set_title(f"Mean of {dim_names[d]}")
            ax.legend()
            ax.grid(True, alpha=0.3)

            ax = axes[1, d]
            if reference is not None:
                ax.plot(
                    reference.time, reference.var[:, d],
                    "k-", lw=2, label="MC reference",
                )
            ax.plot(
                result.time, result.var[:, d],
                "--", color=c, lw=2, label="UT model",
            )
            ax.set_xlabel("Time")
            ax.set_ylabel(f"$\\sigma^2_{{{d + 1}}}$")
            ax.set_title(f"Variance of {dim_names[d]}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    def summary(self) -> None:
        """Print a compact summary of the model configuration."""
        labels = {
            UseCase.PROPAGATE_IC: "Propagate IC",
            UseCase.LEARN_DIFFUSION: "Learn Diffusion",
            UseCase.FULL_SYSTEM_ID: "Full System ID",
        }
        g1p = sum(p.numel() for p in self.g1_net.parameters())
        g2p = sum(p.numel() for p in self.g2_net.parameters())
        tp = sum(
            p.numel()
            for p in self.g1_net.parameters()
            if p.requires_grad
        ) + sum(
            p.numel()
            for p in self.g2_net.parameters()
            if p.requires_grad
        )
        print(f"ContinuousUT [{labels[self.use_case]}]")
        print(
            f"  nx={self.nx}  dt={self.dt}  "
            f"g1={'fixed' if self.g1_fixed else 'learned (MLP)'}  "
            f"g2={'fixed' if self.g2_fixed else 'learned'}"
        )
        print(f"  params: g1={g1p:,}  g2={g2p:,}  trainable={tp:,}")
        print(f"  integrator: neuromancer RK4(h={self.dt})")
        print(f"  trained: {self._trained}")
