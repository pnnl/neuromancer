"""
Continuous-time Unscented Transform for stochastic dynamics.

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
.. [2] J. O'Leary, J. A. Paulson, and A. Mesbah, "Stochastic physics-informed
       neural networks (SPINN): A moment-matching framework for learning hidden
       physics within stochastic differential equations," arXiv:2109.01621, 2021.
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
    "ContinuousUT",
    "UseCase", "TrainingConfig", "UTConfig", "NetworkConfig",
    "PropagationResult",
    "ContinuousUTDynamics", "ContinuousUTIntegrator",
    "ZeroG2", "ConstantG2", "LearnableG2", "SwitchableG2", "DriftWrapper",
    "PreciseLogger",
    "simulate_sde", "generate_moment_data", "estimate_sigma",
    "ut_weights", "cholesky_safe",
    "pack_moments", "unpack_moments",
    "moments_to_ode_state", "ode_state_to_moments",
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
    """Training hyper-parameters for :class:`ContinuousUT`."""
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
    """Unscented-transform scaling parameters."""
    alpha: float = 1.0
    beta: float = 0.0
    kappa: float = 0.0


@dataclass
class NetworkConfig:
    """Architecture hyper-parameters for drift and diffusion networks."""
    g1_hsizes: List[int] = field(default_factory=lambda: [64, 64, 64])
    g2_hsizes: List[int] = field(default_factory=lambda: [64, 64])
    g2_min: float = 1e-6


# ============================================================================
# Result container
# ============================================================================

@dataclass
class PropagationResult:
    """Container for moment-propagation output."""
    time: np.ndarray
    mu: np.ndarray
    var: np.ndarray

    def std(self) -> np.ndarray:
        return np.sqrt(self.var)

    def confidence_band(self, n_sigma: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        s = n_sigma * self.std()
        return self.mu - s, self.mu + s


# ============================================================================
# UT math helpers
# ============================================================================

def ut_weights(n, alpha=1.0, beta=0.0, kappa=0.0, device="cpu", dtype=torch.float32):
    """Compute mean weights, covariance weights, and spread factor."""
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


def cholesky_safe(A, eps=1e-6):
    """Cholesky decomposition with diagonal jitter for stability."""
    n = A.shape[-1]
    return torch.linalg.cholesky(A + eps * torch.eye(n, device=A.device, dtype=A.dtype))


def pack_moments(mu, var):
    """Concatenate mean and marginal variance."""
    return torch.cat([mu, var], dim=-1)


def unpack_moments(xn, nx):
    """Split packed moment vector into mean and variance."""
    return xn[..., :nx], xn[..., nx:]


def moments_to_ode_state(mu, var):
    """Convert (mean, diag-variance) to flat ODE state [mu, vec(P)]."""
    P = torch.diag_embed(var)
    return torch.cat([mu, P.reshape(*mu.shape[:-1], -1)], dim=-1)


def ode_state_to_moments(state, nx):
    """Extract (mean, diag-variance) from flat ODE state."""
    mu = state[..., :nx]
    P = state[..., nx:].reshape(*state.shape[:-1], nx, nx)
    var = torch.clamp(torch.diagonal(P, dim1=-2, dim2=-1), min=1e-8)
    return mu, var


# ============================================================================
# Block classes (all extend neuromancer Block)
# ============================================================================

class ContinuousUTDynamics(Block):
    """ODE RHS for continuous-time UT moment propagation.

    Computes d[m, vec(P)]/dt via sigma-point propagation through drift
    g1 and diffusion g2.
    """

    def __init__(self, g1_net, g2_net, nx, alpha=1.0, beta=0.0, kappa=0.0, g2_min=1e-8):
        super().__init__()
        self.g1_net = g1_net
        self.g2_net = g2_net
        self.nx = nx
        self.g2_min = g2_min
        self.nw = nx
        self.n_aug = nx + self.nw
        self.K = 2 * self.n_aug + 1
        Wm, Wc, gamma = ut_weights(self.n_aug, alpha, beta, kappa)
        self.register_buffer("Wm", Wm)
        self.register_buffer("Wc", Wc)
        self.gamma = gamma
        self.in_features = nx + nx * nx
        self.out_features = nx + nx * nx

    def _sigma_points(self, m, P):
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

    def _dynamics(self, Z):
        B, K, _ = Z.shape
        Z_x, Z_w = Z[:, :, :self.nx], Z[:, :, self.nx:]
        Z_x_flat = Z_x.reshape(B * K, self.nx)
        g1 = self.g1_net(Z_x_flat).view(B, K, self.nx)
        g2 = torch.clamp(self.g2_net(Z_x_flat).view(B, K, self.nx), min=self.g2_min)
        F_x = g1 + torch.sqrt(2 * g2) * Z_w
        F_w = torch.zeros(B, K, self.nw, device=Z.device, dtype=Z.dtype)
        return torch.cat([F_x, F_w], dim=-1)

    def block_eval(self, x):
        """Compute the moment ODE right-hand side."""
        B, nx = x.shape[0], self.nx
        m = x[:, :nx]
        P = x[:, nx:].view(B, nx, nx)
        P = 0.5 * (P + P.transpose(-1, -2))
        Z = self._sigma_points(m, P)
        F_Z = self._dynamics(Z)
        Z_x, F_x = Z[:, :, :nx], F_Z[:, :, :nx]
        Wm = self.Wm.view(1, -1, 1)
        dm_dt = (Wm * F_x).sum(dim=1)
        m_x = (Wm * Z_x).sum(dim=1, keepdim=True)
        m_F = (Wm * F_x).sum(dim=1, keepdim=True)
        Wc = self.Wc.view(1, -1)
        P_ZF = torch.einsum("bk,bki,bkj->bij", Wc.expand(B, -1), Z_x - m_x, F_x - m_F)
        dP_dt = P_ZF + P_ZF.transpose(-1, -2)
        g2_m = torch.clamp(self.g2_net(m), min=self.g2_min)
        dP_dt = dP_dt + 2 * torch.diag_embed(g2_m)
        return torch.cat([dm_dt, dP_dt.view(B, nx * nx)], dim=-1)


class ContinuousUTIntegrator(Block):
    """Integrator wrapper with moment-space I/O."""

    def __init__(self, integrator, nx):
        super().__init__()
        self.integrator = integrator
        self.nx = nx
        self.in_features = 2 * nx
        self.out_features = 2 * nx

    def block_eval(self, xn):
        """Unpack moments, integrate, re-pack."""
        nx = self.nx
        mu, var = unpack_moments(xn, nx)
        ode_state = moments_to_ode_state(mu, var)
        ode_out = self.integrator(ode_state)
        mu_next, var_next = ode_state_to_moments(ode_out, nx)
        return pack_moments(mu_next, var_next)


class ZeroG2(Block):
    """Zero diffusion (Use Case 1)."""
    def __init__(self, nx):
        super().__init__()
        self.in_features = nx
        self.out_features = nx

    def block_eval(self, x):
        return torch.zeros_like(x)


class ConstantG2(Block):
    """Constant diffusion g2 = sigma^2 / 2."""
    def __init__(self, nx, sigma=0.1):
        super().__init__()
        self.in_features = nx
        self.out_features = nx
        self.register_buffer("g2_const", torch.ones(nx) * (sigma**2 / 2))

    def block_eval(self, x):
        return self.g2_const.expand(x.shape[:-1] + self.g2_const.shape).clone()


class LearnableG2(Block):
    """Learnable state-dependent diffusion with softplus positivity."""
    def __init__(self, nx, hsizes=None, min_val=1e-6):
        super().__init__()
        if hsizes is None:
            hsizes = [64, 64]
        self.in_features = nx
        self.out_features = nx
        self.min_val = min_val
        layers = []
        in_dim = nx
        for h in hsizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.SiLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, nx))
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    def block_eval(self, x):
        return self.softplus(self.net(x)) + self.min_val


class SwitchableG2(Block):
    """Switches between constant warm-up and learnable g2."""
    def __init__(self, g2_learnable, g2_constant):
        super().__init__()
        self.g2_learnable = g2_learnable
        self.g2_constant = g2_constant
        self.use_constant = True
        self.in_features = g2_learnable.in_features
        self.out_features = g2_learnable.out_features

    def block_eval(self, x):
        if self.use_constant:
            return self.g2_constant(x)
        return self.g2_learnable(x)

    def enable_learning(self):
        self.use_constant = False


class _FnModule(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self._fn = fn
    def forward(self, x):
        return self._fn(x)


class DriftWrapper(Block):
    """Wraps an nn.Module or callable as a Block for use as g1."""
    def __init__(self, fn, nx):
        super().__init__()
        self.in_features = nx
        self.out_features = nx
        if isinstance(fn, nn.Module):
            self.fn = fn
        else:
            self.fn = _FnModule(fn)

    def block_eval(self, x):
        return self.fn(x)


# ============================================================================
# Logger
# ============================================================================

class PreciseLogger(BasicLogger):
    """Logger with configurable floating-point precision."""
    def __init__(self, args=None, savedir="test", verbosity=10,
                 stdout=("dev_loss", "train_loss"), precision=8):
        super().__init__(args, savedir, verbosity, stdout)
        self.precision = precision

    def log_metrics(self, output, step=None):
        if step is None:
            step = self.step
        else:
            self.step = step
        if step % self.verbosity == 0:
            entries = [f"epoch: {step}"]
            for k, v in output.items():
                try:
                    if k in self.stdout:
                        val = v.item()
                        entries.append(f"{k}: {val:.{self.precision}e}")
                except (ValueError, AttributeError):
                    pass
            filtered = [e for e in entries if "reg_error" not in e]
            print("\t".join(filtered))


# ============================================================================
# Data utilities
# ============================================================================

def simulate_sde(g1_func, sigmas, x0_mean, x0_var, t_final, dt,
                 n_particles=1000, seed=None):
    """Euler-Maruyama SDE simulation returning MC moment statistics."""
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
            drift = g1_func(torch.tensor(x, dtype=torch.float32)).numpy()
            x = x + drift * dt + sigmas * np.random.randn(n_particles, nx) * np.sqrt(dt)
            mu_traj.append(x.mean(0))
            var_traj.append(x.var(0))
    return PropagationResult(np.linspace(0, t_final, n_steps),
                             np.array(mu_traj), np.array(var_traj))


def generate_moment_data(drift_fn, sigma, nx, base_x0, x0_std=0.5,
                         n_ensembles=512, n_particles=512,
                         t_per_ensemble=2.0, dt=0.01, batch_size=64, seed=None):
    """Generate moment-transition training data from SDE ensembles."""
    if seed is not None:
        np.random.seed(seed)
    sigmas = [sigma] * nx
    base_x0 = np.asarray(base_x0)
    all_xn, all_xn_next = [], []
    for i in range(n_ensembles):
        x0 = base_x0 + x0_std * np.random.randn(nx)
        r = simulate_sde(drift_fn, sigmas, x0, np.ones(nx) * 0.01,
                         t_per_ensemble, dt, n_particles)
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
    train_data = {"xn": xn[idx[:n_train]].unsqueeze(1),
                  "xn_next_true": xn_next[idx[:n_train]].unsqueeze(1)}
    dev_data = {"xn": xn[idx[n_train:]].unsqueeze(1),
                "xn_next_true": xn_next[idx[n_train:]].unsqueeze(1)}
    train_ds = DictDataset(train_data, name="train")
    dev_ds = DictDataset(dev_data, name="dev")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=train_ds.collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=dev_ds.collate_fn)
    print(f"  Data: {len(train_ds)} train, {len(dev_ds)} dev")
    return train_loader, dev_loader, dev_data, (xn, xn_next)


def estimate_sigma(xn, xn_next, nx, dt):
    """Estimate diffusion sigma from moment-transition data."""
    var_c = xn[:, nx:].numpy()
    var_n = xn_next[:, nx:].numpy()
    dv = var_n - var_c
    s2 = []
    for d in range(nx):
        pos = dv[:, d][dv[:, d] > 0]
        s2.append(max(np.median(pos) / dt if len(pos) > 0
                      else np.median(np.abs(dv[:, d])) / dt, 1e-8))
    per_dim = np.sqrt(np.array(s2))
    return float(np.mean(per_dim)), per_dim


# ============================================================================
# Main API
# ============================================================================

class ContinuousUT:
    """Unified interface for continuous-time UT uncertainty propagation.

    Factory constructors:
        from_known_dynamics()  -- Use Case 1 (no training)
        from_known_drift()     -- Use Case 2 (learn g2 only)
        from_data()            -- Use Case 3 (learn g1 + g2)

    Internally uses neuromancer Node/System/Problem/Trainer for training.
    """

    def __init__(self, use_case, nx, dt, g1_net, g2_net, g1_fixed, g2_fixed,
                 ut_cfg=None, train_cfg=None, net_cfg=None, sigma_init=None):
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
        self._eval_block = None
        self._trained = (use_case == UseCase.PROPAGATE_IC)

    # ---- Factories ----

    @classmethod
    def from_known_dynamics(cls, drift_fn, nx, dt, ut_cfg=None):
        """Use Case 1: known deterministic dynamics, uncertain IC. No training."""
        g1 = DriftWrapper(drift_fn, nx)
        g2 = ZeroG2(nx)
        return cls(UseCase.PROPAGATE_IC, nx, dt, g1, g2,
                   g1_fixed=True, g2_fixed=True, ut_cfg=ut_cfg)

    @classmethod
    def from_known_drift(cls, drift_fn, nx, dt, sigma_init=None,
                         ut_cfg=None, train_cfg=None, net_cfg=None):
        """Use Case 2: known drift, learn diffusion from data."""
        ncfg = net_cfg or NetworkConfig()
        g1 = DriftWrapper(drift_fn, nx)
        g2 = LearnableG2(nx, hsizes=ncfg.g2_hsizes, min_val=ncfg.g2_min)
        return cls(UseCase.LEARN_DIFFUSION, nx, dt, g1, g2,
                   g1_fixed=True, g2_fixed=False,
                   ut_cfg=ut_cfg, train_cfg=train_cfg, net_cfg=ncfg,
                   sigma_init=sigma_init)

    @classmethod
    def from_data(cls, nx, dt, sigma_init=None, g1_hsizes=None,
                  ut_cfg=None, train_cfg=None, net_cfg=None):
        """Use Case 3: full system ID -- learn both g1 and g2."""
        ncfg = net_cfg or NetworkConfig()
        if g1_hsizes:
            ncfg.g1_hsizes = g1_hsizes
        g1 = MLP(insize=nx, outsize=nx, bias=True, linear_map=nn.Linear,
                 nonlin=nn.SiLU, hsizes=ncfg.g1_hsizes)
        g2 = LearnableG2(nx, hsizes=ncfg.g2_hsizes, min_val=ncfg.g2_min)
        return cls(UseCase.FULL_SYSTEM_ID, nx, dt, g1, g2,
                   g1_fixed=False, g2_fixed=False,
                   ut_cfg=ut_cfg, train_cfg=train_cfg, net_cfg=ncfg,
                   sigma_init=sigma_init)

    # ---- Internal pipeline builders ----

    def _build_loss(self):
        nx = self.nx
        cfg = self.train_cfg
        xn_pred = variable("xn")[:, 1:, :]
        xn_true = variable("xn_next_true")
        mu_loss = cfg.mu_loss_weight * ((xn_pred[:, :, :nx] == xn_true[:, :, :nx]) ^ 2)
        mu_loss.name = "mu_loss"
        var_loss = cfg.var_loss_weight * ((xn_pred[:, :, nx:] == xn_true[:, :, nx:]) ^ 2)
        var_loss.name = "var_loss"
        return PenaltyLoss([mu_loss, var_loss], [])

    def _build_system(self, g1, g2, name="UT"):
        ut = self.ut_cfg
        dynamics = ContinuousUTDynamics(g1, g2, self.nx,
                                        alpha=ut.alpha, beta=ut.beta, kappa=ut.kappa)
        integ = integrators.RK4(dynamics, h=self.dt)
        block = ContinuousUTIntegrator(integ, self.nx)
        node = Node(block, ["xn"], ["xn"], name=name)
        return System([node], name=f"{name}_sys", nsteps=1)

    def _make_logger(self, phase):
        cfg = self.train_cfg
        return PreciseLogger(args=None, savedir=f"{cfg.savedir}/{phase}",
                             verbosity=cfg.verbosity,
                             stdout=["dev_loss", "train_loss"],
                             precision=cfg.log_precision)

    def _run_trainer(self, problem, train_loader, dev_loader, dev_data,
                     optimizer, epochs, label):
        cfg = self.train_cfg
        trainer = Trainer(
            problem, train_loader, dev_loader, dev_data, optimizer,
            patience=cfg.patience, warmup=cfg.warmup, epochs=epochs,
            eval_metric="dev_loss", train_metric="train_loss",
            dev_metric="dev_loss", test_metric="dev_loss",
            logger=self._make_logger(label))
        best = trainer.train()
        problem.load_state_dict(best)
        return best

    # ---- Training ----

    def fit(self, train_loader, dev_loader, dev_data=None, sigma_from_data=None):
        """Train the model using neuromancer Trainer."""
        if self.use_case == UseCase.PROPAGATE_IC:
            print("Use Case 1 -- no training needed.")
            return

        cfg = self.train_cfg
        nx = self.nx
        loss_fn = self._build_loss()

        sigma_w = self.sigma_init
        if sigma_w is None and sigma_from_data is not None:
            xn_all, xn_next_all = sigma_from_data
            sigma_w, per_dim = estimate_sigma(xn_all, xn_next_all, nx, self.dt)
            print(f"Estimated sigma: {sigma_w:.4f} (per dim: {per_dim})")
        sigma_w = sigma_w or 0.1
        g2_const = ConstantG2(nx, sigma=sigma_w)

        if self.use_case == UseCase.FULL_SYSTEM_ID and cfg.g1_warmup_epochs > 0:
            print(f"\n{'=' * 60}")
            print("Phase 1: Warmup -- training g1 with constant g2")
            print(f"{'=' * 60}")
            for p in self.g2_net.parameters():
                p.requires_grad = False
            system = self._build_system(self.g1_net, g2_const, name="warmup")
            problem = Problem([system], loss_fn)
            optimizer = torch.optim.Adam(self.g1_net.parameters(), lr=cfg.lr)
            self._run_trainer(problem, train_loader, dev_loader, dev_data,
                              optimizer, cfg.g1_warmup_epochs, "warmup")

        for p in self.g2_net.parameters():
            p.requires_grad = True
        switchable = SwitchableG2(self.g2_net, g2_const)
        switchable.enable_learning()

        phase = ("g2 only (drift fixed)" if self.use_case == UseCase.LEARN_DIFFUSION
                 else "joint g1 + g2")
        n_ep = (cfg.joint_epochs if self.use_case == UseCase.FULL_SYSTEM_ID
                else cfg.g1_warmup_epochs + cfg.joint_epochs)
        print(f"\n{'=' * 60}")
        print(f"Phase 2: Training {phase} ({n_ep} epochs)")
        print(f"{'=' * 60}")

        system = self._build_system(self.g1_net, switchable, name="joint")
        problem = Problem([system], loss_fn)
        params = list(self.g2_net.parameters())
        if not self.g1_fixed:
            params += list(self.g1_net.parameters())
        optimizer = torch.optim.Adam(params, lr=cfg.lr)
        self._run_trainer(problem, train_loader, dev_loader, dev_data,
                          optimizer, n_ep, "joint")

        self._trained = True
        self._eval_block = None
        print("Training complete.")

    # ---- Inference ----

    def _build_eval_block(self):
        ut = self.ut_cfg
        dynamics = ContinuousUTDynamics(self.g1_net, self.g2_net, self.nx,
                                        alpha=ut.alpha, beta=ut.beta, kappa=ut.kappa)
        integ = integrators.RK4(dynamics, h=self.dt)
        self._eval_block = ContinuousUTIntegrator(integ, self.nx)
        self._eval_block.eval()

    def propagate(self, mu0, var0, n_steps=500):
        """Propagate moments forward n_steps via RK4 integration."""
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
        return PropagationResult(np.linspace(0, n_steps * self.dt, n_steps + 1),
                                 np.array(mus), np.array(vs))

    # ---- Plotting ----

    def plot(self, result, reference=None, title="Moment Propagation",
             dim_names=None, n_sigma=2, save_path=None, show=True):
        """Plot mean and variance trajectories."""
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
                ax.plot(reference.time, reference.mu[:, d], "k-", lw=2, label="MC reference")
            ax.plot(result.time, result.mu[:, d], "--", color=c, lw=2, label="UT model")
            lo, hi = result.confidence_band(n_sigma)
            ax.fill_between(result.time, lo[:, d], hi[:, d], alpha=0.25, color=c)
            ax.set_xlabel("Time"); ax.set_ylabel(f"$\\mu_{{{d + 1}}}$")
            ax.set_title(f"Mean of {dim_names[d]}"); ax.legend(); ax.grid(True, alpha=0.3)
            ax = axes[1, d]
            if reference is not None:
                ax.plot(reference.time, reference.var[:, d], "k-", lw=2, label="MC reference")
            ax.plot(result.time, result.var[:, d], "--", color=c, lw=2, label="UT model")
            ax.set_xlabel("Time"); ax.set_ylabel(f"$\\sigma^2_{{{d + 1}}}$")
            ax.set_title(f"Variance of {dim_names[d]}"); ax.legend(); ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    def summary(self):
        """Print a compact model summary."""
        labels = {UseCase.PROPAGATE_IC: "Propagate IC",
                  UseCase.LEARN_DIFFUSION: "Learn Diffusion",
                  UseCase.FULL_SYSTEM_ID: "Full System ID"}
        g1p = sum(p.numel() for p in self.g1_net.parameters())
        g2p = sum(p.numel() for p in self.g2_net.parameters())
        tp = (sum(p.numel() for p in self.g1_net.parameters() if p.requires_grad)
              + sum(p.numel() for p in self.g2_net.parameters() if p.requires_grad))
        print(f"ContinuousUT [{labels[self.use_case]}]")
        print(f"  nx={self.nx}  dt={self.dt}  "
              f"g1={'fixed' if self.g1_fixed else 'learned (MLP)'}  "
              f"g2={'fixed' if self.g2_fixed else 'learned'}")
        print(f"  params: g1={g1p:,}  g2={g2p:,}  trainable={tp:,}")
        print(f"  integrator: neuromancer RK4(h={self.dt})")
        print(f"  trained: {self._trained}")
