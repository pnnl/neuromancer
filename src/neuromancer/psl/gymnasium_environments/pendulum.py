from neuromancer.psl.gymnasium_environments.gym_utils import wrap, sin_cos_to_t
from neuromancer.psl.base import ODE_NonAutonomous as ODE
from neuromancer.psl import signals

from typing import Union
import numpy as np
import torch

tens_or_float = Union[torch.Tensor, float]


class DifferentiablePendulum(ODE, torch.nn.Module):

    """
    A differentiable implementation of the Pendulum environment from
    Farama Foundation's Gymnasium.
    :param exclude_norms: list of norms to exclude from the stats dictionary
    :param backend: backend to use for the simulation
    :param requires_grad: whether to require gradients for the parameters
    :param seed: seed for the random number generator
    :param set_stats: whether to set the stats dictionary
    """

    def __init__(
        self,
        exclude_norms=["Time"],
        backend="numpy",
        requires_grad=False,
        seed: Union[int, np.random._generator.Generator] = 59,
        set_stats=True,
    ):
        self.rng = np.random.default_rng(seed=seed)
        self.exclude_norms = exclude_norms
        if not hasattr(self, "nsim"):
            self.nsim = 1001
        if not hasattr(self, "ts"):
            self.ts = 0.1
        (
            self.variables,
            self.constants,
            self._params,
            self.meta,
        ) = self.add_missing_parameters()
        self.set_params(self._params, requires_grad=requires_grad)
        self.set_params(self.variables, requires_grad=False)
        self.set_params(self.constants, requires_grad=False, cast=False)
        self.stats = {
            "X": {"max": self.xmax, "min": self.xmin},
            "U": {"max": self.umax, "min": self.umin},
        }
        torch.nn.Module.__init__(self)

    @property
    def umax(self):
        return torch.tensor([self.max_torque], dtype=torch.float32)

    @property
    def umin(self):
        return -self.umax

    @property
    def ymax(self):
        return torch.tensor([1.0, 1.0, self.max_speed], dtype=torch.float32)

    @property
    def ymin(self):
        return -self.ymax

    @property
    def xmax(self):
        return torch.tensor([torch.pi, self.max_speed], dtype=torch.float32)

    @property
    def xmin(self):
        return -self.xmax

    def get_x0(self):
        return torch.tensor(self.rng.uniform(self.xmin, self.xmax), dtype=torch.float32)

    def get_y0(self, x0=None):
        if x0 is None:
            x0 = self.get_x0()
        y0 = torch.cat([torch.cos(x0[0:1]), torch.sin(x0[0:1]), x0[1:2]], axis=0)
        return y0

    @property
    def params(self):
        variables = {
            "x0": torch.tensor([torch.pi, 0.0]),        # Downward position
            "optimal_state": torch.tensor([0, 1, 0]),   # Optimal state
        }
        constants = {
            "ts": 0.05,             # Time step size
            "max_speed": 8.0,       # Maximum speed of the pole
            "max_torque": 2.0,      # Maximum torque
            "g": 10.0,              # Acceleration due to gravity (m/s^2)
            "l": 1.0,               # Length of pole in m
            "m": 1.0,               # Pole mass in kg
            "nu": 1,                # Number of control inputs
            "nx": 2,                # Number of hidden states (theta, theta_dot)
            "ny": 3,                # Number of observable states (cos(theta), sin(theta), theta_dot)
        }
        parameters = {}
        meta = {}
        return variables, constants, parameters, meta

    def add_missing_parameters(self):
        '''
        Add missing parameters to the model
        '''
        variables, constants, parameters, meta = self.params
        if "U" not in variables:
            variables["U"] = torch.empty((1,), dtype=torch.float32)
        return variables, constants, parameters, meta

    def get_U(self, nsim=2, signal=signals.nd_walk, **signal_kwargs):
        """
        Get a sequence of control inputs
        :param nsim: number of time steps
        :param signal: signal generator
        :param signal_kwargs: keyword arguments for the signal generator
        :return: sequence of control inputs
        """
        U = signal(
            nsim=nsim,
            d=self.nu,
            min=self.umin,
            max=self.umax,
            rng=self.rng,
            **signal_kwargs,
        )
        U = torch.tensor(U, dtype=torch.float32)
        return U

    def set_params(self, parameters, requires_grad=False, cast=True):
        """
        Set parameters as attributes of the model
        :param parameters: dictionary of parameters
        :param requires_grad: whether the parameters require gradients
        :param cast: whether to cast the parameters to the backend
        """
        params_shapes = {
            k: v.shape[-1]
            for k, v in parameters.items()
            if hasattr(v, "shape") and len(v.shape) > 0
        }
        for k, v in parameters.items():
            setattr(self, k, v)
        for k, v in params_shapes.items():
            setattr(self, f"n{k}", v)

    def equations(self, t, x, u):
        '''
        Pendulum equations of motion
        :param t: time
        :param x: state vector
        :param u: control input
        :return: time derivative of the state vector
        '''
        th = x[:, 0:1]
        u = torch.clip(u, -self.max_torque, self.max_torque)
        d_thdot_dt = (
            3 * self.g / (2 * self.l) * torch.sin(th) + 3.0 / (self.m * self.l**2) * u
        )
        return d_thdot_dt

    def y_to_x(self, y):
        '''
        Convert observable states to hidden states
        :param y: observable states
        :return: hidden states
        '''
        sin_theta = y[:, 1:2]
        cos_theta = y[:, 0:1]
        theta = torch.atan2(sin_theta, cos_theta)
        x = torch.cat([theta, y[:, 2:3]], axis=1)
        return x

    def forward(self, y, u):
        '''
        Get the next observable state given the current observable state and control input
        :param y: observable state
        :param u: control input
        :return: observable state at the next time step
        '''
        x = self.y_to_x(y)
        d_thdot_dt = self.equations(None, x, u)
        thdot_old = x[:, 1:2]
        thdot_new = torch.clip(
            thdot_old + d_thdot_dt * self.ts, -self.max_speed, self.max_speed
        )
        th_old = x[:, 0:1]
        th_new = th_old + thdot_new * self.ts
        x_new = x.clone()
        x_new[:, 0:1] = wrap(th_new, -torch.pi, torch.pi)
        x_new[:, 1:2] = thdot_new
        Y = torch.cat(
            [
                torch.cos(x_new[:, 0:1]),
                torch.sin(x_new[:, 0:1]),
                x_new[:, 1:2],
            ],
            axis=1,
        )
        X = self.y_to_x(y=Y)

        return Y, X

    def simulate(self, U, x0):
        '''
        Create a rollout of the system that is the same length as the control
        sequence U.
        :param U: control sequence
        :param x0: initial state
        :return: dictionary of hidden states, observable states, and time steps
        '''
        nsim = U.shape[0]
        Time = torch.arange(0, nsim + 1) * self.ts
        X = torch.empty((nsim, self.nx), dtype=torch.float32, device=x0.device)
        x = x0
        for t in range(nsim):
            d_thdot_dt = self.equations(
                None, x.unsqueeze(dim=0), U[t].unsqueeze(dim=0)
            ).ravel()
            thdot = x[1]
            thdot = torch.clip(
                thdot + d_thdot_dt * self.ts, -self.max_speed, self.max_speed
            )
            th = x[0]
            th = th + thdot * self.ts
            x[0] = wrap(th, -torch.pi, torch.pi)
            x[1] = thdot
            X[t, :] = x[:]
        Y = torch.cat([torch.cos(X[:, 0:1]), torch.sin(X[:, 0:1]), X[:, 1:2]], axis=1)
        return {"Y": Y, "X": X, "Time": Time[1:]}

    def get_loss(self, y, u, breakdown=False):
        """
        Note: this should be called on the action y such that
        u = pi(y)
        :param y: observable state
        :param u: control input
        :param breakdown: whether to return a dictionary of contributions to the cost
        :return: total cost

        """
        while len(y.shape) < 3:
            y = y.unsqueeze(0)

        while len(y.shape) > 3:
            y = y.squeeze(0)

        th = sin_cos_to_t(torch.flip(y[:, :, :2], [2]))
        th = wrap(th, -torch.pi, torch.pi)
        thdot = y[:, :, 2]

        if breakdown:
            th_cost = th**2
            thdot_cost = 0.1 * thdot**2
            u_cost = 0.001 * (u**2)
            costs = th**2 + 0.1 * thdot**2 + 0.001 * (u**2)
            cost_dict = {
                "th": th_cost,
                "thdot": thdot_cost,
                "u": u_cost,
                "total": costs,
            }
            return cost_dict

        costs = th**2 + 0.1 * thdot**2 + 0.001 * (u**2)
        return costs
