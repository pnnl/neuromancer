from neuromancer.psl.gymnasium_environments.gym_utils import rk4, wrap, bound, two_D_rk4
from neuromancer.psl.base import ODE_NonAutonomous as ODE
from neuromancer.psl import signals

from typing import Union
import numpy as np
import torch

tens_or_float = Union[torch.Tensor, float]


class DifferentiableAcrobot(ODE, torch.nn.Module):

    """
    A differentiable implementation of the Acrobot environment from
    Farama Foundation's Gymnasium.

    The objective of in the original implementation is try to
    swing the end of the second link above a certain height. But
    in this implementation, the objective is to keep the end of the
    second link above a certain height.

    :param exclude_norms: list of norms to exclude from the model
    :param backend: backend for the model
    :param requires_grad: whether the model requires gradients
    :param seed: random seed
    :param set_stats: whether to set the statistics of the model
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
        return torch.tensor(
            [1.0, 1.0, 1.0, 1.0, self.max_vel_1, self.max_vel_2], dtype=torch.float32
        )

    @property
    def ymin(self):
        return -self.ymax

    @property
    def xmax(self):
        return torch.tensor(
            [torch.pi, torch.pi, self.max_vel_1, self.max_vel_2], dtype=torch.float32
        )

    @property
    def xmin(self):
        return -self.xmax

    @property
    def params(self):
        variables = {
            "x0": torch.tensor([0,] * 4),                       # Downward position
            "optimal_state": torch.tensor([0, 1, 0, 1, 0, 0]),  # Optimal state
        }
        constants = {
            "ts": 0.2,                  # Time step size
            "max_vel_1": 4 * torch.pi,  # Maximum angular velocity for the first link
            "max_vel_2": 9 * torch.pi,  # Maximum angular velocity for the second link
            "max_torque": 1.0,          # Maximum torque
            "g": 9.8,                   # Acceleration due to gravity (m/s^2)
            "link_length_1": 1.0,       # Length of pole in m
            "link_length_2": 1.0,       # Length of pole in m
            "link_com_pos_1": 0.5,      # Center of mass of pole in m
            "link_com_pos_2": 0.5,      # Center of mass of pole in m
            "link_mass_1": 1.0,         # Pole mass in kg
            "link_mass_2": 1.0,         # Pole mass in kg
            "link_moi": 1.0,            # Moment of inertia for both links
            "nu": 1,                    # Number of control inputs
            "nx": 4,                    # Number of hidden states (theta_1, theta_2, vel_1, vel_2)
            "ny": 6,                    # Number of observed states (cos_1, sin_1, cos_2, sin_2, vel_1, vel_1)
        }
        parameters = {}
        meta = {}
        return variables, constants, parameters, meta

    def add_missing_parameters(self):
        """
        Add missing parameters to the model
        """
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

    def get_x0(self, xmin: tens_or_float = None, xmax: tens_or_float = None):
        """
        Get a random initial state
        :param xmin: minimum value for each state
        :param xmax: maximum value for each state
        :return: initial state
        """
        if isinstance(xmin, float):
            xmin = torch.tensor((xmin for n in range(self.nx)), dtype=torch.float32)
        if isinstance(xmax, float):
            xmax = torch.tensor((xmax for n in range(self.nx)), dtype=torch.float32)

        xmin = self.xmin if xmin is None else xmin
        xmax = self.xmax if xmax is None else xmax
        return torch.tensor(self.rng.uniform(xmin, xmax), dtype=torch.float32)

    def get_y0(self, x0=None):
        """
        Get the initial condition as an obeservable state
        :param x0: initial state
        :return: initial observable state
        """

        if x0 is None:
            x0 = self.get_x0()
        y0 = torch.cat(
            [
                torch.cos(x0[0:1]),
                torch.sin(x0[0:1]),
                torch.cos(x0[1:2]),
                torch.sin(x0[1:2]),
                x0[2:],
            ],
            axis=0,
        )
        return y0

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
        """
        Acrobot equations of motion
        :param t: time
        :param x: state
        :param u: control input
        :return: state derivative
        """
        m1 = self.link_mass_1
        m2 = self.link_mass_2
        l1 = self.link_length_1
        lc1 = self.link_com_pos_1
        lc2 = self.link_com_pos_2
        I1 = self.link_moi
        I2 = self.link_moi
        g = 9.8
        a = u
        s = x
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        d1 = (
            m1 * lc1**2
            + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * torch.cos(theta2))
            + I1
            + I2
        )
        d2 = m2 * (lc2**2 + l1 * lc2 * torch.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * torch.cos(theta1 + theta2 - torch.pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2**2 * torch.sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * torch.sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * torch.cos(theta1 - torch.pi / 2)
            + phi2
        )
        ddtheta2 = (
            a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * torch.sin(theta2) - phi2
        ) / (m2 * lc2**2 + I2 - d2**2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return dtheta1, dtheta2, ddtheta1, ddtheta2, 0

    def y_to_x(self, y):
        """
        Convert observable state to hidden state
        :param y: observable state
        :return: hidden state
        """
        sin_theta_1 = y[:, 1:2]
        cos_theta_1 = y[:, 0:1]
        theta_1 = torch.atan2(sin_theta_1, cos_theta_1)

        sin_theta_2 = y[:, 3:4]
        cos_theta_2 = y[:, 2:3]
        theta_2 = torch.atan2(sin_theta_2, cos_theta_2)
        x = torch.cat([theta_1, theta_2, y[:, 4:]], axis=1)
        return x

    def forward(self, y, u):
        """
        Get the next observable state given the current observable state and control input
        :param y: observable state
        :param u: control input
        :return: observable state at the next time step
        """
        x = self.y_to_x(y)
        s_augmented = torch.cat([x, u], dim=1)
        x = two_D_rk4(self.equations, s_augmented, [0, self.ts])
        xnext = torch.cat(
            [
                wrap(x[:, 0:2], -torch.pi, torch.pi),
                bound(x[:, 2:3], -self.max_vel_1, self.max_vel_1),
                bound(x[:, 3:4], -self.max_vel_2, self.max_vel_2),
                x[:, 4:],
            ],
            axis=1,
        )

        Y = torch.cat(
            [
                torch.cos(xnext[:, 0:1]),
                torch.sin(xnext[:, 0:1]),
                torch.cos(xnext[:, 1:2]),
                torch.sin(xnext[:, 1:2]),
                xnext[:, 2:3],
                xnext[:, 3:4],
            ],
            axis=1,
        )
        return Y

    def simulate(self, U, x0):
        '''
        Create a rollout of the system that is the same length as the control
        sequence U.
        :param U: control sequence
        :param x0: initial state
        :return: dictionary of hidden states, observable states, and time steps
        '''
        if len(U.shape) < 2:
            U = U.unsqueeze(0)

        nsim = U.shape[0]
        Time = torch.arange(0, nsim + 1) * self.ts
        X = torch.empty((nsim, self.nx), dtype=torch.float32, device=x0.device)
        x = x0
        U = torch.clip(U, min=-1, max=1)

        for t in range(nsim):
            s_augmented = torch.cat([x, U[t].reshape(-1)], dim=0)
            x = rk4(self.equations, s_augmented, [0, self.ts])

            x[0] = wrap(x[0], -torch.pi, torch.pi)
            x[1] = wrap(x[1], -torch.pi, torch.pi)
            x[2] = bound(x[2], -self.max_vel_1, self.max_vel_1)
            x[3] = bound(x[3], -self.max_vel_2, self.max_vel_2)

            X[t, :] = x[:]

        Y = torch.cat(
            [
                torch.cos(X[:, 0:1]),  # Theta 1
                torch.sin(X[:, 0:1]),
                torch.cos(X[:, 1:2]),  # Theta 2
                torch.sin(X[:, 1:2]),
                X[:, 2:3],  # Angular Vel 1
                X[:, 3:4],  # Angular Vel 2
            ],
            dim=1,
        )

        return {"Y": Y, "X": X, "Time": Time[1:]}

    def get_loss(self, y, u):
        """
        note: this should be called on y caused by u.
        y = p(y_prev, u)

        from farama:
            reward = bool(-cos(s[0]) - cos(s[1] + s[0]) > 1.0)

        cosine sum formula:
            cos(s[1] + s[0]) = cos(s[1]) cos(s[0]) - sin(s[1]) sin(s[0])

        loss = -reward
             = cos(s[0]) + cos(s[1] + s[0]) < -1.0
             = cos0 + (cos0*cos1 - sin0*sin1) < -1.0

        :param y: observable state
        :param u: control input
        :return: loss
        """
        if len(y.shape) < 2:
            y = y.unsqueeze(0)
        elif len(y.shape) > 2:
            y = y.squeeze(0)

        cos0, sin0 = y[:, 0], y[:, 1]
        cos1, sin1 = y[:, 2], y[:, 3]
        terminal = cos0 + (cos0 * cos1 - sin0 * sin1) < -1.0
        terminal = terminal.type(torch.float32)
        loss = 1 - terminal
        return loss
