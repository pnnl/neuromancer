import numpy as np
import torch
import torch.nn as nn

from abc import ABC, abstractmethod


""" single-step integrators for first-order nonautomonomous ODEs """


class Integrator(nn.Module, ABC):

    def __init__(self, block, interp_u=None, h=1.0):
        """
        Integration block.

        :param block:
        :param interp_u:
        :param h:
        """
        super().__init__()
        self.block = block  # block gives dx at this time step.
        self.in_features, self.out_features = block.in_features, block.out_features
        self.interp_u = interp_u  # offline interpolant
        self.h = h
        if self.interp_u is not None:
            self.state = lambda x, tq, t, u: torch.cat([x, self.interp_u(tq, t, u)], dim=-1)
        else:
            self.state = lambda x, tq, t, u: x

    @abstractmethod
    def integrate(self):
        pass

    def forward(self, x, u=torch.empty((1, 1)), t=torch.empty((1, 1))):
        return self.integrate(x, u, t)


class Euler(Integrator):
    def __init__(self, block, interp_u=None, h=1.0):
        """

        :param block:
        :param interp_u:
        :param h:
        """
        super().__init__(block=block, interp_u=interp_u, h=h)

    def integrate(self, x, u, t):
        h = self.h
        k1 = self.block(self.state(x, t, t, u))        # k1 = f(x_i, t_i)
        return x + h*k1


class Euler_Trap(Integrator):
    def __init__(self,  block, interp_u=None, h=1.0):
        """
        Forward Euler (explicit). Trapezoidal rule (implicit).

        :param block: (nn.Module) A state transition model.
        :param h: (float) Time step size.
        """
        super().__init__(block=block, interp_u=interp_u, h=h)

    def integrate(self, x, u, t):
        """

        :param x: (torch.Tensor, shape=[batchsize, SysDim])
        :return x_{t+1}: (torch.Tensor, shape=[batchsize, SysDim])
        """
        pred = x + self.h * self.block(self.state(x, t, t, u))
        corr = x + 0.5 * self.h * (self.block(self.state(x, t, t, u)) + self.block(self.state(pred, self.h+t, t, u)))
        return corr


class RK2(Integrator):
    def __init__(self, block, interp_u=None, h=1.0):
        """

        :param block:
        :param interp_u:
        :param h:
        """
        super().__init__(block=block, interp_u=interp_u, h=h)

    def integrate(self, x, u, t):
        h = self.h
        k1 = self.block(self.state(x, t, t, u))                    # k1 = f(x_i, t_i)
        k2 = self.block(self.state(x + h*k1/2.0, t + h/2, t, u))   # k2 = f(x_i + 0.5*h*k1, t_i + 0.5*h)
        return x + h*k2


class RK4(Integrator):
    def __init__(self, block, interp_u=None, h=1.0):
        """

        :param block:
        :param interp_u:
        :param h:
        """
        super().__init__(block=block, interp_u=interp_u, h=h)

    def integrate(self, x, u, t):
        h = self.h
        k1 = self.block(self.state(x, t, t, u))                    # k1 = f(x_i, t_i)
        k2 = self.block(self.state(x + h*k1/2.0, t + h/2, t, u))   # k2 = f(x_i + 0.5*h*k1, t_i + 0.5*h)
        k3 = self.block(self.state(x + h*k2/2.0, t + h/2, t, u))   # k3 = f(x_i + 0.5*h*k2, t_i + 0.5*h)
        k4 = self.block(self.state(x + h*k3, t + h, t, u))         # k4 = f(y_i + h*k3, t_i + h)
        return x + h*(k1/6.0 + k2/3.0 + k3/3.0 + k4/6.0)


class RK4_Trap(Integrator):
    """
    predictor-corrector integrator for dx = f(x)
    predictor: explicit RK4
    corrector: implicit trapezoidal rule
    """
    def __init__(self, block, interp_u=None, h=1.0):
        super().__init__(block=block, interp_u=interp_u, h=h)

    def integrate(self, x, u, t):
        k1 = self.block(self.state(x, t, t, u))                    # k1 = f(x_i, t_i)
        k2 = self.block(self.state(x + self.h*k1/2.0, t + self.h/2, t, u))   # k2 = f(x_i + 0.5*h*k1, t_i + 0.5*h)
        k3 = self.block(self.state(x + self.h*k2/2.0, t + self.h/2, t, u))   # k3 = f(x_i + 0.5*h*k2, t_i + 0.5*h)
        k4 = self.block(self.state(x + self.h*k3, t + self.h, t, u))         # k4 = f(y_i + h*k3, t_i + h)
        pred = x + self.h*(k1/6.0 + k2/3.0 + k3/3.0 + k4/6.0)
        corr = x + 0.5*self.h*(self.block(self.state(x, t, t, u)) +
                               self.block(self.state(pred, t + self.h, t, u)))
        return corr


class Luther(Integrator):
    def __init__(self, block, interp_u=None, h=1.0):
        """

        :param block:
        :param interp_u:
        :param h:
        """
        super().__init__(block=block, interp_u=interp_u, h=h)

    def integrate(self, x, u, t):
        q = 21**0.5     # constant
        h = self.h         
        k1 = self.block(self.state(x, t, t, u))                    # k1 = f(x_i, t_i)
        k2 = self.block(self.state(x + h*k1, t + h, t, u))
        k3 = self.block(self.state(x + h*(3/8*k1 + 1/8*k2), t + h, t, u))
        k4 = self.block(self.state(x + h*(8/27*k1 + 2/27*k2 + 8/27*k3), t + h, t, u))
        k5 = self.block(self.state(x + h*((-21 + 9*q)/392*k1 +
                                          (-56 + 8*q)/392*k2 + (336 - 48*q)/392*k3 +
                                          (-63 + 3*q)/392*k4), t + h, t, u))
        k6 = self.block(self.state(x + h*((-1155 - 255*q)/1960*k1 +
                                          (-280-40*q)/1960*k2 - 320*q/1960*k3 +
                                          (63 + 363*q)/1960*k4 +
                                        (2352 + 392*q)/1960*k5), t + h, t, u))
        k7 = self.block(self.state(x + h*((330 + 105*q)/180*k1 + 120/180*k2 +
                                          (-200 + 280*q)/180*k3 + (126 - 189*q)/180*k4 +
                                          (-686 - 126*q)/180*k5 + (490 - 70*q)/180*k6), t + h, t, u))
        return x + h*(1/20*k1 + 16/45*k3 + 49/180*k5 + 49/180*k6 + 1/20*k7)


""" integrator reporting local truncation error"""

class Runge_Kutta_Fehlberg(Integrator):
    """
    The Runge–Kutta–Fehlberg method has two methods of orders 5 and 4. Therefore, we can calculate the local truncation error to
    determine if current time step size is suitable or not.
    # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Adaptive_Runge%E2%80%93Kutta_methods
    """

    def __init__(self, block, interp_u=None, h=1.0):
        """

        :param block: (nn.Module) A state transition model.
        :param interp_u:
        :param h: (float) Time step size.
        """
        super().__init__(block=block, interp_u=interp_u, h=h)
        self.local_error = []

    def integrate(self, x, u, t):
        """

        :param x: (torch.Tensor, shape=[batchsize, SysDim])
        :return x_{t+1}: (torch.Tensor, shape=[batchsize, SysDim])
        """
        h = self.h
        k1 = self.block(self.state(x, t, t, u))
        k2 = self.block(self.state(x + h*k1/4, t + h/4, t, u))
        k3 = self.block(self.state(x + 3 * h * k1 / 32 + 9 * h * k2 / 32, t + 3*h / 32 + 9 * h / 32, t, u))
        k4 = self.block(self.state(x + h * k1 * 1932 / 2197 - 7200 / 2197 * h * k2 + 7296 / 2197 * h * k3,
                                   t + h * 1932 / 2197 - 7200 / 2197 * h + 7296 / 2197 * h, t, u))
        k5 = self.block(self.state(x + h * k1 * 439 / 216 - 8 * h * k2 + 3680 / 513 * h * k3 - 845 / 4104 * h * k4,
                                   t + h * 439 / 216 - 8 * h + 3680 / 513 * h - 845 / 4104 * h, t, u))
        k6 = self.block(self.state(x - 8 / 27 * h * k1 + 2 * h * k2 - 3544 / 2565 * h * k3 +
                                   1859 / 4104 * h * k4 - 11 / 40 * h * k5,
                                   t - 8 / 27 * h + 2 * h - 3544 / 2565 * h + 1859 / 4104 * h - 11 / 40 * h, t, u))
        x_t1_high = x + h * (
                    k1 * 16 / 135 + k3 * 6656 / 12825 + k4 * 28561 / 56430 - 9 / 50 * k5 + k6 * 2 / 55)  # high order
        x_t1_low = x + h * (k1 * 25 / 216 + k3 * 1408 / 2565 + k4 * 2197 / 4104 - 1 / 5 * k5)  # low order
        self.local_error.append(x_t1_high - x_t1_low)
        return x_t1_high


# TODO: to be updated for non-autonomous systems - currently works only for autonomous
class MultiStep_PredictorCorrector(Integrator):
    def __init__(self, block, interp_u=None, h=1.0):
        """
        an only work for autonomous system for now,
        because there is dimensionality issue in self.state

        :param block:
        :param interp_u:
        :param h:
        """
        super().__init__(block=block, interp_u=interp_u, h=h)

    def integrate(self, x, u, t):
        """
        :param x: (torch.Tensor, shape=[nsteps, batchsize, SysDim]) where
                    x[0:1, :, :] = x_{t-3},
                    x[1:2, :, :] = x_{t-2},
                    x[2:3, :, :] = x_{t-1},
                    x[3:4, :, :] = x_{t}
        :param u: (torch.Tensor, shape=[nsteps, batchsize, nu])
        :param t: (torch.Tensor, shape=[nsteps, batchsize, 1])
        :return x_{t+1}: (torch.Tensor, shape=[batchsize, SysDim])
        """

        assert x.shape[0] == 4, "This four-step method requires x.shape[0] = 4."
        x0 = x[0, :, :]
        x1 = x[1, :, :]
        x2 = x[2, :, :]
        x3 = x[3, :, :]     # current state
        # assert u.shape[0] == 4,  "This four-step method requires u.shape[0] = 4."
        # u0 = u[0, :, :]
        # u1 = u[1, :, :]
        # u2 = u[2, :, :]
        # u3 = u[3, :, :]
        # assert t.shape[0] == 4,  "This four-step method requires t.shape[0] = 4."
        # t0 = t[0, :, :]
        # t1 = t[1, :, :]
        # t2 = t[2, :, :]
        # t3 = t[3, :, :]

        x4_pred = x3 + self.h*(55/24*self.block(self.state(x3, t, t, u)) -
                               59/24*self.block(self.state(x2, t-self.h, t, u)) +
                               37/24*self.block(self.state(x1, t-self.h*2, t, u)) -
                               9/24*self.block(self.state(x0, t-self.h*3, t, u)))
        x4_corr = x3 + self.h*(251/720*self.block(self.state(x4_pred, t+self.h, t, u)) +
                               646/720*self.block(self.state(x3, t, t, u)) -
                               264/720*self.block(self.state(x2, t-self.h, t, u)) +
                               106/720*self.block(self.state(x1, t-self.h*2, t, u)) -
                               19/720*self.block(self.state(x0, t-self.h*3, t, u)))
        return x4_corr  # (overlapse moving windows #, state dim) -> 2D tensor


integrators = {'Euler': Euler,
               'Euler_Trap': Euler_Trap,
               'RK2': RK2,
               'RK4': RK4,
               'RK4_Trap': RK4_Trap,
               'Luther': Luther,
               'Runge_Kutta_Fehlberg': Runge_Kutta_Fehlberg,
               }
