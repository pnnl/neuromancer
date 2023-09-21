"""
Single-step integrators for first-order nonautomonomous ODEs
"""
import warnings
import torch
import torch.nn as nn

from torchdiffeq import odeint_adjoint as odeint
import torchdiffeq
from abc import ABC, abstractmethod


class Integrator(nn.Module, ABC):

    def __init__(self, block, interp_u=None, h=1.0):
        """
        Integration block.

        :param block: (nn.Module or Callable) right hand side of the ODE system to be integrated
        :param h: (float) integration step size
        """
        super().__init__()
        self.block = block  # block gives dx at this time step.
        self.in_features, self.out_features = block.in_features, block.out_features
        self.h = h

        if interp_u is not None:
            warnings.warn('interp_u argument is deprecated, it has no effect, and will be removed in the next release.', FutureWarning)

    @abstractmethod
    def integrate(self, x, *args):
        pass

    def forward(self, x, *args):
        """
        This function needs x only for autonomous systems. x is 2D.
        It needs x and u for nonautonomous system w/ online interpolation. x and u are 2D tensors.
        """
        return self.integrate(x, *args)

    def reg_error(self):
        return sum([k.reg_error() for k in self.children() if hasattr(k, "reg_error")])


def rms_norm(tensor):
    return tensor.pow(2).mean().sqrt()


def make_norm(state):
    state_size = state.numel()
    def norm(aug_state):
        y = aug_state[1:1 + state_size]
        adj_y = aug_state[1 + state_size:1 + 2 * state_size]
        return max(rms_norm(y), rms_norm(adj_y))
    return norm


class DiffEqIntegrator(Integrator):
    """

    """
    def __init__(self, block, interp_u=None, h=0.001, method='euler'):
        """

        :param block:(nn.Module) A state transition model.
        :param interp_u: Function for interpolating control input values - deprecated
        :param h: (float) integration step size
        :param method: (str) Can be dopri8, dopri5, bosh3, fehlberg2, adaptive_heun, euler,
        midpoint, rk4, explicit_adams, implicit_adams, explicit_adams, implicit_adams, fixed_adams
        """
        super().__init__(block, interp_u=interp_u, h=h)
        self.method = method
        self.adjoint_params = torchdiffeq._impl.adjoint.find_parameters(self.block)

    def integrate(self, x, *args):
        t_0 = 0.0
        timepoints = torch.tensor([t_0, t_0 + self.h])
        rhs_fun = lambda t, x: self.block(x, *args)
        solution = odeint(rhs_fun, x, timepoints, method=self.method,
                          adjoint_params=self.adjoint_params,
                          adjoint_options=dict(norm=make_norm(x)))
        x_t = solution[-1]
        return x_t


class Euler(Integrator):
    def __init__(self, block, interp_u=None, h=1.0):
        """

        :param block: (nn.Module) A state transition model.
        :param h: (float) integration step size
        """
        super().__init__(block=block, interp_u=interp_u, h=h)

    def integrate(self, x, *args):
        h = self.h
        k1 = self.block(*[x, *args])        # k1 = f(x_i, t_i)
        return x + h*k1


class Euler_Trap(Integrator):
    def __init__(self, block, interp_u=None, h=1.0):
        """
        Forward Euler (explicit). Trapezoidal rule (implicit).

        :param block: (nn.Module) A state transition model.
        :param h: (float) integration step size
        """
        super().__init__(block=block, interp_u=interp_u, h=h)

    def integrate(self, x, *args):
        """

        :param x: (torch.Tensor, shape=[batchsize, SysDim])
        :return x_{t+1}: (torch.Tensor, shape=[batchsize, SysDim])
        """
        pred = x + self.h * self.block(x, *args)
        corr = x + 0.5 * self.h * (self.block(x, *args) + self.block(pred, *args))
        return corr


class RK2(Integrator):
    def __init__(self, block, interp_u=None, h=1.0):
        """

        :param block: (nn.Module) A state transition model.
        :param h: (float) integration step size
        """
        super().__init__(block=block, interp_u=interp_u, h=h)

    def integrate(self, x, *args):
        h = self.h
        k1 = self.block(x, *args)                    # k1 = f(x_i, t_i)
        k2 = self.block(x + h*k1/2.0, *args)         # k2 = f(x_i + 0.5*h*k1, t_i + 0.5*h)
        return x + h*k2


class RK4(Integrator):
    def __init__(self, block, interp_u=None, h=1.0):
        """

        :param block: (nn.Module) A state transition model.
        :param h: (float) integration step size
        """
        super().__init__(block=block, interp_u=interp_u, h=h)

    def integrate(self, x, *args):
        h = self.h
        k1 = self.block(x, *args)                    # k1 = f(x_i, t_i)
        k2 = self.block(x + h*k1/2.0, *args)         # k2 = f(x_i + 0.5*h*k1, t_i + 0.5*h)
        k3 = self.block(x + h*k2/2.0, *args)         # k3 = f(x_i + 0.5*h*k2, t_i + 0.5*h)
        k4 = self.block(x + h*k3, *args)             # k4 = f(y_i + h*k3, t_i + h)
        return x + h*(k1/6.0 + k2/3.0 + k3/3.0 + k4/6.0)


class RK4_Trap(Integrator):
    """
    predictor-corrector integrator for dx = f(x)
    predictor: explicit RK4
    corrector: implicit trapezoidal rule
    """
    def __init__(self, block, interp_u=None, h=1.0):
        """

        :param block: (nn.Module) A state transition model.
        :param h: (float) integration step size
        """
        super().__init__(block=block, interp_u=interp_u, h=h)

    def integrate(self, x, *args):
        k1 = self.block(x, *args)                     # k1 = f(x_i, t_i)
        k2 = self.block(x + self.h*k1/2.0, *args)     # k2 = f(x_i + 0.5*h*k1, t_i + 0.5*h)
        k3 = self.block(x + self.h*k2/2.0, *args)     # k3 = f(x_i + 0.5*h*k2, t_i + 0.5*h)
        k4 = self.block(x + self.h*k3, *args)         # k4 = f(y_i + h*k3, t_i + h)
        pred = x + self.h*(k1/6.0 + k2/3.0 + k3/3.0 + k4/6.0)
        corr = x + 0.5*self.h*(self.block(x, *args) + self.block(pred, *args) )
        return corr


class Luther(Integrator):
    def __init__(self, block, interp_u=None, h=1.0):
        """

        :param block: (nn.Module) A state transition model.
        :param h: (float) integration step size
        """
        super().__init__(block=block, interp_u=interp_u, h=h)

    def integrate(self, x, *args):
        q = 21**0.5     # constant
        h = self.h         
        k1 = self.block(x, *args)                    # k1 = f(x_i, t_i)
        k2 = self.block(x + h*k1, *args)
        k3 = self.block(x + h*(3/8*k1 + 1/8*k2), *args)
        k4 = self.block(x + h*(8/27*k1 + 2/27*k2 + 8/27*k3), *args)
        k5 = self.block(x + h*((-21 + 9*q)/392*k1 +
                                          (-56 + 8*q)/392*k2 + (336 - 48*q)/392*k3 +
                                          (-63 + 3*q)/392*k4), *args)
        k6 = self.block(x + h*((-1155 - 255*q)/1960*k1 +
                                          (-280-40*q)/1960*k2 - 320*q/1960*k3 +
                                          (63 + 363*q)/1960*k4 +
                                        (2352 + 392*q)/1960*k5), *args)
        k7 = self.block(x + h*((330 + 105*q)/180*k1 + 120/180*k2 +
                                          (-200 + 280*q)/180*k3 + (126 - 189*q)/180*k4 +
                                          (-686 - 126*q)/180*k5 + (490 - 70*q)/180*k6), *args)
        return x + h*(1/20*k1 + 16/45*k3 + 49/180*k5 + 49/180*k6 + 1/20*k7)


class Runge_Kutta_Fehlberg(Integrator):
    """
    The Runge–Kutta–Fehlberg method has two methods of orders 5 and 4. Therefore, we can calculate the local truncation error to
    determine if current time step size is suitable or not.
    # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Adaptive_Runge%E2%80%93Kutta_methods
    """

    def __init__(self, block, interp_u=None, h=1.0):
        """

        :param block: (nn.Module) A state transition model.
        :param h: (float) integration step size
        """
        super().__init__(block=block, interp_u=interp_u, h=h)
        self.local_error = []

    def integrate(self, x, *args):
        """

        :param x: (torch.Tensor, shape=[batchsize, SysDim])
        :return x_{t+1}: (torch.Tensor, shape=[batchsize, SysDim])
        """
        h = self.h
        k1 = self.block(x, *args)
        k2 = self.block(x + h*k1/4, *args)
        k3 = self.block(x + 3 * h * k1 / 32 + 9 * h * k2 / 32, *args)
        k4 = self.block(x + h * k1 * 1932 / 2197 - 7200 / 2197 * h * k2 + 7296 / 2197 * h * k3, *args)
        k5 = self.block(x + h * k1 * 439 / 216 - 8 * h * k2 + 3680 / 513 * h * k3 - 845 / 4104 * h * k4, *args)
        k6 = self.block(x - 8 / 27 * h * k1 + 2 * h * k2 - 3544 / 2565 * h * k3 +
                                   1859 / 4104 * h * k4 - 11 / 40 * h * k5, *args)
        x_t1_high = x + h * (
                    k1 * 16 / 135 + k3 * 6656 / 12825 + k4 * 28561 / 56430 - 9 / 50 * k5 + k6 * 2 / 55)  # high order
        x_t1_low = x + h * (k1 * 25 / 216 + k3 * 1408 / 2565 + k4 * 2197 / 4104 - 1 / 5 * k5)  # low order
        self.local_error.append(x_t1_high - x_t1_low)
        return x_t1_high


class MultiStep_PredictorCorrector(Integrator):
    def __init__(self, block, interp_u=None, h=1.0):
        """
        :param block: (nn.Module) A state transition model.
        :param h: (float) integration step size
        """
        super().__init__(block=block, interp_u=interp_u, h=h)

    def integrate(self, x, *args):
        """
        :param x: (torch.Tensor, shape=[batchsize, nsteps, SysDim]) where
                    x[0:1, :, :] = x_{t-3},
                    x[1:2, :, :] = x_{t-2},
                    x[2:3, :, :] = x_{t-1},
                    x[3:4, :, :] = x_{t}
        :return x_{t+1}: (torch.Tensor, shape=[batchsize, SysDim])
        """
        assert x.shape[1] == 4, "This four-step method requires x.shape[1] = 4."
        x0 = x[:, 0, :]
        x1 = x[:, 1, :]
        x2 = x[:, 2, :]
        x3 = x[:, 3, :]     # current state
        # Predictor: linear multistep Adams–Bashforth method (explicit)
        x4_pred = x3 + self.h*(55/24*self.block(x3, *args) -
                               59/24*self.block(x2, *args) +
                               37/24*self.block(x1, *args) -
                               9/24*self.block(x0, *args))
        # Corrector: linear multistep Adams–Moulton method (implicit)
        x4_corr = x3 + self.h*(251/720*self.block(x4_pred, *args) +
                               646/720*self.block(x3, *args) -
                               264/720*self.block(x2, *args) +
                               106/720*self.block(x1, *args) -
                               19/720*self.block(x0, *args))
        return x4_corr  # (overlapse moving windows #, state dim) -> 2D tensor


class LeapFrog(Integrator):
    def __init__(self, block, interp_u=None, h=1.0):
        """
        Leapfrog integration for ddx = f(x)
        https://en.wikipedia.org/wiki/Leapfrog_integration
        :param block: (nn.Module) A state transition model.
        :param h: (float) integration step size
        """
        super().__init__(block=block, interp_u=interp_u, h=h)

    def integrate(self, X, *args):
        """
        :param X: (torch.Tensor, shape=[batchsize, 2*SysDim]) where X[:, :SysDim] = x_t and X[:, SysDim:] = \dot{x}_t
        :return X_{t+1}: (torch.Tensor, shape=[batchsize, 2*SysDim]) where X_{t+1}[:, :SysDim] = x_{t+1} and X_{t+1}[:, SysDim:] = \dot{x}_{t+1}
        """
        SysDim = X.shape[-1]//2
        x = X[:, :SysDim]  # x at t = i*h
        dx = X[:, SysDim:2*SysDim]  # dx at t = i*h
        x_1 = x + dx*self.h + 0.5*self.block(x, *args)*self.h**2  # x at t = (i + 1)*h
        ddx_1 = self.block(x_1, *args)  # ddx at t = (i + 1)*h.
        dx_1 = dx + 0.5*(self.block(x, *args) + ddx_1)*self.h  # dx at t = (i + 1)*h
        return torch.cat([x_1, dx_1], dim=-1)


class Yoshida4(Integrator):
    def __init__(self, block, interp_u=None, h=1.0):
        """
        4th order Yoshida integrator for ddx = f(x). One step under the 4th order Yoshida integrator requires four intermediary steps. 
        https://en.wikipedia.org/wiki/Leapfrog_integration#4th_order_Yoshida_integrator
        :param block: (nn.Module) A state transition model.
        :param h: (float) integration step size
        """
        super().__init__(block=block, interp_u=interp_u, h=h)

    def integrate(self, X, *args):
        """
        :param X: (torch.Tensor, shape=[batchsize, 2*SysDim]) where X[:, :SysDim] = x_t and X[:, SysDim:] = \dot{x}_t
        :return X_{t+1}: (torch.Tensor, shape=[batchsize, 2*SysDim]) where X_{t+1}[:, :SysDim] = x_{t+1} and X_{t+1}[:, SysDim:] = \dot{x}_{t+1}
        """
        SysDim = X.shape[-1]//2
        x = X[:, :SysDim]  # x at t = i*h
        dx = X[:, SysDim:2*SysDim]  # dx at t = i*h
        # constants
        w0 = -2**(1/3)/(2 - 2**(1/3))
        w1 = 1/(2 - 2**(1/3))
        c1 = w1/2
        c4 = c1
        c2 = (w0 + w1)/2
        c3 = c2
        d1 = w1
        d3 = d1
        d2 = w0
        # intermediate step 1
        x_1 = x + c1*dx*self.h
        dx_1 = dx + d1*self.block(x_1, *args)*self.h
        # intermediate step 2
        x_2 = x_1 + c2*dx_1*self.h
        dx_2 = dx_1 + d2*self.block(x_2, *args)*self.h
        # intermediate step 3
        x_3 = x_2 + c3*dx_2*self.h
        dx_3 = dx_2 + d3*self.block(x_3, *args)*self.h
        # intermediate step 4
        x_4 = x_3 + c4*dx_3*self.h
        dx_4 = dx_3
        return torch.cat([x_4, dx_4], dim=-1)


integrators = {'Euler': Euler,
               'Euler_Trap': Euler_Trap,
               'RK2': RK2,
               'RK4': RK4,
               'RK4_Trap': RK4_Trap,
               'Luther': Luther,
               'Runge_Kutta_Fehlberg': Runge_Kutta_Fehlberg,
               'DiffEqIntegrator': DiffEqIntegrator}

integrators_multistep = {'MultiStep_PredictorCorrector': MultiStep_PredictorCorrector}  

integrators_second_order = {'LeapFrog': LeapFrog,
                            'Yoshida4': Yoshida4}