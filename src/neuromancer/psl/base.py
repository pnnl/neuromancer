"""
Base classes for dynamic systems.

"""

import requests, functools, os
from abc import ABC, abstractmethod
import scipy, torch, torchdiffeq, numpy
import numpy as np
from neuromancer.psl.norms import StandardScaler, normalize, denormalize
from neuromancer.psl.signals import sines
import matplotlib.pyplot as plt
from typing import Union


def download(url, dst):
    """
    Function is used by FileEmulator and BuildingEnvelope classes to retrieve
    data we don't want to host on github.

    :param url: (str) Url to retrieve data from
    :param dst: (str pathlike) Where to store the data on disk
    """
    if not os.path.exists(dst):
        print(f'Downloading file {url} and saving to {dst}...')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        r = requests.get(url)
        with open(dst, 'wb') as f:
            f.write(r.content)


def grad(tensor, requires_grad):
    """
    Helper function to set gradient of tensors for pytorch backend
    :param tensor:
    :param requires_grad:
    :return:
    """
    tensor.requires_grad = requires_grad
    return tensor


class Backend:
    numpy_backend = {'odeint': functools.partial(scipy.integrate.odeint, tfirst=True),
                     'cat': numpy.concatenate,
                     'cast': numpy.array,
                     'core': numpy,
                     'grad': lambda x, requires_grad: x,
                     }
    torch_backend = {'odeint': torchdiffeq.odeint,
                     'cat': torch.cat,
                     'cast': torch.tensor,
                     'core': torch,
                     'grad': grad,
                     }
    backends = {'torch': torch_backend,
                'numpy': numpy_backend}

    def __init__(self, backend):
        """
        backend: can be torch or numpy
        """
        self.backend = backend
        for k, v in Backend.backends[backend].items():
            setattr(self, k, v)


class EquationWrapper:
    """
    The interface for odeint methods in torch and scipy does not handle exogenous inputs.
    This wrapper allows us to index control inputs by time point.
    """

    def __init__(self, Time, U, equations, backend):
        """

        :param Time: (1-D array of timepoints)
        :param U: (2-D array of control actions)
        :param equations: (Callable) Function with signature (t, x, u)
        """
        if backend.backend == 'numpy':
            self.ufunc = scipy.interpolate.interp1d(Time, U, kind='previous', axis=0, fill_value='extrapolate')
        elif backend.backend == 'torch':
            self.ufunc = scipy.interpolate.interp1d(Time, numpy.array(U.numpy(force=True)), kind='previous', axis=0, fill_value='extrapolate')
        self.equations = equations
        self.B = backend

    def __call__(self, t, x):
        return self.equations(t, x, self.B.cast(self.ufunc(t)))


def cast_backend(method):
    """
    Decorator to cast numerics to appropriate backend.

    :param method:
    :return:
    """
    @functools.wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        method_output = method(self, *method_args, **method_kwargs)
        if type(method_output) is dict:
            return {k: self.B.cast(v, dtype=self.B.core.float32) if k != 'ts' else v
                    for k, v in method_output.items()}
        return self.B.cast(method_output, dtype=self.B.core.float32)

    return _impl


class EmulatorBase(ABC, torch.nn.Module):
    def __init__(self, exclude_norms=['Time'], backend='numpy', requires_grad=False,
                 seed: Union[int,np.random._generator.Generator]=59, set_stats=True):
        """

        :param ts: (float) Time step for numerical integration
        :param seed: (int) Fixes random seed
        :param exclude_norms: (List of str) Keys to any data that should not be normalized such as 'Time'
        :param backend: (str) Can be 'torch' or 'numpy'
        :param requires_grad: (bool) Set this to true for parameter tuning with a pytorch backend
        """
        super().__init__()
        self.B = Backend(backend)
        self.rng = np.random.default_rng(seed=seed)
        self.exclude_norms = exclude_norms
        if not hasattr(self, 'nsim'):
            self.nsim = 1001
        if not hasattr(self, 'ts'):
            self.ts = 0.1
        self.variables, self.constants, self._params, self.meta = self.add_missing_parameters()
        self.set_params(self._params, requires_grad=requires_grad)
        self.set_params(self.variables, requires_grad=False)
        self.set_params(self.constants, requires_grad=False, cast=False)
        if set_stats:
            self.set_stats()

    def change_backend(self, backend=torch):
        self.B = Backend(backend)
        self.set_params(self._params)
        self.set_params(self.variables)

    def add_missing_parameters(self):
        return self.params

    def set_params(self, parameters, requires_grad=False, cast=True):
        if cast:
            parameters = {k: self.B.grad(self.B.cast(v, dtype=self.B.core.float32), requires_grad)
                          for k, v in parameters.items()}
        params_shapes = {k: v.shape[-1] for k, v in parameters.items()
                         if hasattr(v, 'shape') and len(v.shape) > 0}
        for k, v in parameters.items():
            setattr(self, k, v)
        for k, v in params_shapes.items():
            setattr(self, f'n{k}', v)

    @property
    def params(self):
        return {}, {}, {}, {}

    @property
    @params.setter
    def params(self, params):
        assert isinstance(params, dict), 'Need to set params with dictionary {str: numeric}'
        self._params = {**self._params, **params}
        self.set_params(params)

    @abstractmethod
    def simulate(self):
        pass

    def normalize(self, data, normalizer=None, key=None):
        """

        :param data: (Tensor, ndarray, or dict of tensor or ndarray)
        :param normalizer:
        :param key:
        :return:
        """
        if type(data) is not dict:
            return normalize(data, self.normalizers[key])
        if normalizer is None:
            norms = self.normalizers
        else:
            norms = normalizer
        return normalize(data, norms)

    def denormalize(self, data, normalizer=None, key=None):
        """

        :param data:
        :param normalizer:
        :param key:
        :return:
        """
        if type(data) is not dict:
            return denormalize(data, self.normalizers[key])
        if normalizer is None:
            norms = self.normalizers
        else:
            norms = normalizer
        return denormalize(data, norms)

    def set_stats(self, x0=None, U=None, D=None, nsim=None, sim=None):
        """
        Get a hyperbox defined by min and max values on each of nx axes. Used to sample initial conditions for simulations.
        Box is generated by simulating system with step size ts for nsim steps and then taking the min and max along each axis

        :param system: (psl.ODE_NonAutonomous)
        :param ts: (float) Timestep interval size
        :param nsim: (int) Number of simulation steps to use in defining box
        """
        if sim is None:
            if hasattr(self, 'x0'):
                x0 = self.x0 if x0 is None else self.B.cast(x0)
            nsim = self.nsim if nsim is None else nsim
            if hasattr(self, 'U'):
                U = self.U if U is None else self.B.cast(U)
                sim = self.simulate(x0=x0, U=U)
            elif hasattr(self, 'D'):
                D = self.D if D is None else self.B.cast(D)
                sim = self.simulate(x0=x0, U=U, D=D)
            elif hasattr(self, 'x0'):
                sim = self.simulate(x0=x0, nsim=nsim)
            else:
                sim = self.simulate(nsim=nsim)
        if self.B.core is numpy:
            self.stats = {k: {'min': v.min(axis=0), 'max': v.max(axis=0),
                              'mean': v.mean(axis=0), 'var': v.var(axis=0),
                              'std': v.std(axis=0)} for k, v in sim.items() if k not in self.exclude_norms}
        elif self.B.core is torch:
            self.stats = {k: {'min': v.min(axis=0)[0].detach(), 'max': v.max(axis=0)[0].detach(),
                              'mean': v.mean(axis=0).detach(), 'var': v.var(axis=0).detach(),
                              'std': v.std(axis=0).detach()} for k, v in sim.items() if k not in self.exclude_norms}

        self.stats_data = sim
        shapes = {k.lower(): v.shape[-1] for k, v in sim.items()
                  if hasattr(v, 'shape')}
        for k, v in shapes.items():
            setattr(self, f'n{k}', v)
        self.normalizers = {k: StandardScaler(v) for k, v in self.stats.items()}

    @cast_backend
    def get_x0(self):
        """
        Randomly sample an initial condition

        :param box: Dictionary with keys 'min' and 'max' and values np.arrays with shape=(nx,)
        """
        return self.rng.uniform(low=self.stats['X']['min'], high=self.stats['X']['max'])

    def _plot(self, data=None):
        """
        By default will plot the data used to generate initial system statistics from the canonical 1000 step simulation.

        :param data: (dict {str: Tensor or ndarray}) Will plot data from any system simulation given via this argument
        :param figname: (str) Optional name for figure. By default uses class name and saves figure as .png.
        """
        plt.close()
        data = self.stats_data if data is None else data
        figsize = 25
        nrows = self.ny
        nrows = nrows + self.nu if hasattr(self, 'nu') else nrows
        fig, ax = plt.subplots(nrows, figsize=(figsize, figsize))
        plt.xticks(fontsize=figsize)
        labels = [f'$y_{k}$' for k in range(data['Y'].shape[-1])]
        for row, (y, label) in enumerate(zip(data['Y'].T, labels)):
            axe = ax[row]
            axe.set_ylabel(label, rotation=0, labelpad=20, fontsize=figsize)
            axe.plot(y)
            axe.tick_params(labelbottom=False, labelsize=figsize)
        if hasattr(self, 'nu'):
            labels = [f'$u_{k}$' for k in range(data['U'].shape[-1])]
            for row, (u, label) in enumerate(zip(data['U'].T, labels)):
                axe = ax[row + self.ny]
                axe.set_ylabel(label, rotation=0, labelpad=20, fontsize=figsize)
                axe.plot(u)
                axe.tick_params(labelbottom=False, labelsize=figsize)
        axe.tick_params(labelbottom=True, labelsize=figsize)
        plt.tight_layout()

    def show(self, data=None, figname=None):
        """
        By default will plot the data used to generate initial system statistics from the canonical 1000 step simulation.

        :param data: (dict {str: Tensor or ndarray}) Will plot data from any system simulation given via this argument
        :param figname: (str) Optional name for figure. By default uses class name and saves figure as .png.
        """
        self._plot(data)
        if figname is not None:
            plt.savefig(figname)
            plt.close()
        else:
            plt.show()

    def save_random_state(self):
        """ Save random state for later use """
        self.rng_state = self.rng.bit_generator.state

    def restore_random_state(self, rng_state=None):
        """ Load random state """
        self.rng.bit_generator.state = self.rng_state if rng_state is None else \
                                       rng_state


class ODE_NonAutonomous(EmulatorBase):
    """
    base class non-autonomous ODE
    """

    def add_missing_parameters(self):
        variables, constants, parameters, meta = self.params
        if 'U' not in variables:
            variables['U'] = self.get_U(self.nsim + 1)
        return variables, constants, parameters, meta

    def get_simulation_args(self, nsim, Time, ts, x0, U):
        ts = self.ts if ts is None else ts
        if nsim is None:
            if Time is not None:
                nsim = len(Time) - 1
            elif U is not None:
                nsim = len(U) - 1
            else:
                nsim = self.nsim
        Time = self.B.core.arange(0, nsim + 1) * ts if Time is None else self.B.cast(Time)
        U = self.get_U(nsim + 1) if U is None else self.B.cast(U)
        x0 = self.get_x0() if x0 is None else self.B.cast(x0)
        assert x0.shape[0] == self.nx0, f"Mismatch in x0 size, nx: {self.nx0}, x0: {x0.shape[0]}"
        return Time, ts, x0, U

    @abstractmethod
    def equations(self):
        pass

    @cast_backend
    def forward(self, x, u):
        """
        For compatibility with the System class for open/closed loop simulations

        :param x: 2d Matrix (1, nx) # for torch backend generalize to (batchsize, nx)
        :param t: (1, 1)
        :param u: (1, nu)
        :return: x_next (1, nx)
        """
        U = self.B.cat([u, u], axis=0)
        Time = self.B.core.arange(0, 2) * self.ts
        equation = EquationWrapper(Time, U, self.equations, self.B)
        if self.B.core is torch:
            xdot = self.B.odeint(equation, x.flatten(), Time, options={"grid_points": Time, "eps": 1e-6})
        else:
            xdot = self.B.odeint(equation, x.flatten(), Time)
        return xdot[-1].reshape(1, -1)

    @cast_backend
    def simulate(self, nsim=None, Time=None, ts=None, x0=None, U=None):
        """
        :param nsim: (int) Number of steps for open loop response
        :param Time: (Sequence of float) Optional timesteps to integrate over.
        :param ts: (float) step size, sampling time
        :param x0: (float) state initial conditions
        :return: Dictionary containing  X, Y, U
        """
        Time, ts, x0, U = self.get_simulation_args(nsim, Time, ts, x0, U)
        equation = EquationWrapper(Time, U, self.equations, self.B)
        if self.B.core is torch:
            X = self.B.odeint(equation, x0, Time, options={"grid_points": Time, "eps": 1e-6})
        else:
            X = self.B.odeint(equation, x0, Time)
        return {'Y': X[1:], 'X': X[1:], 'U': U[1:], 'Time': Time[1:]}

    @cast_backend
    def get_U(self, nsim, umin=None, umax=None, signal=None, **signal_kwargs):
        """
        For sampling a sequence of control actions
        :param nsim: length of sequence
        :return: Matrix nsim X nU

        """
        if signal is None:
            return self.rng.normal(loc=self.stats['U']['mean'], scale=self.stats['U']['std'],
                                   size=(nsim, self.nu))
        if umin is None:
            umin = self.umin if hasattr(self, 'umin') else self.stats['U']['min']
        if umax is None:
            umax = self.umax if hasattr(self, 'umax') else self.stats['U']['max']
        d = umin.ravel().shape[0]
        return signal(nsim=nsim, d=d, min=umin, rng=self.rng, max=umax, **signal_kwargs)

    @cast_backend
    def get_R(self, nsim):
        """
        For sampling a sequence of reference trajectories

        :param nsim: (int) Length of sequence
        :return: Matrix nsim X nx0
        """
        return sines(nsim, self.nx0,
                     min=self.stats['X']['max'], max=self.stats['X']['min'], rng=self.rng)


class ODE_Autonomous(EmulatorBase):
    """
    base class autonomous ODE
    """

    @abstractmethod
    def equations(self):
        pass

    @cast_backend
    def forward(self, x, t):
        x, t = x.flatten(), t.flatten()[0]
        dT = [t, t+self.ts]
        xdot = self.B.odeint(self.equations, x, dT)
        return xdot[-1].reshape(1, -1)

    @cast_backend
    def simulate(self, nsim=None, Time=None, ts=None, x0=None):

        """
        :param nsim: (int) Number of steps for open loop response
        :param ninit: (float) initial simulation time
        :param ts: (float) step size, sampling time
        :param Time: (Sequence of float) Optional timesteps to integrate over.
        :param x0: (float) state initial conditions
        :return: The response matrices, i.e. X
        """
        nsim = nsim if nsim is not None else self.nsim
        ts = ts if ts is not None else self.ts
        x0 = x0 if x0 is not None else self.x0
        Time = Time if Time is not None else self.B.core.arange(0, nsim+1) * ts
        assert x0.shape[0] % self.nx0 == 0, "Mismatch in x0 size"
        X = self.B.odeint(self.equations, x0, Time)
        return {'Y': X[1:], 'X': X[1:], 'Time': Time[1:]}




