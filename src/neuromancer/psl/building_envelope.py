import os
from scipy.io import loadmat
import numpy as np
from neuromancer.psl.base import ODE_NonAutonomous, cast_backend, download
from neuromancer.psl.signals import periodic, noise, step
import functools


class BuildingEnvelope(ODE_NonAutonomous):
    """
    building envelope heat transfer model
    linear building envelope dynamics and bilinear heat flow input dynamics for
    different building types are downloaded as needed and stored at buildings_parameters/*.mat
    Models obtained from: https://github.com/drgona/BeSim
    """

    systems = ['SimpleSingleZone', 'Reno_full', 'Reno_ROM40', 'RenoLight_full',
               'RenoLight_ROM40', 'Old_full', 'Old_ROM40',
               'HollandschHuys_full', 'HollandschHuys_ROM100']

    T_dist_idx = {'Reno_full': [40], 'Reno_ROM40': [40],
                  'RenoLight_full': [40], 'RenoLight_ROM40': [40],
                  'Old_full': [40], 'Old_ROM40': [40],
                  'HollandschHuys_full': [221], 'HollandschHuys_ROM100': [221],
                  'SimpleSingleZone': [0]}

    @property
    def params(self):
        """

        :return: Four dicts (str: numeric)
                              parameters (could be optimized),
                              variables (exogenous inputs),
                              constants (don't vary from system to system),
                              Meta-data (physical units, system type, etc.)
        """
        p = loadmat(self.path)
        self.p = p
        nx = p['Ad'].shape[0]
        x0 = p['x0'].reshape(nx).astype(np.float32) if self.system == 'SimpleSingleZone' else np.zeros(nx, dtype=np.float32)
        self.dT_max = p['dT_max'].flatten().astype(np.float32)  # maximal temperature difference deg C
        self.dT_min = p['dT_min'].flatten().astype(np.float32)  # minimal temperature difference deg C
        self.nq = p['Bd'].shape[1]
        self.mf_max = p['mf_max'].flatten().astype(np.float32)  # maximal nominal mass flow l/h
        self.mf_min = p['mf_min'].flatten().astype(np.float32)
        self.d_idx = self.T_dist_idx[self.system]
        self.n_mf = p['Bd'].shape[1]
        self.n_dT = p['dT_max'].shape[0]
        variables = {'x0': x0,
                     'U': self.get_U(self.nsim + 1), # control actions
                     'D': p['disturb'][self.nsim + 1],
                     '_D': p['disturb'],
                     'D_obs': p['disturb'][self.nsim + 1][self.d_idx],
                     }
        constants = {'ts': 0.01,
                     # Heat flow equation constants
                     'rho': 0.997,  # density  of water kg/1l
                     'cp': 4185.5,  # specific heat capacity of water J/(kg/K)
                     'time_reg': 1. / 3600.,  # time regularization of the mass flow 1 hour = 3600 seconds
                     'nx': nx,
                     'ny': p['Cd'].shape[0],
                     'nq': p['Bd'].shape[1],
                     'nd': p['Ed'].shape[1],
                    }
        parameters = {'A': p['Ad'].astype(np.float32),  # Linear system parameters
                      'Beta': p['Bd'].astype(np.float32),  # Variable B used by backend
                      'C': p['Cd'].astype(np.float32),
                      'E': p['Ed'].astype(np.float32),
                      'G': p['Gd'].astype(np.float32),
                      'F': p['Fd'].astype(np.float32),
                      'y_ss': p['y_ss'].reshape(-1).astype(np.float32),  # Physical unit offset for observation
                      }
        meta = {'type': p['type'], 'HC_system': p['HC_system']}
        return variables, constants, parameters, meta

    @property
    def umax(self):
        """
        maximal nominal mass flow l/h, maximal temperature difference deg C
        """
        return np.concatenate([self.mf_max, self.dT_max])

    @property
    def umin(self):
        """
        minimal nominal mass flow l/h, minimal temperature difference deg C
        """
        return np.concatenate([self.mf_min, self.dT_min])

    def __init__(self, seed=59, exclude_norms=['Time'],
                 backend='numpy', requires_grad=False,
                 system='Reno_full', set_stats=True, *args, **kwargs):
        self.nsim = 6000
        self.system = system
        download(self.url, self.path)
        super().__init__(seed=seed, exclude_norms=exclude_norms,
                         backend=backend, requires_grad=requires_grad, set_stats=set_stats)

    def get_xy(self):
        x0 = self.x0
        y0 = self.C @ x0 + self.F.ravel() - self.y_ss
        return x0, y0

    @property
    def path(self):
        """
        Path where model parameter file is stored
        """
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'building_parameters', f'{self.system}.mat')

    @property
    def url(self):
        """
        Remote github location for model parameter data
        """
        return f'https://github.com/pnnl/psl/raw/master/psl/parameters/buildings/{self.system}.mat'

    @cast_backend
    def get_U(self, nsim, signal=None, **signal_kwargs):
        if signal is not None:
            return super().get_U(nsim=nsim, signal=signal, **signal_kwargs)

        return periodic(nsim, self.umin.shape[-1], max=(self.umax+self.umin) / 2., min=self.umin, periods=int(np.ceil(nsim / 48.)),
                        form='sin', rng=self.rng) + noise(nsim, self.umin.shape[-1], rng=self.rng)

    @cast_backend
    def get_q(self, u):
        m_flow = u[0:self.n_mf]
        dT = u[self.n_mf:self.n_mf + self.n_dT]
        q = m_flow * self.rho * self.cp * self.time_reg * dT
        return q

    @cast_backend
    def get_D(self, nsim):
        start_idx = self.rng.integers(0, len(self._D)-1-nsim)
        return self._D[start_idx:start_idx+nsim]

    @cast_backend
    def get_D_obs(self, nsim):
        start_idx = self.rng.integers(0, len(self._D) - 1 - nsim)
        return self._D[start_idx:start_idx + nsim, self.d_idx]

    @cast_backend
    def get_R(self, nsim):
        s = step(nsim, self.ny, randsteps=int(np.ceil(nsim / 24.)), min=self.stats['Y']['min'],
                 max=self.stats['Y']['max'], rng=self.rng)
        return s

    def forward(self, x, u, d):
        """
        For compatibility with the System class for open/closed loop simulations

        :param x: 2d Matrix (1, nx) # for torch backend generalize to (batchsize, nx)
        :param u: (1, nu)
        :param d: (1, nd)
        :return: x_next (1, nx), y_next (1, ny)
        """
        x, u, d = x.T, u.T, d.T
        x, y = self.equations(x, u, d)
        return x.T, y.T - self.y_ss

    def equations(self, x, u, d):
        G = self.G.ravel() if len(x.shape) == 1 else self.G
        F = self.F.ravel() if len(x.shape) == 1 else self.F
        q = self.get_q(u)
        x = self.A @ x + self.Beta @ q + self.E @ d + G
        y = self.C @ x + F
        return x, y

    def get_simulation_args(self, nsim, x0, U, D):
        nsim = self.nsim if nsim is None else nsim
        x0 = self.get_x0() if x0 is None else x0
        D = self.get_D(nsim+1) if D is None else D
        U = self.get_U(nsim+1) if U is None else U
        Time = self.B.core.arange(0, nsim + 1) * self.ts
        return nsim, x0, U, D, Time

    def simulate(self, nsim=None, U=None, D=None, x0=None, *args, **kwargs):
        """
        Simulate at a minimum needs the number of simulation steps. You can optionally supply U, D, and x0
        with or without nsim. If supplying U and D need to supply an extra time step of data.

        :param nsim: (int) Number of simulation steps
        :param U: (2D array or tensor)
        :param D:
        :param x0:
        :return:
        """
        nsim, x, U, D, Time = self.get_simulation_args(nsim, x0, U, D)
        X, Y = [], []
        for k in range(nsim):
            x, y = self.equations(x, U[k, :], D[k, :])
            X.append(x)
            Y.append(y - self.y_ss)
        Dhidden = D[1:]
        Dout = D[1:, self.d_idx]
        out = {'X': self.B.core.stack(X), 'Y': self.B.core.stack(Y), 'U': U[1:], 'D': Dout, 'Dhidden': Dhidden, 'Time': Time[1:]}
        return out


class LinearBuildingEnvelope(BuildingEnvelope):

    @cast_backend
    def get_U(self, nsim, signal=None, **signal_kwargs):
        if signal is not None:
            return super().get_U(nsim=nsim, signal=signal, **signal_kwargs)

        return periodic(nsim,
                        self.nq,
                        max=(self.umax+self.umin) / 2.,
                        min=self.umin,
                        periods=int(np.ceil(nsim / 48.)),
                        rng=self.rng) + noise(nsim, self.nq, rng=self.rng)

    @property
    def umax(self):
        return self.p['umax'].squeeze().astype(np.float32)  # max heat per zone

    @property
    def umin(self):
        return self.p['umin'].squeeze().astype(np.float32)  # min heat per zone

    def get_q(self, u):
        return u


bilinear_systems = {system: functools.partial(BuildingEnvelope, system=system) for system in BuildingEnvelope.systems}
linear_systems = {f'Linear{system}': functools.partial(LinearBuildingEnvelope, system=system) for system in BuildingEnvelope.systems}

systems = {**bilinear_systems, **linear_systems}


if __name__ == '__main__':
    for n, system in systems.items():
        print(n)
        s = system(backend='torch')
        out = s.simulate(nsim=5)


        print({k: v.shape for k, v in out.items()})

    for n, system in systems.items():
        print(n)
        s = system()
        out = s.simulate(nsim=5)
        print({k: v.shape for k, v in out.items()})
