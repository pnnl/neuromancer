

# machine learning data science imports
import numpy as np
import torch
import torch.nn as nn
import random

from neuromancer.component import Component
from neuromancer.dataset import normalize_01 as normalize
import psl
import dill
import itertools


# TODO: move to dev
class NonlinearExpansion(Component):
    def __init__(self, keys=None, name='nlin_expand', device='cpu', order=2):
        super().__init__()
        self.name = name
        self.keys = keys
        self.device = device
        self.order = order
        self.output_keys = [f'{key}_{name}' for key in keys]

    def forward(self, data):
        nlin_data = dict()
        for key in self.keys:
            expansions = []
            for i, o in enumerate(range(1, self.order+1)):
                comb = itertools.combinations_with_replacement(range(data[key].shape[-1]), r=o)
                for j, c in enumerate(comb):
                    subset = torch.index_select(data[key], -1, torch.tensor(c, dtype=torch.long))
                    subset = torch.prod(subset, -1).unsqueeze(-1)
                    expansions.append(subset)
            nlin_data[f'{key}_{self.name}'] = torch.concat(expansions, dim=-1).to(self.device)

        return nlin_data


# TODO: generalize SignalGeneratorDynamics = move to dev for now
class SignalGeneratorDynamics(Component):
    def __init__(self, dynamics, estimator, nsteps, xmax=1.0, xmin=0.0,
                 name='signal_dynamics'):
        super().__init__(input_keys=[], output_keys=['Yp', 'Yf'], name=name)
        """
        dynamics 

        """
        self.nsteps = nsteps
        self.nbatch = None
        self.nu = dynamics.nu
        self.nd = dynamics.nd
        self.ny = dynamics.ny
        self.nx = dynamics.nx
        self.dynamics_input_keys = [k for k in dynamics.input_keys if 'x0' not in k]
        self.estimator_input_keys = estimator.input_keys
        self.data_dims_in = {k: v[-1] for k, v in estimator.data_dims.items() if k in estimator.input_keys}
        self.xmax, self.xmin, self.name = xmax, xmin, name
        self.estimator, self.dynamics = estimator, dynamics
        freeze_weight(self.estimator)
        freeze_weight(self.dynamics)

    def forward(self, data):
        with torch.no_grad():
            key = list(data.keys())[0]
            current_nbatch = data[key].shape[1]
            if self.nbatch is None:
                self.nbatch = current_nbatch
            state_estimator_input = dict()
            for input_signal, dim in self.data_dims_in.items():
                numPeriods = random.randint(self.nsteps // 4, self.nsteps // 2)
                if numPeriods > self.nsteps:
                    numPeriods = self.nsteps
                sample = psl.Periodic(nx=dim, nsim=self.nsteps,
                                      numPeriods=numPeriods,
                                      xmax=self.xmax, xmin=self.xmin)
                state_estimator_input[input_signal] = torch.tensor(sample).view(self.nsteps, 1, dim).float()
            estimator_output = self.estimator(state_estimator_input)
            ysignals = []
            dynamics_input = dict()
            x0 = estimator_output[f'x0_{self.estimator.name}']
            for i in range(self.nbatch + 5):
                for dim, name in zip([self.ny, self.nu, self.nd], self.dynamics_input_keys):
                    if dim > 0:
                        dynamics_input[name] = torch.tensor(psl.Periodic(nx=dim, nsim=self.nsteps,
                                                                         numPeriods=random.randint(self.nsteps // 4,
                                                                                                   self.nsteps // 2),
                                                                         xmax=self.xmax, xmin=self.xmin)).view(
                            self.nsteps, 1, dim).float()
                dynamics_input[f'x0_{self.estimator.name}'] = x0
                dynamics_output = self.dynamics({**estimator_output, **dynamics_input})
                x0 = dynamics_output[self.dynamics_model.output_keys[0]][-1]
                ysignals.append(dynamics_output[self.dynamics_model.output_keys[1]])

            if self.nbatch == current_nbatch:
                Yp = torch.stack(ysignals[:-5], dim=1).view(self.nsteps, self.nbatch, self.ny)
                Yf = torch.stack(ysignals[1:-4], dim=1).view(self.nsteps, self.nbatch, self.ny)
            else:
                Yp = torch.cat(ysignals[:-1]).squeeze(1)
                Yf = torch.cat(ysignals[1:]).squeeze(1)
                end_step_Yp = Yp.shape[0] - self.nsteps
                end_step_Yf = Yf.shape[0] - self.nsteps
                Yp = torch.stack([Yp[k:k + self.nsteps, :] for k in range(0, end_step_Yp)]).transpose(1,
                                                                                                      0)  # nsteps X nsamples X nfeatures
                Yf = torch.stack([Yf[k:k + self.nsteps, :] for k in range(0, end_step_Yf)]).transpose(1,
                                                                                                      0)  # nsteps X nsamples X nfeatures
                Yp = Yp[:, :2993, :]
                Yf = Yf[:, :2993, :]
        return {'Yp': Yp, 'Yf': Yf}
