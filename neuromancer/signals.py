"""
signal generator components for training

"""


# machine learning data science imports
import numpy as np
import torch
import torch.nn as nn
import random

from neuromancer.datasets import normalize, EmulatorDataset, FileDataset, systems
import psl
from collections import defaultdict
import dill



def freeze_weight(model, module_names=['']):
    """
    ['parent->child->child']
    :param component:
    :param module_names:
    :return:
    """
    modules = dict(model.named_modules())
    for name in module_names:
        freeze_path = name.split('->')
        if len(freeze_path) == 1:
            modules[name].requires_grad_(False)
        else:
            parent = modules[freeze_path[0]]
            freeze_weight(parent, ['->'.join(freeze_path[1:])])


class SignalGeneratorDynamics(nn.Module):
    def __init__(self, dynamics, estimator, nsteps, xmax=1.0, xmin=0.0, name='signal_dynamics'):
        super().__init__()
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
            # nsim = data[key].shape[1]*data[key].shape[0] + self.nsteps
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
            # TODO: this is a hack!!! FIX in FY21
            for i in range(self.nbatch+5):
                for dim, name in zip([self.ny, self.nu, self.nd], self.dynamics_input_keys):
                    if dim > 0:
                        dynamics_input[name] = torch.tensor(psl.Periodic(nx=dim, nsim=self.nsteps,
                                                            numPeriods=random.randint(self.nsteps//4, self.nsteps//2),
                                                            xmax=self.xmax, xmin=self.xmin)).view(self.nsteps, 1, dim).float()
                dynamics_input[f'x0_{self.estimator.name}'] = x0
                dynamics_output = self.dynamics({**estimator_output, **dynamics_input})
                x0 = dynamics_output[f'X_pred_{self.dynamics.name}'][-1]
                ysignals.append(dynamics_output[f'Y_pred_{self.dynamics.name}'])

            if self.nbatch == current_nbatch:
                # TODO: this is a hack!!! FIX in FY21
                Yp = torch.stack(ysignals[:-5], dim=1).view(self.nsteps, self.nbatch, self.ny)
                Yf = torch.stack(ysignals[1:-4], dim=1).view(self.nsteps, self.nbatch, self.ny)
            else:
                # TODO: this is a hack!!! FIX in FY21
                # print(ysignals[0].shape)
                # print(len(ysignals))
                Yp = torch.cat(ysignals[:-1]).squeeze(1)
                Yf = torch.cat(ysignals[1:]).squeeze(1)
                # print(Yp.shape)
                # print(Yf.shape)
                end_step_Yp = Yp.shape[0] - self.nsteps
                end_step_Yf = Yf.shape[0] - self.nsteps
                # print(end_step_Yp)
                # print(end_step_Yf)
                Yp = torch.stack([Yp[k:k + self.nsteps, :] for k in range(0, end_step_Yp)]).transpose(1, 0)  # nsteps X nsamples X nfeatures
                Yf = torch.stack([Yf[k:k + self.nsteps, :] for k in range(0, end_step_Yf)]).transpose(1, 0)  # nsteps X nsamples X nfeatures
                Yp = Yp[:, :2993, :]
                Yf = Yf[:, :2993, :]
        return {self.name + 'p': Yp, self.name + 'f': Yf}


class SignalGenerator(nn.Module):

    def __init__(self, nsteps, nx, xmax, xmin, name='signal'):
        super().__init__()
        self.nsteps, self.nx = nsteps, nx
        self.xmax, self.xmin, self.name = xmax, xmin, name
        self.iter = 0
    def get_xmax(self):
        return self.xmax

    def get_xmin(self):
        return self.xmin
# nstep X nsamples x nfeatures
#
#     end_step = data.shape[0] - nsteps
#     data = np.asarray([data[k:k+nsteps, :] for k in range(0, end_step)])  # nchunks X nsteps X nfeatures
#     return data.transpose(1, 0, 2)  # nsteps X nsamples X nfeatures

    def forward(self, data):

        key = list(data.keys())[0]
        nsim = data[key].shape[1]*data[key].shape[0] + self.nsteps
        nbatch = data[key].shape[1]

        xmax, xmin = self.get_xmax(), self.get_xmin()
        R = self.sequence_generator(nsim, xmax, xmin)
        R, _, _ = normalize(R)

        Rp, Rf = R[:-self.nsteps], R[self.nsteps:]

        if self.iter == 0:
            self.nstep_batch_size = nbatch
        if nbatch == self.nstep_batch_size:
            Rp = torch.tensor(Rp, dtype=torch.float32).view(nbatch, -1, self.nx).transpose(0, 1)
            Rf = torch.tensor(Rf, dtype=torch.float32).view(nbatch, -1, self.nx).transpose(0, 1)
        else:
            end_step = Rp.shape[0] - self.nsteps
            Rp = np.asarray([Rp[k:k+self.nsteps, :] for k in range(0, end_step)])  # nchunks X nsteps X nfeatures
            Rp = torch.tensor(Rp.transpose(1, 0, 2), dtype=torch.float32)  # nsteps X nsamples X nfeatures
            Rf = np.asarray([Rf[k:k + self.nsteps, :] for k in range(0, end_step)])  # nchunks X nsteps X nfeatures
            Rf = torch.tensor(Rf.transpose(1, 0, 2), dtype=torch.float32)  # nsteps X nsamples X nfeatures
        return {self.name + 'p': Rp, self.name + 'f': Rf}


class WhiteNoisePeriodicGenerator(SignalGenerator):

    def __init__(self, nsteps, nx, xmax=(0.1, 0.5), xmin=0.0, min_period=5, max_period=30, name='period'):
        super().__init__(nsteps, nx, xmax, xmin, name=name)

        self.white_noise_generator = lambda nsim, xmin, xmax: psl.WhiteNoise(nx=self.nx, nsim=nsim,
                                                                             xmax=xmax, xmin=xmin)[:nsim]
        self.period_generator = lambda nsim, xmin, xmax: psl.Periodic(nx=self.nx, nsim=nsim,
                                                                      numPeriods=random.randint(min_period, max_period),
                                                                      xmax=xmax, xmin=xmin)[:nsim]
        self.sequence_generator = lambda nsim, xmin, xmax: self.period_generator(nsim, xmin, xmax) + self.white_noise_generator(nsim, xmin, 1.0 - xmax)

    def get_xmax(self):
        return random.uniform(*self.xmax)


class PeriodicGenerator(SignalGenerator):

    def __init__(self, nsteps, nx, xmax, xmin, min_period=5, max_period=30, name='period'):
        super().__init__(nsteps, nx, xmax, xmin, name=name)
        self.sequence_generator = lambda nsim: psl.Periodic(nx=self.nx, nsim=nsim, numPeriods=random.randint(min_period, max_period),
                                                            xmax=self.xmax, xmin=self.xmin)


class WhiteNoiseGenerator(SignalGenerator):
    def __init__(self, nsteps, nx, xmax, xmin, name='period'):
        super().__init__(nsteps, nx, xmax, xmin, name=name)
        self.sequence_generator = lambda nsim: psl.WhiteNoise(nx=self.nx, nsim=nsim,
                                                              xmax=self.xmax, xmin=self.xmin)


class AddGenerator(SignalGenerator):
    def __init__(self, SG1, SG2, nsteps, nx, xmax, xmin, name='period'):
        super().__init__(nsteps, nx, xmax, xmin, name=name)
        assert SG1.nsteps == SG2.nsteps, 'Nsteps must match to compose sequence generators'
        assert SG1.nx == SG2.nx, 'Nx must match to compose sequence generators'
        self.sequence_generator = lambda nsim: SG1.sequence_generator(nsim) + SG2.sequence_generator(nsim)


if __name__ == '__main__':

    model_file = './datasets/Flexy_air/best_model_flexy1.pth'
    load_model = torch.load(model_file, pickle_module=dill, map_location=torch.device('cpu'))
    estimator = load_model.components[0]
    dynamics = load_model.components[1]
    nsteps = 32
    ny = load_model.components[1].fy.out_features
    dataset = FileDataset(system='flexy_air', nsim=10000,
                          norm=['U', 'D', 'Y'], nsteps=nsteps, device='cpu')
    dataset.min_max_norms['Ymin'] = dataset.min_max_norms['Ymin'][0]
    dataset.min_max_norms['Ymax'] = dataset.min_max_norms['Ymax'][0]
    nsim = dataset.data['Y'].shape[0]
    nu = dataset.data['U'].shape[1]
    new_sequences = {'Y_max': 0.8 * np.ones([nsim, 1]), 'Y_min': 0.2 * np.ones([nsim, 1]),
                     'U_max': np.ones([nsim, nu]), 'U_min': np.zeros([nsim, nu]),
                     'R': psl.Periodic(nx=1, nsim=nsim, numPeriods=20, xmax=0.7, xmin=0.3)}
    dataset.add_data(new_sequences)
    dynamics_generator = SignalGeneratorDynamics(dynamics, estimator, nsteps, xmax=1.0, xmin=0.0, name='Y_ctrl_')
    out = dynamics_generator(dataset.train_data)
    print(out['Y_ctrl_p'].shape)