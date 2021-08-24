"""
signal generator components which allow you to generate dynamic data for training nm.problem
or add noise to existing data

TODO: normalize to be mode general thant normalize_01

"""

# machine learning data science imports
import numpy as np
import torch
import random

from neuromancer.component import Component
from neuromancer.dataset import normalize_01 as normalize
import psl


class SequenceGenerator(Component):
    """
        Component object for generating nsteps long sequence signal for nx variables
        Can be used as a source for generating dynamic sequences in the training data
    """
    # TODO: does not work with output keys
    def __init__(self, nsteps, nx, xmax, xmin, output_keys=['Yp', 'Yf'], name='signal'):
        super().__init__(input_keys=[], output_keys=output_keys, name=name)
        self.nsteps, self.nx = nsteps, nx
        self.xmax, self.xmin, self.name = xmax, xmin, name
        self.iter = 0

    def get_xmax(self):
        return self.xmax

    def get_xmin(self):
        return self.xmin

    def forward(self, data):
        key = list(data.keys())[0]
        nsim = data[key].shape[1] * data[key].shape[0] + self.nsteps
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
        return {self.output_keys[0]: Rp, self.output_keys[1]: Rf}


class WhiteNoisePeriodicGenerator(SequenceGenerator):
    """
        Sum of wite noise with periodic signal
        Periodic signal with random number of periods within: min_period <= nr periods <= max_period
    """

    def __init__(self, nsteps, nx, xmax=(0.1, 0.5), xmin=0.0, min_period=5, max_period=30,
                 output_keys=['Yp', 'Yf'], name='period'):
        super().__init__(nsteps, nx, xmax, xmin, output_keys=output_keys, name=name)

        self.white_noise_generator = lambda nsim, xmin, xmax: psl.WhiteNoise(nx=self.nx, nsim=nsim,
                                                                             xmax=xmax, xmin=xmin)[:nsim]
        self.period_generator = lambda nsim, xmin, xmax: psl.Periodic(nx=self.nx, nsim=nsim,
                                                                      numPeriods=random.randint(min_period, max_period),
                                                                      xmax=xmax, xmin=xmin)[:nsim]
        self.sequence_generator = lambda nsim, xmin, xmax: self.period_generator(nsim, xmin, xmax) \
                                                           + self.white_noise_generator(nsim, xmin, 1.0 - xmax)

    def get_xmax(self):
        return random.uniform(*self.xmax)


class PeriodicGenerator(SequenceGenerator):
    """
        Generates periodic signal with random number of periods within: min_period <= nr periods <= max_period
    """

    def __init__(self, nsteps, nx, xmax, xmin, min_period=5, max_period=30,
                 output_keys=['Yp', 'Yf'],
                 name='period'):
        super().__init__(nsteps, nx, xmax, xmin, output_keys=output_keys, name=name)
        self.sequence_generator = lambda nsim: psl.Periodic(nx=self.nx, nsim=nsim, numPeriods=random.randint(min_period, max_period),
                                                            xmax=self.xmax, xmin=self.xmin)


class DataNoiseGenerator(Component):
    """
        Generates random signal to be added to variables the dataset labeled by input_keys
    """
    def __init__(self, ratio=0.05, input_keys=[], name='noise'):
        super().__init__(input_keys=input_keys, output_keys=input_keys, name=name)
        self.name = name
        self.ratio = ratio

    def forward(self, data):
        noisy_data = dict()
        for key in self.input_keys:
            noisy_data[key] = data[key] + self.ratio*torch.randn(data[key].shape)
        return noisy_data


class WhiteNoiseGenerator(SequenceGenerator):
    """
        Generates white noise nsteps long for nx variables
    """
    def __init__(self, nsteps, nx, xmax, xmin, output_keys=[], name='period'):
        super().__init__(nsteps, nx, xmax, xmin, output_keys=output_keys, name=name)
        self.sequence_generator = lambda nsim: psl.WhiteNoise(nx=self.nx, nsim=nsim,
                                                              xmax=self.xmax, xmin=self.xmin)


class AddGenerator(SequenceGenerator):
    """
        Generates a sequence as a sum of two other SequenceGenerators SG1 and SG2
    """
    def __init__(self, SG1, SG2, nsteps, nx, xmax, xmin, output_keys=[], name='period'):
        super().__init__(nsteps, nx, xmax, xmin, output_keys=output_keys, name=name)
        assert SG1.nsteps == SG2.nsteps, 'Nsteps must match to compose sequence generators'
        assert SG1.nx == SG2.nx, 'Nx must match to compose sequence generators'
        self.sequence_generator = lambda nsim: SG1.sequence_generator(nsim) + SG2.sequence_generator(nsim)


"""
# TODO: make this a proper unit test
if __name__ == '__main__':
    import os
    model_file = os.path.join(psl.resource_path, "Flexy_air/ape_models/best_model.pth")
    load_model = torch.load(model_file, pickle_module=dill, map_location=torch.device('cpu'))
    estimator = load_model.components[0]
    dynamics = load_model.components[1]
    dynamics.fyu = None
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
"""
