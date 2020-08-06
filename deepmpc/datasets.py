"""
# TODO Mini-batching
# TODO: Generalize batching with sliding window size. 1 = moving horizon, Nstep = contiguous chunks, generally 1< k < Nstep for midstep

"""
# python base imports
import os
from typing import Dict

# machine learning/data science imports
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# ecosystem imports
import slip as emulators

# local imports
import deepmpc.plot as plot


def min_max_denorm(M, Mmin, Mmax):
    """
    denormalize min max norm
    :param M: (2-d np.array) Data to be normalized
    :param Mmin: (int) Minimum value
    :param Mmax: (int) Maximum value
    :return: (2-d np.array) Un-normalized data
    """
    M_denorm = M*(Mmax - Mmin) + Mmin
    return np.nan_to_num(M_denorm)


def batch_mh_data(data, nsteps):
    """
    moving horizon batching

    :param data: np.array shape=(nsim, dim)
    :param nsteps: (int) n-step prediction horizon
    :return: np.array shape=(nsteps, nsamples, dim)
    """
    end_step = data.shape[0] - nsteps
    data = np.asarray([data[k:k+nsteps, :] for k in range(0, end_step)])  # nchunks X nsteps X nfeatures
    return data.transpose(1, 0, 2)  # nsteps X nsamples X nfeatures


def batch_data(data, nsteps):
    """

    :param data: np.array shape=(nsim, dim)
    :param nsteps: (int) n-step prediction horizon
    :return: np.array shape=(nsteps, nsamples, dim)
    """
    nsplits = (data.shape[0]) // nsteps
    leftover = (data.shape[0]) % nsteps
    data = np.stack(np.split(data[:data.shape[0] - leftover], nsplits))  # nchunks X nsteps X nfeatures
    return data.transpose(1, 0, 2)  # nsteps X nsamples X nfeatures


def unbatch_mh_data(data):
    """
    Data put back together into original sequence from moving horizon dataset.

    :param data: (torch.Tensor or np.array, shape=(nsteps, nsamples, dim)
    :return:  (torch.Tensor, shape=(nsim, 1, dim)
    """
    data_unmove = np.asarray([data[0, k, :] for k in range(0, data.shape[1])])
    if isinstance(data, torch.Tensor):
        data_unmove = torch.Tensor(data_unmove)
    return data_unmove.reshape(-1, 1, data_unmove.shape[-1])


def unbatch_data(data):
    """
    Data put back together into original sequence.

    :param data: (torch.Tensor or np.array, shape=(nsteps, nsamples, dim)
    :return:  (torch.Tensor, shape=(nsim, 1, dim)
    """
    if isinstance(data, torch.Tensor):
        return data.transpose(1, 0).reshape(-1, 1, data.shape[-1])
    else:
        return data.transpose(1, 0, 2).reshape(-1, 1, data.shape[-1])


class DataDict(dict):
    """
    So we can add a name property to the dataset dictionaries
    """
    pass


class Dataset:

    def __init__(self, system=None, nsim=1000, norm=['Y'], batch_type='batch',
                 nsteps=1, device='cpu', sequences=dict(), name='openloop',
                 savedir='test'):
        """

        :param system: (str) Identifier for dataset.
        :param nsim: (int) Total number of time steps in data sequence
        :param norm: (str) String of letters corresponding to data to be normalized, e.g. ['Y','U','D']
        :param batch_type: (str) Type of the batch generator, expects: 'mh' or 'batch'
        :param nsteps: (int) N-step prediction horizon for batching data
        :param device: (str) String identifier of device to place data on, e.g. 'cpu', 'cuda:0'
        :param sequences: (dict str: np.array) Dictionary of supplemental data
        :param name: (str) String identifier of dataset type, must be ['static', 'openloop', 'closedloop']
        """
        assert not (system is None and len(sequences) == 0), 'Trying to instantiate an empty dataset.'
        self.name = name
        self.savedir = savedir
        self.system, self.nsim, self.norm, self.nsteps, self.device = system, nsim, norm, nsteps, device
        self.batch_type = batch_type
        self.sequences = sequences
        self.data = self.load_data()
        self.add_data(sequences)
        self.min_max_norms, self.dims,  self.nstep_data, self.shift_data = dict(), dict(), dict(), dict()
        for k, v in self.data.items():
            self.dims[k] = v.shape
        self.dims['nsim'] = v.shape[0]
        self.dims['nsteps'] = self.nsteps

        assert set(norm) & set(self.data.keys()) == set(norm), \
            f'Specified keys to normalize: {list(set(norm))} are not in dataset keys: {list(self.data.keys())}'
        self.data = self.norm_data(self.data, self.norm)
        self.make_nstep()
        self.make_loop()

    def norm_data(self, data, norm):
        for k, v in data.items():
            v = v.reshape(v.shape[0], -1)
            if k in norm:
                v, vmin, vmax = self.normalize(v)
                self.min_max_norms.update({k + 'min': vmin, k + 'max': vmax})
                data[k] = v
        return data

    def make_nstep(self, overwrite=False):
        for k, v in self.data.items():
            if k + 'p' not in self.shift_data or overwrite:
                self.dims[k + 'p'], self.dims[k + 'f'] = v.shape, v.shape
                self.shift_data[k + 'p'] = v[:-self.nsteps]
                self.shift_data[k + 'f'] = v[self.nsteps:]

                if self.batch_type == 'mh':
                    self.nstep_data[k + 'p'] = batch_mh_data(self.shift_data[k + 'p'], self.nsteps)
                    self.nstep_data[k + 'f'] = batch_mh_data(self.shift_data[k + 'f'], self.nsteps)
                else:
                    self.nstep_data[k + 'p'] = batch_data(self.shift_data[k + 'p'], self.nsteps)
                    self.nstep_data[k + 'f'] = batch_data(self.shift_data[k + 'f'], self.nsteps)
                self.data[k] = v
        plot.plot_traj(self.data, figname=os.path.join(self.savedir, f'{self.system}.png'))
        self.train_data, self.dev_data, self.test_data = self.split_train_test_dev(self.nstep_data)

    def make_loop(self):
        if self.name == 'openloop':
            if self.batch_type == 'mh':
                self.train_loop = self.unbatch_mh(self.train_data)
                self.dev_loop = self.unbatch_mh(self.dev_data)
                self.test_loop = self.unbatch_mh(self.test_data)
            else:
                self.train_loop = self.unbatch(self.train_data)
                self.dev_loop = self.unbatch(self.dev_data)
                self.test_loop = self.unbatch(self.test_data)

            all_loop = {k: np.concatenate([self.train_loop[k], self.dev_loop[k], self.test_loop[k]]).squeeze(1)
                        for k in self.train_data.keys()}
            for k in self.train_data.keys():
                assert np.array_equal(all_loop[k], self.shift_data[k][:all_loop[k].shape[0]]), \
                    f'Reshaped data {k} is not equal to truncated original data'
            plot.plot_traj(all_loop, figname=os.path.join(self.savedir, f'{self.system}_open.png'))
            plt.close('all')

        elif self.name == 'closedloop':
            nstep_data = dict()
            for k, v in self.data.items():
                nstep_data[k + 'p'] = batch_mh_data(self.shift_data[k + 'p'], self.nsteps)
                nstep_data[k + 'f'] = batch_mh_data(self.shift_data[k + 'f'], self.nsteps)
            self.train_loop, self.dev_loop, self.test_loop = self.split_train_test_dev(nstep_data)

        self.train_data.name, self.dev_data.name, self.test_data.name = 'nstep_train', 'nstep_dev', 'nstep_test'
        self.train_loop.name, self.dev_loop.name, self.test_loop.name = 'loop_train', 'loop_dev', 'loop_test'
        for dset in self.train_data, self.dev_data, self.test_data, self.train_loop, self.dev_loop, self.test_loop:
            for k, v in dset.items():
                dset[k] = torch.tensor(v, dtype=torch.float32).to(self.device)

    def add_data(self, sequences, norm=[]):
        for k, v in sequences.items():
            assert v.shape[0] == self.dims['nsim']
        for k in norm:
            self.norm.append(k)
        sequences = self.norm_data(sequences, norm)
        self.data = {**self.data, **sequences}

    def del_data(self, keys):
        for k in keys:
            del self.data[k]

    def add_variable(self, var: Dict[str, int]):
        for k, v in var.items():
            self.dims[k] = v

    def del_variable(self, keys):
        for k in keys:
            del self.dims[k]

    def load_data(self):
        assert self.system is None and len(self.sequences) > 0, \
            'User must provide data via the sequences argument for basic Dataset. ' +\
            'Use FileDataset or EmulatorDataset for predefined datasets'
        return dict()

    def normalize(self, M, Mmin=None, Mmax=None):
            """
            :param M: (2-d np.array) Data to be normalized
            :param Mmin: (int) Optional minimum. If not provided is inferred from data.
            :param Mmax: (int) Optional maximum. If not provided is inferred from data.
            :return: (2-d np.array) Min-max normalized data
            """
            Mmin = M.min(axis=0).reshape(1, -1) if Mmin is None else Mmin
            Mmax = M.max(axis=0).reshape(1, -1) if Mmax is None else Mmax
            M_norm = (M - Mmin) / (Mmax - Mmin)
            return np.nan_to_num(M_norm), Mmin.squeeze(), Mmax.squeeze()

    def split_train_test_dev(self, data):
        """

        :param data: (dict, str: 3-d np.array) Complete dataset. dims=(nsteps, nsamples, dim)
        :return: (3-tuple) Dictionarys for train, dev, and test sets
        """
        train_data, dev_data, test_data = DataDict(), DataDict(), DataDict()
        train_idx = (list(data.values())[0].shape[1] // 3)
        dev_idx = train_idx * 2
        for k, v in data.items():
            if data is not None:
                train_data[k] = v[:, :train_idx, :]
                dev_data[k] = v[:, train_idx:dev_idx, :]
                test_data[k] = v[:, dev_idx:, :]
        return train_data, dev_data, test_data

    def unbatch(self, data):
        """

        :param data: (dict, str: 3-d np.array) Data broken into samples of n-step prediction horizon sequences
        :return: (dict, str: 3-d np.array) Data put back together into original sequence. dims=(nsim, 1, dim)
        """
        unbatched_data = DataDict()
        for k, v in data.items():
            unbatched_data[k] = unbatch_data(v)
        return unbatched_data

    def unbatch_mh(self, data):
        """

        :param data: (dict, str: 3-d np.array) Data broken into samples of n-step prediction horizon sequences
        :return: (dict, str: 3-d np.array) Data put back together into original sequence. dims=(nsim, 1, dim)
        """
        unbatched_data = DataDict()
        for k, v in data.items():
            unbatched_data[k] = unbatch_mh_data(v)
        return unbatched_data


class EmulatorDataset(Dataset):

    def load_data(self):
        """
        dataset creation from the emulator. system argument to init should be the name of a registered emulator
        return: (dict, str: 2-d np.array)
        """
        systems = emulators.systems  # list of available emulators
        model = systems[self.system](nsim=self.nsim)  # instantiate model class
        X, Y, U, D = model.simulate(nsim=self.nsim)  # simulate open loop
        data = dict()
        for d, k in zip([Y, U, D], ['Y', 'U', 'D']):
            if d is not None:
                data[k] = d
        return data


class FileDataset(Dataset):

    def load_data(self):
        """
        Load data from files. system argument to init should be a file path to the dataset file
        :return: (dict, str: 2-d np.array)
        """
        ninit = 0
        Y, U, D = None, None, None
        resource_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
        systems_datapaths = {'tank': os.path.join(resource_path, 'NLIN_SISO_two_tank/NLIN_two_tank_SISO.mat'),
                             'vehicle3': os.path.join(resource_path, 'NLIN_MIMO_vehicle/NLIN_MIMO_vehicle3.mat'),
                             'aero': os.path.join(resource_path, 'NLIN_MIMO_Aerodynamic/NLIN_MIMO_Aerodynamic.mat'),
                             'flexy_air': os.path.join(resource_path, 'Flexy_air/flexy_air_data.csv')}
        if self.system in systems_datapaths.keys():
            file_path = systems_datapaths[self.system]
        else:
            file_path = self.system
        if not os.path.exists(file_path):
            raise ValueError(f'No file at {file_path}')
        file_type = file_path.split(".")[-1]
        if file_type == 'mat':
            file = loadmat(file_path)
            Y = file.get("y", None)  # outputs
            U = file.get("u", None)  # inputs
            D = file.get("d", None)  # disturbances
        elif file_type == 'csv':
            data = pd.read_csv(file_path)
            Y = data.filter(regex='y').values if data.filter(regex='y').values.size != 0 else None
            U = data.filter(regex='u').values if data.filter(regex='u').values.size != 0 else None
            D = data.filter(regex='d').values if data.filter(regex='d').values.size != 0 else None
        data = dict()
        for d, k in zip([Y, U, D], ['Y', 'U', 'D']):
            if d is not None:
                data[k] = d[ninit:self.nsim, :]
        return data


class DatasetMPP:

    def __init__(self, norm=[], device='cpu', sequences=dict(), name='mpp',
                 savedir='test'):
        """

        :param system: (str) Identifier for dataset.
        :param nsim: (int) Total number of time steps in data sequence
        :param norm: (str) String of letters corresponding to data to be normalized, e.g. ['Y','U','D']
        :param batch_type: (str) Type of the batch generator, expects: 'mh' or 'chunk'
        :param nsteps: (int) N-step prediction horizon for batching data
        :param device: (str) String identifier of device to place data on, e.g. 'cpu', 'cuda:0'
        :param sequences: (dict str: np.array) Dictionary of supplemental data
        :param name: (str) String identifier of dataset type, must be ['static', 'openloop', 'closedloop']
        """
        self.name = name
        self.savedir = savedir
        self.norm, self.device = norm, device
        self.min_max_norms, self.dims, self.data = dict(), dict(), dict()
        self.sequences = sequences
        self.add_data(sequences)
        for k, v in self.data.items():
            self.dims[k] = v.shape[1]
        self.data = self.norm_data(self.data, self.norm)
        self.make_train_data()

    def norm_data(self, data, norm):
        for k, v in data.items():
            v = v.reshape(v.shape[0], -1)
            if k in norm:
                v, vmin, vmax = self.normalize(v)
                self.min_max_norms.update({k + 'min': vmin, k + 'max': vmax})
                data[k] = v
        return data

    def make_train_data(self):
        for k, v in self.data.items():
            self.dims[k] = v.shape[1]
        self.train_data, self.dev_data, self.test_data = self.split_train_test_dev(self.data)
        self.train_data.name, self.dev_data.name, self.test_data.name = 'train', 'dev', 'test'
        for dset in self.train_data, self.dev_data, self.test_data:
            for k, v in dset.items():
                dset[k] = torch.tensor(v, dtype=torch.float32).to(self.device)

    def add_data(self, sequences, norm=[]):
        for k in norm:
            self.norm.append(k)
        sequences = self.norm_data(sequences, norm)
        self.data = {**self.data, **sequences}

    def del_data(self, keys):
        for k in keys:
            self.data.pop(k)

    def add_variable(self, var: Dict[str, int]):
        for k, v in var.items():
            self.dims[k] = v

    def del_variable(self, keys):
        for k in keys:
            self.dims.pop(k)

    def normalize(self, M, Mmin=None, Mmax=None):
            """
            :param M: (2-d np.array) Data to be normalized
            :param Mmin: (int) Optional minimum. If not provided is inferred from data.
            :param Mmax: (int) Optional maximum. If not provided is inferred from data.
            :return: (2-d np.array) Min-max normalized data
            """
            Mmin = M.min(axis=0).reshape(1, -1) if Mmin is None else Mmin
            Mmax = M.max(axis=0).reshape(1, -1) if Mmax is None else Mmax
            M_norm = (M - Mmin) / (Mmax - Mmin)
            return np.nan_to_num(M_norm), Mmin.squeeze(), Mmax.squeeze()

    def split_train_test_dev(self, data):
        """

        :param data: (dict, str: 2-d np.array) Complete dataset. dims=(nsamples, dim)
        :return: (3-tuple) Dictionarys for train, dev, and test sets
        """
        train_data, dev_data, test_data = DataDict(), DataDict(), DataDict()
        train_idx = (list(data.values())[0].shape[0] // 3)
        dev_idx = train_idx * 2
        for k, v in data.items():
            if data is not None:
                train_data[k] = v[:train_idx, :]
                dev_data[k] = v[train_idx:dev_idx, :]
                test_data[k] = v[dev_idx:, :]
        return train_data, dev_data, test_data


systems = {'tank': 'datafile',
               'vehicle3': 'datafile',
               'aero': 'datafile',
               'flexy_air': 'datafile',
               'TwoTank': 'emulator',
               'LorenzSystem': 'emulator',
               'Lorenz96': 'emulator',
               'VanDerPol': 'emulator',
               'ThomasAttractor': 'emulator',
               'RosslerAttractor': 'emulator',
               'LotkaVolterra': 'emulator',
               'Brusselator1D': 'emulator',
               'ChuaCircuit': 'emulator',
               'Duffing': 'emulator',
               'UniversalOscillator': 'emulator',
               'HindmarshRose': 'emulator',
               'SimpleSingleZone': 'emulator',
               'Pendulum-v0': 'emulator',
               'CartPole-v1': 'emulator',
               'Acrobot-v1': 'emulator',
               'MountainCar-v0': 'emulator',
               'MountainCarContinuous-v0': 'emulator',
               'Reno_full': 'emulator',
               'Reno_ROM40': 'emulator',
               'RenoLight_full': 'emulator',
               'RenoLight_ROM40': 'emulator',
               'Old_full': 'emulator',
               'Old_ROM40': 'emulator',
               'HollandschHuys_full': 'emulator',
               'HollandschHuys_ROM100': 'emulator',
               'Infrax_full': 'emulator',
               'Infrax_ROM100': 'emulator',
               'CSTR': 'emulator'}


if __name__ == '__main__':

    for system, data_type in systems.items():
        print(system)
        if data_type == 'emulator':
            dataset = EmulatorDataset(system)
        elif data_type == 'datafile':
            dataset = FileDataset(system)

    # testing adding sequences
    nsim, ny = dataset.data['Y'].shape
    new_sequences = {'Ymax': 25*np.ones([nsim, ny]), 'Ymin': np.zeros([nsim, ny])}
    dataset.add_data(new_sequences, norm=['Ymax', 'Ymin'])
    dataset.make_nstep()
    dataset.make_loop()

