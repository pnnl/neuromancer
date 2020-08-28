"""

"""
# python base imports
import os
from typing import Dict
import warnings

# machine learning/data science imports
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# ecosystem imports
import psl as emulators

# local imports
import neuromancer.plot as plot


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


def batch_data_exp_idx(data, idx, nsteps):
    """
    batch data from multiple indexed experiments

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

    def __init__(self, system=None, nsim=10000, ninit=0, norm=['Y'], batch_type='batch',
                 nsteps=1, device='cpu', sequences=dict(), name='openloop',
                 savedir='test'):
        """

        :param system: (str) Identifier for dataset.
        :param nsim: (int) Total number of time steps in data sequence
        :param ninit: (int) Time step to begin dataset at.
        :param norm: (list) List of strings corresponding to data to be normalized, e.g. ['Y','U','D']
        :param batch_type: (str) Type of the batch generator, expects: 'mh' or 'batch'
        :param nsteps: (int) N-step prediction horizon for batching data
        :param device: (str) String identifier of device to place data on, e.g. 'cpu', 'cuda:0'
        :param sequences: (dict str: np.array) Dictionary of supplemental data
        :param name: (str) String identifier of dataset type, must be ['static', 'openloop', 'closedloop']
        :param savedir: (str) Where to save plots of dataset time sequences.

         returns: Dataset Object with public properties:
                    train_data: dict(str: Tensor)
                    dev_data: dict(str: Tensor)
                    test_data: dict(str: Tensor)
                    train_loop: dict(str: Tensor)
                    dev_loop: dict(str: Tensor)
                    test_loop: dict(str: Tensor)
                    dims: dict(str: tuple)
        """
        assert not (system is None and len(sequences) == 0), 'Trying to instantiate an empty dataset.'
        self.name = name
        self.savedir = savedir
        os.makedirs(self.savedir, exist_ok=True)
        self.system, self.nsim, self.ninit, self.norm, self.nsteps, self.device = system, nsim, ninit, norm, nsteps, device
        self.batch_type = batch_type
        self.sequences = sequences
        self.data = self.load_data()
        self.data = {**self.data, **self.sequences}
        self.min_max_norms, self.dims, self.nstep_data, self.shift_data = dict(), dict(), dict(), dict()
        for k, v in self.data.items():
            self.dims[k] = v.shape
        assert len(set([k[0] for k in self.dims.values()])) == 1, f'Sequence lengths are not equal: {self.dims}'
        self.dims['nsim'] = v.shape[0]
        self.dims['nsteps'] = self.nsteps
        self.data = self.norm_data(self.data, self.norm)
        # self.train_data, self.dev_data, self.test_data = self.make_nstep()
        # self.train_loop, self.dev_loop, self.test_loop = self.make_loop()

    def norm_data(self, data, norm):
        """
        Min-max normalize some variables in the dataset.
        :param data: (dict {str: np.array})
        :param norm: List of variable names to normalize.
        :return:
        """
        for k in norm:
            if k not in data:
                print(f'Warning: Key to normalize: {k} is not in dataset keys: {list(self.data.keys())}')
        for k, v in data.items():
            v = v.reshape(v.shape[0], -1)
            if k in norm:
                v, vmin, vmax = self.normalize(v)
                self.min_max_norms.update({k + 'min': vmin, k + 'max': vmax})
                data[k] = v
        return data

    def normalize(self, M, Mmin=None, Mmax=None):
            """
            :param M: (2-d np.array) Data to be normalized
            :param Mmin: (int) Optional minimum. If not provided is inferred from data.
            :param Mmax: (int) Optional maximum. If not provided is inferred from data.
            :return: (2-d np.array) Min-max normalized data
            """
            Mmin = M.min(axis=0).reshape(1, -1) if Mmin is None else Mmin
            Mmax = M.max(axis=0).reshape(1, -1) if Mmax is None else Mmax
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                M_norm = (M - Mmin) / (Mmax - Mmin)
            return np.nan_to_num(M_norm), Mmin.squeeze(), Mmax.squeeze()

    def make_nstep(self, overwrite=False):
        """

        :param overwrite: Whether to overwrite a dataset sequence if it already exists in the dataset.
        :return: train_data (dict str: 3-way torch.Tensor} Dictionary with values of shape Nsteps X Nbatches X dim
                 dev_data  see train_data
                 test_data see train_data
        """
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
        plot.plot_traj(self.data, figname=os.path.join(self.savedir, f'{self.system}.png'))
        train_data, dev_data, test_data = self.split_train_test_dev(self.nstep_data)
        train_data.name, dev_data.name, test_data.name = 'nstep_train', 'nstep_dev', 'nstep_test'
        return train_data, dev_data, test_data

    def make_loop(self):
        """
        Unbatches data to original format with extra 1-dimension at the batch dimension.
        Length of sequences has been shortened to account for shift of data for sequence to sequence modeling.
        Length of sequences has been potentially shortened to be evenly divided by nsteps:
            nsim = (nsim - shift) - (nsim % nsteps)

        :return: train_loop (dict str: 3-way np.array} Dictionary with values of shape nsim % X 1 X dim
                 dev_loop  see train_data
                 test_loop  see train_data
        """
        if self.name == 'openloop':
            if self.batch_type == 'mh':
                train_loop = self.unbatch_mh(self.train_data)
                dev_loop = self.unbatch_mh(self.dev_data)
                test_loop = self.unbatch_mh(self.test_data)
            else:
                train_loop = self.unbatch(self.train_data)
                dev_loop = self.unbatch(self.dev_data)
                test_loop = self.unbatch(self.test_data)

            all_loop = {k: np.concatenate([train_loop[k], dev_loop[k], test_loop[k]]).squeeze(1)
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
            train_loop, dev_loop, test_loop = self.split_train_test_dev(nstep_data)

        train_loop.name, dev_loop.name, test_loop.name = 'loop_train', 'loop_dev', 'loop_test'
        for dset in train_loop, dev_loop, test_loop, self.train_data, self.dev_data, self.test_data:
            for k, v in dset.items():
                dset[k] = torch.tensor(v, dtype=torch.float32).to(self.device)
        return train_loop, dev_loop, test_loop

    def add_data(self, sequences, norm=[], overwrite=False):
        """
        Add new sequences to dataset.

        :param sequences: dict {'str': 2-way np.array} Dictionary of nsim X dim sequences.
        :param norm: Sequences to normalize.
        :param overwrite: Whether to allow overwriting a portion of the dataset.
        :return:
        """
        for k, v in sequences.items():
            assert v.shape[0] == self.dims['nsim']
            self.dims[k] = v.shape
        for k in norm:
            self.norm.append(k)
        sequences = self.norm_data(sequences, norm)
        self.data = {**self.data, **sequences}
        self.train_data, self.dev_data, self.test_data = self.make_nstep(overwrite=overwrite)
        self.train_loop, self.dev_loop, self.test_loop = self.make_loop()

    def del_data(self, keys):
        """
        Delete a sequence from the dataset.

        :param keys: Key for sequence to delete
        :return:
        """
        for k in keys:
            del self.data[k]

    def add_variable(self, var: Dict[str, int]):
        """
        Manually add dataset dimensions to the dataset
        :param var: (dict {str: tuple})
        """
        for k, v in var.items():
            self.dims[k] = v

    def del_variable(self, keys):
        """
        Manually delete dataset dimensions from the dataset

        :param keys:
        """
        for k in keys:
            del self.dims[k]

    def load_data(self):
        """
        Load a dataset of sequences. Arrays must be nsim X ndim
        :return: dict (str: np.array)
        """
        assert self.system is None and len(self.sequences) > 0, \
            'User must provide data via the sequences argument for basic Dataset. ' +\
            'Use FileDataset or EmulatorDataset for predefined datasets'
        return dict()

    def split_train_test_dev(self, data):
        """

        :param data: (dict, str: 3-d np.array) Complete dataset. dims=(nsteps, nsamples, dim)
        :return: (3-tuple) Dictionarys for train, dev, and test sets
        """
        train_data, dev_data, test_data = DataDict(), DataDict(), DataDict()
        train_idx = (list(data.values())[0].shape[1] // 3)
        dev_idx = train_idx * 2
        for k, v in data.items():
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
        model = systems[self.system](nsim=self.nsim, ninit=self.ninit)  # instantiate model class
        return model.simulate()  # simulate open loop


class FileDataset(Dataset):

    def load_data(self):
        """
        Load data from files. system argument to init should be the name of a registered dataset in systems_datapaths
        :return: (dict, str: 2-d np.array)
        """
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
            exp_id = file.get("exp_id", None)  # experiment run id
        elif file_type == 'csv':
            data = pd.read_csv(file_path)
            Y = data.filter(regex='y').values if data.filter(regex='y').values.size != 0 else None
            U = data.filter(regex='u').values if data.filter(regex='u').values.size != 0 else None
            D = data.filter(regex='d').values if data.filter(regex='d').values.size != 0 else None
            exp_id = data.filter(regex='exp_id').values if data.filter(regex='exp_id').values.size != 0 else None
        data = dict()
        for d, k in zip([Y, U, D, exp_id], ['Y', 'U', 'D', 'exp_id']):
            if d is not None:
                data[k] = d[self.ninit:self.nsim+self.ninit, :]
        return data


def ld_to_dl(ld):
    return DataDict([(k, torch.cat([dic[k] for dic in ld], dim=1)) for k in ld[0]])


class MultiExperimentDataset(FileDataset):
    def __init__(self, system='fsw_phase_4', nsim=10000000, ninit=0, norm=['Y'], batch_type='batch',
                 nsteps=1, device='cpu', sequences=dict(), name='openloop',
                 savedir='test', split=[.5, .25]):
        """
        :param split: (2-tuple of float) First index is proportion of experiments from train, second is proportion from dev,
                       leftover are for test set.
        """

        super().__init__(system=system, nsim=nsim, ninit=ninit, norm=norm, batch_type=batch_type,
                         nsteps=nsteps, device=device, sequences=sequences, name=name,
                         savedir=savedir)
        plot.plot_traj(self.data, figname=os.path.join(self.savedir, f'{self.system}.png'))
        self.experiments = self.split_data_by_experiment()
        nstep_data = []
        loop_data = []
        for d in self.experiments:
            shift_data, nsteps = self._make_nstep(d)
            nstep_data.append(nsteps)
            loop_data.append(self._make_loop(nsteps, shift_data))
            for k, v in nstep_data[-1].items():
                nstep_data[-1][k] = torch.tensor(v, dtype=torch.float32).to(self.device)
                self.dims[k] = v.shape
            for k, v in loop_data[-1].items():
                loop_data[-1][k] = torch.tensor(v, dtype=torch.float32).to(self.device)

        num_exp = len(nstep_data)
        num_train = int(split[0]*num_exp)
        num_dev = int(split[1]*num_exp)
        num_test = num_exp - num_dev - num_train
        assert num_train + num_dev + num_test == num_exp
        assert num_test > 0

        self.train_data, self.train_loop = nstep_data[:num_train], loop_data[:num_train]
        self.dev_data, self.dev_loop = nstep_data[num_train:num_train+num_dev], loop_data[num_train:num_train+num_dev]
        self.test_data, self.test_loop = nstep_data[num_train+num_dev:], loop_data[num_train+num_dev:]

        for dset, name in zip([self.train_data, self.dev_data, self.test_data], ['nstep_train', 'nstep_dev', 'nstep_test']):
            dset = ld_to_dl(dset)
            dset.name = name

        for dset, name in zip([self.train_loop, self.dev_loop, self.test_loop], ['loop_train', 'loop_dev', 'loop_test']):
            for d in dset:
                d.name = name

    def split_data_by_experiment(self):
        exp_ids = self.data['exp_id']
        experiments = []
        for id in np.unique(exp_ids):
            experiments.append(dict())
            for k in self.data:
                experiments[-1][k] = self.data[k][self.data['exp_id'].squeeze() == id]
        return experiments

    def _make_nstep(self, data, overwrite=False):
        """

        :param overwrite: Whether to overwrite a dataset sequence if it already exists in the dataset.
        :return: train_data (dict str: 3-way torch.Tensor} Dictionary with values of shape Nsteps X Nbatches X dim
                 dev_data  see train_data
                 test_data see train_data
        """
        shift_data = dict()
        nstep_data = dict()
        for k, v in data.items():

            shift_data[k + 'p'] = v[:-self.nsteps]
            shift_data[k + 'f'] = v[self.nsteps:]
            if self.batch_type == 'mh':
                nstep_data[k + 'p'] = batch_mh_data(shift_data[k + 'p'], self.nsteps)
                nstep_data[k + 'f'] = batch_mh_data(shift_data[k + 'f'], self.nsteps)
            else:
                nstep_data[k + 'p'] = batch_data(shift_data[k + 'p'], self.nsteps)
                nstep_data[k + 'f'] = batch_data(shift_data[k + 'f'], self.nsteps)
        return shift_data, nstep_data

    def _make_loop(self, nstep_data, shift_data):
        """
        Unbatches data to original format with extra 1-dimension at the batch dimension.
        Length of sequences has been shortened to account for shift of data for sequence to sequence modeling.
        Length of sequences has been potentially shortened to be evenly divided by nsteps:
            nsim = (nsim - shift) - (nsim % nsteps)

        :return: train_loop (dict str: 3-way np.array} Dictionary with values of shape nsim % X 1 X dim
                 dev_loop  see train_data
                 test_loop  see train_data
        """
        if self.name == 'openloop':
            if self.batch_type == 'mh':
                loop = self.unbatch_mh(nstep_data)
            else:
                loop = self.unbatch(nstep_data)

            for k in loop.keys():
                assert np.array_equal(loop[k].squeeze(1), shift_data[k][:loop[k].shape[0]]), \
                    f'Reshaped data {k} is not equal to truncated original data'
            plot.plot_traj({k: v.squeeze(1) for k, v in loop.items()}, figname=os.path.join(self.savedir, f'{self.system}_open.png'))
            plt.close('all')

        return loop


class DatasetMPP(Dataset):

    def __init__(self, norm=[], device='cpu', sequences=dict(), name='mpp'):
        """

        :param norm: (str) String of letters corresponding to data to be normalized, e.g. ['Y','U','D']
        :param device: (str) String identifier of device to place data on, e.g. 'cpu', 'cuda:0'
        :param sequences: (dict str: np.array) Dictionary of supplemental data
        :param name: (str) String identifier of dataset type, must be ['static', 'openloop', 'closedloop']
        """
        self.name = name
        self.norm, self.device = norm, device
        self.min_max_norms, self.dims = dict(), dict()
        self.sequences = sequences
        self.add_data(sequences)
        for k, v in self.data.items():
            self.dims[k] = v.shape[1]
        self.data = self.norm_data(self.data, self.norm)
        self.make_train_data()

    def make_train_data(self):
        for k, v in self.data.items():
            self.dims[k] = v.shape[1]
        self.train_data, self.dev_data, self.test_data = self.split_train_test_dev(self.data)
        self.train_data.name, self.dev_data.name, self.test_data.name = 'train', 'dev', 'test'
        for dset in self.train_data, self.dev_data, self.test_data:
            for k, v in dset.items():
                dset[k] = torch.tensor(v, dtype=torch.float32).to(self.device)


resource_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
systems_datapaths = {'tank': os.path.join(resource_path, 'NLIN_SISO_two_tank/NLIN_two_tank_SISO.mat'),
                     'vehicle3': os.path.join(resource_path, 'NLIN_MIMO_vehicle/NLIN_MIMO_vehicle3.mat'),
                     'aero': os.path.join(resource_path, 'NLIN_MIMO_Aerodynamic/NLIN_MIMO_Aerodynamic.mat'),
                     'flexy_air': os.path.join(resource_path, 'Flexy_air/flexy_air_data.csv'),
                     'EED_building': os.path.join(resource_path, 'EED_building/EED_building.csv'),
                     'fsw_phase_1': os.path.join(resource_path, 'FSW/by_step/fsw_data_step1.csv'),
                     'fsw_phase_2': os.path.join(resource_path, 'FSW/by_step/fsw_data_step2.csv'),
                     'fsw_phase_3': os.path.join(resource_path, 'FSW/by_step/fsw_data_step3.csv'),
                     'fsw_phase_4': os.path.join(resource_path, 'FSW/by_step/fsw_data_step4.csv'),
                     }


systems = {'fsw_phase_1': 'datafile',
           'fsw_phase_2': 'datafile',
           'fsw_phase_3': 'datafile',
           'fsw_phase_4': 'datafile',
           'tank': 'datafile',
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
           'CSTR': 'emulator',
           'UAV3D_kin': 'emulator'}


if __name__ == '__main__':

    print('FSW')
    dataset = MultiExperimentDataset()
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

