"""

"""
import os
from typing import Dict
import warnings
import random

from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

import psl

import neuromancer.plot as plot
from neuromancer.data.normalization import norm_fns
from neuromancer.data.batch import (
    batch_data, batch_mh_data, batch_data_exp_idx,
    unbatch_data, unbatch_mh_data
)


class DataDict(dict):
    """
    So we can add a name property to the dataset dictionaries
    """
    pass


class Dataset:
    def __init__(self, system=None, nsim=10000, ninit=0, norm=['Y'], batch_type='batch',
                 nsteps=1, device='cpu', sequences=dict(), name='openloop',
                 savedir='test', norm_type='zero-one'):
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
        self.norm_type = norm_type
        self.norm_fn = norm_fns[self.norm_type]
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
        self.train_data, self.dev_data, self.test_data = self.make_nstep()
        self.train_loop, self.dev_loop, self.test_loop = self.make_loop()

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
                v, vmin, vmax = self.norm_fn(v)
                self.min_max_norms.update({k + 'min': vmin, k + 'max': vmax})
                data[k] = v
        return data

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
        plt.close('all')
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


class FileDataset(Dataset):
    def load_data(self):
        """
        Load data from files. system argument to init should be the name of a registered dataset in systems_datapaths
        :return: (dict, str: 2-d np.array)
        """
        if self.system in psl.datasets.keys():
            file_path = psl.datasets[self.system]
        else:
            file_path = self.system
        if not os.path.exists(file_path):
            raise ValueError(f'No file at {file_path}')
        file_type = file_path.split(".")[-1].lower()
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


class MultiExperimentDataset(FileDataset):
    # TODO: This will break if nsim is small enough to exclude some experiments
    def __init__(self, system='fsw_phase_2', nsim=10000000, ninit=0, norm=['Y'], batch_type='batch',
                 nsteps=1, device='cpu', sequences=dict(), name='openloop',
                 savedir='test', split={'train': [0], 'dev': [0], 'test': [0]}, norm_type='zero-one'):
        """
        :param split: (2-tuple of float) First index is proportion of experiments from train, second is proportion from dev,
                       leftover are for test set.

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
        self.norm_type = norm_type
        self.norm_fn = norm_fns[norm_type]
        self.savedir = savedir
        os.makedirs(self.savedir, exist_ok=True)
        self.system, self.nsim, self.ninit, self.norm, self.nsteps, self.device = system, nsim, ninit, norm, nsteps, device
        self.batch_type = batch_type
        self.min_max_norms = dict()
        self.data = self.load_data()
        self.data = {**self.data, **sequences}
        self.data = self.norm_data(self.data, self.norm)
        plot.plot_traj(self.data, figname=os.path.join(self.savedir, f'{self.system}.png'))
        plt.close('all')
        self.experiments = self.split_data_by_experiment()
        self.nstep_data, self.loop_data = self.make_nstep_loop()
        self.nstep_data, self.loop_data = self.to_tensor(self.nstep_data), self.to_tensor(self.loop_data)
        self.split_train_test_dev(split)
        self.train_data = self.listDict_to_dictTensor(self.train_data)
        self.dev_data = self.listDict_to_dictTensor(self.dev_data)
        self.test_data = self.listDict_to_dictTensor(self.test_data)
        self.dims = self.get_dims()
        self.name_data()

    def listDict_to_dictTensor(self, ld):
        return DataDict([(k, torch.cat([dic[k] for dic in ld], dim=1)) for k in ld[0]])

    def split_train_test_dev(self, split):
        if type(split) is dict:
            # TODO fix this indexing hack for more general indexing outside of FSW context
            self.train_data = np.array(self.nstep_data)[np.array(split['train']) - 1]
            self.train_loop = np.array(self.loop_data)[np.array(split['train']) - 1]

            print(len(self.train_data))
            self.dev_data = np.array(self.nstep_data)[np.array(split['dev']) - 1]
            self.dev_loop = np.array(self.loop_data)[np.array(split['dev']) - 1]

            self.test_data = np.array(self.nstep_data)[np.array(split['test']) - 1]
            self.test_loop = np.array(self.loop_data)[np.array(split['test']) - 1]

        elif type(split) is list:
            num_exp = len(self.nstep_data)
            num_train = int(split[0] * num_exp)
            num_dev = int(split[1] * num_exp)
            num_test = num_exp - num_dev - num_train
            assert num_train + num_dev + num_test == num_exp
            assert num_test > 0

            self.train_data = self.nstep_data[:num_train]
            self.train_loop = self.loop_data[:num_train]
            self.dev_data = self.nstep_data[num_train:num_train + num_dev]
            self.dev_loop = self.loop_data[num_train:num_train + num_dev]
            self.test_data = self.nstep_data[num_train + num_dev:]
            self.test_loop = self.loop_data[num_train + num_dev:]

        else:
            raise ValueError('Split must be a list of two floating point values summing to 1, or a dictionary with keys'
                             'train, val, test and values integer experiment ids.')

    def get_dims(self):
        self.dims = dict()
        for k, v in self.data.items():
            self.dims[k] = v.shape
            self.dims[k + 'p'] = v.shape
            self.dims[k + 'f'] = v.shape
        assert len(set([k[0] for k in self.dims.values()])) == 1, f'Sequence lengths are not equal: {self.dims}'
        self.dims['nsim'] = v.shape[0]
        self.dims['nsteps'] = self.nsteps
        return self.dims

    def split_data_by_experiment(self):
        exp_ids = self.data['exp_id']
        experiments = []
        for id in np.unique(exp_ids):
            experiments.append(dict())
            for k in self.data:
                experiments[-1][k] = self.data[k][self.data['exp_id'].squeeze() == id]
        return experiments

    def to_tensor(self, data):
        for i in range(len(data)):
            for k, v in data[i].items():
                data[i][k] = torch.tensor(v, dtype=torch.float32).to(self.device)
        return data

    def make_nstep_loop(self):
        nstep_data, loop_data = [], []
        for d in self.experiments:
            shift_data, nsteps = self._make_nstep(d)
            nstep_data.append(nsteps)
            loop_data.append(self._make_loop(nsteps, shift_data))
        return nstep_data, loop_data

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

    def name_data(self):
        for dset, name in zip([self.train_data, self.dev_data, self.test_data], ['nstep_train', 'nstep_dev', 'nstep_test']):
            dset.name = name

        for dset, name in zip([self.train_loop, self.dev_loop, self.test_loop], ['loop_train', 'loop_dev', 'loop_test']):
            for d in dset:
                d.name = name


def _check_data(data):
    return not np.any(np.isnan(data))


class EmulatorDataset(Dataset):
    def __init__(self, system=None, nsim=10000, ninit=0, norm=['Y'], batch_type='batch',
                 nsteps=1, device='cpu', sequences=dict(), name='openloop',
                 savedir='test', norm_type='zero-one', seed=59):
        self.simulator_seed = seed
        super().__init__(system, nsim, ninit, norm, batch_type, nsteps, device, sequences, name, savedir, norm_type)

    def load_data(self):
        """
        dataset creation from the emulator. system argument to init should be the name of a registered emulator
        return: (dict, str: 2-d np.array)
        """
        model = psl.emulators[self.system](nsim=self.nsim, ninit=self.ninit, seed=self.simulator_seed)  # instantiate model class
        sim = model.simulate()  # simulate open loop
        i = 1
        while not _check_data(sim['Y']):
            print(f'Emulator generated invalid data, resimulating with seed={self.simulator_seed+i}...')
            del sim['Y']
            del model
            model = psl.emulators[self.system](nsim=self.nsim, ninit=self.ninit, seed=self.simulator_seed+i)  # instantiate model class
            sim = model.simulate()
            i += 1

        return sim


class MultiExperimentEmulatorDataset(MultiExperimentDataset):
    def __init__(self, system='LorenzSystem', nsim=20, ninit=0, norm=['Y', 'X'], batch_type='batch',
                 nsteps=1, device='cpu', sequences=dict(), name='openloop',
                 savedir='test', split=[.5, .25], nexp=5):
        """
        :param split: (2-tuple of float) First index is proportion of experiments from train, second is proportion from dev,
                       leftover are for test set.
        :param sequences: List of (dict str: np.array) List of dictionaries of supplemental data. Should be nexp long.

         returns: Dataset Object with public properties:
                    train_data: dict(str: Tensor)
                    dev_data: dict(str: Tensor)
                    test_data: dict(str: Tensor)
                    train_loop: dict(str: Tensor)
                    dev_loop: dict(str: Tensor)
                    test_loop: dict(str: Tensor)
                    dims: dict(str: tuple)
        """
        self.nexp = nexp
        assert nsim % nexp == 0, 'nsim must evenly divide nexp'

        super().__init__(system=system, nsim=nsim, ninit=ninit, norm=norm, batch_type=batch_type,
                         nsteps=nsteps, device=device, sequences=sequences, name=name,
                         savedir=savedir, split=split)

    def _load_data(self, x0=None, nsim=None):

        if nsim is None:
            nsim = self.nsim
        model = psl.systems[self.system](nsim=nsim, ninit=self.ninit)  # instantiate model class
        return model.simulate(x0=x0, nsim=nsim)  # simulate open loop

    def merge_data(self, experiments):
        data = dict()
        for k in experiments[0]:
            data[k] = np.concatenate([d[k] for d in experiments])
        return data

    def load_data(self):
        """
        dataset creation from the emulator. system argument to init should be the name of a registered emulator
        return: (dict, str: 2-d np.array)
        """

        experiments = []
        initial_data = self._load_data()
        _ = self.norm_data(initial_data, self.norm)
        for i in range(self.nexp):
            x0 = np.array([random.uniform(self.min_max_norms['Xmin'][j], self.min_max_norms['Xmax'][j])
                           for j in range(self.min_max_norms['Xmin'].shape[0])])
            experiments.append({**self._load_data(x0=x0, nsim=self.nsim//self.nexp),
                                'exp_id': i*np.ones([self.nsim//self.nexp, 1])})
        return self.merge_data(experiments)


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

systems = {
    **{k: "datafile" for k in psl.datasets},
    **{k: "emulator" for k in psl.systems},
}

if __name__ == '__main__':
    print("Testing EmulatorDataset with psl.systems...")
    for system in psl.systems:
        print(f"  {system}")
        dataset = EmulatorDataset(system)

    print("\nTesting FileDataset with psl.datasets...")
    for system in psl.datasets:
        print(f"  {system}")
        dataset = FileDataset(system)

    print("\nTesting adding sequences...")
    nsim, ny = dataset.data['Y'].shape
    new_sequences = {'Ymax': 25*np.ones([nsim, ny]), 'Ymin': np.zeros([nsim, ny])}
    dataset.add_data(new_sequences, norm=['Ymax', 'Ymin'])

    print("\nTesting MultiExperimentEmulatorDataset...")
    # FIXME: this fails for UAV3D_kin and UAV2D_kin
    for system in [k for k, v in systems.items() if v == 'emulator'
                                                    and k not in ['CartPole-v1',
                                                                  'Acrobot-v1',
                                                                  'MountainCar-v0',
                                                                  'Pendulum-v0',
                                                                  'MountainCarContinuous-v0']]:
        print(f"  {system}")
        try:
            dataset = MultiExperimentEmulatorDataset(system=system)
        except Exception as e:
            print("Error encountered:", e)

    print("\nTesting MultiExperimentDataset on FSW data...")
    for system in ['fsw_phase_1', 'fsw_phase_2', 'fsw_phase_3', 'fsw_phase_4']:
        print(f"  {system}")
        dataset = MultiExperimentDataset(system, split=psl.datasplits['pid'])
