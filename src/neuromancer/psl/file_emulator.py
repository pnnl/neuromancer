"""
This module implements an emulator interface for recorded datasets.
The FileEmulator class facilitates sampling a wide range of initial conditions
across the state space. Users can interact with a FileEmulator almost exactly how
they interact with any other psl Emulator objects.

Data files are expected to be in .csv or .mat format with columns named by prefix
indicating x) state, y) observation, u) system input, d) system disturbance.

See psl/psl/data for example files of recorded datasets.
"""
import os, functools
import numpy as np
import pandas as pd
from scipy.io import loadmat
from neuromancer.psl.base import EmulatorBase, download


def _extract_var(data, regex):
    filtered = data.filter(regex=regex).values
    return filtered if filtered.size != 0 else None


SUPPORTED_EXTENSIONS = {".csv", ".mat"}


datasets = {"vehicle3": "NLIN_MIMO_vehicle/NLIN_MIMO_vehicle3.mat",
             "aero": "NLIN_MIMO_Aerodynamic/NLIN_MIMO_Aerodynamic.mat",
             "flexy_air": "Flexy_air/flexy_air_data.csv",
             "EED_building": "EED_building/EED_building.csv",
             }


def read_file(file_or_dir):
    if os.path.isdir(file_or_dir):
        files = [
            os.path.join(file_or_dir, x)
            for x in os.listdir(file_or_dir)
            if os.path.splitext(x)[1].lower() in SUPPORTED_EXTENSIONS
        ]
        return [_read_file(x) for x in sorted(files)]

    return _read_file(file_or_dir)


def _read_file(file_path):
    """Read data from MAT or CSV file into data dictionary.

    :param file_path: (str) path to a MAT or CSV file to load.
    """
    file_type = file_path.split(".")[-1].lower()
    if file_type == "mat":
        f = loadmat(file_path)
        Y = f.get("y", None)  # outputs
        X = f.get("x", None)
        U = f.get("u", None)  # inputs
        D = f.get("d", None)  # disturbances
        id_ = f.get("exp_id", None)  # experiment run id
        Time = f.get("Time", None)
    elif file_type == "csv":
        data = pd.read_csv(file_path)
        Y = _extract_var(data, "^y[0-9]+$")
        X = _extract_var(data, "^x[0-9]+$")
        U = _extract_var(data, "^u[0-9]+$")
        D = _extract_var(data, "^d[0-9]+$")
        id_ = _extract_var(data, "^exp_id")
        Time = _extract_var(data, "^Time$")
    else:
        print(f"error: unsupported file type: {file_type}")

    assert any([v is not None for v in [Y, X, U, D]])

    if id_ is None and Time is None:
        return {
            k: v for k, v in zip(["Y", "X", "U", "D"], [Y, X, U, D]) if v is not None
        }
    elif Time is None:
        return [
            {k: v[id_.flatten() == i, ...] for k, v in zip(["Y", "X", "U", "D"], [Y, X, U, D]) if v is not None}
            for i in sorted(set(id_.flatten()))
        ]
    elif id_ is None:
        return {
            k: v for k, v in zip(["Time", "Y", "X", "U", "D"], [Time, Y, X, U, D]) if v is not None
        }
    else:
        return [
            {k: v[id_.flatten() == i, ...] for k, v in zip(["Time", "Y", "X", "U", "D"], [Time, Y, X, U, D]) if
             v is not None}
            for i in sorted(set(id_.flatten()))
        ]


class FileEmulator(EmulatorBase):
    """
    An emulator interface for recorded datasets. The FileEmulator class facilitates
    sampling a wide range of initial conditions across the state space.
    """
    def __init__(self, seed=59, path=None, system=None):
        """

        :param seed: (int) Random seed to initialize the system.
        :param path: (str) Filepath to file stored locally
        :param system: (str) System name for dataset registered in the global datasets variable in this file.
        """
        if path is not None:
            self._path = path
        if system is not None:
            self.system = system
        self.data = self.retrieve_data()
        data_keys = list(self.data.keys())
        assert 'X' in data_keys or 'Y' in data_keys, f'Missing data: {data_keys}. Must have Y or X.'
        if 'Y' not in data_keys:
            self.data['Y'] = self.data['X']
            data_keys.append('Y')
        elif 'X' not in data_keys:
            self.data['X'] = self.data['Y']
            data_keys.append('X')
        self.ts = np.diff(self.data['Time'], axis=0).mean() if 'Time' in data_keys else 0.1
        self.data_keys = data_keys
        self.nsim = 100
        self.batch = self.get_batch(self.nsim, startidx=0)
        self.nx = self.batch['X'].shape[-1]
        self.ny = self.batch['Y'].shape[-1]
        if 'U' in self.data_keys:
            self.nu = self.batch['U'].shape[-1]
        elif 'D' in self.data_keys:
            self.nd = self.batch['D'].shape[-1]
        super().__init__(seed=seed)

    @property
    def params(self):
        return {}, {}, {}, {}

    def retrieve_data(self):
        if hasattr(self, '_path'):
            return read_file(self._path)
        else:
            download(self.url, self.path)
            return read_file(self.path)

    @property
    def path(self):
        """
        Path where model parameter file is stored
        """
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', f'{self.system}.{datasets[self.system].split(".")[-1]}')

    @property
    def url(self):
        """
        Remote github location for model parameter data
        """
        return f'https://github.com/pnnl/psl/raw/master/psl/data/{datasets[self.system]}'

    def get_batch(self, nsim, startidx=0):
        batch_data = {k: v[startidx: startidx + nsim] for k, v in self.data.items()}
        Time = np.arange(startidx, startidx+nsim) * self.ts
        return {**batch_data, 'Time': Time}

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def simulate(self, nsim=None, x0=None):
        """
        :param nsim: (int) Number of steps for open loop response
        :param ninit: (float) initial simulation time
        :param x0: (float) state initial conditions
        :return: X, Y, U, D
        """
        nsim = nsim if nsim is not None else self.nsim
        if x0 is None:
            return self.get_batch(nsim, startidx=np.random.randint(0, self.data['Y'].shape[0]-nsim-1))
        else:
            idx = self.find_nearest(self.data['X'][:-nsim-1], x0)
            self.batch = self.get_batch(nsim, startidx=idx)
            return self.batch


systems = {system: functools.partial(FileEmulator, system=system) for system in datasets}


if __name__ == '__main__':
    for name, system in systems.items():
        print(name)
        sys = system()
        data = sys.simulate(nsim=100)
        print({k: v.shape for k, v in data.items()})
        for i in range(100):
            data = sys.simulate(nsim=100)
            assert all([data[k].shape[0] == 100 for k in data])



