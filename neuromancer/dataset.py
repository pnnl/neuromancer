from glob import glob
import math
import os
from typing import Union
import warnings

import numpy as np
import pandas as pd
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate


def _is_multisequence_data(data):
    return isinstance(data, list) and all([isinstance(x, dict) for x in data])


def _is_sequence_data(data):
    return isinstance(data, dict) and len({x.shape[0] for x in data.values()}) == 1


def _extract_var(data, regex):
    filtered = data.filter(regex=regex).values
    return filtered if filtered.size != 0 else None


SUPPORTED_EXTENSIONS = {".csv", ".mat"}


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
    elif file_type == "csv":
        data = pd.read_csv(file_path)
        Y = _extract_var(data, "^y[0-9]+$")
        X = _extract_var(data, "^x[0-9]+$")
        U = _extract_var(data, "^u[0-9]+$")
        D = _extract_var(data, "^d[0-9]+$")
        id_ = _extract_var(data, "^exp_id")
    else:
        print(f"error: unsupported file type: {file_type}")

    assert any([v is not None for v in [Y, X, U, D]])

    if id_ is None:
        return {
            k: v for k, v in zip(["Y", "X", "U", "D"], [Y, X, U, D]) if v is not None
        }
    else:
        return [
            {k: v[id_.flatten() == i, ...] for k, v in zip(["Y", "X", "U", "D"], [Y, X, U, D]) if v is not None}
            for i in sorted(set(id_.flatten()))
        ]


def batch_tensor(x: torch.Tensor, steps: int, mh: bool = False):
    return x.unfold(0, steps, 1 if mh else steps)


def unbatch_tensor(x: torch.Tensor, mh: bool = False):
    return (
        torch.cat((x[:, :, :, 0], x[-1, :, :, 1:]), dim=0)
        if mh
        else torch.cat(torch.unbind(x, 0), dim=-1)
    )


def _get_sequence_time_slices(data):
    seq_lens = []
    for i, d in enumerate(data):
        seq_lens.append(None)
        for v in d.values():
            seq_lens[i] = seq_lens[i] or v.shape[0]
            assert seq_lens[i] == v.shape[0], \
                "sequence lengths within a dictionary must be equal"
    slices = []
    i = 0
    for seq_len in seq_lens:
        slices.append(slice(i, i + seq_len, 1))
        i += seq_len
    return slices


def _validate_keys(data):
    keys = set(data[0].keys())
    for d in data[1:]:
        other_keys = set(d.keys())
        assert len(keys - other_keys) == 0 and len(other_keys - keys) == 0, \
            "list of dictionaries must have matching keys across all dictionaries."
        keys = other_keys
    return keys


class SequenceDataset(Dataset):
    def __init__(
        self,
        data,
        nsteps=1,
        moving_horizon=False,
        name="data",
    ):
        """Dataset for handling sequential data and transforming it into the dictionary structure
        used by NeuroMANCER models.

        :param data: (dict str: np.array) dictionary mapping variable names to tensors of shape
            (T, Dk), where T is number of time steps and Dk is dimensionality of variable k.
        :param nsteps: (int) N-step prediction horizon for batching data.
        :param moving_horizon: (bool) if True, generate batches using sliding window with stride 1;
            else use stride N.
        :param name: (str) name of dataset split.

        .. note:: To generate train/dev/test datasets and DataLoaders for each, see the
            `get_sequence_dataloaders` function.

        .. warning:: This dataset class requires the use of a special collate function that must be
            provided to PyTorch's DataLoader class; see the `collate_fn` method of this class.

        .. todo:: Add support for categorical features.
        .. todo:: Add support for memory-mapped and/or streaming data.
        .. todo:: Add support for data augmentation?
        .. todo:: Clean up data validation code.
        """

        super().__init__()
        self.name = name

        self.multisequence = _is_multisequence_data(data)
        assert _is_sequence_data(data) or self.multisequence, \
            "data must be provided as a dictionary or list of dictionaries"

        if isinstance(data, dict):
            data = [data]

        keys = _validate_keys(data)

        # _sslices used to slice out sequences from a multi-sequence dataset
        self._sslices = _get_sequence_time_slices(data)
        assert all([nsteps < (sl.stop - sl.start) for sl in self._sslices]), \
            f"length of time series data must be greater than nsteps"

        self.nsteps = nsteps

        self.variables = list(keys)
        self.full_data = torch.cat(
            [torch.cat([torch.tensor(d[k], dtype=torch.float) for k in self.variables], dim=1) for d in data],
            dim=0,
        )
        self.nsim = self.full_data.shape[0]
        self.dims = {k: (self.nsim, *data[0][k].shape[1:],) for k in self.variables}

        # _vslices used to slice out sequences of individual variables from full_data and batched_data
        i = 0
        self._vslices = {}
        for k, v in self.dims.items():
            self._vslices[k] = slice(i, i + v[1], 1)
            i += v[1]

        self.dims = {
            **self.dims,
            **{k + "p": (self.nsim - 1, v[1]) for k, v in self.dims.items()},
            **{k + "f": (self.nsim - 1, v[1]) for k, v in self.dims.items()},
            "nsim": self.nsim,
            "nsteps": nsteps,
        }

        self.batched_data = torch.cat(
            [batch_tensor(self.full_data[s, ...] , nsteps, mh=moving_horizon) for s in self._sslices],
            dim=0,
        )
        self.batched_data = self.batched_data.permute(0, 2, 1)

    def __len__(self):
        """Gives the number of N-step batches in the dataset."""
        return len(self.batched_data) - 1

    def __getitem__(self, i):
        """Fetch a single N-step sequence from the dataset."""
        datapoint = {
            **{
                k + "p": self.batched_data[i, :, self._vslices[k]]
                for k in self.variables
            },
            **{
                k + "f": self.batched_data[i + 1, :, self._vslices[k]]
                for k in self.variables
            },
        }
        datapoint['index'] = i
        return datapoint

    def _get_full_sequence_impl(self, start=0, end=None):
        """Returns the full sequence of data as a dictionary. Useful for open-loop evaluation.
        """
        if end is not None and end < 0:
            end = self.full_data.shape[0] + end
        elif end is None:
            end = self.full_data.shape[0]

        return {
            **{
                k + "p": self.full_data[start : end - self.nsteps, self._vslices[k]].unsqueeze(1)
                for k in self.variables
            },
            **{
                k + "f": self.full_data[start + self.nsteps : end, self._vslices[k]].unsqueeze(1)
                for k in self.variables
            },
            "name": "loop_" + self.name,
        }

    def get_full_sequence(self):
        return (
            [self._get_full_sequence_impl(start=s.start, end=s.stop) for s in self._sslices]
            if self.multisequence
            else self._get_full_sequence_impl()
        )

    def get_full_batch(self):
        return {
            **{
                k + "p": self.batched_data[:-1, :, self._vslices[k]].transpose(0, 1)
                for k in self.variables
            },
            **{
                k + "f": self.batched_data[1:, :, self._vslices[k]].transpose(0, 1)
                for k in self.variables
            },
            "name": "nstep_" + self.name,
        }

    def collate_fn(self, batch):
        """Batch collation for dictionaries of samples generated by this dataset. This wraps the
        default PyTorch batch collation function and does some light post-processing to transpose
        the data for NeuroMANCER models and add a "name" field.

        :param batch: (dict str: torch.Tensor) dataset sample.
        """
        batch = default_collate(batch)
        return {
            **{k: v.transpose(0, 1) if k != 'index' else v for k, v in batch.items()},
            "name": "nstep_" + self.name,
        }

    def __repr__(self):
        varinfo = "\n    ".join([f"{x}: {d}" for x, d in self.dims.items() if x not in {"nsteps", "nsim"}])
        seqinfo = f"    nsequences: {len(self._sslices)}\n" if self.multisequence else ""
        return (
            f"{type(self).__name__}:\n"
            f"  multi-sequence: {self.multisequence}\n"
            f"{seqinfo}"
            f"  variables (shapes):\n"
            f"    {varinfo}\n"
            f"  nsim: {self.nsim}\n"
            f"  nsteps: {self.nsteps}\n"
            f"  nsamples: {len(self)}\n"
        )


class SequenceDataset_MultiStep(Dataset):
    def __init__(
        self,
        data,
        nsteps_p=1,
        nsteps_f=1,
        moving_horizon=False,
        name="data",
    ):
        """

        :param nsteps_p: (int) N-step input horizon for batching data.
        :param nsteps_f: (int) N-step output horizon for batching data.
        """

        super().__init__()
        self.name = name

        self.multisequence = _is_multisequence_data(data)
        assert _is_sequence_data(data) or self.multisequence, \
            "data must be provided as a dictionary or list of dictionaries"

        if isinstance(data, dict):
            data = [data]

        keys = _validate_keys(data)

        # _sslices used to slice out sequences from a multi-sequence dataset
        self._sslices = _get_sequence_time_slices(data)
        assert all([nsteps_p < (sl.stop - sl.start) for sl in self._sslices]), \
            f"length of time series data must be greater than nsteps_p" 
        assert all([nsteps_f < (sl.stop - sl.start) for sl in self._sslices]), \
            f"length of time series data must be greater than nsteps_f"
        self.nsteps_p = nsteps_p
        self.nsteps_f = nsteps_f

        self.variables = list(keys)
        self.full_data = torch.cat(
            [torch.cat([torch.tensor(d[k], dtype=torch.float) for k in self.variables], dim=1) for d in data],
            dim=0,
        )
        self.nsim = self.full_data.shape[0]
        self.dims = {k: (self.nsim, *data[0][k].shape[1:],) for k in self.variables}

        # _vslices used to slice out sequences of individual variables from full_data and batched_data
        i = 0
        self._vslices = {}
        for k, v in self.dims.items():
            self._vslices[k] = slice(i, i + v[1], 1)
            i += v[1]

        self.dims = {
            **self.dims,
            **{k + "p": (self.nsim - 1, v[1]) for k, v in self.dims.items()},
            **{k + "f": (self.nsim - 1, v[1]) for k, v in self.dims.items()},
            "nsim": self.nsim,
            "nsteps_p": nsteps_p,
            "nsteps_f": nsteps_f,
        }

        self.batched_data_p = torch.cat(
            [batch_tensor(self.full_data[s.start:s.stop-nsteps_p, ...],
                          nsteps_p, mh=moving_horizon) for s in self._sslices],
            dim=0,
        )
        self.batched_data_p = self.batched_data_p.permute(0, 2, 1)

        self.batched_data_f = torch.cat(
            [batch_tensor(self.full_data[s.start+nsteps_p:s.stop, ...],
                          nsteps_f, mh=moving_horizon) for s in self._sslices],
            dim=0,
        )
        self.batched_data_f = self.batched_data_f.permute(0, 2, 1)

    def __len__(self):
        """Gives the number of N-step batches in the dataset."""
        return min([len(self.batched_data_p) - 1, len(self.batched_data_f) - 1])

    def __getitem__(self, i):
        """Fetch a single N-step sequence from the dataset."""
        return {
            **{
                k + "p": self.batched_data_p[i, :, self._vslices[k]]
                for k in self.variables
            },
            **{
                # k + "f": self.batched_data_f[i + 1, :, self._vslices[k]]
                k + "f": self.batched_data_f[i, :, self._vslices[k]]
                for k in self.variables
            },
        }

    def _get_full_sequence_impl(self, start=0, end=None):
        """Returns the full sequence of data as a dictionary. Useful for open-loop evaluation.
        """
        if end is not None and end < 0:
            end = self.full_data.shape[0] + end
        elif end is None:
            end = self.full_data.shape[0]
        return {
            **{
                k + "p": self.full_data[start : end - self.nsteps_f, self._vslices[k]].unsqueeze(1)
                for k in self.variables
            },
            **{
                k + "f": self.full_data[start + self.nsteps_p : end, self._vslices[k]].unsqueeze(1)
                for k in self.variables
            },
            "name": "loop_" + self.name,
        }

    def get_full_sequence(self):
        return (
            [self._get_full_sequence_impl(start=s.start, end=s.stop) for s in self._sslices]
            if self.multisequence
            else self._get_full_sequence_impl()
        )

    def get_full_batch(self):
        return {
            **{
                k + "p": self.batched_data_p[:-1, :, self._vslices[k]].transpose(0, 1)
                for k in self.variables
            },
            **{
                k + "f": self.batched_data_f[1:, :, self._vslices[k]].transpose(0, 1)
                for k in self.variables
            },
            "name": "nstep_" + self.name,
        }

    def collate_fn(self, batch):
        """Batch collation for dictionaries of samples generated by this dataset. This wraps the
        default PyTorch batch collation function and does some light post-processing to transpose
        the data for NeuroMANCER models and add a "name" field.

        :param batch: (dict str: torch.Tensor) dataset sample.
        """
        batch = default_collate(batch)
        return {
            **{k: v.transpose(0, 1) for k, v in batch.items()},
            "name": "nstep_" + self.name,
        }

    def __repr__(self):
        varinfo = "\n    ".join([f"{x}: {d}" for x, d in self.dims.items() if x not in {"nsteps", "nsim"}])
        seqinfo = f"    nsequences: {len(self._sslices)}\n" if self.multisequence else ""
        return (
            f"{type(self).__name__}:\n"
            f"  multi-sequence: {self.multisequence}\n"
            f"{seqinfo}"
            f"  variables (shapes):\n"
            f"    {varinfo}\n"
            f"  nsim: {self.nsim}\n"
            f"  nsteps_p: {self.nsteps_p}\n"
            f"  nsteps_f: {self.nsteps_f}\n"
            f"  batches_p: {self.batched_data_p.shape[0]}\n"
            f"  batches_f: {self.batched_data_f.shape[0]}\n"
        )


def get_sequence_dataloaders_multistep(
    data, nsteps_p, nsteps_f, moving_horizon=False, norm_type="zero-one", split_ratio=None, num_workers=0,
):

    #data, _ = normalize_data(data, norm_type)
    train_data, dev_data, test_data = split_sequence_data(data, nsteps_p+nsteps_f, moving_horizon, split_ratio)
    
    train_data = SequenceDataset_MultiStep(
        train_data,
        nsteps_p=nsteps_p,
        nsteps_f=nsteps_f,
        moving_horizon=moving_horizon,
        name="train",
    )

    dev_data = SequenceDataset_MultiStep(
        dev_data,
        nsteps_p=nsteps_p,
        nsteps_f=nsteps_f,
        moving_horizon=moving_horizon,
        name="dev",
    )
    test_data = SequenceDataset_MultiStep(
        test_data,
        nsteps_p=nsteps_p,
        nsteps_f=nsteps_f,
        moving_horizon=moving_horizon,
        name="test",
    )

    train_loop = train_data.get_full_sequence()
    dev_loop = dev_data.get_full_sequence()
    test_loop = test_data.get_full_sequence()

    train_data = DataLoader(
        train_data,
        batch_size=len(train_data),
        shuffle=False,
        collate_fn=train_data.collate_fn,
        num_workers=num_workers,
    )
    dev_data = DataLoader(
        dev_data,
        batch_size=len(dev_data),
        shuffle=False,
        collate_fn=dev_data.collate_fn,
        num_workers=num_workers,
    )
    test_data = DataLoader(
        test_data,
        batch_size=len(test_data),
        shuffle=False,
        collate_fn=test_data.collate_fn,
        num_workers=num_workers,
    )

    return (train_data, dev_data, test_data), (train_loop, dev_loop, test_loop), train_data.dataset.dims


class StaticDataset(Dataset):
    def __init__(
        self,
        data,
        name="data",
    ):
        """Dataset for handling static data and transforming it into the dictionary structure
        used by NeuroMANCER models.

        :param data: (dict str: np.array) dictionary mapping variable names to tensors of shape
            (N, Dk), where N is the number of samples and Dk is dimensionality of variable k.
        :param name: (str) name of dataset split.

        .. warning:: This dataset class requires the use of a special collate function that must be
            provided to PyTorch's DataLoader class; see the `collate_fn` method of this class.

        .. todo:: Add support for categorical features.
        .. todo:: Add support for memory-mapped and/or streaming data.
        .. todo:: Add support for data augmentation?
        .. todo:: Clean up data validation code.
        """

        super().__init__()
        self.name = name

        self.variables = list(data.keys())
        self.full_data = torch.cat([torch.tensor(data[k], dtype=torch.float) for k in self.variables], dim=1)

        self.nsamples = self.full_data.shape[0]
        self.dims = {k: (self.nsamples, *data[k].shape[1:],) for k in self.variables}

        # _vslices used to slice out sequences of individual variables from full_data
        i = 0
        self._vslices = {}
        for k, v in self.dims.items():
            self._vslices[k] = slice(i, i + v[1], 1)
            i += v[1]

        self.dims = {
            **self.dims,
            "nsamples": self.nsamples,
        }

    def __len__(self):
        """Gives the number of samples in the dataset."""
        return self.nsamples

    def __getitem__(self, i):
        """Fetch a single sample from the dataset."""
        datapoint = {
            k: self.full_data[i, self._vslices[k]]
            for k in self.variables
        }
        datapoint['index'] = i
        return datapoint

    def get_full_batch(self):
        batch = {
            k: self.full_data[:, self._vslices[k]]
            for k in self.variables
        }
        batch["name"] = self.name
        return batch

    def collate_fn(self, batch):
        """Batch collation for dictionaries of samples generated by this dataset. This wraps the
        default PyTorch batch collation function and simply adds a "name" field to a batch.

        :param batch: (dict str: torch.Tensor) dataset sample.
        """
        batch = default_collate(batch)
        batch["name"] = self.name
        return batch

    def __repr__(self):
        varinfo = "\n    ".join([f"{x}: {d}" for x, d in self.dims.items() if x != "nsamples"])
        return (
            f"{type(self).__name__}:\n"
            f"  variables (shapes):\n"
            f"    {varinfo}\n"
            f"  nsamples: {self.nsamples}\n"
        )


def normalize_data(data, norm_type, stats=None):
    """Normalize data, optionally using arbitrary statistics (e.g. computed from train split).

    :param data: (dict str: np.array) data dictionary.
    :param norm_type: (str) type of normalization to use; can be "zero-one", "one-one", or "zscore".
    :param stats: (dict str: np.array) statistics to use for normalization. Default is None, in which
        case stats are inferred by underlying normalization function.
    """
    multisequence = _is_multisequence_data(data)
    assert _is_sequence_data(data) or multisequence, \
        "data must be provided as a dictionary or list of dictionaries"

    if not multisequence:
        data = [data]

    if stats is None:
        norm_fn = lambda x, _: norm_fns[norm_type](x)
    else:
        norm_fn = lambda x, k: norm_fns[norm_type](
            x,
            stats[k + "_min"].reshape(1, -1),
            stats[k + "_max"].reshape(1, -1),
        )

    keys = data[0].keys()
    slices = _get_sequence_time_slices(data)
    data = {k: np.concatenate([v[k] for v in data], axis=0) for k in keys}

    norm_data = [norm_fn(v, k) for k, v in data.items()]
    norm_data, stat0, stat1 = zip(*norm_data)

    stats = {
        **{k + "_min": v for k, v in zip(data.keys(), stat0)},
        **{k + "_max": v for k, v in zip(data.keys(), stat1)},
    }
    data = [{k: v[sl, ...] for k, v in zip(data.keys(), norm_data)} for sl in slices]

    return data if multisequence else data[0], stats


def split_sequence_data(data, nsteps, moving_horizon=False, split_ratio=None):
    """Split a data dictionary into train, development, and test sets. Splits data into thirds by
    default, but arbitrary split ratios for train and development can be provided.

    :param data: (dict str: np.array or list[str: np.array]) data dictionary.
    :param nsteps: (int) N-step prediction horizon for batching data; used here to ensure split
        lengths are evenly divisible by N.
    :param moving_horizon: (bool) whether batches use a sliding window with stride 1; else stride of
        N is assumed.
    :param split_ratio: (list float) Two numbers indicating percentage of data included in train and
        development sets (out of 100.0). Default is None, which splits data into thirds.

    .. todo:: polymorphic split arg for supporting split percentages or indices for multiseq case
        and determine best way to split given a dict of file names
    """
    multisequence = _is_multisequence_data(data)
    assert _is_sequence_data(data) or multisequence, \
        "data must be provided as a dictionary or list of dictionaries"

    nsim = len(data) if multisequence else min(v.shape[0] for v in data.values())
    split_mod = nsteps if not multisequence else 1
    if split_ratio is None:
        split_len = nsim // 3
        split_len -= split_len % split_mod
        train_slice = slice(0, split_len + nsteps * (not multisequence))
        dev_slice = slice(split_len, split_len * 2 + nsteps * (not multisequence))
        test_slice = slice(split_len * 2, nsim)
    else:
        dev_start = math.ceil(split_ratio[0] * nsim / 100.)
        test_start = dev_start + math.ceil(split_ratio[1] * nsim / 100.)
        train_slice = slice(0, dev_start)
        dev_slice = slice(dev_start, test_start)
        test_slice = slice(test_start, nsim)

    if not multisequence:
        train_data = {k: v[train_slice] for k, v in data.items()}
        dev_data = {k: v[dev_slice] for k, v in data.items()}
        test_data = {k: v[test_slice] for k, v in data.items()}
    else:
        train_data = data[train_slice]
        dev_data = data[dev_slice]
        test_data = data[test_slice]

    return train_data, dev_data, test_data


def split_static_data(data, split_ratio=None):
    """Split a data dictionary into train, development, and test sets. Splits data into thirds by
    default, but arbitrary split ratios for train and development can be provided.

    :param data: (dict str: np.array or list[str: np.array]) data dictionary.
    :param split_ratio: (list float) Two numbers indicating percentage of data included in train and
        development sets (out of 100.0). Default is None, which splits data into thirds.
    """

    nsim = min(v.shape[0] for v in data.values())
    if split_ratio is None:
        split_len = nsim // 3
        train_slice = slice(0, split_len)
        dev_slice = slice(split_len, split_len * 2)
        test_slice = slice(split_len * 2, nsim)
    else:
        dev_start = math.ceil(split_ratio[0] * nsim / 100.)
        test_start = dev_start + math.ceil(split_ratio[1] * nsim / 100.)
        train_slice = slice(0, dev_start)
        dev_slice = slice(dev_start, test_start)
        test_slice = slice(test_start, nsim)

    train_data = {k: v[train_slice] for k, v in data.items()}
    dev_data = {k: v[dev_slice] for k, v in data.items()}
    test_data = {k: v[test_slice] for k, v in data.items()}

    return train_data, dev_data, test_data


def standardize(M, mean=None, std=None):
    mean = M.mean(axis=0).reshape(1, -1) if mean is None else mean
    std = M.std(axis=0).reshape(1, -1) if std is None else std
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        M_norm = (M - mean) / std
    return np.nan_to_num(M_norm), mean.squeeze(0), std.squeeze(0)


def normalize_01(M, Mmin=None, Mmax=None):
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
    return np.nan_to_num(M_norm), Mmin.squeeze(0), Mmax.squeeze(0)


def normalize_11(M, Mmin=None, Mmax=None):
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
        M_norm = 2 * ((M - Mmin) / (Mmax - Mmin)) - 1
    return np.nan_to_num(M_norm), Mmin.squeeze(0), Mmax.squeeze(0)


def denormalize_01(M, Mmin, Mmax):
    """
    denormalize min max norm
    :param M: (2-d np.array) Data to be normalized
    :param Mmin: (int) Minimum value
    :param Mmax: (int) Maximum value
    :return: (2-d np.array) Un-normalized data
    """
    M_denorm = M * (Mmax - Mmin) + Mmin
    return M_denorm


def denormalize_11(M, Mmin, Mmax):
    """
    denormalize min max norm
    :param M: (2-d np.array) Data to be normalized
    :param Mmin: (int) Minimum value
    :param Mmax: (int) Maximum value
    :return: (2-d np.array) Un-normalized data
    """
    M_denorm = ((M + 1) / 2) * (Mmax - Mmin) + Mmin
    return M_denorm


def destandardize(M, mean, std):
    return M * std + mean


norm_fns = {
    "zscore": standardize,
    "zero-one": normalize_01,
    "one-one": normalize_11,
}

denorm_fns = {
    "zscore": destandardize,
    "zero-one": denormalize_01,
    "one-one": denormalize_11,
}
