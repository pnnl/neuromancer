import math

import numpy as np
import pandas as pd
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

from neuromancer.data.normalization import norm_fns


def _is_multisequence_data(data):
    return isinstance(data, list) and all([isinstance(x, dict) for x in data])


def _is_sequence_data(data):
    return isinstance(data, dict) and len({x.shape[0] for x in data.values()}) == 1


def _extract_var(data, regex):
    filtered = data.filter(regex=regex).values
    return filtered if filtered.size != 0 else None


def read_file(file_path):
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
        torch.cat((x[:, :, :, 0], x[-1, :, :, 1:].permute(2, 0, 1)), dim=0)
        if mh
        else torch.cat(torch.unbind(x, 0), dim=-1).permute(2, 0, 1)
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
            (T, Dk), where T is number of time steps and Dk is dimensionality of variable k
        :param nsteps: (int) N-step prediction horizon for batching data
        :param moving_horizon: (bool) if True, generate batches using sliding window with stride 1;
            else use stride N
        :param name: (str) name of dataset split

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
            f"length of time series data ({v.shape[0]}) must be greater than nsteps ({nsteps})"

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
        return {
            **{
                k + "p": self.batched_data[i, :, self._vslices[k]]
                for k in self.variables
            },
            **{
                k + "f": self.batched_data[i + 1, :, self._vslices[k]]
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
                k + "p": self.full_data[start : end - 1, self._vslices[k]].unsqueeze(1)
                for k in self.variables
            },
            **{
                k + "f": self.full_data[start + 1 : end, self._vslices[k]].unsqueeze(1)
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
            f"  nsteps: {self.nsteps}\n"
            f"  batches: {len(self)}\n"
        )


def normalize_data(data, norm_type, stats=None):
    """Normalize data, optionally using arbitrary statistics (e.g. computed from train split).

    :param data: (dict str: np.array) data dictionary
    :param norm_type: (str) type of normalization to use; can be "zero-one", "one-one", or "zscore"
    :param stats: (dict str: np.array) statistics to use for normalization. Default is None, in which
        case stats are inferred by underlying normalization function
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


def split_data(data, split_ratio=None):
    """Split a data dictionary into train, development, and test sets. Splits data into thirds by
    default, but arbitrary split ratios for train and development can be provided.

    :param data: (dict str: np.array or list[str: np.array]) data dictionary
    :param split_ratio: (list float) Two numbers indicating percentage of data included in train and
        development sets (out of 100.0). Default is None, which splits data into thirds.

    .. todo:: add real support for multi-experiment datasets
    .. todo:: fix split boundary cutoff issue when len(data) % nsteps != 0
    """
    multisequence = _is_multisequence_data(data)
    assert _is_sequence_data(data) or multisequence, \
        "data must be provided as a dictionary or list of dictionaries"

    nsim = len(data) if multisequence else min(v.shape[0] for v in data.values())
    if split_ratio is None:
        split_len = nsim // 3
        train_offs = slice(0, split_len)
        dev_offs = slice(split_len, split_len * 2)
        test_offs = slice(split_len * 2, nsim)
    else:
        dev_start = math.ceil(split_ratio[0] / 100.) * nsim
        test_start = dev_start + math.ceil(split_ratio[1] / 100.) * nsim
        train_offs = slice(0, dev_start)
        dev_offs = slice(dev_start, test_start)
        test_offs = slice(test_start, nsim)

    if not multisequence:
        train_data = {k: v[train_offs] for k, v in data.items()}
        dev_data = {k: v[dev_offs] for k, v in data.items()}
        test_data = {k: v[test_offs] for k, v in data.items()}
    else:
        train_data = data[train_offs]
        dev_data = data[dev_offs]
        test_data = data[test_offs]

    return train_data, dev_data, test_data


def get_sequence_dataloaders(
    data, nsteps, moving_horizon=False, norm_type="zero-one", split_ratio=None, num_workers=0,
):
    """This will generate dataloaders and open-loop sequence dictionaries for a given dictionary of
    data. Dataloaders are hard-coded for full-batch training to match NeuroMANCER's original
    training setup.

    :param data: (dict str: np.array) data dictionary
    :param nsteps: (int) length of windowed subsequences for N-step training
    :param norm_type: (str) type of normalization; see function `normalize_data` for more info.
    :param split_ratio: (list float) percentage of data in train and development splits; see
        function `split_data` for more info.
    """

    data, _ = normalize_data(data, norm_type)
    train_data, dev_data, test_data = split_data(data, split_ratio)

    train_data = SequenceDataset(train_data, nsteps=nsteps, name="train")
    dev_data = SequenceDataset(dev_data, nsteps=nsteps, name="dev")
    test_data = SequenceDataset(test_data, nsteps=nsteps, name="test")

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
