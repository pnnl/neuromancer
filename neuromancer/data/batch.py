"""
Batching and unbatching utilities for NeuroMANCER Datasets.
"""

import torch
import numpy as np


def batch_mh_data(data, nsteps):
    """
    moving horizon batching

    :param data: np.array shape=(nsim, dim)
    :param nsteps: (int) n-step prediction horizon
    :return: np.array shape=(nsteps, nsamples, dim)
    """
    end_step = data.shape[0] - nsteps
    data = np.asarray(
        [data[k : k + nsteps, :] for k in range(0, end_step)]
    )  # nchunks X nsteps X nfeatures
    return data.transpose(1, 0, 2)  # nsteps X nsamples X nfeatures


def batch_data(data, nsteps):
    """

    :param data: np.array shape=(nsim, dim)
    :param nsteps: (int) n-step prediction horizon
    :return: np.array shape=(nsteps, nsamples, dim)
    """
    nsplits = (data.shape[0]) // nsteps
    leftover = (data.shape[0]) % nsteps
    data = np.stack(
        np.split(data[: data.shape[0] - leftover], nsplits)
    )  # nchunks X nsteps X nfeatures
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
    data = np.stack(
        np.split(data[: data.shape[0] - leftover], nsplits)
    )  # nchunks X nsteps X nfeatures
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
