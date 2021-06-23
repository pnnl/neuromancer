import warnings

import numpy as np
from typing import Union


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