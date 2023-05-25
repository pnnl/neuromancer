from plum import dispatch
import torch
import numpy
from typing import Union, Dict


class StandardScaler:
    """
    This class subsumes some boiler plate code translating between numpy and pytorch.
    All normalized and denormalized data should be returned as the same type.
    """
    def __init__(self, stats):
        self.stats = {}
        if isinstance(stats['mean'], torch.Tensor):
            self.stats[numpy.ndarray] = {k: v.detach().numpy() for k, v in stats.items()}
            self.stats[torch.Tensor] = stats
        elif isinstance(stats['mean'], numpy.ndarray):
            self.stats[numpy.ndarray] = stats
            self.stats[torch.Tensor] = {k: torch.tensor(v, dtype=torch.float32) for k, v in stats.items()}

    def transform(self, X):
        stats = self.stats[type(X)]
        if len(X.shape) > 2:
            shp = X.shape
            return ((X.reshape(-1, shp[-1]) - stats['mean']) / stats['std']).reshape(shp)
        else:
            return (X - stats['mean']) / stats['std']

    def inverse_transform(self, Z):
        stats = self.stats[type(Z)]
        if len(Z.shape) > 2:
            shp = Z.shape
            return (Z.reshape(-1, shp[-1]) * stats['std'] + stats['mean']).reshape(shp)
        else:
            return Z * stats['std'] + stats['mean']


Data = Union[torch.Tensor, numpy.ndarray]
DataDict = Dict[str, Data]
NormDict = Dict[str, StandardScaler]


@dispatch
def normalize(data: DataDict, norms: NormDict) -> DataDict:  # pylint: disable=function-redefined
    """

    """
    return {k: norms[k].transform(v) if k in norms else v for k, v in data.items()}


@dispatch
def normalize(data: Data, norm: StandardScaler) -> Data:  # pylint: disable=function-redefined
    """

    """
    return norm.transform(data)


@dispatch
def denormalize(data: DataDict, norms: NormDict) -> DataDict:  # pylint: disable=function-redefined
    """

    """
    return {k: norms[k].inverse_transform(v) if k in norms else v for k, v in data.items()}


@dispatch
def denormalize(data: Data, norm: StandardScaler) -> Data:  # pylint: disable=function-redefined
    """

    """
    return norm.inverse_transform(data)


