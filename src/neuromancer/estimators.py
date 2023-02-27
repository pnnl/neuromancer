"""

State estimators for SSM models
    + x: states (x0 - initial conditions)
    + u: control inputs
    + y: measured outputs
    + d: uncontrolled inputs (measured disturbances)

"""

# pytorch imports
import torch
import torch.nn as nn

# ecosystem imports
import neuromancer.slim as slim

# local imports
import neuromancer.blocks as blocks
from neuromancer.component import Component, check_key_subset


class TimeDelayEstimator(Component):

    def __init__(self, data_dims, nsteps=1, window_size=1, input_keys=[], name='estimator'):
        """

        :param data_dims: dict {str: tuple of ints) Data structure describing dimensions of input variables
        :param nsteps: (int) Prediction horizon
        :param window_size: (int) Size of sequence history to use as input to the state estimator.
        :param input_keys: (List of str) List of input variable names
        :param name: (str) Name for tracking output of module.
        """
        output_keys = [f"{k}_{name}" if name is not None else k for k in ["x0", "reg_error"]]
        super().__init__(input_keys=input_keys, output_keys=output_keys, name=name)

        assert window_size <= nsteps, f'Window size {window_size} longer than sequence length {nsteps}.'
        check_key_subset(set(input_keys), set(data_dims.keys()))
        self.name, self.data_dims = name, data_dims
        self.nsteps, self.window_size = nsteps, window_size
        self.nx = data_dims['x0'][-1]
        self.ny = data_dims['Yp'][-1]
        data_dims_in = {k: v for k, v in data_dims.items() if k in input_keys}
        self.sequence_dims_sum = sum(v[-1] for k, v in data_dims_in.items() if len(v) == 2)
        self.static_dims_sum = sum(v[-1] for k, v in data_dims_in.items() if len(v) == 1)
        self.in_features = self.static_dims_sum + window_size * self.sequence_dims_sum
        self.out_features = self.nx

    def reg_error(self):
        """

        :return: A scalar value of regularization error associated with submodules
        """
        error = sum([k.reg_error() for k in self.children() if hasattr(k, 'reg_error')])
        if not isinstance(error, torch.Tensor):
            error = torch.Tensor(error)
        return error

    def features(self, data):
        """
        Compile a feature vector using data features corresponding to self.input_keys

        :param data: (dict {str: torch.Tensor})
        :return: (torch.Tensor)
        """
        check_key_subset(self.input_keys, set(data.keys()))
        featlist = []
        for k in self.input_keys:
            assert self.data_dims[k][-1] == data[k].shape[-1], \
                f'Input feature {k} expected {self.data_dims[k][-1]} but got {data[k].shape[-1]}'
            if len(data[k].shape) == 2:
                featlist.append(data[k])
            elif len(data[k].shape) == 3:
                assert data[k].shape[1] >= self.nsteps, \
                    f'Sequence too short for estimator calculation. Should be at least {self.nsteps}'
                featlist.append(data[k][:, self.nsteps - self.window_size:self.nsteps, :].reshape(data[k].shape[0], -1))
            else:
                raise ValueError(f'Input {k} has {len(data[k].shape)} dimensions. Should have 2 or 3 dimensions')
        features = torch.cat(featlist, dim=1)
        return features

    def forward(self, data):
        """

        :param data: (dict {str: torch.tensor)}
        :return: (dict {str: torch.tensor)}
        """
        features = self.features(data)
        output = {name: tensor for tensor, name
                  in zip([self.net(features),  self.reg_error()], self.output_keys)}
        return output


class FullyObservable(TimeDelayEstimator):
    def __init__(self, data_dims, nsteps=1, window_size=1, bias=False,
                 linear_map=slim.Linear, nonlin=nn.Identity, hsizes=[],
                 input_keys=['Yp'], linargs=dict(), name='fully_observable'):
        """
        Dummmy estimator to use consistent API for fully and partially observable systems
        """
        super().__init__(data_dims, nsteps=nsteps, window_size=window_size,
                         input_keys=input_keys, name=name)
        self.net = nn.Identity()

    def features(self, data):
        return data[self.input_keys[0]][:, self.nsteps-1, :]

    def reg_error(self):
        return torch.tensor(0.0)


class FullyObservableAugmented(FullyObservable):
    def __init__(self, data_dims, nsteps=1, window_size=1, d0=0.0, bias=False,
                 linear_map=slim.Linear, nonlin=nn.Identity, hsizes=[],
                 input_keys=['Yp'], linargs=dict(), name='fully_observable_aug'):
        """
        Dummmy estimator to use consistent API for fully observable systems with augmented state space with disturbances
        """
        super().__init__(data_dims, nsteps=nsteps, window_size=window_size,
                         input_keys=input_keys, name=name)
        self.net = nn.Identity()
        self.nd = self.nx - self.ny   # dimensions of the augmented states
        self.d0 = d0   # fixed initial conditions of the augmented state

    def features(self, data):
        nsamples, ny = data['Yp'][:, 0, :].shape
        augmented_state = self.d0*torch.ones([nsamples, self.nd])
        return torch.cat([data['Yp'][:, self.nsteps - 1, :], augmented_state], 1)


class LinearEstimator(TimeDelayEstimator):
    def __init__(self, data_dims, nsteps=1, window_size=1, bias=False,
                 linear_map=slim.Linear, nonlin=nn.Identity, hsizes=[],
                 input_keys=['Yp'], linargs=dict(), name='linear_estim'):
        """

        See base class for arguments
        """
        super().__init__(data_dims, nsteps=nsteps, window_size=window_size, input_keys=input_keys, name=name)
        self.net = linear_map(self.in_features, self.out_features, bias=bias, **linargs)


class MLPEstimator(TimeDelayEstimator):
    """

    """
    def __init__(self, data_dims, nsteps=1, window_size=1, bias=False,
                 linear_map=slim.Linear, nonlin=nn.GELU, hsizes=[64],
                 input_keys=['Yp'], linargs=dict(), name='MLP_estim'):
        """
        See base class for arguments
        """
        super().__init__(data_dims, nsteps=nsteps, window_size=window_size, input_keys=input_keys, name=name)
        self.net = blocks.MLP(self.in_features, self.out_features, bias=bias,
                              linear_map=linear_map, nonlin=nonlin, hsizes=hsizes, linargs=linargs)


class MLPAugmentedEstimator(MLPEstimator):
    """
    Appends observed variables to state estimated from a Multi-layer Perceptron
    """
    def __init__(self, data_dims, nsteps=1, window_size=1, bias=False,
                 linear_map=slim.Linear, nonlin=nn.GELU, hsizes=[64],
                 input_keys=['Yp'], linargs=dict(), name='MLP_estim'):
        """
        See base class for arguments
        """
        super().__init__(data_dims, nsteps=nsteps, window_size=window_size, bias=bias,
                         linear_map=linear_map, nonlin=nonlin, hsizes=hsizes,
                         input_keys=input_keys, linargs=linargs, name=name)
        self.net = blocks.MLP(self.in_features, self.nx - self.ny, bias=bias,
                              linear_map=linear_map, nonlin=nonlin, hsizes=hsizes, linargs=linargs)

    def forward(self, data):
        X = data[self.input_keys[0]][:, self.nsteps - 1, :]
        output = super().forward(data)
        output[self.output_keys[0]] = torch.cat([X, output[self.output_keys[0]]], dim=-1)
        return output


class ResMLPEstimator(TimeDelayEstimator):
    """

    """
    def __init__(self, data_dims, nsteps=1, window_size=1, bias=False,
                 linear_map=slim.Linear, nonlin=nn.GELU, hsizes=[64],
                 input_keys=['Yp'], linargs=dict(), name='ResMLP_estim'):
        """
        see base class for arguments
        """
        super().__init__(data_dims, nsteps=nsteps, window_size=window_size, input_keys=input_keys, name=name)
        self.net = blocks.ResMLP(self.in_features, self.out_features, bias=bias,
                                 linear_map=linear_map, nonlin=nonlin, hsizes=hsizes, linargs=linargs)


class RNNEstimator(TimeDelayEstimator):
    def __init__(self, data_dims, nsteps=1, window_size=1, bias=False,
                 linear_map=slim.Linear, nonlin=nn.GELU, hsizes=[64],
                 input_keys=['Yp'], linargs=dict(), name='RNN_estim'):
        """
        see base class for arguments
        """
        super().__init__(data_dims, nsteps=nsteps, window_size=window_size, input_keys=input_keys, name=name)
        self.in_features = self.sequence_dims_sum
        self.net = blocks.RNN(self.in_features, self.out_features, hsizes=hsizes,
                              bias=bias, nonlin=nonlin, linear_map=linear_map, linargs=linargs)

    def forward(self, data):
        features = torch.cat([data[k][:, self.nsteps-self.window_size:self.nsteps, :] for k in self.input_keys], dim=2)
        output = {name: tensor for tensor, name
                  in zip([self.net(features),  self.reg_error()], self.output_keys)}
        return output


estimators = {'fullobservable': FullyObservable,
              'linear': LinearEstimator,
              'mlp': MLPEstimator,
              'rnn': RNNEstimator,
              'residual_mlp': ResMLPEstimator,
              'mlpaugmented': MLPAugmentedEstimator,
              'fullobservableaugmented': FullyObservableAugmented}
