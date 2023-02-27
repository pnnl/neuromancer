"""

policies for SSM models

    + x: states
    + u: control inputs
    + d: uncontrolled inputs (measured disturbances)
    + r: reference signals

generic mapping:

    + u = policy(x,u,d,r)
"""

# machine learning/data science imports
import torch
import torch.nn as nn
# ecosystem imports
import neuromancer.slim as slim
# local imports
import neuromancer.blocks as blocks
from neuromancer.component import Component



class Policy(Component):

    def __init__(self, data_dims, nsteps=1,forecast = False, input_keys=["x0"], name="policy"):
        """

        :param data_dims: dict {str: tuple of ints) Data structure describing dimensions of input variables
        :param nsteps: (int) Prediction horizon
        :param forecast: (bool) Should time dimensions of inputs be preserved, (e.g. a forecast of disturbances to inform the control policy)
        :param input_keys: (List of str) List of input variable names
        :param name: (str) Name for tracking output of module.
        """
        output_keys = [f"{k}_{name}" if name is not None else k for k in ["U_pred", "reg_error"]]
        super().__init__(input_keys=input_keys, output_keys=output_keys, name=name)

        self.name, self.data_dims = name, data_dims
        self.nsteps = nsteps
        self.forecast = forecast
        self.nu = data_dims["U"][-1]
        data_dims_in = {k: v for k, v in data_dims.items() if k in input_keys}
        self.sequence_dims_sum = sum(v[-1] for k, v in data_dims_in.items() if len(v) == 2)
        self.static_dims_sum = sum(v[-1] for k, v in data_dims_in.items() if len(v) == 1)
        if self.forecast ==False:
            self.in_features = self.static_dims_sum + nsteps * self.sequence_dims_sum
            self.out_features = nsteps * self.nu
        else:
            self.in_features = self.static_dims_sum + self.sequence_dims_sum
            self.out_features = self.nu


    def reg_error(self):
        """

        :return: A scalar value of regularization error associated with submodules
        """
        error = sum([k.reg_error() for k in self.children() if hasattr(k, "reg_error")])
        if not isinstance(error, torch.Tensor):
            error = torch.Tensor(error)
        return error

    def features(self, data):
        """
        Compile a feature vector using data features corresponding to self.input_keys

        :param data: (dict {str: torch.Tensor})
        :return: (torch.Tensor)
        """
        featlist = []
    
        for k in self.input_keys:
            assert self.data_dims[k][-1] == data[k].shape[-1], \
                f"Input feature {k} expected {self.data_dims[k][-1]} but got {data[k].shape[-1]}"
            if len(data[k].shape) == 2:
                if self.forecast == False: featlist.append(data[k])
                else: featlist.append(torch.tile(torch.unsqueeze(data[k],1),(1,self.nsteps,1)))
            elif len(data[k].shape) == 3:
                assert data[k].shape[1] >= self.nsteps, \
                    f"Sequence too short for policy calculation. Should be at least {self.nsteps}"
                if self.forecast == False: featlist.append(data[k][:, :self.nsteps, :].reshape(data[k].shape[0], -1))
                else: featlist.append(data[k][:, :self.nsteps, :])
            else:
                raise ValueError(f"Input {k} has {len(data[k].shape)} dimensions. Should have 2 or 3 dimensions")
            
            if self.forecast == False: feat_tensor = torch.cat(featlist, dim=1)
            else: feat_tensor = torch.cat(featlist, dim=2)
        return feat_tensor

    def forward(self, data):
        """

        :param data: (dict {str: torch.tensor)}
        :return: (dict {str: torch.tensor)}
        """
        features = self.features(data)
        Uf = self.net(features)
        Uf = Uf.reshape(features.shape[0], self.nsteps, -1)
        output = {name: tensor for tensor, name
                  in zip([Uf, self.reg_error()], self.output_keys)}
        return output


class Compensator(Policy):
    def __init__(self, data_dims, policy_output_keys, nsteps=1, input_keys=['Ep'], name='compensator'):
        """
            Additive compensator for the nominal policy: e.g. for online updates
        :param data_dims: dict {str: tuple of ints) Data structure describing dimensions of input variables
        :param policy_output_keys: output keys of the original policy to add upon
        :param nsteps: (int) Prediction horizon
        :param input_keys: (List of str) List of input variable names
        :param name: (str) Name for tracking output of module.
        """
        super().__init__(data_dims, nsteps=nsteps, input_keys=input_keys, name=name)
        self.policy_output_keys = policy_output_keys
        assert len(input_keys) == 1, \
            f'One input key expected but got {len(input_keys)}. ' \
            f'Required format input_keys=[\'error signal\'].'

    def forward(self, data):
        """

        :param data: (dict {str: torch.tensor)}
        :return: (dict {str: torch.tensor)}
        """
        U_nominal = data[self.policy_output_keys]
        features = self.features(data)
        U_compensator = self.net(features).reshape(features.shape[0], -1)
        Uf = U_nominal + U_compensator
        output = {name: tensor for tensor, name
                  in zip([Uf, self.reg_error()], self.output_keys)}
        return output


class LinearCompensator(Compensator):
    def __init__(self, data_dims, policy_output_keys, nsteps=1, bias=False,
                 linear_map=slim.Linear, nonlin=None, hsizes=None,
                 input_keys=['Ep'], linargs=dict(), name='linear_compensator'):
        """

        :param data_dims:
        :param policy_output_keys: output keys of the original policy to add upon
        :param nsteps:
        :param bias:
        :param linear_map:
        :param nonlin:
        :param hsizes:
        :param input_keys:
        :param linargs:
        :param name:
        """
        super().__init__(data_dims, policy_output_keys, nsteps=nsteps, input_keys=input_keys, name=name)
        self.net = linear_map(self.in_features, self.out_features, bias=bias, **linargs)


class LinearPolicy(Policy):
    def __init__(self, data_dims, nsteps=1, bias=False,
                 linear_map=slim.Linear, nonlin=None, hsizes=None,
                 input_keys=["x0"], linargs=dict(), name="linear_policy"):
        """
        :param data_dims: dict {str: tuple of ints) Data structure describing dimensions of input variables
        :param nsteps: (int) Prediction horizon
        :param bias: (bool) Whether to use bias in MLP
        :param Linear: (class) slim.Linear class for subcomponents
        :param nonlin: (class) Pytorch elementwise activation function class for subcomponents
        :param hsizes: (List [int]) Sizes of hidden layers in MLP
        :param input_keys: (List of str) List of input variable names
        :param linargs: (dict) Arguments for instantiating linear layers.
        :param name: (str) Name for tracking output of module.
        """
        super().__init__(data_dims, nsteps=nsteps, input_keys=input_keys, name=name)
        self.net = linear_map(self.in_features, self.out_features, bias=bias, **linargs)


class MLPPolicy(Policy):
    def __init__(self, data_dims, nsteps=1, bias=True,
                 linear_map=slim.Linear, nonlin=nn.GELU, hsizes=[64],
                 input_keys=["x0"], linargs=dict(), name="MLP_policy"):
        """

        See LinearPolicy for arguments
        """
        super().__init__(data_dims, nsteps=nsteps, input_keys=input_keys, name=name)
        self.net = blocks.MLP(insize=self.in_features, outsize=self.out_features, bias=bias,
                              linear_map=linear_map, nonlin=nonlin, hsizes=hsizes, linargs=linargs)

class MLP_boundsPolicy(Policy):
    def __init__(self, data_dims, nsteps=1, bias=True,
                 linear_map=slim.Linear, nonlin=nn.GELU, hsizes=[64],
                 min=0.0, max=1.0, method='sigmoid_scale',
                 input_keys=["x0"], linargs=dict(), name="MLP_policy"):
        """

        See LinearPolicy for arguments
        """
        super().__init__(data_dims, nsteps=nsteps, input_keys=input_keys, name=name)
        self.net = blocks.MLP_bounds(insize=self.in_features, outsize=self.out_features, bias=bias,
                              linear_map=linear_map, nonlin=nonlin, hsizes=hsizes,
                              min=min, max=max, method=method, linargs=linargs)


class RNNPolicy(Policy):
    def __init__(self, data_dims, nsteps=1, bias=True,
                 linear_map=slim.Linear, nonlin=nn.GELU, hsizes=[64],
                 input_keys=["x0"], linargs=dict(), name="RNN_policy"):
        """
        See LinearPolicy for arguments
        """
        super().__init__(data_dims, nsteps=nsteps, input_keys=input_keys, name=name)
        self.in_features = self.sequence_dims_sum + self.static_dims_sum
        self.net = blocks.RNN(self.in_features, self.out_features, hsizes=hsizes,
                              bias=bias, nonlin=nonlin, linear_map=linear_map, linargs=linargs)

    def forward(self, data):
        """

        :param data: (dict {str: torch.tensor)}
        :return: (dict {str: torch.tensor)}
        """
        features = torch.cat([
            *[data[k][:, :self.nsteps, :] for k in self.input_keys if len(data[k].shape) == 3],
            *[torch.cat([data[k] for _ in range(self.nsteps)], dim=1).reshape(data[k].shape[0], self.nsteps, data[k].shape[1])
              for k in self.input_keys if len(data[k].shape) == 2]], dim=-1)
        print('features', features.shape)
        Uf = self.net(features).reshape(features.shape[0], self.nsteps, -1)
        output = {name: tensor for tensor, name
                  in zip([Uf, self.net.reg_error()], self.output_keys)}
        return output


class ConvolutionalForecastPolicy(Policy):
    def __init__(self, data_dims, nsteps=1, hsizes=[20], linear_map=slim.Linear,
                 nonlin=nn.Tanh, linargs=dict(), forecast=True, kernel_size=1,
                 input_keys=["x0", "Df"], name="CNV_policy"):
        """
        :param nsteps: (int) Prediction horizon and horizon over which forecast data is available
        :param kernel_size: (int) Number of time-steps to include in the convolution kernel.

        """
        super().__init__(data_dims, nsteps=nsteps, forecast=forecast,
                         input_keys=input_keys, name=name)
        
        self.h_dim = self.in_features
        self.kernel_size = kernel_size
        self.Conv_Tanh_stack = nn.Sequential(
            nn.Conv1d(self.in_features, self.h_dim,
                      self.kernel_size, stride=1, padding='same'),
            nn.Tanh(),
            nn.Conv1d(self.h_dim, self.h_dim,
                      self.kernel_size, stride=1, padding='same'),
            nn.Tanh(), 
            nn.Conv1d(self.h_dim, self.h_dim,
                      self.kernel_size, stride=1, padding='same')
        )
        self.output_block = blocks.MLP(insize=self.h_dim,
                                       outsize=self.out_features, bias=True,
                                       linear_map=linear_map, nonlin=nonlin,
                                       hsizes=hsizes, linargs=linargs)
        def net(features):
            feats = torch.transpose(features, 1, 2)
            output = self.Conv_Tanh_stack(feats)
            output = torch.transpose(output, 1, 2)
            output = self.output_block(output)
            return output

        self.net = net



policies = [LinearPolicy, MLPPolicy, RNNPolicy]

