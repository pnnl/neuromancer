"""
policies for SSM models
x: states
u: control inputs
d: uncontrolled inputs (measured disturbances)
r: reference signals

generic mapping:
u = policy(x,u,d,r)
"""

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
#local imports
import linear
from blocks import MLP
from rnn import RNN, RNNCell


def check_keys(k1, k2):
    assert k1 - k2 == set(), \
        f'Missing values in dataset. Input_keys: {k1}, data_keys: {k2}'


class Policy(nn.Module):
    def __init__(self, data_dims, nsteps=1, bias=False,
                 Linear=linear.Linear, nonlin=None, hsizes=None, num_layers=None, input_keys={'x0'}):
        """

        :param nx: (int) dimension of state
        :param nu: (int) dimension of inputs
        :param nd: (int) dimension of disturbances
        :param ny: (int) dimension of observation/reference
        :param N: (int) prediction horizon
        """
        super().__init__()
        check_keys(set(set(input_keys), data_dims.keys()))

        nu = data_dims['U']
        nx = data_dims['x0']
        data_dims = {k: v for k, v in data_dims if k in input_keys}
        sequence_dims_sum = sum(v for k, v in data_dims.items() if k != 'x0')

        self.input_size = nx + nsteps * sequence_dims_sum
        self.output_size = nsteps * nu
        self.input_keys = set(input_keys)

    def reg_error(self):
        return self.net.reg_error()

    def forward(self, data):
        check_keys(self.input_keys, set(data.keys()))
        features = data['x0']
        for k in self.input_keys - {'x0'}:
            new_feat = torch.cat([step for step in data[k]], dim=1)
            features = torch.cat([features, new_feat], dim=1)
        Uf, reg_error = self.net(features)
        return {'Uf': Uf, f'{self.name}_reg_error': self.reg_error()}


class LinearPolicy(Policy):
    def __init__(self, data_dims, nsteps=1, bias=False,
                 Linear=linear.Linear, nonlin=None, hsizes=None, num_layers=None, input_keys={'x0'}):
        """

        :param data_dims:
        :param nsteps:
        :param bias:
        :param Linear:
        :param nonlin:
        :param hsizes:
        :param num_layers:
        :param input_keys:
        """
        super().__init__(data_dims, nsteps=nsteps, bias=bias,
                         Linear=Linear, nonlin=nonlin, hsizes=hsizes,
                         num_layers=num_layers, input_keys=input_keys)
        self.net = Linear(self.input_size, self.output_size, bias=bias)

    def forward(self, data):
        output = super().forward(data)
        output['Uf'] = torch.cat([u.reshape(self.nsteps, 1, -1) for u in output['Uf']], dim=1)
        return output


class MLPPolicy(nn.Module):
    def __init__(self, data_dims, nsteps=1, bias=False,
                 Linear=linear.Linear, nonlin=None, hsizes=None, num_layers=None, input_keys={'x0'}):
        super().__init__(data_dims, nsteps=nsteps, bias=bias,
                         Linear=Linear, nonlin=nonlin, hsizes=hsizes,
                         num_layers=num_layers, input_keys=input_keys)
        self.net = MLP(insize=self.input_size, outsize=self.output_size, bias=bias,
                       Linear=Linear, nonlin=nonlin, hsizes=hsizes)

    def forward(self, data):
        output = super().forward(data)
        output['Uf'] = torch.cat([u.reshape(self.nsteps, 1, -1) for u in output['Uf']], dim=1)
        return output


class RNNPolicy(nn.Module):
    def __init__(self, data_dims, nsteps=1, bias=False,
                 Linear=linear.Linear, nonlin=None, hsizes=None, num_layers=None, input_keys={'x0'}):
        super().__init__(data_dims, nsteps=nsteps, bias=bias,
                         Linear=Linear, nonlin=nonlin, hsizes=hsizes,
                         num_layers=num_layers, input_keys=input_keys)
        self.net = RNN(self.input_size, self.output_size//self.nsteps, num_layers=num_layers,
                       bias=bias, nonlin=nonlin, Linear=Linear)

    def forward(self, data):
        check_keys(self.input_keys, set(data.keys()))
        features = torch.cat([data[k] for k in self.input_keys], dim=2)
        x_feats = torch.stack([data['x0'] for d in features.shape[1]], dim=1)
        features = torch.cat([features, x_feats], dim=1)
        return self.net(features)


# similar structure to Linear Kalman Filter
class LQRPolicy(nn.Module):
    pass


policies = [LinearPolicy, MLPPolicy, RNNPolicy]


if __name__ == '__main__':
    nx, ny, nu, nd = 15, 7, 5, 3
    N = 10
    samples = 100
    # Data format: (N,samples,dim)
    x = torch.rand(samples, nx)
    D = torch.rand(N, samples, nd)
    R = torch.rand(N, samples, ny)

    for pol in policies:
        p = pol(nx, nu, nd, ny, N)
        p_out = p(x, D, R)
        print(p_out[0].shape, p_out[1].shape)