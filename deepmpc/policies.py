"""
TODO: Better high level comments
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
import blocks



def check_keys(k1, k2):
    assert set(k1) - set(k2) == set(), \
        f'Missing values in dataset. Input_keys: {set(k1)}, data_keys: {set(k2)}'


class SolutionMap(nn.Module):
    def __init__(self, data_dims, input_keys=['x'], output_keys=['z'], bias=False,
                 Linear=linear.Linear, nonlin=F.gelu, hsizes=[64], linargs=dict(), name='sol_map'):
        """
        Solution map for multiparametric programming problems
        :param data_dims:
        :param input_keys:
        :param output_keys:
        :param name:
        """
        super().__init__()
        check_keys(set(input_keys), set(data_dims.keys()))
        self.name = name
        self.input_keys = input_keys
        output_keys.append(f'{self.name}_reg_error')
        self.output_keys = output_keys
        data_dims_in = {k: v for k, v in data_dims.items() if k in input_keys}
        self.input_size = sum(v for k, v in data_dims_in.items())
        self.output_size = data_dims[output_keys[0]]
        self.net = blocks.MLP(insize=self.input_size, outsize=self.output_size, bias=bias,
                              Linear=Linear, nonlin=nonlin, hsizes=hsizes, linargs=linargs)

    def reg_error(self):
        """

        :return:
        """
        return self.net.reg_error()

    def forward(self, data):
        """

        :param data:
        :return:
        """
        check_keys(self.input_keys, set(data.keys()))
        features = data[self.input_keys[0]]
        for k in self.input_keys[1:]:
            features = torch.cat([features, data[k]], dim=1)
        Z = self.net(features)
        return {self.output_keys[0]: Z, self.output_keys[1]: self.net.reg_error()}



class Policy(nn.Module):
    def __init__(self, data_dims, nsteps=1, input_keys=['x0'], output_keys=['U_pred'], name='policy'):
        """

        :param data_dims:
        :param nsteps:
        :param input_keys:
        :param name:
        """
        super().__init__()
        check_keys(set(input_keys), set(data_dims.keys()))
        self.name = name
        self.nsteps = nsteps
        self.nu = data_dims[output_keys[0]]
        self.nx = data_dims[input_keys[0]]
        data_dims_in = {k: v for k, v in data_dims.items() if k in input_keys}
        self.sequence_dims_sum = sum(v for k, v in data_dims_in.items() if k != input_keys[0])
        self.input_size = self.nx + nsteps * self.sequence_dims_sum
        self.output_size = nsteps * self.nu
        self.input_keys = input_keys
        output_keys.append(f'{self.name}_reg_error')
        self.output_keys = output_keys

    def reg_error(self):
        """

        :return:
        """
        return self.net.reg_error()

    def forward(self, data):
        """

        :param data:
        :return:
        """
        check_keys(self.input_keys, set(data.keys()))
        features = data[self.input_keys[0]]
        for k in self.input_keys[1:]:
            new_feat = torch.cat([step for step in data[k]], dim=1)
            features = torch.cat([features, new_feat], dim=1)
        Uf = self.net(features)
        Uf = torch.cat([u.reshape(self.nsteps, 1, -1) for u in Uf], dim=1)
        # return {'U_pred': Uf, f'{self.name}_reg_error': self.net.reg_error()}
        return {self.output_keys[0]: Uf, self.output_keys[1]: self.net.reg_error()}

class LinearPolicy(Policy):
    def __init__(self, data_dims, nsteps=1, bias=False,
                 Linear=linear.Linear, nonlin=None, hsizes=None,
                 input_keys=['x0'], output_keys=['U_pred'],
                 linargs=dict(), name='linear_policy'):
        """

        :param data_dims:
        :param nsteps:
        :param bias:
        :param Linear:
        :param nonlin:
        :param hsizes:
        :param input_keys:
        """
        super().__init__(data_dims, nsteps=nsteps, input_keys=input_keys,
                         output_keys=output_keys, name=name)
        self.net = Linear(self.input_size, self.output_size, bias=bias, **linargs)


class MLPPolicy(Policy):
    def __init__(self, data_dims, nsteps=1, bias=False,
                 Linear=linear.Linear, nonlin=F.gelu, hsizes=[64],
                 input_keys=['x0'], output_keys=['U_pred'],
                 linargs=dict(), name='MLP_policy'):
        """

        :param data_dims:
        :param nsteps:
        :param bias:
        :param Linear:
        :param nonlin:
        :param hsizes:
        :param input_keys:
        :param linargs:
        """
        super().__init__(data_dims, nsteps=nsteps, input_keys=input_keys,
                         output_keys=output_keys, name=name)
        self.net = blocks.MLP(insize=self.input_size, outsize=self.output_size, bias=bias,
                              Linear=Linear, nonlin=nonlin, hsizes=hsizes, linargs=linargs)


class RNNPolicy(Policy):
    def __init__(self, data_dims, nsteps=1, bias=False,
                 Linear=linear.Linear, nonlin=F.gelu, hsizes=[64],
                 input_keys=['x0'], output_keys=['U_pred'],
                 linargs=dict(), name='RNN_policy'):
        """

        :param data_dims:
        :param nsteps:
        :param bias:
        :param Linear:
        :param nonlin:
        :param hsizes:
        :param input_keys:
        :param linargs:
        """
        super().__init__(data_dims, nsteps=nsteps, input_keys=input_keys,
                         output_keys=output_keys, name=name)
        self.input_size = self.sequence_dims_sum + self.nx
        self.net = blocks.RNN(self.input_size, self.output_size, hsizes=hsizes,
                              bias=bias, nonlin=nonlin, Linear=Linear, linargs=linargs)

    def forward(self, data):
        """

        :param data:
        :return:
        """
        check_keys(self.input_keys, set(data.keys()))
        features = torch.cat([data[k] for k in self.input_keys[1:]], dim=2)
        x_feats = torch.stack([data[self.input_keys[0]] for d in range(features.shape[0])], dim=0)
        features = torch.cat([features, x_feats], dim=2)
        Uf = self.net(features)
        Uf = torch.cat([u.reshape(self.nsteps, 1, -1) for u in Uf], dim=1)
        return {self.output_keys[0]: Uf, self.output_keys[1]: self.net.reg_error()}
        # return {'U_pred': Uf, f'{self.name}_reg_error': self.net.reg_error()}


# similar structure to Linear Kalman Filter
class LQRPolicy(nn.Module):
    pass


policies = [LinearPolicy, MLPPolicy, RNNPolicy]

# TODO: current handling of input key x0 via self.input_keys[0] is a tricky one, what if we would like to have more static features?

if __name__ == '__main__':
    nx, ny, nu, nd = 15, 7, 5, 3
    N = 10
    samples = 100
    # Data format: (N,samples,dim)
    x = torch.rand(samples, nx)
    D = torch.rand(N, samples, nd)
    R = torch.rand(N, samples, ny)
    data = {'x0': x, 'D': D, 'R': R}
    data_dims = {'x0': nx, 'D': nd, 'R': ny, 'U': nu}

    for pol in policies:
        p = pol(data_dims, nsteps=10, input_keys=['x0', 'D', 'R'], output_keys=['U_pred'])
        print(p.name)
        p_out = p(data)
        print(p_out['U_pred'].shape, p_out[f'{p.name}_reg_error'].shape)
