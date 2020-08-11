"""

policies for SSM models
x: states
u: control inputs
d: uncontrolled inputs (measured disturbances)
r: reference signals

generic mapping:
u = policy(x,u,d,r)
"""

# machine learning/data science imports
import torch
import torch.nn as nn

# ecosystem imports
import slim

# local imports
import neuromancer.blocks as blocks


def check_keys(k1, k2):
    assert set(k1) - set(k2) == set(), \
        f'Missing values in dataset. Input_keys: {set(k1)}, data_keys: {set(k2)}'


class SolutionMap(nn.Module):
    def __init__(self, data_dims, input_keys=['x'], bias=False,
                 Linear=slim.Linear, nonlin=nn.GELU, hsizes=[64], linargs=dict(), name='sol_map'):
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
        data_dims_in = {k: v for k, v in data_dims.items() if k in input_keys}
        self.input_size = sum(v[-1] for k, v in data_dims_in.items())
        self.output_size = data_dims['z'][-1]
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
        return {f'z_{self.name}': Z, f'reg_error_{self.name}': self.net.reg_error()}


# class Policy(nn.Module):
#     def __init__(self, data_dims, nsteps=1, input_keys=['x0'], name='policy'):
#         """
#
#         :param data_dims:
#         :param nsteps:
#         :param input_keys:
#         :param name:
#         """
#         super().__init__()
#         check_keys(set(input_keys), set(data_dims.keys()))
#         self.name = name
#         self.nsteps = nsteps
#         self.nu = data_dims['U'][-1]
#         data_dims_in = {k: v for k, v in data_dims.items() if k in input_keys}
#         self.sequence_dims_sum = sum(v[-1] for k, v in data_dims_in.items() if len(v) == 2)
#         self.static_dims_sum = sum(v[-1] for k, v in data_dims_in.items() if len(v) == 1)
#         self.input_size = self.static_dims_sum + nsteps * self.sequence_dims_sum
#         self.output_size = nsteps * self.nu
#         self.input_keys = input_keys
#
#     def reg_error(self):
#         """
#
#         :return:
#         """
#         return self.net.reg_error()
#
#     def forward(self, data):
#         """
#
#         :param data:
#         :return:
#         """
#         check_keys(self.input_keys, set(data.keys()))
#         featlist = []
#         for k in self.input_keys:
#             if len(data[k].shape) == 2:
#                 featlist.append(data[k])
#             elif len(data[k].shape) == 3:
#                 featlist.append(torch.cat([step for step in data[k]], dim=1))
#         features = torch.cat(featlist, dim=1)
#         print('features', features.shape)
#         print('input_size', self.input_size)
#         Uf = self.net(features)
#         Uf = torch.cat([u.reshape(self.nsteps, 1, -1) for u in Uf], dim=1)
#         return {f'U_pred_{self.name}': Uf, f'reg_error_{self.name}': self.net.reg_error()}
#

class Policy(nn.Module):
    def __init__(self, data_dims, nsteps=1, window_size=1, input_keys=['x0'], name='policy'):
        """

        :param data_dims: dict {str: tuple of ints) Data structure describing dimensions of input variables
        :param nsteps: (int) Prediction horizon
        :param input_keys: (List of str) List of input variable names
        :param name: (str) Name for tracking output of module.
        """
        super().__init__()
        check_keys(set(input_keys), set(data_dims.keys()))
        self.name, self.data_dims = name, data_dims
        self.nsteps, self.window_size = nsteps, window_size
        self.data_dims = data_dims
        self.nu = data_dims['U'][-1]
        data_dims_in = {k: v for k, v in data_dims.items() if k in input_keys}
        self.sequence_dims_sum = sum(v[-1] for k, v in data_dims_in.items() if len(v) == 2)
        self.static_dims_sum = sum(v[-1] for k, v in data_dims_in.items() if len(v) == 1)
        self.input_size = self.static_dims_sum + window_size * self.sequence_dims_sum
        self.output_size = nsteps * self.nu
        self.input_keys = input_keys

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
        check_keys(self.input_keys, set(data.keys()))
        featlist = []
        for k in self.input_keys:
            assert self.data_dims[k][-1] == data[k].shape[-1], \
                f'Input feature {k} expected {self.data_dims[k][-1]} but got {data[k].shape[-1]}'
            if len(data[k].shape) == 2:
                featlist.append(data[k])
            elif len(data[k].shape) == 3:
                assert len(data[k]) >= self.nsteps, \
                    f'Sequence too short for policy calculation. Should be at least {self.nsteps}'
                featlist.append(
                    torch.cat([step for step in data[k][self.nsteps - self.window_size:self.nsteps]], dim=1))
            else:
                raise ValueError(f'Input {k} has {len(data[k].shape)} dimensions. Should have 2 or 3 dimensions')
        return torch.cat(featlist, dim=1)

    def forward(self, data):
        """

        :param data: (dict {str: torch.tensor)}
        :return: (dict {str: torch.tensor)}
        """
        features = self.features(data)
        Uf = self.net(features)
        Uf = torch.cat([u.reshape(self.nsteps, 1, -1) for u in Uf], dim=1)
        return {f'U_pred_{self.name}': Uf, f'reg_error_{self.name}': self.reg_error()}


class LinearPolicy(Policy):
    def __init__(self, data_dims, nsteps=1, bias=False,
                 Linear=slim.Linear, nonlin=None, hsizes=None,
                 input_keys=['x0'], linargs=dict(), name='linear_policy'):
        """
        See base class for arguments
        """
        super().__init__(data_dims, nsteps=nsteps, input_keys=input_keys, name=name)
        self.net = Linear(self.input_size, self.output_size, bias=bias, **linargs)


class MLPPolicy(Policy):
    def __init__(self, data_dims, nsteps=1, bias=False,
                 Linear=slim.Linear, nonlin=nn.GELU, hsizes=[64],
                 input_keys=['x0'], linargs=dict(), name='MLP_policy'):
        """

        See base class for arguments
        """
        super().__init__(data_dims, nsteps=nsteps, input_keys=input_keys, name=name)
        self.net = blocks.MLP(insize=self.input_size, outsize=self.output_size, bias=bias,
                              Linear=Linear, nonlin=nonlin, hsizes=hsizes, linargs=linargs)


class RNNPolicy(Policy):
    def __init__(self, data_dims, nsteps=1, bias=False,
                 Linear=slim.Linear, nonlin=nn.GELU, hsizes=[64],
                 input_keys=['x0'], linargs=dict(), name='RNN_policy'):
        """
        See arguments for policy
        """
        super().__init__(data_dims, nsteps=nsteps, input_keys=input_keys, name=name)
        self.input_size = self.sequence_dims_sum + self.static_dims_sum
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
        return {f'U_pred_{self.name}': Uf, f'reg_error_{self.name}': self.net.reg_error()}


policies = [LinearPolicy, MLPPolicy, RNNPolicy]


if __name__ == '__main__':
    nx, ny, nu, nd = 15, 7, 5, 3
    N = 10
    samples = 100
    # Data format: (N,samples,dim)
    x = torch.rand(samples, nx)
    D = torch.rand(N, samples, nd)
    R = torch.rand(N, samples, ny)
    data = {'x0': x, 'D': D, 'R': R}
    data_dims = {'x0': (nx,), 'D': (N, nd,), 'R': (N, ny), 'U': (N, nu)}

    for pol in policies:
        p = pol(data_dims, nsteps=10, input_keys=['x0', 'D', 'R'], name='policy')
        print(p.name)
        p_out = p(data)
        print(p_out[f'U_pred_{p.name}'].shape, p_out[f'reg_error_{p.name}'].shape)
