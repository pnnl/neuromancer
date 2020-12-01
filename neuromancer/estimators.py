"""
# TODO: we can have sequence to sequence RNN block for seq2seq estimator

state estimators for SSM models
x: states (x0 - initial conditions)
u: control inputs
ym: measured outputs
d: uncontrolled inputs (measured disturbances)

generic mapping:
x = estim(ym,x0,u,d)
"""

# pytorch imports
import torch
import torch.nn as nn

# ecosystem imports
import slim

# local imports
import neuromancer.blocks as blocks
from neuromancer.dynamics import BlockSSM


def check_keys(k1, k2):
    """
    Check that all elements in k1 are contained in k2

    :param k1: iterable of str
    :param k2: iterable of str
    """
    assert set(k1) - set(k2) == set(), \
        f'Missing values in dataset. Input_keys: {set(k1)}, data_keys: {set(k2)}'


class TimeDelayEstimator(nn.Module):
    def __init__(self, data_dims, nsteps=1, window_size=1, input_keys=['Yp'], name='estimator'):
        """

        :param data_dims: dict {str: tuple of ints) Data structure describing dimensions of input variables
        :param nsteps: (int) Prediction horizon
        :param window_size: (int) Size of sequence history to use as input to the state estimator.
        :param input_keys: (List of str) List of input variable names
        :param name: (str) Name for tracking output of module.
        """
        super().__init__()
        assert window_size <= nsteps, f'Window size {window_size} longer than sequence length {nsteps}.'
        check_keys(set(input_keys), set(data_dims.keys()))
        self.name, self.data_dims = name, data_dims
        self.nsteps, self.window_size = nsteps, window_size
        self.nx = data_dims['x0'][-1]
        data_dims_in = {k: v for k, v in data_dims.items() if k in input_keys}
        self.sequence_dims_sum = sum(v[-1] for k, v in data_dims_in.items() if len(v) == 2)
        self.static_dims_sum = sum(v[-1] for k, v in data_dims_in.items() if len(v) == 1)
        self.in_features = self.static_dims_sum + window_size * self.sequence_dims_sum
        self.out_features = self.nx
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
                    f'Sequence too short for estimator calculation. Should be at least {self.nsteps}'
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
        return {f'x0_{self.name}': self.net(features), f'reg_error_{self.name}': self.reg_error()}


class seq2seqTimeDelayEstimator(TimeDelayEstimator):
    def __init__(self, data_dims, nsteps=1, window_size=1, input_keys=['Yp'], timedelay=0, name='estimator'):
        """

        :param data_dims: dict {str: tuple of ints) Data structure describing dimensions of input variables
        :param nsteps: (int) Prediction horizon
        :param window_size: (int) Size of sequence history to use as input to the state estimator.
        :param input_keys: (List of str) List of input variable names
        :param name: (str) Name for tracking output of module.
        """
        super().__init__(data_dims, nsteps=nsteps, window_size=window_size, input_keys=input_keys, name=name)
        self.nx = data_dims['x0'][-1]
        self.timedelay = timedelay
        self.nx_td = self.nx * (1+self.timedelay)
        self.out_features = self.nx_td

    def forward(self, data):
        """

        :param data: (dict {str: torch.tensor)}
        :return: (dict {str: torch.tensor)}
        """
        features = self.features(data)
        Xtd = self.net(features).view(self.timedelay+1, -1, self.nx)
        return {f'Xtd_{self.name}': Xtd, f'reg_error_{self.name}': self.reg_error()}


class FullyObservable(TimeDelayEstimator):
    def __init__(self, data_dims, nsteps=1, window_size=1, bias=False,
                 linear_map=slim.Linear, nonlin=nn.Identity, hsizes=[],
                 input_keys=['Yp'], linargs=dict(), name='fully_observable'):
        """
        Dummmy estimator to use consistent API for fully and partially observable systems
        """
        super().__init__(data_dims, nsteps=nsteps, window_size=window_size, input_keys=input_keys, name=name)
        self.net = nn.Identity()

    def features(self, data):
        return data['Yp'][self.nsteps-1]


class LinearEstimator(TimeDelayEstimator):
    def __init__(self, data_dims, nsteps=1, window_size=1, bias=False,
                 linear_map=slim.Linear, nonlin=nn.Identity, hsizes=[],
                 input_keys=['Yp'], linargs=dict(), name='linear_estim'):
        """

        See base class for arguments
        """
        super().__init__(data_dims, nsteps=nsteps, window_size=window_size, input_keys=input_keys, name=name)
        self.net = linear_map(self.in_features, self.out_features, bias=bias, **linargs)


class seq2seqLinearEstimator(seq2seqTimeDelayEstimator):
    def __init__(self, data_dims, nsteps=1, window_size=1, bias=False,
                 linear_map=slim.Linear, nonlin=nn.Identity, hsizes=[], timedelay=0,
                 input_keys=['Yp'], linargs=dict(), name='linear_estim'):
        """

        See base class for arguments
        """
        super().__init__(data_dims, nsteps=nsteps, window_size=window_size, input_keys=input_keys,
                         timedelay=timedelay, name=name)
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


class seq2seqMLPEstimator(seq2seqTimeDelayEstimator):
    """

    """
    def __init__(self, data_dims, nsteps=1, window_size=1, bias=False,
                 linear_map=slim.Linear, nonlin=nn.GELU, hsizes=[64], timedelay=0,
                 input_keys=['Yp'], linargs=dict(), name='MLP_estim'):
        """
        See base class for arguments
        """
        super().__init__(data_dims, nsteps=nsteps, window_size=window_size, input_keys=input_keys,
                         timedelay=timedelay, name=name)
        self.net = blocks.MLP(self.in_features, self.out_features, bias=bias,
                              linear_map=linear_map, nonlin=nonlin, hsizes=hsizes, linargs=linargs)


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


class seq2seqResMLPEstimator(seq2seqTimeDelayEstimator):
    """

    """
    def __init__(self, data_dims, nsteps=1, window_size=1, bias=False,
                 linear_map=slim.Linear, nonlin=nn.GELU, hsizes=[64], timedelay=0,
                 input_keys=['Yp'], linargs=dict(), name='ResMLP_estim'):
        """
        see base class for arguments
        """
        super().__init__(data_dims, nsteps=nsteps, window_size=window_size, input_keys=input_keys,
                         timedelay=timedelay, name=name)
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
        features = torch.cat([data[k][self.nsteps-self.window_size:self.nsteps] for k in self.input_keys], dim=2)
        return {f'x0_{self.name}': self.net(features), f'reg_error_{self.name}': self.net.reg_error()}


class seq2seqRNNEstimator(seq2seqTimeDelayEstimator):
    def __init__(self, data_dims, nsteps=1, window_size=1, bias=False,
                 linear_map=slim.Linear, nonlin=nn.GELU, hsizes=[64], timedelay=0,
                 input_keys=['Yp'], linargs=dict(), name='RNN_estim'):
        """
        see base class for arguments
        """
        super().__init__(data_dims, nsteps=nsteps, window_size=window_size, input_keys=input_keys,
                         timedelay=timedelay, name=name)
        self.in_features = self.sequence_dims_sum
        self.net = blocks.RNN(self.in_features, self.out_features, hsizes=hsizes,
                              bias=bias, nonlin=nonlin, linear_map=linear_map, linargs=linargs)

    def forward(self, data):
        features = torch.cat([data[k][self.nsteps-self.window_size:self.nsteps] for k in self.input_keys], dim=2)
        Xtd = self.net(features).view(self.timedelay+1, -1, self.nx)
        return {f'x0_{self.name}': Xtd, f'reg_error_{self.name}': self.net.reg_error()}


class LinearKalmanFilter(nn.Module):
    """
    Time-Varying Linear Kalman Filter
    """
    def __init__(self, model=None, name='kalman_estim'):
        """

        :param model: Dynamics model. Should be a block dynamics model with potential input non-linearity.
        :param name: Identifier for tracking output.
        """
        super().__init__()
        assert model is not None
        assert isinstance(model, BlockSSM)
        assert isinstance(model.fx, slim.LinearBase)
        assert isinstance(model.fy, slim.LinearBase)
        self.model = model
        self.name = name
        self.Q_init = nn.Parameter(torch.eye(model.nx), requires_grad=False)
        self.R_init = nn.Parameter(torch.eye(model.ny), requires_grad=False)
        self.P_init = nn.Parameter(torch.eye(model.nx), requires_grad=False)
        self.L_init = nn.Parameter(torch.zeros(model.nx, model.ny), requires_grad=False)
        self.x0_estim = nn.Parameter(torch.zeros(1, model.nx), requires_grad=False)

    def reg_error(self):
        return torch.tensor(0.0)

    def forward(self, data):
        x = self.x0_estim
        Q = self.Q_init
        R = self.R_init
        P = self.P_init
        L = self.L_init  # KF gain
        eye = torch.eye(self.model.nx).to(data['Yp'].device)

        # State estimation loop on past data
        Yp, U, D = data['Yp'], data['Up'], data['Dp']
        for ym, u, d in zip(Yp, U[:len(Yp)], D[:len(Yp)]):
            # PREDICT STEP:
            x = self.model.fx(x) + self.model.fu(u) + self.model.fd(d)
            y = self.model.fy(x)
            # estimation error covariance
            P = torch.mm(self.model.fx.effective_W(), torch.mm(P, self.model.fx.effective_W().T)) + Q
            # UPDATE STEP:
            x = x + torch.mm((ym - y), L.T)
            L_inverse_part = torch.inverse(R + torch.mm(self.model.fy.effective_W().T,
                                                        torch.mm(P, self.model.fy.effective_W())))
            L = torch.mm(torch.mm(P, self.model.fy.effective_W()), L_inverse_part)
            P = eye - torch.mm(L, torch.mm(self.model.fy.effective_W().T, P))
        return {f'x0_{self.name}': x, f'reg_error_{self.name}': self.reg_error()}


estimators = {'fullyObservable': FullyObservable,
              'linear': LinearEstimator,
              'mlp': MLPEstimator,
              'rnn': RNNEstimator,
              'residual_mlp': ResMLPEstimator}

seq2seq_estimators = {'seq2seq_linear': seq2seqLinearEstimator,
                      'seq2seq_mlp': seq2seqMLPEstimator,
                      'seq2seq_rnn': seq2seqRNNEstimator,
                      'seq2seq_residual_mlp': seq2seqResMLPEstimator}

if __name__ == '__main__':
    nx, ny, nu, nd = 15, 7, 5, 2
    N = 40

    samples = 1
    # Data format: (N,samples,dim)
    Y = torch.rand(N, samples, ny)
    U = torch.rand(N, samples, nu)
    D = torch.rand(N, samples, nd)
    data = {'Yp': Y, 'Up': U, 'Dp': D}
    data_dims = {'x0': (nx,), 'Yp': (N, ny), 'Up': (N, nu), 'Dp': (N, nd)}
    input_keys = ['Yp']

    for bias in [True, False]:
        for name, est in estimators.items():
            print(name)
            e = est(data_dims, input_keys=input_keys)
            e_out = e(data)
            for k, v in e_out.items():
                print(f'{k}: {v.shape}')
            for lin in set(slim.maps.values()) - slim.square_maps:
                print(lin)
                e = est(data_dims, input_keys=input_keys, bias=bias, linear_map=lin)
                e_out = e(data)
                for k, v in e_out.items():
                    print(f'{k}: {v.shape}')

    for bias in [True, False]:
        for name, est in estimators.items():
            print(name)
            e = est(data_dims, nsteps=N, window_size=N, input_keys=input_keys)
            e_out = e(data)
            for k, v in e_out.items():
                print(f'{k}: {v.shape}')
            for lin in set(slim.maps.values()) - slim.square_maps:
                print(lin)
                e = est(data_dims, nsteps=N, window_size=N, input_keys=input_keys, bias=bias, linear_map=lin)
                e_out = e(data)
                for k, v in e_out.items():
                    print(f'{k}: {v.shape}')

    for bias in [True, False]:
        for name, est in estimators.items():
            print(name)
            e = est(data_dims, nsteps=N, window_size=N-1, input_keys=input_keys)
            e_out = e(data)
            for k, v in e_out.items():
                print(f'{k}: {v.shape}')
            for lin in set(slim.maps.values()) - slim.square_maps:
                print(lin)
                e = est(data_dims, nsteps=N, window_size=N-1, input_keys=input_keys, bias=bias, linear_map=lin)
                e_out = e(data)
                for k, v in e_out.items():
                    print(f'{k}: {v.shape}')


    for bias in [True, False]:
        for name, est in seq2seq_estimators.items():
            print(name)
            e = est(data_dims, nsteps=N, window_size=N, timedelay=N-1, input_keys=input_keys)
            e_out = e(data)
            for k, v in e_out.items():
                print(f'{k}: {v.shape}')
            for lin in set(slim.maps.values()) - slim.square_maps:
                print(lin)
                e = est(data_dims, nsteps=N, window_size=N, timedelay=N-1, input_keys=input_keys, bias=bias, linear_map=lin)
                e_out = e(data)
                for k, v in e_out.items():
                    print(f'{k}: {v.shape}')

    for bias in [True, False]:
        for name, est in seq2seq_estimators.items():
            print(name)
            e = est(data_dims, nsteps=N, window_size=N-1, timedelay=0, input_keys=input_keys)
            e_out = e(data)
            for k, v in e_out.items():
                print(f'{k}: {v.shape}')
            for lin in set(slim.maps.values()) - slim.square_maps:
                print(lin)
                e = est(data_dims, nsteps=N, window_size=N-1, timedelay=0, input_keys=input_keys, bias=bias, linear_map=lin)
                e_out = e(data)
                for k, v in e_out.items():
                    print(f'{k}: {v.shape}')


    print('Kalman filter')
    fx, fu, fd = [slim.Linear(insize, nx) for insize in [nx, nu, nd]]
    fy = slim.Linear(nx, ny)
    model = BlockSSM(fx, fy, fu, fd)
    est = LinearKalmanFilter(model=model)
    est_out = est(data)
    for k, v in est_out.items():
        print(f'{k}: {v.shape}')






