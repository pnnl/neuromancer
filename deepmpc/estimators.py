"""
# TODO: Better high level comments
state estimators for SSM models
x: states (x0 - initial conditions)
u: control inputs
ym: measured outputs
d: uncontrolled inputs (measured disturbances)

generic mapping:
x = estim(ym,x0,u,d)
"""
from typing import Dict
# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
#local imports
import linear
import blocks
from dynamics import BlockSSM


def check_keys(k1, k2):
    assert k1 - k2 == set(), \
        f'Missing values in dataset. Input_keys: {k1}, data_keys: {k2}'


class Estimator(nn.Module):
    def __init__(self, data_dims, nsteps=1, input_keys={'Yp'}, name='policy'):
        super().__init__()
        check_keys(set(input_keys), set(data_dims.keys()))
        self.name = name
        self.nsteps = nsteps
        self.nx = data_dims['X']

        data_dims = {k: v for k, v in data_dims.items() if k in input_keys}
        self.sequence_dims_sum = sum(v for k, v in data_dims.items())

        self.input_size = self.sequence_dims_sum
        self.output_size = self.nx
        self.input_keys = set(input_keys)

    def reg_error(self):
        return torch.tensor(0.0)

    def forward(self, data):
        """

        :param data:
        :return:
        """
        check_keys(self.input_keys, set(data.keys()))
        features = torch.cat([data[k][-1] for k in self.input_keys], dim=1)
        return {'x0': self.net(features), f'{self.name}_reg_error': self.net.reg_error()}


class FullyObservable(nn.Module):
    def __init__(self, *args, name='full_observable', **linargs):
        """

        :param name:
        :param args:
        :param linargs:
        """
        super().__init__()
        self.name = name

    def reg_error(self):
        return torch.tensor(0.0)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """

        :param data:
        :return:
        """
        return {'x0': data['Yp'][-1], f'{self.name}_reg_error': self.reg_error()}


class LinearEstimator(Estimator):
    def __init__(self, data_dims, nsteps=1, bias=False,
                 Linear=linear.Linear, nonlin=F.gelu, hsizes=[64], input_keys={'Yp'},
                 linargs=dict(), name='linear_estim'):
        """

        :param data_dims:
        :param nsteps:
        :param bias:
        :param Linear:
        :param nonlin:
        :param hsizes:
        :param input_keys:
        :param linargs:
        :param name:
        """
        super().__init__(data_dims, nsteps=nsteps, input_keys=input_keys, name=name)
        self.net = Linear(self.input_size, self.output_size, bias=bias, **linargs)

    def reg_error(self):
        return self.linear.reg_error()


class MLPEstimator(Estimator):
    """

    """
    def __init__(self, data_dims, nsteps=1, bias=False,
                 Linear=linear.Linear, nonlin=F.gelu, hsizes=[64], input_keys={'Yp'},
                 linargs=dict(), name='MLP_estim'):
        """

        :param data_dims:
        :param nsteps:
        :param bias:
        :param Linear:
        :param nonlin:
        :param hsizes:
        :param input_keys:
        :param linargs:
        :param name:
        """
        super().__init__(data_dims, nsteps=nsteps, input_keys=input_keys, name=name)
        self.net = blocks.MLP(self.input_size, self.output_size, bias=bias,
                              Linear=Linear, nonlin=nonlin, hsizes=hsizes, linargs=linargs)

    def reg_error(self):
        return self.net.reg_error()


class RNNEstimator(Estimator):
    def __init__(self, data_dims, nsteps=1, bias=False,
                 Linear=linear.Linear, nonlin=F.gelu, hsizes=[64], input_keys={'Yp'},
                 linargs=dict(), name='RNN_estim'):
        """

        :param data_dims:
        :param nsteps:
        :param bias:
        :param Linear:
        :param nonlin:
        :param hsizes:
        :param input_keys:
        :param linargs:
        :param name:
        """
        super().__init__(data_dims, nsteps=nsteps, input_keys=input_keys, name=name)
        self.input_size = self.sequence_dims_sum
        self.net = blocks.RNN(self.input_size, self.output_size, hsizes=hsizes,
                              bias=bias, nonlin=nonlin, Linear=Linear, linargs=linargs)

    def reg_error(self):
        return self.net.reg_error()

    def forward(self, data):
        features = torch.cat([data[k] for k in self.input_keys], dim=2)
        return {'x0': self.net(features), f'{self.name}_reg_error': self.reg_error()}


class LinearKalmanFilter(nn.Module):
    """
    Time-Varying Linear Kalman Filter
    """
    def __init__(self, insize=None, outsize=None, model=None, name='kalman_estim'):
        """

        :param insize: Dummy variable for consistent API
        :param outsize: Dummy variable for consistent API
        :param model: Dynamics model
        """
        super().__init__()
        assert model is not None
        assert isinstance(model, BlockSSM)
        assert isinstance(model.fx, linear.LinearBase)
        assert isinstance(model.fy, linear.LinearBase)
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

        return {'x0': x, f'{self.name}_reg_error': self.reg_error()}


class ExtendedKalmanFilter(nn.Module):
    """
    Extended = Kalman Filter
    TODO: Implement extended Kalman filter
    """
    pass


estimators = [FullyObservable, LinearEstimator, MLPEstimator, RNNEstimator]

if __name__ == '__main__':
    nx, ny, nu, nd = 15, 7, 5, 2
    N = 40
    samples = 1
    # Data format: (N,samples,dim)
    Y = torch.rand(N, samples, ny)
    U = torch.rand(N, samples, nu)
    D = torch.rand(N, samples, nd)
    data = {'Yp': Y, 'Up': U, 'Dp': D}
    data_dims = {'X': nx, 'Yp': ny, 'Up': nu, 'Dp': nd}
    input_keys = {'Yp'}
    for bias in [True, False]:
        for est in estimators:
            e = est(data_dims, nsteps=N, input_keys=input_keys)
            e_out = e(data)
            for k, v in e_out.items():
                print(f'{k}: {v.shape}')
            for lin in set(linear.maps.values()) - linear.square_maps:
                print(lin)
                e = est(data_dims, nsteps=N, input_keys=input_keys, bias=bias, Linear=lin)
                e_out = e(data)
                for k, v in e_out.items():
                    print(f'{k}: {v.shape}')

    print('Kalman filter')
    fx, fu, fd = [linear.Linear(insize, nx) for insize in [nx, nu, nd]]
    fy = linear.Linear(nx, ny)
    model = BlockSSM(nx, nu, nd, ny, fx, fy, fu, fd)
    est = LinearKalmanFilter(nx, ny, model=model)
    est_out = est(data)
    for k, v in est_out.items():
        print(f'{k}: {v.shape}')






