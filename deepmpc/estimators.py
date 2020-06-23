"""
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
import torch.nn.functional as F
#local imports
import linear
from blocks import MLP
from rnn import RNN, RNNCell
from ssm import BlockSSM


class FullyObservable(nn.Module):
    def __init__(self, *args, **linargs):
        super().__init__()

    def reg_error(self):
        return torch.zeros(1)

    def forward(self, Ym, *args):
        return Ym[-1], self.reg_error()


class LinearEstimator(nn.Module):
    def __init__(self, insize, outsize, bias=False, Linear=linear.Linear, **linargs):
        super().__init__()
        self.linear = Linear(insize, outsize, bias=bias, **linargs)

    def reg_error(self):
        return self.linear.reg_error()

    def forward(self, Ym, *args):
        return self.linear(Ym[-1]), self.reg_error()


class MLPEstimator(nn.Module):
    def __init__(self, insize, outsize, bias=True,
                 Linear=linear.Linear, nonlin=F.relu, hsizes=[64], **linargs):
        super().__init__()
        self.net = MLP(insize, outsize, bias=bias,
                       Linear=Linear, nonlin=nonlin, hsizes=hsizes, **linargs)

    def reg_error(self):
        return self.net.reg_error()

    def forward(self, Ym, *args):
        return self.net(Ym[-1]), self.reg_error()


class RNNEstimator(nn.Module):
    def __init__(self, input_size, hidden_size, bias=False, num_layers=1,
                 nonlin=F.gelu, Linear=linear.Linear, **linargs):
        super().__init__()
        self.RNN = RNN(input_size, hidden_size, num_layers=num_layers,
                       bias=bias, nonlin=nonlin, Linear=Linear, **linargs)

    def reg_error(self):
        return self.RNN.reg_error()

    def forward(self, Ym, *args):
        return self.RNN(Ym)[0][-1].to(Ym.device), self.reg_error().to(Ym.device)


class LinearKalmanFilter(nn.Module):
    """
    Linear Time-Varyig Linear Kalman Filter
    """
    def __init__(self, insize, outsize, model=None):
        super().__init__()
        assert model is not None
        assert isinstance(model, BlockSSM)
        assert isinstance(model.fx, linear.LinearBase)
        assert isinstance(model.fy, linear.LinearBase)
        self.model = model
        self.Q_init = nn.Parameter(torch.eye(model.nx), requires_grad=False)
        self.R_init = nn.Parameter(torch.eye(model.ny), requires_grad=False)
        self.P_init = nn.Parameter(torch.eye(model.nx), requires_grad=False)
        self.L_init = nn.Parameter(torch.zeros(model.nx, model.ny), requires_grad=False)
        self.x0_estim = nn.Parameter(torch.zeros(1, model.nx), requires_grad=False)

    def reg_error(self):
        return torch.zeros(1)

    def forward(self, Ym, U, D):
        x = self.x0_estim
        Q = self.Q_init
        R = self.R_init
        P = self.P_init
        L = self.L_init  # KF gain
        eye = torch.eye(self.model.nx).to(Ym.device)

        # State estimation loop on past data
        for ym, u, d in zip(Ym, U[:len(Ym)], D[:len(Ym)]):
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

        return x, self.reg_error()


class ExtendedKalmanFilter(nn.Module):
    """
    Extended = Kalman Filter
    TODO: Implement extended Kalman filter
    """
    pass

estimators = [FullyObservable, LinearEstimator, MLPEstimator, RNNEstimator]

if __name__ == '__main__':
    nx, ny, nu, nd = 15, 7, 5, 3
    N = 40
    samples = 100
    # Data format: (N,samples,dim)
    Y = torch.rand(N, samples, ny)
    U = torch.rand(N, samples, nu)
    D = torch.rand(N, samples, nd)

    for bias in [True, False]:
        for est in estimators:
            e = est(ny, 15)
            e_out = e(Y, U, D)
            print(e_out[0].shape, e_out[1].shape)
            e = est(ny, 3)
            e_out = e(Y, U, D)
            print(e_out[0].shape, e_out[1].shape)
            for lin in set(linear.maps.values()) - linear.square_maps:
                e = est(ny, 15, bias=bias, Linear=lin)
                e_out = e(Y, U, D)
                print(e_out[0].shape, e_out[1].shape)
                e = est(ny, 3, bias=bias, Linear=lin)
                e_out = e(Y, U, D)
                print(e_out[0].shape, e_out[1].shape)

    fx, fu, fd = [linear.Linear(insize, nx) for insize in [nx, nu, nd]]
    fy = linear.Linear(nx, ny)
    model = BlockSSM(nx, nu, nd, ny, fx, fy, fu, fd)
    est = LinearKalmanFilter(nx, ny, model=model)
    est_out = est(Y, U, D)
    print(est_out[0].shape, est_out[1].shape)







