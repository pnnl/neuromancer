"""
state estimators for SSM model of the building dynamics
TODO: generalize Kalman Filter
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


class LinearEstimator(nn.Module):
    def __init__(self, insize, outsize, bias=False, Linear=linear.Linear, **linargs):
        super().__init__()
        self.linear = Linear(insize, outsize, bias=bias, **linargs)

    def reg_error(self):
        return self.linear.reg_error()

    def forward(self, Ym, *args):
        return self.linear(Ym[-1])


class MLPEstimator(nn.Module):
    def __init__(self, insize, outsize, bias=True,
                 Linear=linear.Linear, nonlin=F.relu, hsizes=[64], **linargs):
        super().__init__()
        self.net = MLP(insize, outsize, bias=bias,
                       Linear=Linear, nonlin=nonlin, hsizes=hsizes, **linargs)

    def reg_error(self):
        return self.net.reg_error()

    def forward(self, Ym, *args):
        return self.net(Ym[-1])


class RNNEstimator(nn.Module):
    def __init__(self, input_size, hidden_size, bias=False, num_layers=1,
                 nonlinearity=F.gelu, Linear=linear.Linear, **linargs):
        super().__init__()
        self.RNN = RNN(input_size, hidden_size, num_layers=num_layers,
                       bias=bias, nonlinearity=nonlinearity, Linear=Linear, **linargs)

    def reg_error(self):
        return self.RNN.reg_error()

    def forward(self, Ym, *args):
        return self.RNN(Ym)[0][-1]


class KalmanFilterEstimator(nn.Module):

    def __init__(self, model):
        super().__init__()
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
        return 0.0

    def forward(self, Ym, U, D):
        x = self.x0_estim
        Q = self.Q_init
        R = self.R_init
        P = self.P_init
        L = self.L_init  # KF gain
        eye = torch.eye(self.model.nx).to(Ym.device)

        # State estimation loop on past data
        for ym, u, d in zip(Ym, U, D):
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

        return x


estimators = [LinearEstimator, MLPEstimator, RNNEstimator]

if __name__ == '__main__':
    nx, ny, nu, nd = 15, 7, 5, 3
    Y = torch.rand(100, 40, ny)
    U = torch.rand(100, 40, nu)
    D = torch.rand(100, 40, nd)

    for est in estimators:
        for lin in set(linear.maps) - linear.square_maps:
            e = est(ny, 15)
            print(e(Y, U, D).shape)
            e = est(ny, 3)
            print(e(Y, U, D).shape)

    fx, fu, fd = [linear.Linear(insize, nx) for insize in [nx, nu, nd]]
    fy = linear.Linear(nx, ny)
    ssm = BlockSSM(nx, nu, nd, ny, fx, fu, fd, fy)
    est = KalmanFilterEstimator(ssm)
    print(est(Y, U, D).shape)








