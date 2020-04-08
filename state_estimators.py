import torch
import torch.nn as nn
import torch.nn.functional as F

from linear import PerronFrobeniusLinear
from rnn import RNN, RNNCell


class LinearEstimator(nn.Module):
    def __init__(self, insize, outsize, bias=False):
        super().__init__()
        self.linear = nn.Linear(insize, outsize, bias=bias)

    @property
    def regularization_error(self):
        return 0.0

    def forward(self, Ym, M_flow, DT, D):
        return self.linear(Ym[-1])


class PerronFrobeniusEstimator(nn.Module):
    def __init__(self, insize, outsize, bias=False):
        super().__init__()
        self.linear = PerronFrobeniusLinear(insize, outsize, bias=bias)

    @property
    def regularization_error(self):
        return 0.0

    def forward(self, Ym, M_flow, DT, D):
        return self.linear(Ym[-1])


class MLPEstimator(nn.Module):
    def __init__(self, insize, outsize, hiddensize, bias=False, nonlinearity=F.gelu):
        super().__init__()
        self.layer1 = PerronFrobeniusLinear(insize, hiddensize, bias=bias)
        self.layer2 = PerronFrobeniusLinear(hiddensize, outsize, bias=bias)
        self.nlin = nonlinearity

    @property
    def regularization_error(self):
        return 0.0

    def forward(self, Ym, M_flow, DT, D):
        return self.nlin(self.layer2(self.nlin(self.layer1(Ym[-1]))))


class RNNEstimator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1,
                       bias=False, nonlinearity=F.gelu, cell=RNNCell):
        super().__init__()
        self.RNN = RNN(input_size, hidden_size, num_layers=num_layers,
                       bias=bias, nonlinearity=nonlinearity, cell=cell)

    @property
    def regularization_error(self):
        return self.RNN.spectral_error

    def forward(self, Ym, M_flow, DT, D):
        return self.RNN(Ym)[0][-1]


class KalmanFilterEstimator(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.Q_init = nn.Parameter(torch.eye(model.nx), requires_grad=False)
        self.R_init = nn.Parameter(torch.eye(model.ny), requires_grad=False)
        self.P_init = nn.Parameter(torch.eye(model.nx), requires_grad=False)
        self.L_init = nn.Parameter(torch.zeros(model.nx, model.ny), requires_grad=False)
        self.x0_estim = nn.Parameter(torch.zeros(1, model.nx), requires_grad=False)

    @property
    def regularization_error(self):
        return 0.0

    def forward(self, Ym, M_flow, DT, D):
        x = self.x0_estim
        Q = self.Q_init
        R = self.R_init
        P = self.P_init
        L = self.L_init  # KF gain
        eye = torch.eye(self.model.nx).to(Ym.device)

        # State estimation loop on past data
        for ym, m_flow, dT, d in zip(Ym, M_flow, DT, D):
            u = self.model.heat_flow(m_flow, dT)
            x = self.model.A(x) + self.model.B(u) + self.model.E(d)
            y = self.model.C(x)
            # estimation error covariance
            P = torch.mm(self.model.A.effective_W(), torch.mm(P, self.model.A.effective_W().T)) + Q
            # UPDATE STEP:
            x = x + torch.mm((ym - y), L.T)
            L_inverse_part = torch.inverse(R + torch.mm(self.model.C.effective_W().T, torch.mm(P, self.model.C.effective_W())))
            L = torch.mm(torch.mm(P, self.model.C.effective_W()), L_inverse_part)
            P = eye - torch.mm(L, torch.mm(self.model.C.effective_W().T, P))

        return x





