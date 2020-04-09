# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
# local imports
from linear import Linear, SVDLinear, PerronFrobeniusLinear, NonnegativeLinear, SpectralLinear
from data import BuildingDAE


def heat_flow(m_flow, dT):
    U = 1.1591509722222224 * m_flow * dT
    return U


class HeatFlow(nn.Module):

    def __init__(self):
        super().__init__()
        self.rho = torch.nn.Parameter(torch.tensor(0.997), requires_grad=False)  # density  of water kg/1l
        self.cp = torch.nn.Parameter(torch.tensor(4185.5),
                                     requires_grad=False)  # specific heat capacity of water J/(kg/K)
        self.time_reg = torch.nn.Parameter(torch.tensor(1 / 3600), requires_grad=False)

    def forward(self, m_flow, dT):
        return m_flow * self.rho * self.cp * self.time_reg * dT


class MLPHeatFlow(nn.Module):
    def __init__(self, insize, outsize, hiddensize, bias=False, nonlinearity=F.gelu):
        super().__init__()
        self.layer1 = Linear(insize, hiddensize, bias=bias)
        self.layer2 = Linear(hiddensize, outsize, bias=bias)
        self.nlin = nonlinearity

    def __call__(self, m_flow, dT):
        return self.nlin(self.layer2(self.nlin(self.layer1((torch.cat([m_flow, dT], dim=1))))))


class SSM(nn.Module):
    def __init__(self, nx, ny, n_m, n_dT, nu, nd, n_hidden, bias=False, heatflow='white',
                 xmin=0, xmax=35, umin=-5000, umax=5000,
                 Q_dx=1e2, Q_dx_ud=1e5, Q_con_x=1e1, Q_con_u=1e1, Q_spectral=1e2, rom=True):
        super().__init__()
        self.nx, self.ny, self.nu, self.nd, self.n_hidden = nx, ny, nu, nd, n_hidden
        self.A = Linear(nx, nx, bias=bias)
        self.B = Linear(nu, nx, bias=bias)
        self.E = Linear(nd, nx, bias=bias)
        self.C = Linear(nx, ny, bias=bias)
        self.x0_correct = torch.nn.Parameter(torch.zeros(1, nx)) # state initialization term, corresponding to G

        if heatflow == 'white':
            self.heat_flow = heat_flow
        elif heatflow == 'grey':
            self.heat_flow = HeatFlow()
        elif heatflow == 'black':
            self.heat_flow = MLPHeatFlow(n_m + n_dT, nu, n_hidden, bias=bias)
        else:
            raise ValueError('heatflow argument must be black, grey, or white')
        #  Regularization Initialization
        dxmax_val = 0.5
        dxmin_val = -0.5
        self.xmin, self.xmax, self.umin, self.umax = xmin, xmax, umin, umax
        self.dxmax = nn.Parameter(dxmax_val * torch.ones(1, nx), requires_grad=False)
        self.dxmin = nn.Parameter(dxmin_val * torch.ones(1, nx), requires_grad=False)
        self.sxmin, self.sxmax, self.sumin, self.sumax, self.sdx_x, self.dx_u, self.dx_d, self.spectral_error = [[] for i in range(8)]
        # weights on one-step difference of states
        self.Q_dx = Q_dx / n_hidden  # penalty on smoothening term dx
        self.Q_dx_ud = Q_dx_ud / n_hidden  # penalty on constrained maximal influence of u and d on x
        # state and input constraints weight
        self.Q_con_x = Q_con_x / n_hidden
        self.Q_con_u = Q_con_u / nu
        # (For SVD) Weights on orthogonality violation of matrix decomposition factors
        self.Q_spectral = Q_spectral

    def regularize(self, x_0, x, u, d):
        # Barrier penalties
        self.sxmin.append(F.relu(-x + self.xmin))
        self.sxmax.append(F.relu(x - self.xmax))
        self.sumin.append(F.relu(-u + self.umin))
        self.sumax.append(F.relu(u - self.umax))
        # one step state residual penalty
        self.sdx_x.append(x - x_0)
        # penalties on max one-step infuence of controls and disturbances on states
        self.dx_u.append(F.relu(-self.B(u) + self.dxmin) + F.relu(self.B(u) - self.dxmax))
        self.dx_d.append(F.relu(-self.E(d) + self.dxmin) + F.relu(self.E(d) - self.dxmax))

    @property
    def regularization_error(self):
        if self.training:
            sxmin, sxmax, sumin, sumax = (torch.stack(self.sxmin), torch.stack(self.sxmax),
                                          torch.stack(self.sumin), torch.stack(self.sumax))
            xmin_loss = self.Q_con_x*F.mse_loss(sxmin,  self.xmin * torch.ones(sxmin.shape).to(sxmin.device))
            xmax_loss = self.Q_con_x*F.mse_loss(sxmax,  self.xmax * torch.ones(sxmax.shape).to(sxmax.device))
            umin_loss = self.Q_con_u*F.mse_loss(sumin, self.umin * torch.ones(sumin.shape).to(sumin.device))
            umax_loss = self.Q_con_u*F.mse_loss(sumax, self.umax * torch.ones(sumax.shape).to(sumax.device))
            sdx, dx_u, dx_d = torch.stack(self.sdx_x), torch.stack(self.dx_u), torch.stack(self.dx_d)
            sdx_loss = self.Q_dx*F.mse_loss(sdx, torch.zeros(sdx.shape).to(sdx.device))
            dx_u_loss = self.Q_dx_ud*F.mse_loss(dx_u, torch.zeros(dx_u.shape).to(dx_u.device))
            dx_d_loss = self.Q_dx_ud*F.mse_loss(dx_d, torch.zeros(dx_d.shape).to(dx_u.device))
            self.sxmin, self.sxmax, self.sumin, self.sumax, self.sdx_x, self.dx_u, self.dx_d, self.spectral_error = [[] for i in range(8)]
            return torch.sum(torch.stack([xmin_loss, xmax_loss, umin_loss, umax_loss, sdx_loss, dx_u_loss, dx_d_loss]))
        else:
            return 0

    def forward(self, x, M_flow, DT, D):
        """
        """
        X, Y, U = [], [], []
        x = x + self.x0_correct
        for m_flow, dT, d in zip(M_flow, DT, D):
            x_prev = x  # previous state memory
            u = self.heat_flow(m_flow, dT)
            x = self.A(x) + self.B(u) + self.E(d)
            y = self.C(x)
            X.append(x)
            Y.append(y)
            U.append(u)
            if self.training:
                self.regularize(x_prev, x, u, d)
        return torch.stack(X), torch.stack(Y), torch.stack(U), self.regularization_error


class SVDSSM(SSM):
    def __init__(self, nx, ny, n_m, n_dT, nu, nd, n_hidden, bias=False, heatflow='white',
                 xmin=0, xmax=35, umin=-5000, umax=5000,
                 Q_dx=1e2, Q_dx_ud=1e5, Q_con_x=1e1, Q_con_u=1e1, Q_spectral=1e2, rom=True):

        super().__init__(nx, ny, n_m, n_dT, nu, nd, n_hidden, bias=bias, heatflow=heatflow,
                         xmin=xmin, xmax=xmax, umin=umin, umax=umax,
                         Q_dx=Q_dx, Q_dx_ud=Q_dx_ud, Q_con_x=Q_con_x, Q_con_u=Q_con_u, Q_spectral=Q_spectral, rom=rom)
        # Module initialization
        self.A = SVDLinear(nx, nx, bias=bias, sigma_min=0.6, sigma_max=1.0)
        self.E = PerronFrobeniusLinear(nd, nx, bias=bias, sigma_min=0.05, sigma_max=1)
        self.B = NonnegativeLinear(nu, nx, bias=bias)
        self.C = SVDLinear(nx, ny, bias=bias, sigma_min=0.9, sigma_max=1)

    def forward(self, x, M_flow, DT, D):
        """
        """
        X, Y, U, regularization_error = super().forward(x, M_flow, DT, D)
        return X, Y, U, regularization_error + self.A.spectral_error + self.C.spectral_error


class PerronFrobeniusSSM(SSM):
    def __init__(self, nx, ny, n_m, n_dT, nu, nd, n_hidden, bias=False, heatflow='white',
                 xmin=0, xmax=35, umin=-5000, umax=5000,
                 Q_dx=1e2, Q_dx_ud=1e5, Q_con_x=1e1, Q_con_u=1e1, Q_spectral=1e2, rom=True):
        super().__init__(nx, ny, n_m, n_dT, nu, nd, n_hidden, bias=bias, heatflow=heatflow,
                         xmin=xmin, xmax=xmax, umin=umin, umax=umax,
                         Q_dx=Q_dx, Q_dx_ud=Q_dx_ud, Q_con_x=Q_con_x, Q_con_u=Q_con_u, Q_spectral=Q_spectral, rom=rom)
        # Module initialization
        self.A = PerronFrobeniusLinear(nx, nx, bias=bias, sigma_min=0.95, sigma_max=1.0)
        self.E = PerronFrobeniusLinear(nd, nx, bias=bias, sigma_min=0.05, sigma_max=1)
        self.B = NonnegativeLinear(nu, nx, bias=bias)
        self.C = PerronFrobeniusLinear(nx, ny, bias=bias, sigma_min=0.9, sigma_max=1)


class SpectralSSM(PerronFrobeniusSSM):
    def __init__(self, nx, ny, n_m, n_dT, nu, nd, n_hidden, bias=False, heatflow='white',
                 xmin=0, xmax=35, umin=-5000, umax=5000,
                 Q_dx=1e2, Q_dx_ud=1e5, Q_con_x=1e1, Q_con_u=1e1, Q_spectral=1e2, rom=True):
        super().__init__(nx, ny, n_m, n_dT, nu, nd, n_hidden, bias=bias, heatflow=heatflow,
                         xmin=xmin, xmax=xmax, umin=umin, umax=umax,
                         Q_dx=Q_dx, Q_dx_ud=Q_dx_ud, Q_con_x=Q_con_x, Q_con_u=Q_con_u, Q_spectral=Q_spectral, rom=rom)
        self.A = SpectralLinear(nx, nx, bias=bias, n_U_reflectors=nx, n_V_reflectors=nx, sigma_min=0.6, sigma_max=1.0)


class SSMGroundTruth(SSM):
    def __init__(self, nx, ny, n_m, n_dT, nu, nd, n_hidden, bias=False,
                 heatflow='white', # dummy args for common API in training script
                 xmin=0, xmax=35, umin=-5000, umax=5000,
                 Q_dx=1e2, Q_dx_ud=1e5, Q_con_x=1e1, Q_con_u=1e1, Q_spectral=1e2, rom=True):
        super().__init__(nx, ny, n_m, n_dT, nu, nd, n_hidden, bias=bias)
        bd = BuildingDAE(rom=rom)

        self.G = nn.Parameter(torch.tensor(bd.G, dtype=torch.float32), requires_grad=False)
        self.F = nn.Parameter(torch.tensor(bd.F, dtype=torch.float32), requires_grad=False)

        with torch.no_grad():
            self.A.linear.weight.copy_(torch.tensor(bd.A))
            self.B.linear.weight.copy_(torch.tensor(bd.B))
            self.E.linear.weight.copy_(torch.tensor(bd.E))
            self.C.linear.weight.copy_(torch.tensor(bd.C))

        for p in self.parameters():
            p.requires_grad = False

    @property
    def regularization_error(self):
        return 0.0

    def forward(self, x, M_flow, DT, D):
        """
        """
        X, Y, U = [], [], []
        for m_flow, dT, d in zip(M_flow, DT, D):
            u = heat_flow(m_flow, dT)
            x = self.A(x) + self.B(u) + self.E(d) + self.G.T
            y = self.C(x) + self.F.T - 273.15
            X.append(x)
            Y.append(y)
            U.append(u)
        return torch.stack(X), torch.stack(Y), torch.stack(U), self.regularization_error


