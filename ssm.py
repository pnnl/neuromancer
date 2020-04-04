import torch
import torch.nn as nn
import torch.nn.functional as F
from linear import SVDLinear, PerronFrobeniusLinear, NonnegativeLinear
from rnn import RNN


class SSM_black_con(nn.Module):
    def __init__(self, nx, ny, n_m, n_dT, nu, nd, n_hidden, bias=False):
        super().__init__()
        self.A = SVDLinear(nx, nx, 0.6, 1)
        self.nlin = {'gelu': F.gelu, 'relu': F.relu, 'softplus': F.softplus}
        self.x0_correct = torch.nn.Parameter(torch.zeros(1, nx))  # state initialization term, corresponding to G
        self.E = PerronFrobeniusLinear(nd, nx, 0.05, 1)

        self.B = NonnegativeLinear(nu, nx)
        self.C = SVDLinear(nx, ny, 0.9, 1)

        self.hf1 = nn.Linear(n_m + n_dT, n_hidden, bias=bias)
        self.hf2 = nn.Linear(n_hidden, nu, bias=bias)
        ###        state estim layers
        self.Xestim1 = PerronFrobeniusLinear(ny, nx, 0.9, 1)
        self.Xestim2 = PerronFrobeniusLinear(nx, nx, 0.9, 1)
        self.nx = nx
        #        KF init
        self.Q_init = nn.Parameter(torch.eye(nx), requires_grad=False)
        self.R_init = nn.Parameter(torch.eye(ny), requires_grad=False)
        self.P_init = nn.Parameter(torch.eye(nx), requires_grad=False)
        self.L_init = nn.Parameter(torch.zeros(nx, ny), requires_grad=False)
        self.x0_estim = nn.Parameter(torch.zeros(1, nx), requires_grad=False)

        self.rnn = RNN(ny, nx, args.num_rnn_layers, bias=args.rnn_bias,
                       nonlinearity=self.nlin[args.rnn_activation], stable=args.rnn_type)
        #  thresholds on maximal one-step change of rate of state variables through u and d
        dxmax_val = 0.5
        dxmin_val = -0.5
        self.dxmax = nn.Parameter(dxmax_val * torch.ones(1, nx), requires_grad=False)
        self.dxmin = nn.Parameter(dxmin_val * torch.ones(1, nx), requires_grad=False)

    def forward(self, Ym, M_flow_p, DT_p, D_p, M_flow, DT, D, XMIN, XMAX, UMIN, UMAX):
        """
        """
        X = []
        Y = []
        U = []
        Sx_min = []
        Sx_max = []
        Su_min = []
        Su_max = []
        Sdx_x = []  # violations of max one-step difference of states
        Sdx_u = []  # violations of max one-step influence of control inputs on states
        Sdx_d = []  # violations of max one-step influence of disturbances on states
        SpectErr = []
        # P is estimation error covariance which can be set to be a diagonal matrix
        x_estim = self.x0_estim
        Q = self.Q_init
        R = self.R_init
        P = self.P_init
        L = self.L_init  # KF gain

        #        State estimation loop on past data
        for ym, m_flow_p, dT_p, d_p in zip(Ym, M_flow_p, DT_p, D_p):
            u_p = F.relu(self.hf2(F.relu(self.hf1(torch.cat([m_flow_p, dT_p], dim=1)))))
            Ax_estim, spectralerror = self.A(x_estim)
            x_estim = Ax_estim + self.B(u_p) + self.E(d_p)
            y_estim, spectralerror = self.C(x_estim)
            #           estimation error covariance
            P = torch.mm(self.A.effective_W(), torch.mm(P, self.A.effective_W().T)) + Q

            #     UPDATE STEP:
            x_estim = x_estim + torch.mm((ym - y_estim), L.T)
            L_inverse_part = torch.inverse(R + torch.mm(self.C.effective_W().T, torch.mm(P, self.C.effective_W())))
            L = torch.mm(torch.mm(P, self.C.effective_W()), L_inverse_part)
            P = torch.eye(self.nx) - torch.mm(L, torch.mm(self.C.effective_W().T, P))

        #        past estimated state is our initial condition for prediction
        x = x_estim

        #        PREDICT STEP
        for m_flow, dT, d, xmin, xmax, umin, umax in zip(M_flow, DT, D, XMIN, XMAX, UMIN, UMAX):
            x_prev = x  # previous state memory
            u = F.relu(self.hf2(F.relu(self.hf1(torch.cat([m_flow, dT], dim=1)))))
            Ax, SpectralErrorA = self.A(x)
            x = Ax + self.B(u) + self.E(d)
            y, SpectralErrorC = self.C(x)
            SpectralError = SpectralErrorA + SpectralErrorC
            #  GEQ constraints  x>= xmin
            sxmin = F.relu(-x + xmin)
            #  LEQ constraints x <= xmax
            sxmax = F.relu(x - xmax)

            #  GEQ constraints  u>= umin
            sumin = F.relu(-u + umin)
            #  LEQ constraints u <= umax
            sumax = F.relu(u - umax)

            # smooth operator
            dx_x = x - x_prev  # one step state residual to be penalized in the objective
            # constraints on max one-step infuence of controls and disturbances on states
            dx_u = F.relu(-self.B(u) + self.dxmin) + F.relu(self.B(u) - self.dxmax)
            dx_d = F.relu(-self.E(d) + self.dxmin) + F.relu(self.E(d) - self.dxmax)

            Sx_min.append(sxmin)
            Sx_max.append(sxmax)
            Su_min.append(sumin)
            Su_max.append(sumax)
            Sdx_x.append(dx_x)
            Sdx_u.append(dx_u)
            Sdx_d.append(dx_d)
            X.append(x)
            Y.append(y)
            U.append(u)
            SpectErr.append(SpectralError)

        X_out = torch.stack(X)
        Y_out = torch.stack(Y)
        U_out = torch.stack(U)
        Sx_min_out = torch.stack(Sx_min)
        Sx_max_out = torch.stack(Sx_max)
        Su_min_out = torch.stack(Su_min)
        Su_max_out = torch.stack(Su_max)
        Sdx_x_out = torch.stack(Sdx_x)
        Sdx_u_out = torch.stack(Sdx_u)
        Sdx_d_out = torch.stack(Sdx_d)
        SpectErr_out = torch.stack(SpectErr)
        return X_out, Y_out, U_out, Sx_min_out, Sx_max_out, Su_min_out, Su_max_out, Sdx_x_out, Sdx_u_out, Sdx_d_out, SpectErr_out


class SSM_gray_con(nn.Module):
    def __init__(self, nx, ny, n_m, n_dT, nu, nd, n_hidden, bias=False):
        super().__init__()
        #        self.A = PerronFrobeniusIntervalWeight(nx, nx, 0.9, 1)
        self.A = SpectralWeight(nx, nx, 0.1, 0.90)  # TODO explore different ranges as tuning factor
        self.nlin = {'gelu': F.gelu, 'relu': F.relu, 'softplus': F.softplus}
        self.x0_correct = torch.nn.Parameter(torch.zeros(1, nx))  # state initialization term, corresponding to G
        #        self.E = nn.Linear(nd, nx, bias=bias)
        self.E = PerronFrobeniusIntervalWeight(nd, nx, 0.05, 1)
        self.B = Linear_nonNeg(nu, nx)
        #        self.C = PerronFrobeniusIntervalWeight(nx, ny, 0.9, 1)
        self.C = SpectralWeight(nx, ny, 0.9, 1)

        self.hf = nn.Bilinear(n_m, n_dT, nu, bias=bias)

        ###        state estim layer
        self.Xestim1 = PerronFrobeniusIntervalWeight(ny, nx, 0.9, 1)
        self.Xestim2 = PerronFrobeniusIntervalWeight(nx, nx, 0.9, 1)

        #        self.rnn = torch.nn.RNN(ny, nx, 1, nonlinearity='relu')
        self.rnn = RNN(ny, nx, args.num_rnn_layers, bias=args.rnn_bias,
                       nonlinearity=self.nlin[args.rnn_activation], stable=args.rnn_type)

        #  thresholds on maximal one-step change of rate of state variables through u and d
        dxmax_val = 0.5
        dxmin_val = -0.5
        self.dxmax = nn.Parameter(dxmax_val * torch.ones(1, nx), requires_grad=False)
        self.dxmin = nn.Parameter(dxmin_val * torch.ones(1, nx), requires_grad=False)

    def forward(self, Ym, M_flow, DT, D, XMIN, XMAX, UMIN, UMAX):
        """
        """
        X = []
        Y = []
        U = []
        Sx_min = []
        Sx_max = []
        Su_min = []
        Su_max = []
        Sdx_x = []  # violations of max one-step difference of states
        Sdx_u = []  # violations of max one-step influence of control inputs on states
        Sdx_d = []  # violations of max one-step influence of disturbances on states
        SpectErr = []

        #        UPDATE STEP
        RNN_out = self.rnn(Ym)
        x = RNN_out[0][-1]
        #        x = x + self.x0_correct # state initialization correction term

        #        Pedict STEP
        for m_flow, dT, d, xmin, xmax, umin, umax in zip(M_flow, DT, D, XMIN, XMAX, UMIN, UMAX):
            x_prev = x  # previous state memory
            x = x + self.x0_correct  # state correction term
            u = self.hf(m_flow, dT)

            Ax, SpectralErrorA = self.A(x)
            x = Ax + self.B(u) + self.E(d)
            y, SpectralErrorC = self.C(x)
            SpectralError = SpectralErrorA + SpectralErrorC

            #  GEQ constraints  x>= xmin
            sxmin = F.relu(-x + xmin)
            #  LEQ constraints x <= xmax
            sxmax = F.relu(x - xmax)

            #  GEQ constraints  u>= umin
            sumin = F.relu(-u + umin)
            #  LEQ constraints u <= umax
            sumax = F.relu(u - umax)

            # smooth operator
            dx_x = x - x_prev  # one step state residual to be penalized in the objective
            # constraints on max one-step infuence of controls and disturbances on states
            dx_u = F.relu(-self.B(u) + self.dxmin) + F.relu(self.B(u) - self.dxmax)
            dx_d = F.relu(-self.E(d) + self.dxmin) + F.relu(self.E(d) - self.dxmax)

            Sx_min.append(sxmin)
            Sx_max.append(sxmax)
            Su_min.append(sumin)
            Su_max.append(sumax)
            Sdx_x.append(dx_x)
            Sdx_u.append(dx_u)
            Sdx_d.append(dx_d)
            X.append(x)
            Y.append(y)
            U.append(u)
            SpectErr.append(SpectralError)

        X_out = torch.stack(X)
        Y_out = torch.stack(Y)
        U_out = torch.stack(U)
        Sx_min_out = torch.stack(Sx_min)
        Sx_max_out = torch.stack(Sx_max)
        Su_min_out = torch.stack(Su_min)
        Su_max_out = torch.stack(Su_max)
        Sdx_x_out = torch.stack(Sdx_x)
        Sdx_u_out = torch.stack(Sdx_u)
        Sdx_d_out = torch.stack(Sdx_d)
        SpectErr_out = torch.stack(SpectErr)
        return X_out, Y_out, U_out, Sx_min_out, Sx_max_out, Su_min_out, Su_max_out, Sdx_x_out, Sdx_u_out, Sdx_d_out, SpectErr_out


class SSM_white_con(nn.Module):
    def __init__(self, nx, ny, n_m, n_dT, nu, nd, n_hidden, bias=False):
        super().__init__()
        #        self.A = PerronFrobeniusIntervalWeight(nx, nx, 0.9, 1)
        self.A = SpectralWeight(nx, nx, 0.1, 0.90)  # TODO explore different ranges as tuning factor
        self.x0_correct = torch.nn.Parameter(torch.zeros(1, nx))  # state initialization term, corresponding to G
        self.nlin = {'gelu': F.gelu, 'relu': F.relu, 'softplus': F.softplus}
        #        self.E = nn.Linear(nd, nx, bias=bias)
        self.E = PerronFrobeniusIntervalWeight(nd, nx, 0.05, 1)
        self.B = Linear_nonNeg(nu, nx)
        self.C = SpectralWeight(nx, ny, 0.9, 1)

        self.rho = torch.nn.Parameter(torch.tensor(0.997), requires_grad=False)  # density  of water kg/1l
        self.cp = torch.nn.Parameter(torch.tensor(4185.5),
                                     requires_grad=False)  # specific heat capacity of water J/(kg/K)
        self.time_reg = torch.nn.Parameter(torch.tensor(1 / 3600),
                                           requires_grad=False)  # time regularization of the mass flow 1 hour = 3600 seconds

        ###        state estim layer
        self.Xestim1 = PerronFrobeniusIntervalWeight(ny, nx, 0.9, 1)
        self.Xestim2 = PerronFrobeniusIntervalWeight(nx, nx, 0.9, 1)

        #        self.rnn = torch.nn.RNN(ny, nx, 1, nonlinearity='relu')
        self.rnn = RNN(ny, nx, args.num_rnn_layers, bias=args.rnn_bias,
                       nonlinearity=self.nlin[args.rnn_activation], stable=args.rnn_type)

        #  thresholds on maximal one-step change of rate of state variables through u and d
        dxmax_val = 0.5
        dxmin_val = -0.5
        self.dxmax = nn.Parameter(dxmax_val * torch.ones(1, nx), requires_grad=False)
        self.dxmin = nn.Parameter(dxmin_val * torch.ones(1, nx), requires_grad=False)

    def heat_flow(self, m_flow, dT):
        U = m_flow * self.rho * self.cp * self.time_reg * dT
        return U

    def forward(self, Ym, M_flow, DT, D, XMIN, XMAX, UMIN, UMAX):
        """
        """
        X = []
        Y = []
        U = []
        Sx_min = []
        Sx_max = []
        Su_min = []
        Su_max = []
        Sdx_x = []  # violations of max one-step difference of states
        Sdx_u = []  # violations of max one-step influence of control inputs on states
        Sdx_d = []  # violations of max one-step influence of disturbances on states
        SpectErr = []
        #            naive stable estimator, mapping ym onto x
        #            TODO: extend mapping of ym,d,u onto x
        #            TODO: moving horizon estimation, mapping Y history over N steps
        #            x = self.Xestim2(F.relu(self.Xestim1(ym)))
        #        x = self.Xestim1(ym)
        #        x = x + self.x0_correct # state initialization correction term

        #        UPDATE STEP
        RNN_out = self.rnn(Ym)
        x = RNN_out[0][-1]
        #        x = x + self.x0_correct # state initialization correction term

        #        PREDICT STEP
        for m_flow, dT, d, xmin, xmax, umin, umax in zip(M_flow, DT, D, XMIN, XMAX, UMIN, UMAX):
            x_prev = x  # previous state memory
            x = x + self.x0_correct  # state correction term
            u = self.heat_flow(m_flow, dT)

            #            Ax, SpectralError = self.A(x)
            #            x = Ax + self.B(u) + self.E(d)
            #            y = self.C(x)

            Ax, SpectralErrorA = self.A(x)
            x = Ax + self.B(u) + self.E(d)
            y, SpectralErrorC = self.C(x)
            SpectralError = SpectralErrorA + SpectralErrorC

            #  GEQ constraints  x>= xmin
            sxmin = F.relu(-x + xmin)
            #  LEQ constraints x <= xmax
            sxmax = F.relu(x - xmax)

            #  GEQ constraints  u>= umin
            sumin = F.relu(-u + umin)
            #  LEQ constraints u <= umax
            sumax = F.relu(u - umax)

            # smooth operator
            dx_x = x - x_prev  # one step state residual to be penalized in the objective
            # constraints on max one-step infuence of controls and disturbances on states
            dx_u = F.relu(-self.B(u) + self.dxmin) + F.relu(self.B(u) - self.dxmax)
            dx_d = F.relu(-self.E(d) + self.dxmin) + F.relu(self.E(d) - self.dxmax)

            Sx_min.append(sxmin)
            Sx_max.append(sxmax)
            Su_min.append(sumin)
            Su_max.append(sumax)
            Sdx_x.append(dx_x)
            Sdx_u.append(dx_u)
            Sdx_d.append(dx_d)
            X.append(x)
            Y.append(y)
            U.append(u)
            SpectErr.append(SpectralError)

        X_out = torch.stack(X)
        Y_out = torch.stack(Y)
        U_out = torch.stack(U)
        Sx_min_out = torch.stack(Sx_min)
        Sx_max_out = torch.stack(Sx_max)
        Su_min_out = torch.stack(Su_min)
        Su_max_out = torch.stack(Su_max)
        Sdx_x_out = torch.stack(Sdx_x)
        Sdx_u_out = torch.stack(Sdx_u)
        Sdx_d_out = torch.stack(Sdx_d)
        SpectErr_out = torch.stack(SpectErr)
        return X_out, Y_out, U_out, Sx_min_out, Sx_max_out, Su_min_out, Su_max_out, Sdx_x_out, Sdx_u_out, Sdx_d_out, SpectErr_out

