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


# this option could be used for system ID with given U
class StaticPolicy(nn.Module):
    pass

class LinearPolicy(nn.Module):
    def __init__(self, nx, nu, nd, ny, N=1, bias=False,
                 Linear=linear.Linear, **linargs):
        """

        :param nx: (int) dimension of state
        :param nu: (int) dimension of inputs
        :param nd: (int) dimension of disturbances
        :param ny: (int) dimension of observation/reference
        :param N: (int) prediction horizon
        """
        super().__init__()
        self.nx, self.nu, self.nd, self.ny, self.N = nx, nu, nd, ny, N
        self.linear = Linear(nx+N*(nd+ny), N*nu, bias=bias, **linargs)

#   shall we create separate module called constraints and passing various options as arguments here?
    def regularize(self):
        pass

    def reg_error(self):
        return self.linear.reg_error()

    def forward(self, x, D, R, *args):
        D = D.reshape(-1, self.N * self.nd)
        R = R.reshape(-1, self.N * self.ny)
        x = x.reshape(-1, self.nx)
        xi = torch.cat((x, D, R), 1)
        return self.linear(xi), self.reg_error()


class MLPPolicy(nn.Module):
    def __init__(self, nx, nu, nd, ny, N=1, bias=True,
                 Linear=linear.Linear, nonlin=F.relu, hsizes=[64], **linargs):
        super().__init__()
        self.nx, self.nu, self.nd, self.ny, self.N = nx, nu, nd, ny, N
        self.net = MLP(nx+N*(nd+ny), N*nu, bias=bias,
                       Linear=Linear, nonlin=nonlin, hsizes=hsizes, **linargs)

    def reg_error(self):
        return self.net.reg_error()

    def forward(self, x, D, R, *args):
        D = D.reshape(-1, self.N * self.nd)
        R = R.reshape(-1, self.N * self.ny)
        x = x.reshape(-1, self.nx)
        xi = torch.cat((x, D, R), 1)
        return self.net(xi), self.reg_error()



class RNNPolicy(nn.Module):
    def __init__(self, nx, nu, nd, ny, N=1, bias=False, num_layers=1,
                 nonlinearity=F.gelu, Linear=linear.Linear, **linargs):
        super().__init__()
        self.nx, self.nu, self.nd, self.ny, self.N = nx, nu, nd, ny, N
        self.RNN = RNN(nx+N*(nd+ny), N*nu, num_layers=num_layers,
                       bias=bias, nonlinearity=nonlinearity, Linear=Linear, **linargs)

    def reg_error(self):
        return self.RNN.reg_error()

    def forward(self, x, D, R, *args):
        D = D.reshape(-1, self.N * self.nd)
        R = R.reshape(-1, self.N * self.ny)
        x = x.reshape(-1, self.nx)
        xi = torch.cat((x, D, R), 1)
        return self.RNN(xi)[0].reshape(-1, self.N * self.nu), self.reg_error()


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