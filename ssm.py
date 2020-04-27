"""
SSM
"""

"""
unstructured dynamical models
x+ = f(x,u,d)
y =  m(x)
    
Block dynamical models 
x+ = f1(x) o f2(x)  o g(u) o h(d)
y =  m(x)

o = operator, e.g., +, or *
any operation perserving dimensions

estimator and policy - possibly structured
x = estim(y,u,d)
u = policy(x,u,d)

generic closed-loop dynamics: 
    
SSM
x+ = fx(x) o fu(u) o fd(d)
y =  fy(x)
        
state_estimators.py
x = estim(y_p,u_p,d_p)
        
policies.py
x = policy(x,d)
"""
# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
# local imports
from blocks import MLP


# TODO: generic HW-SSM
def get_modules(model):
    return {name: module for name, module in model.named_modules()
            if len(list(module.named_children())) == 0}


# smart ways of initializing the weights?
class BlockSSM(nn.Module):
    def __init__(self, nx, nu, nd, ny, fx, fu, fd, fy,
                 xou=torch.add, xod=torch.add):
        """

        :param nx: (int) dimension of state
        :param nu: (int) dimension of inputs
        :param nd: (int) dimension of disturbances
        :param ny: (int) dimension of observation
        :param fx: function R^{nx} -> R^(nx}
        :param fu: function R^{nu} -> R^(nx}
        :param fd: function R^{nd} -> R^(nx}
        :param fy: function R^{nx} -> R^(ny}
        :param xou: Shape preserving binary operator (e.g. +, -, *)
        :param xod: Shape preserving binary operator (e.g. +, -, *)
        """
        """
        generic system dynamics:   
        # x+ = fx(x) o fu(u) o fd(d)
        # y =  fy(x)
        """
        super().__init__()
        assert fx.in_features == nx, "Mismatch in input function size"
        assert fx.out_features == nx, "Mismatch in input function size"
        assert fd.in_features == nd, "Mismatch in disturbance function size"
        assert fd.out_features == nx, "Mismatch in disturbance function size"
        assert fu.in_features == nu, "Mismatch in control input function size"
        assert fu.out_features == nx, "Mismatch in control input function size"
        assert fy.in_features == nx, "Mismatch in observable output function size"
        assert fy.out_features == ny, "Mismatch in observable output function size"

        self.nx, self.nu, self.nd, self.ny = nx, nu, nd, ny
        self.fx, self.fu, self.fd, self.fy = fx, fu, fd, fy
        # block operators
        self.xou = xou
        self.xod = xod       
        
        # Regularization Initialization
        self.xmin, self.xmax, self.umin, self.umax, self.uxmin, self.uxmax, self.dxmin, self.dxmax = self.con_init()
        self.Q_dx, self.Q_dx_ud, self.Q_con_x, self.Q_con_u, self.Q_sub = self.reg_weight_init()
      
        # slack variables for calculating constraints violations/ regularization error
        self.sxmin, self.sxmax, self.sumin, self.sumax, self.sdx_x, self.dx_u, self.dx_d, self.d_sub = [0.0]*8

    def con_init(self):
            return [-1, 1, -1, 1, -1, 1, -1, 1]
        
    def reg_weight_init(self):
        return [0.2]*5

    def running_mean(self, mu, x, n):
        return mu + (1/n)*(x - mu)
    
#    include regularization in each module in the framework
    def regularize(self, x_prev, x, fu, fd, N):
        
        # Barrier penalties
        self.sxmin = self.running_mean(self.sxmin, torch.mean(F.relu(-x + self.xmin)), N)
        self.sxmax = self.running_mean(self.sxmax, torch.mean(F.relu(x - self.xmax)), N)
        self.sumin = self.running_mean(self.sumin, torch.mean(F.relu(-fu + self.umin)), N)
        self.sumax = self.running_mean(self.sumax, torch.mean(F.relu(fu - self.umax)), N)
        # one step state residual penalty
        self.sdx_x = self.running_mean(self.sdx_x, torch.mean((x - x_prev)*(x - x_prev)), N)
        # penalties on max one-step infuence of controls and disturbances on states
        self.dx_u = self.running_mean(self.dx_u,
                                      torch.mean(F.relu(-fu + self.uxmin) + F.relu(fu - self.uxmax)), N)
        self.dx_d = self.running_mean(self.dx_d,
                                      torch.mean(F.relu(-fd + self.dxmin) + F.relu(fd - self.dxmax)), N)
        self.d_sub = self.running_mean(self.d_sub, sum([k.reg_error() for k in
                                                        [self.fx, self.fu, self.fd, self.fy]
                                                        if hasattr(k, 'reg_error')]), N)

    def reg_error(self):
        error = sum([self.Q_con_x*self.sxmin, self.Q_con_x*self.sxmax,
                     self.Q_con_u*self.sumin, self.Q_con_u*self.sumax,
                     self.Q_dx*self.sdx_x, self.Q_dx_ud*self.dx_u,
                     self.Q_dx_ud*self.dx_d, self.Q_sub*self.d_sub])
        self.sxmin, self.sxmax, self.sumin, self.sumax, self.sdx_x, self.dx_u, self.dx_d, self.d_sub = [0.0]*8
        return error

    def forward(self, x, U, D):
        """
        """      
        X, Y = [], []
        N = 0
        for u, d in zip(U, D):
            N += 1
            x_prev = x
            fu = self.fu(u)
            fd = self.fd(d)
            x = self.fx(x)     
            x = self.xou(x, fu)
            x = self.xod(x, fd)
            y = self.fy(x)
            X.append(x)
            Y.append(y)
            self.regularize(x_prev, x, fu, fd, N)
        return torch.stack(X), torch.stack(Y), self.reg_error()


if __name__ == '__main__':
    nx, ny, nu, nd = 15, 7, 5, 3
    fx, fu, fd = [MLP(insize, nx, hsizes=[64, 64, 64]) for insize in [nx, nu, nd]]
    fy = MLP(nx, ny, hsizes=[64, 64, 64])
    ssm = BlockSSM(nx, nu, nd, ny, fx, fu, fd, fy)
    x = torch.rand(25, nx)
    U = torch.rand(100, 25, nu)
    D = torch.rand(100, 25, nd)
    output = ssm(x, U, D)
    print(output[0].shape, output[1].shape, output[2])
