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
x+ = f_x(x) o f_u(u) o f_d(d)
y =  f_y(x)
        
state_estimators.py
x = estim(y_p,u_p,d_p)
        
policies.py
x = policy(x,d)
"""
# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: generic HW-SSM
def get_modules(model):
    return {name: module for name, module in model.named_modules() if len(list(module.named_children())) == 0}


# smart ways of initializing the weights?
class BlockSSM(nn.Module):
    def __init__(self, nx, ny, nu, nd, f_x, f_u, f_d, f_y,
                 xou=torch.add, xod=torch.add):
        """
        generic system dynamics:   
        # x+ = f_x(x) o f_u(u) o f_d(d)
        # y =  f_y(x)
        """
        super().__init__()
        assert f_x.in_features == nx, "Mismatch in input function size"
        assert f_x.out_features == nx, "Mismatch in input function size"
        assert f_d.in_features == nd, "Mismatch in disturbance function size"
        assert f_d.out_features == nx, "Mismatch in disturbance function size"
        assert f_u.in_features == nu, "Mismatch in control input function size"
        assert f_u.out_features == nx, "Mismatch in control input function size"
        assert f_y.in_features == nx, "Mismatch in observable output function size"
        assert f_y.out_features == ny, "Mismatch in observable output function size"

        self.nx, self.ny, self.nu, self.nd = nx, ny, nu, nd
        self.f_x, self.f_u, self.f_d, self.f_y = f_x, f_u, f_d, f_y
        # block operators
        self.xou = xou
        self.xod = xod       
        
        # Regularization Initialization
        self.xmin, self.xmax, self.umin, self.umax, self.uxmin, self.uxmax, self.dxmin, self.dxmax = self.con_init()
        self.Q_dx, self.Q_dx_ud, self.Q_con_x, self.Q_con_u, self.Q_sub = self.reg_weight_init()
      
        # slack variables for calculating constraints violations/ regularization error
        self.sxmin, self.sxmax, self.sumin, self.sumax, self.sdx_x, self.dx_u, self.dx_d, self.d_sub = [0.0]*8

    def con_init(self):
            return [-1,1,-1,1,-1,1,-1,1]  
        
    def reg_weight_init(self):
        return [0.2]*5

    def running_mean(self, mu, x, n):
        return mu + (1/n)*(x - mu)
    
#    include regularization in each module in the framework
    def regularize(self, x_prev, x, f_u, f_d, N):
        
        # Barrier penalties
        self.sxmin = self.running_mean(self.sxmin, torch.mean(F.relu(-x + self.xmin)), N)  #self.sxmin*(n-1)/n + self.Q_con_x*F.relu(-x + self.xmin)
        self.sxmax = self.running_mean(self.sxmax, torch.mean(F.relu(x - self.xmax)), N)
        self.sumin = self.running_mean(self.sumin, torch.mean(F.relu(-f_u + self.umin)), N)
        self.sumax = self.running_mean(self.sumax, torch.mean(F.relu(f_u - self.umax)), N)
        # one step state residual penalty
        self.sdx_x = self.running_mean(self.sdx_x, torch.mean((x - x_prev)*(x - x_prev)), N)
        # penalties on max one-step infuence of controls and disturbances on states
        self.dx_u = self.running_mean(self.dx_u,  torch.mean(F.relu(-f_u + self.uxmin) + F.relu(f_u - self.uxmax)), N)
        self.dx_d = self.running_mean(self.dx_d,  torch.mean(F.relu(-f_d + self.dxmin) + F.relu(f_d - self.dxmax)), N)
        self.d_sub = self.running_mean(torch.sum([k.reg_error() for k in get_modules(self).values() if hasattr(k, 'reg_error')]))

    def reg_error(self):
        error = torch.sum(torch.stack([self.Q_con_x*self.sxmin, self.Q_con_x*self.sxmax, self.Q_con_u*self.sumin, self.Q_con_u*self.sumax, 
                                       self.Q_dx*self.sdx_x, self.Q_dx_ud*self.dx_u, self.Q_dx_ud*self.dx_d]))
        error += self.Q_sub*self.d_sub   # error of elements in the list of all submodules
        self.sxmin, self.sxmax, self.sumin, self.sumax, self.sdx_x, self.dx_u, self.dx_d, self.d_sub = [0.0]*8
        return error

    def forward(self, x, U, D):
        """
        """      
        # prediction on future moving window    
        X, Y = [], []
        N = 0
        for u, d in zip(U, D):
            N += 1
            x_prev = x  # previous state memory
            f_u = self.f_u(u)
            f_d = self.f_d(d)
            x = self.f_x(x)     
            x = self.xou(x,f_u)
            x = self.xod(x,f_d)
            y = self.f_y(x)
            X.append(x)
            Y.append(y)
            self.regularize(x_prev, x, f_u, f_d, N)
        return torch.stack(X), torch.stack(Y), self.reg_error