"""
Open and closed loop dynamical models
x: states (x0 - initial conditions)
y: predicted outputs
ym: measured outputs
u: control inputs
d: uncontrolled inputs (measured disturbances)
r: reference signals

generic closed-loop dynamics:
x+ = fx(x) o fu(u) o fd(d)
y =  fy(x)
x = estim(yp,up,dp,x_prev)
u = policy(x,u,d,r)

Dynamical models from ssm.py
estimator from estimators.py
policy from  policies.py
"""

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import linear
import estimators
import policies
import ssm
from blocks import MLP

# TODO: solve the issue with non-uniform forward pass definition among the estimators and policies
# either via wrapper or via standardized foward pass

class OpenLoop(nn.Module):
    def __init__(self, model=ssm.BlockSSM, estim=estimators.LinearEstimator, **linargs):
        """
        :param model: SSM mappings, see ssm.py
        :param estim: state estimator mapping, see estimators.py

        input data trajectories:
        Y: measured outputs p (past)
        U: control inputs p (past), f (future)
        D: measured disturbances p (past), f (future)
        """
        super().__init__()
        self.model = model
        self.estim = estim

    def forward(self, Yp, Up, Uf, Dp, Df):

        x0, reg_error_estim = self.estim(Yp, Up, Dp)
        Xf, Yf, reg_error_model = self.model(x0, Uf, Df)
        reg_error = reg_error_model + reg_error_estim
        return Xf, Yf, reg_error


class ClosedLoop(nn.Module):
    def __init__(self, model=ssm.BlockSSM, estim=estimators.LinearEstimator,
                 policy=policies.LinearPolicy, **linargs):
        """
        :param model: SSM mappings, see ssm.py
        :param estim: state estimator mapping, see estimators.py

        input data trajectories:
        Y: measured outputs p (past)
        U: control inputs  f (future)
        D: measured disturbances p (past), f (future)
        R: desired references f (future)
        """
        super().__init__()
        self.model = model
        self.estim = estim
        self.policy = policy

    def forward(self, Yp, Up, Dp, Df, Rf):

        x0, reg_error_estim = self.estim(Yp, Up, Dp)
        Uf, reg_error_policy = self.policy(x0, Df, Rf)
        Xf, Yf, reg_error_model = self.model(x0, Uf, Df)
        reg_error = reg_error_model + reg_error_policy + reg_error_estim
        return Xf, Yf, Uf, reg_error

if __name__ == '__main__':
    nx, ny, nu, nd = 15, 7, 5, 3
    Np = 2
    Nf = 2
    Yp = torch.rand(100, Np, ny)
    Up = torch.rand(100, Np, nu)
    Uf = torch.rand(100, Nf, nu)
    Dp = torch.rand(100, Np, nd)
    Df = torch.rand(100, Nf, nd)
    Rf = torch.rand(100, Nf, ny)
    x0 = torch.rand(100, 1, nx)

    fx, fu, fd = [MLP(insize, nx, hsizes=[64, 64, 64]) for insize in [nx, nu, nd]]
    fy = MLP(nx, ny, hsizes=[64, 64, 64])
    model = ssm.BlockSSM(nx, nu, nd, ny, fx, fu, fd, fy)
    est = estimators.LinearEstimator(ny,nx)
    pol = policies.LinearPolicy(nx, nu, nd, ny, Nf)

    # TODO: issue with the estimator switching 0th index with 1st index
    est_out = est(Yp, Up, Dp)
    print(est_out[0].shape, est_out[1].shape)
    pol_out = pol(x0, Df, Rf)
    print(pol_out[0].shape, pol_out[1].shape)

    ol = OpenLoop(model,est)
    cl = ClosedLoop(model,est,pol)



    ol_out = ol(Yp, Up, Uf, Dp, Df)
    print(ol_out[0].shape, ol_out[1].shape, ol_out[2].shape)
    cl_out = cl(Yp, Up, Dp, Df, Rf)
    print(cl_out[0].shape, cl_out[1].shape, cl_out[2].shape, cl_out[3].shape)