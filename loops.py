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
#local imports
import estimators
import policies
import ssm
from blocks import MLP

# TODO: solve the issue with non-uniform forward pass definition among the estimators and policies
# either via wrapper or via standardized foward pass through *args

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
        Uf = Uf.unsqueeze(2).reshape(Uf.shape[0], self.model.nu, -1)
        Uf = Uf.permute(2,0,1)
        # Uf = Uf.reshape(Rf.shape[0], -1, self.model.nu)  # not sure if it does not shuffle
        Xf, Yf, reg_error_model = self.model(x0, Uf, Df)
        reg_error = reg_error_model + reg_error_policy + reg_error_estim
        return Xf, Yf, Uf, reg_error

if __name__ == '__main__':
    nx, ny, nu, nd = 15, 7, 5, 3
    Np = 2
    Nf = 10
    samples = 100
    # Data format: (N,samples,dim)
    x = torch.rand(samples, nx)
    Yp = torch.rand(Np, samples, ny)
    Up = torch.rand(Np, samples, nu)
    Uf = torch.rand(Nf, samples, nu)
    Dp = torch.rand(Np, samples, nd)
    Df = torch.rand(Nf, samples, nd)
    Rf = torch.rand(Nf, samples, ny)
    x0 = torch.rand(samples, nx)

    # block  SSM
    fx, fu, fd = [MLP(insize, nx, hsizes=[64, 64, 64]) for insize in [nx, nu, nd]]
    fy = MLP(nx, ny, hsizes=[64, 64, 64])
    model1 = ssm.BlockSSM(nx, nu, nd, ny, fx, fy, fu, fd)
    model_out = model1(x0, Uf, Df)
    print(model_out[0].shape, model_out[1].shape, model_out[2])

    # black box SSM
    fxud = MLP(nx + nu + nd, nx, hsizes=[64, 64, 64])
    model2 = ssm.BlackSSM(nx, nu, nd, ny, fxud, fy)
    model_out = model2(x0, Uf, Df)
    print(model_out[0].shape, model_out[1].shape, model_out[2])

    # TODO: issue with the estimator switching 0th index with 1st index
    est = estimators.LinearEstimator(ny, nx)
    est_out = est(Yp, Up, Dp)
    print(est_out[0].shape, est_out[1].shape)

    pol = policies.LinearPolicy(nx, nu, nd, ny, Nf)
    pol_out = pol(x0, Df, Rf)
    print(pol_out[0].shape, pol_out[1].shape)

    ol = OpenLoop(model1,est)
    ol_out = ol(Yp, Up, Uf, Dp, Df)
    print(ol_out[0].shape, ol_out[1].shape, ol_out[2].shape)

    cl = ClosedLoop(model1, est, pol)
    cl_out = cl(Yp, Up, Dp, Df, Rf)
    print(cl_out[0].shape, cl_out[1].shape, cl_out[2].shape, cl_out[3].shape)

    ol = OpenLoop(model2, est)
    ol_out = ol(Yp, Up, Uf, Dp, Df)
    print(ol_out[0].shape, ol_out[1].shape, ol_out[2].shape)

    cl = ClosedLoop(model2, est, pol)
    cl_out = cl(Yp, Up, Dp, Df, Rf)
    print(cl_out[0].shape, cl_out[1].shape, cl_out[2].shape, cl_out[3].shape)