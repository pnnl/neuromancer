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

Dynamical models from dynamics.py
estimator from estimators.py
policy from  policies.py
"""

# pytorch imports
import torch
import torch.nn as nn
#local imports
import estimators
import policies
import dynamics
from blocks import MLP

# TODO: custom loss functions with templates
# IDEA: user should have opportunity to define custom loss functions easily via high level API


class OpenLoop(nn.Module):
    def __init__(self, model=dynamics.BlockSSM, estim=estimators.LinearEstimator, Q_e=1.0, **linargs):
        """
        :param model: SSM mappings, see dynamics.py
        :param estim: state estimator mapping, see estimators.py

        input data trajectories:
        Y: measured outputs p (past)
        U: control inputs p (past), f (future)
        D: measured disturbances p (past), f (future)
        nsamples: prediction horizon length
        """
        super().__init__()
        self.model = model
        self.estim = estim
        self.Q_e = Q_e
        self.criterion =  torch.nn.MSELoss()

    def n_step(self, data):
        Yp, Yf, Up, Uf, Dp, Df = data.values()
        X_pred, Y_pred, reg_error = self.forward(Yp, Up, Uf, Dp, Df, nsamples=Yf.shape[0])
        U_pred = Uf
        loss = self.criterion(Y_pred.squeeze(), Yf.squeeze())
        return {f'{data.name}_nstep_obj_loss': loss,
                f'{data.name}_nstep_reg_error': reg_error,
                f'{data.name}_nstep_loss': loss + reg_error,
                'X_pred': X_pred,
                'Y_pred': Y_pred,
                'U_pred': U_pred,
                'Df': Df}

    def loop_step(self, data):
        Yp, Yf, Up, Uf, Dp, Df = data.values()
        X_pred, Y_pred, reg_error = self.forward(Yp, Up, Uf, Dp, Df, nsamples=Yf.shape[0])
        U_pred = Uf
        loss = self.criterion(Y_pred.squeeze(), Yf.squeeze())
        return {f'{data.name}_loop_obj_loss': loss,
                f'{data.name}_loop_reg_error': reg_error,
                f'{data.name}_loop_loss': loss + reg_error,
                'X_pred': X_pred,
                'Y_pred': Y_pred,
                'U_pred': U_pred,
                'Df': Df}

    def forward(self, Yp, Up, Uf, Dp, Df, nsamples=1):
        x0, reg_error_estim = self.estim(Yp, Up, Dp)
        Xf, Yf, reg_error_model = self.model(x=x0, U=Uf, D=Df, nsamples=nsamples)
        # Calculate mse for smoother state estimator predictions. Last prediction of SSM for a batch should equal
        # the state estimation of the next sequential batch. Warning: This will not perform as expected
        # if batches are shuffled in SGD (we are using full GD so we are okay here.
        estim_error = self.Q_e*torch.nn.functional.mse_loss(x0[1:], Xf[-1, :-1:, :])
        reg_error = reg_error_model + reg_error_estim + estim_error
        return Xf, Yf, reg_error


class ClosedLoop(nn.Module):
    def __init__(self, model=dynamics.BlockSSM, estim=estimators.LinearEstimator,
                 policy=policies.LinearPolicy, Q_e=1.0, **linargs):
        """
        :param model: SSM mappings, see dynamics.py
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
        self.Q_e = Q_e

    def forward(self, Yp, Up, Dp, Df, Rf, nsamples=1):
        x0, reg_error_estim = self.estim(Yp, Up, Dp)
        # TODO: how do we make policy design more flexible for the user?
        Uf, reg_error_policy = self.policy(x0, Df, Rf)
        Uf = Uf.unsqueeze(2).reshape(Uf.shape[0], self.model.nu, -1)
        Uf = Uf.permute(2,0,1)
        # Uf = Uf.reshape(Rf.shape[0], -1, self.model.nu)  # not sure if it does not shuffle
        Xf, Yf, reg_error_model = self.model(x=x0, U=Uf, D=Df, nsamples=nsamples)
        # Calculate mse for smoother state estimator predictions. Last prediction of SSM for a batch should equal
        # the state estimation of the next sequential batch. Warning: This will not perform as expected
        # if batches are shuffled in SGD (we are using full GD so we are okay here.
        estim_error = self.Q_e * torch.nn.functional.mse_loss(x0[1:], Xf[-1, :-1:, :])
        reg_error = reg_error_model + reg_error_policy + reg_error_estim + estim_error
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
    model1 = dynamics.BlockSSM(nx, nu, nd, ny, fx, fy, fu, fd)
    model_out = model1(x0, Uf, Df)

    # black box SSM
    fxud = MLP(nx + nu + nd, nx, hsizes=[64, 64, 64])
    model2 = dynamics.BlackSSM(nx, nu, nd, ny, fxud, fy)
    model_out = model2(x0, Uf, Df)

    # TODO: issue with the estimator switching 0th index with 1st index
    est = estimators.LinearEstimator(ny, nx)
    est_out = est(Yp, Up, Dp)

    pol = policies.LinearPolicy(nx, nu, nd, ny, Nf)
    pol_out = pol(x0, Df, Rf)

    ol = OpenLoop(model1,est)
    ol_out = ol(Yp, Up, Uf, Dp, Df)

    cl = ClosedLoop(model1, est, pol)
    cl_out = cl(Yp, Up, Dp, Df, Rf)

    ol = OpenLoop(model2, est)
    ol_out = ol(Yp, Up, Uf, Dp, Df)

    cl = ClosedLoop(model2, est, pol)
    cl_out = cl(Yp, Up, Dp, Df, Rf)
