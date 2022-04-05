import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn

import psl

from neuromancer.integrators import integrators
from neuromancer.problem import Problem
from neuromancer.callbacks import Callback
from neuromancer.loggers import MLFlowLogger
from neuromancer.trainer import Trainer
from neuromancer.constraint import Loss
from neuromancer.component import Component
from neuromancer.blocks import MLP
from neuromancer.activations import activations
from neuromancer.dataset import DictDataset
from auto_timestepper import plot_traj, TSCallback, truncated_mse


def get_x0():
    return np.array([5, 5, 25], dtype=np.float32) + np.random.rand(3) * 30 - 15


class Validator:

    def __init__(self, netG, modelSystem):
        self.x0s = [get_x0() for i in range(10)]
        X, U, T = [], [], []
        for x0 in self.x0s:
            x, u, t = modelSystem.simulate(ts=args.ts, nsim=1000, x0=x0)
            X.append(x), U.append(u), T.append(t)

        self.reals = {'X': torch.tensor(np.stack(X), dtype=torch.float32),
                      'U': torch.tensor(np.stack(U), dtype=torch.float32),
                      'T': torch.tensor(np.stack(T), dtype=torch.float32)}
        self.netG = netG

    def __call__(self):
        nsteps = self.netG.nsteps
        self.netG.nsteps = 1000
        with torch.no_grad():
            simulation = self.netG.forward(self.reals)
            simulation = np.nan_to_num(simulation['X_ssm'].detach().numpy(),
                                       copy=True, nan=200000., posinf=None, neginf=None)
            mses = ((self.reals['X'] - simulation)**2).mean(axis=(1, 2))
            truncs = truncated_mse(self.reals['X'], simulation)
        best = np.argmax(truncs)
        self.netG.nsteps = nsteps
        return truncs.mean(), mses.mean(), simulation[best], self.reals['X'][best]


def lorenz_control(x, t, u, sigma=10, beta=2.66667, rho=28):
    return [
        sigma * (x[1] - x[0]) + u[0],
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2] - u[1],
    ]


class LorenzControl:

    def __init__(self):
        pass

    # Control input
    def u_fun(self, t):
        return np.array([np.sin(2 * t), np.sin(8*t)])

    def simulate(self, ts=0.01, nsim=100, x0=[-8, 8, 27]):

        t = np.random.uniform(low=0, high=np.pi)
        x = x0
        tt = np.array([t, t + ts])
        u = self.u_fun(t)
        X, U, T = [x], [self.u_fun(t)], [t]
        for i in range(nsim-1):
            x = odeint(lorenz_control, x, tt, args=(u,))[-1]
            X.append(x)
            T.append(tt[-1])
            u = self.u_fun(tt[-1])
            U.append(u)
            tt += ts
        x_train, u_train, t_train = np.stack(X), np.stack(U), np.stack(T)
        return x_train, u_train, t_train


modelSystem = LorenzControl()


def get_data(nsteps):
    X, U, T = [], [], []
    for _ in range(args.nsim):
        x, u, t = modelSystem.simulate(ts=args.ts, nsim=nsteps, x0=get_x0())
        X.append(x)
        U.append(u)
        T.append(t)
    X, U, T = np.stack(X), np.stack(U), np.stack(T)
    x, u, t = modelSystem.simulate(ts=args.ts, nsim=args.nsim*nsteps, x0=get_x0())

    nx, nu = X.shape[-1], U.shape[-1]
    x, u, t = x.reshape(args.nsim, nsteps, nx), u.reshape(args.nsim, nsteps, nu), t.reshape(args.nsim, nsteps)
    X, U, T = np.concatenate([X, x], axis=0), np.concatenate([U, u], axis=0), np.concatenate([T, t], axis=0)

    train_data = DictDataset({'X': torch.Tensor(X, device=device),
                              'U': torch.Tensor(U, device=device),
                              'T': torch.Tensor(T, device=device)}, name='train')
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              collate_fn=train_data.collate_fn, shuffle=True)

    dev_data = DictDataset({'X': torch.Tensor(X[0:1], device=device),
                            'U': torch.Tensor(U[0:1], device=device),
                            'T': torch.Tensor(T[0:1], device=device)}, name='dev')
    dev_loader = DataLoader(dev_data, num_workers=1, batch_size=args.batch_size,
                            collate_fn=dev_data.collate_fn, shuffle=False)
    test_loader = dev_loader
    return nx, train_loader, dev_loader, test_loader


class SSMIntegrator(Component):
    def __init__(self, integrator, nsteps):
        super().__init__(['X'], ['X_ssm'], name='ssm')
        self.integrator = integrator
        self.nsteps = nsteps

    def forward(self, data):
        x = data['X'][:, 0, :]
        U = data['U']
        X = [x]
        for i in range(self.nsteps - 1):
            x = self.integrator(x, u=U[:, i, :])
            X.append(x)
        return {'X_ssm': torch.stack(X, dim=1)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', type=int, default=1000,
                        help='Number of epochs of training.')
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-nsteps', type=int, default=4)
    parser.add_argument('-stepper', default='Euler', choices=[k for k in integrators])
    parser.add_argument('-batch_size', type=int, default=100)
    parser.add_argument('-nsim', type=int, default=1000)
    parser.add_argument('-ts', type=float, default=0.01)
    parser.add_argument('-q_mse', type=float, default=2.0)
    parser.add_argument('-logdir', default='test')
    parser.add_argument("-exp", type=str, default="test",
           help="Will group all run under this experiment name.")
    parser.add_argument("-location", type=str, default="mlruns",
           help="Where to write mlflow experiment tracking stuff")
    parser.add_argument("-run", type=str, default="neuromancer",
           help="Some name to tell what the experiment run was about.")

    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    os.makedirs(args.logdir, exist_ok=True)

    nx, train_data, dev_data, test_data = get_data(args.nsteps)
    fx = MLP(5, 3, bias=False, linear_map=nn.Linear, nonlin=activations['elu'], hsizes=[128, 128, 128, 128])
    interp_u = lambda tq, t, u: u
    integrator = integrators[args.stepper](fx, h=args.ts, interp_u=interp_u)
    ssm = SSMIntegrator(integrator, nsteps=args.nsteps)
    opt = optim.Adam(ssm.parameters(), args.lr, betas=(0.0, 0.9))
    validator = Validator(ssm, modelSystem)
    callback = TSCallback(validator, args.logdir)
    objective = Loss(['X', 'X_ssm'], F.mse_loss, weight=args.q_mse, name='mse')
    problem = Problem([ssm], objective)
    logger = MLFlowLogger(args, savedir=args.logdir, stdout=['train_mse', 'eval_mse', 'eval_tmse'])
    trainer = Trainer(problem, train_data, dev_data, test_data, opt, logger,
                                  callback=callback,
                                  epochs=args.epochs,
                                  patience=args.epochs,
                      train_metric='train_mse',
                      dev_metric='dev_mse',
                      test_metric='test_mse',
                      eval_metric='eval_tmse')

    lr = args.lr
    nsteps = args.nsteps
    for i in range(5):
        print(f'training {nsteps} objective, lr={lr}')
        trainer.train()
        lr/= 2.0
        nsteps *= 2
        nx, train_data, dev_data, test_data = get_data(nsteps)
        trainer.train_data, trainer.dev_data, trainer.test_data = train_data, dev_data, test_data
        ssm.nsteps = nsteps
        opt.param_groups[0]['lr'] = lr





