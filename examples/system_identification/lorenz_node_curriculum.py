import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn

import neuromancer.psl as psl

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
from neuromancer.loss import PenaltyLoss


def lorenz_x0(batchsize, nx):
    sample = torch.tensor([5., 5., 25.], dtype=torch.float32).view(1, nx) + torch.rand(batchsize,                                                                                  nx) * 30. - 15.
    return sample


boxes = pickle.load(open('boxes.pkl', 'rb'))


def get_x0(system):
    if system == 'LorenzSystem':
        return np.array([5, 5, 25], dtype=np.float32) + np.random.rand(3) * 30 - 15
    return np.random.uniform(low=boxes[system]['min'], high=boxes[system]['max'])


def plot_traj(true_traj, pred_traj, figname='open_loop.png'):
    true_traj, pred_traj = true_traj.transpose(1, 0), pred_traj.transpose(1, 0)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(len(true_traj), 1)
    labels = [f'$x_{k}$' for k in range(len(true_traj))]
    for row, (t1, t2, label) in enumerate(zip(true_traj, pred_traj, labels)):
        axe = ax if len(true_traj) == 1 else ax[row]
        axe.set_ylabel(label, rotation=0, labelpad=20)
        axe.plot(t1, label='True', c='c')
        axe.plot(t2, label='Pred', c='m')
        axe.tick_params(labelbottom=False)
    axe.tick_params(labelbottom=True)
    axe.set_xlabel('Time')
    axe.legend()
    plt.savefig(figname)


def truncated_mse(true, pred):
    diffsq = (true - pred) ** 2
    truncs = diffsq > 1.0
    tmse = truncs * np.ones_like(diffsq) + ~truncs * diffsq
    return tmse.mean()



class Validator:

    def __init__(self, netG, modelSystem):
        self.x0s = [get_x0(args.system) for i in range(10)]
        X = []
        for x0 in self.x0s:
            x = modelSystem.simulate(ts=args.ts, nsim=1000, x0=x0)['X'][:-1, :]
            X.append(x)

        self.reals = {'X': torch.tensor(np.stack(X), dtype=torch.float32)}
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


class SSMIntegrator(Component):
    def __init__(self, integrator, nsteps):
        super().__init__(['X'], ['X_ssm'], name='ssm')
        self.integrator = integrator
        self.nsteps = nsteps

    def forward(self, data):
        x = data['X'][:, 0, :]
        X = [x]
        for i in range(self.nsteps - 1):
            x = self.integrator(x)
            X.append(x)
        return {'X_ssm': torch.stack(X, dim=1)}


class TSCallback(Callback):
    def __init__(self, validator, logdir, figname='test/open_loop.png'):
        self.validator = validator
        self.logdir = logdir
        self.figname=figname

    def begin_eval(self, trainer, output):
        tmse, mse, sim, real = self.validator()
        output['eval_tmse'] = tmse
        output['eval_mse'] = mse
        plot_traj(real, sim, figname=self.figname)
        plt.close()


def get_data(nsteps):
    train_set = np.stack([
        modelSystem.simulate(ts=args.ts, nsim=nsteps, x0=get_x0(args.system))['X'][:-1, :]
        for _ in range(args.nsim)
    ], axis=0)
    nx = train_set[0].shape[-1]

    train_set_long = modelSystem.simulate(ts=args.ts, nsim=args.nsim * nsteps,
                                          x0=get_x0(args.system))['X'][:-1, :]
    train_set_long = train_set_long.reshape(args.nsim, nsteps, nx)
    train_set = np.concatenate([train_set, train_set_long])

    train_data = DictDataset({'X': torch.Tensor(train_set, device=device)}, name='train')

    train_data = torch.utils.data.DataLoader(train_data,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=1,
                                             collate_fn=train_data.collate_fn)

    dev_data = DictDataset({'X': torch.Tensor(train_set, device=device)[0:1]}, name='dev')
    dev_data = torch.utils.data.DataLoader(dev_data, batch_size=1, num_workers=1, collate_fn=dev_data.collate_fn)
    test_data = dev_data
    return nx, train_data, dev_data, test_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', type=int, default=10,
                        help='Number of epochs of training.')
    parser.add_argument('-system', type=str, default='LorenzSystem',
                        choices=["LorenzSystem", "VanDerPol", "ThomasAttractor", "RosslerAttractor",
                                 "LotkaVolterra", "Brusselator1D", "ChuaCircuit"])
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

    modelSystem = psl.systems[args.system]()

    nx, train_data, dev_data, test_data = get_data(args.nsteps)
    fx = MLP(nx, nx, bias=False, linear_map=nn.Linear, nonlin=activations['elu'], hsizes=[128, 128, 128, 128])
    integrator = integrators[args.stepper](fx, h=args.ts)
    ssm = SSMIntegrator(integrator, nsteps=args.nsteps)
    opt = optim.Adam(ssm.parameters(), args.lr, betas=(0.0, 0.9))
    validator = Validator(ssm, modelSystem)
    callback = TSCallback(validator, args.logdir)
    objective = Loss(['X', 'X_ssm'], F.mse_loss, weight=args.q_mse, name='mse')
    loss = PenaltyLoss([objective], [])
    # construct constrained optimization problem
    problem = Problem([ssm], loss)
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





