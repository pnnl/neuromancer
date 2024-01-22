"""
This is a training script for system id using NODEs with Euler integration for any nonautonomous psl models.

A few things that are included:
    + Multiple objectives for finite difference, one step ahead, nstep ahead, latent and observed state MSE
    + Optional statewise scaling of MSE-loss by state variance
    + Optional normalization
    + Configurable curriculum learning
    + Training data with many sampled initial conditions for independent n-step simulations as well as a single long trajectory
"""
import os

import sklearn
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

from neuromancer.modules.blocks import MLP
from neuromancer.dynamics.integrators import Euler
from neuromancer.psl.autonomous import systems
from neuromancer.problem import Problem
from neuromancer.loggers import MLFlowLogger
from neuromancer.trainer import Trainer
from neuromancer.constraint import Loss, variable
from neuromancer.loss import PenaltyLoss
from neuromancer.dataset import DictDataset
from neuromancer.callbacks import Callback
from neuromancer.system import Node, System


def plot_traj(true_traj, pred_traj, figname='open_loop.png'):
    true_traj, pred_traj = true_traj.transpose(1, 0), pred_traj.transpose(1, 0)
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


class TSCallback(Callback):
    def __init__(self, validator, logdir):
        self.validator = validator
        self.logdir = logdir

    def begin_eval(self, trainer, output):
        tmse, mse, sim, real = self.validator()
        output['eval_tmse'] = tmse
        output['eval_mse'] = mse
        plot_traj(real, sim, figname=f'{self.logdir}/{self.validator.figname}steps_open.png')
        plt.close()


class Validator:

    def __init__(self, netG, sys, args, normalize=False):
        """
        Used for evaluating model performance

        :param netG: (nn.Module) Some kind of neural network state space model
        :param sys: (psl.ODE_NonAutonomous) Ground truth ODE system
        :param normalize: (bool) Whether to normalized data. Will denorm before plotting and loss calculation if True
        """
        self.figname = ''
        self.x0s = [sys.get_x0() for i in range(10)]
        self.normalize = normalize
        self.sys = sys
        X = []
        for x0 in self.x0s:
            sim = sys.simulate(ts=args.ts, nsim=1000, x0=x0)
            X.append(torch.tensor(sim['X'], dtype=torch.float32))
            if normalize:
                X[-1] = sys.normalize(X[-1], key='X')

        def mse_loss(input, target):
            return torch.mean((input - target) ** 2, axis=(1, 2))

        self.mse = mse_loss

        self.reals = {'X': torch.stack(X)}
        self.netG = netG

    def __call__(self):
        """
        Runs the model in it's current state on 10 validation set trajectories with
        different initial conditions and control sequences.

        :return: truncated_mse, mse over all validation rollouts
        and x, x_pred trajectories for best validation rollout
        """
        with torch.no_grad():
            self.reals['xn'] = self.reals['X'][:, 0:1, :]
            simulation = self.netG.forward(self.reals)
            x = self.reals['X']
            xprime = simulation['xn'][:, :-1, :]
            xprime = torch.nan_to_num(xprime, nan=200000., posinf=None, neginf=None)
            if self.normalize:
                x = self.sys.denormalize(x, key='X')
                xprime = self.sys.denormalize(xprime, key='X')
            print('dev', x.shape, xprime.shape)
            mses = self.mse(x, xprime)
            truncs = truncated_mse(x, xprime)
        best = np.argmax(mses)
        return truncs.mean(), mses.mean(), xprime[best].detach().numpy(), x[best].detach().numpy()


def get_data(nsteps, sys, nsim, bs, normalize=False):
    """
    :param nsteps: (int) Number of timesteps for each batch of training data
    :param sys: (psl.ODE_NonAutonomous)
    :param normalize: (bool) Whether to normalize the data

    """
    X = []
    for _ in range(nsim):
        sim = sys.simulate(nsim=nsteps, x0=sys.get_x0())
        X.append(sim['X'])

    X = np.stack(X)
    sim = sys.simulate(nsim=nsim * nsteps, x0=sys.get_x0())

    nx = X.shape[-1]
    x = sim['X'].reshape(nsim, nsteps, nx)
    X = np.concatenate([X, x], axis=0)

    X = torch.tensor(X, dtype=torch.float32)
    if normalize:
            X = sys.normalize(X, key='X')
    train_data = DictDataset({'X': X, 'xn': X[:, 0:1, :]}, name='train')
    train_loader = DataLoader(train_data, batch_size=bs,
                              collate_fn=train_data.collate_fn, shuffle=True)
    dev_data = DictDataset({'X': X[0:1], 'xn': X[0:1, 0:1, :]}, name='dev')
    dev_loader = DataLoader(dev_data, num_workers=1, batch_size=bs,
                            collate_fn=dev_data.collate_fn, shuffle=False)
    test_loader = dev_loader
    return nx, train_loader, dev_loader, test_loader


class EulerIntegrator(nn.Module):
    """
    Simple black-box NODE
    """
    def __init__(self, nx, hsize, nlayers, ts):
        super().__init__()
        self.dx = MLP(nx, nx, bias=True, linear_map=nn.Linear, nonlin=nn.ELU,
                      hsizes=[hsize for h in range(nlayers)])
        self.integrator = Euler(self.dx, h=torch.tensor(ts))

    def forward(self, xn):
        """

        :param xn: (Tensor, shape=(batchsize, nx)) State
        :param u: (Tensor, shape=(batchsize, nu)) Control action
        :return: (Tensor, shape=(batchsize, nx)) xn+1
        """
        return self.integrator(xn)


def get_node(nx, args):
    integrator = EulerIntegrator(nx, args.hsize, args.nlayers, torch.tensor(args.ts))
    nodes = [Node(integrator, ['xn'], ['xn'])]
    system = System(nodes)
    return system


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-system', default='LorenzSystem', choices=[k for k in systems],
                        help='You can use any of the systems from psl.nonautonomous with this script')
    parser.add_argument('-epochs', type=int, default=100,
                        help='Number of epochs of training.')
    parser.add_argument('-normalize', action='store_true', help='Whether to normalize data')
    parser.add_argument('-lr', type=float, default=0.001,
                        help='Learning rate for gradient descent.')
    parser.add_argument('-nsteps', type=int, default=4,
                        help='Prediction horizon for optimization objective. During training will roll out for nsteps from and initial condition')
    parser.add_argument('-batch_size', type=int, default=100)
    parser.add_argument('-nsim', type=int, default=1000,
                        help="The script will generate an nsim long time series for training and testing and 10 nsim long time series for validation")
    parser.add_argument('-logdir', default='test',
                        help='Plots and best models will be saved here. Also will be moved to the location directory for mlflow artifact logging')
    parser.add_argument("-exp", type=str, default="test",
                        help="Will group all run under this experiment name.")
    parser.add_argument("-location", type=str, default="mlruns",
                        help="Where to write mlflow experiment tracking stuff")
    parser.add_argument("-run", type=str, default="neuromancer",
                        help="Some name to tell what the experiment run was about.")
    parser.add_argument('-hsize', type=int, default=128, help='Size of hiddens states')
    parser.add_argument('-nlayers', type=int, default=4, help='Number of hidden layers for MLP')
    parser.add_argument('-iterations', type=int, default=3,
                        help='How many episodes of curriculum learning by doubling the prediction horizon and halving the learn rate each episode')
    parser.add_argument('-eval_metric', type=str, default='eval_mse')
    args = parser.parse_args()
    os.makedirs(args.logdir, exist_ok=True)

    sys = systems[args.system]()
    args.ts = sys.ts
    nx, train_data, dev_data, test_data = get_data(args.nsteps, sys, args.nsim, args.batch_size, normalize=args.normalize)

    ssm = get_node(nx, args)
    opt = optim.Adam(ssm.parameters(), args.lr, betas=(0.0, 0.9))
    validator = Validator(ssm, sys, args, normalize=args.normalize)
    callback = TSCallback(validator, args.logdir)

    xpred = variable('xn')[:, :-1, :]
    xtrue = variable('X')
    loss = (xpred == xtrue) ^ 2
    loss.update_name('loss')

    obj = PenaltyLoss([loss], [])
    problem = Problem([ssm], obj)

    logout = ['loss', 'fd', 'mse_test', 'mae_test', 'r2_test', 'eval_mse']
    logger = MLFlowLogger(args, savedir=args.logdir, stdout=['train_loss', 'eval_mse'], logout=logout)
    trainer = Trainer(problem, train_data, dev_data, test_data, opt, logger,
                      callback=callback,
                      epochs=args.epochs,
                      patience=args.epochs*args.iterations,
                      train_metric='train_loss',
                      dev_metric='dev_loss',
                      test_metric='test_loss',
                      eval_metric=args.eval_metric)

    # Model training
    lr = args.lr
    nsteps = args.nsteps
    for i in range(args.iterations):
        print(f'training {nsteps} objective, lr={lr}')
        validator.figname = str(nsteps)
        best_model = trainer.train()
        trainer.model.load_state_dict(best_model)
        lr /= 2.0
        nsteps *= 2
        nx, train_data, dev_data, test_data = get_data(nsteps, sys, args.nsim, args.batch_size)
        trainer.train_data, trainer.dev_data, trainer.test_data = train_data, dev_data, test_data
        opt.param_groups[0]['lr'] = lr

    # Test set results
    x0 = sys.get_x0()
    sim = sys.simulate(ts=args.ts, nsim=1000, x0=x0)
    X = torch.tensor(sim['X'], dtype=torch.float32)
    X = X.view(1, *X.shape)
    if args.normalize:
        X = sys.normalize(X, key='X')

    true_traj = X
    with torch.no_grad():
        pred_traj = ssm.forward({'X': X, 'xn': X[:, 0:1, :]})['xn'][:, :-1, :]
    if args.normalize:
        pred_traj = sys.denormalize(pred_traj, key='X')
        true_traj = sys.denormalize(true_traj, key='X')

    pred_traj = pred_traj.detach().numpy().reshape(-1, nx)
    true_traj = true_traj.detach().numpy().reshape(-1, nx)

    mae = sklearn.metrics.mean_absolute_error(true_traj, pred_traj)
    mse = sklearn.metrics.mean_squared_error(true_traj, pred_traj, squared=False)
    r2 = sklearn.metrics.r2_score(true_traj, pred_traj)
    print(f'mae: {mae}\tmse: {mse}\tr2: {r2}')
    logger.log_metrics({f'mse_test': mse,
                        f'mae_test': mae,
                        f'r2_test': r2})

    np.save(os.path.join(args.logdir, f'test_true_loop.npy'), true_traj)
    np.save(os.path.join(args.logdir, f'test_pred_loop.npy'), pred_traj)

    pred_traj, true_traj = pred_traj.transpose(1, 0), true_traj.transpose(1, 0)

    figsize = 25
    plt.xticks(fontsize=figsize)
    fig, ax = plt.subplots(nx, figsize=(figsize, figsize))
    labels = [f'$y_{k}$' for k in range(len(true_traj))]
    for row, (t1, t2, label) in enumerate(zip(true_traj, pred_traj, labels)):
        if nx > 1:
            axe = ax[row]
        else:
            axe = ax
        axe.set_ylabel(label, rotation=0, labelpad=20, fontsize=figsize)
        axe.plot(t1, label='True', c='c')
        axe.plot(t2, label='Pred', c='m')
        axe.tick_params(labelbottom=False, labelsize=figsize)
    axe.tick_params(labelbottom=True, labelsize=figsize)
    axe.legend(fontsize=figsize)
    plt.tight_layout()
    plt.savefig(os.path.join(args.logdir, '../open_loop.png'))
    logger.log_artifacts({})


