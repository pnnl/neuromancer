"""
This is a training script for system id using NODEs with Euler integration for any building envelope psl models.

A few things that are included:
    + Multiple objectives for finite difference, one step ahead, nstep ahead, latent and observed state MSE
    + Configurable curriculum learning
    + Training data with many sampled initial conditions for independent n-step simulations as well as a single long trajectory
"""

"""
python building_system_train.py -model node -normalize -epochs 100 -nsteps 128 -iterations 3 -lr 0.01 -qstep 1. -hsize 64 -nlayers 3

"""
import dill
import os

import sklearn
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import torch.nn as nn

from neuromancer.modules.activations import SoftExponential
from neuromancer.psl.building_envelope import systems
from neuromancer.problem import Problem
from neuromancer.loggers import MLFlowLogger
from neuromancer.trainer import Trainer
from neuromancer.constraint import Loss, variable
from neuromancer.loss import PenaltyLoss
from neuromancer.dataset import DictDataset
from neuromancer.modules.blocks import MLP
from neuromancer.dynamics.integrators import Euler
from neuromancer.system import Node, System
from neuromancer.callbacks import Callback
import neuromancer.psl as psl

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


class TSCallback(Callback):
    def __init__(self, validator, logdir):
        self.validator = validator
        self.logdir = logdir

    def begin_eval(self, trainer, output):
        mse, sim, real = self.validator()
        output['eval_mse'] = mse
        plot_traj(real, sim, figname=f'{self.logdir}/{self.validator.figname}steps_open.png')
        plt.close()


class Validator:

    def __init__(self, netG, sys, args):
        """
        Used for evaluating model performance

        :param netG: (nn.Module) Some kind of neural network state space model
        :param sys: (psl.ODE_NonAutonomous) Ground truth ODE system
        :param normalize: (bool) Whether to normalized data. Will denorm before plotting and loss calculation if True
        """
        self.figname = ''
        self.x0s = [sys.get_x0() for i in range(10)]
        self.sys = sys
        Y, U, D = [], [], []
        nsim = (args.nsim//args.nsteps)*args.nsteps
        for x0 in self.x0s:
            sim = sys.simulate(nsim=500, x0=x0, U=sys.get_U(500+1, type=args.input))
            Y.append(torch.tensor(sim['Y'], dtype=torch.float32))
            U.append(torch.tensor(sim['U'], dtype=torch.float32))
            D.append(torch.tensor(sim['D'], dtype=torch.float32))

            Y[-1] = sys.normalize(Y[-1], key='Y')
            U[-1] = sys.normalize(U[-1], key='U')
            D[-1] = sys.normalize(D[-1], key='D')

        def mse_loss(input, target):
            return torch.mean((input - target) ** 2, axis=(1, 2))

        self.mse = mse_loss

        self.reals = {'Y': torch.stack(Y), 'U': torch.stack(U), 'D': torch.stack(D)}
        self.netG = netG

    def __call__(self):
        """
        Runs the model in it's current state on 10 validation set trajectories with
        different initial conditions and control sequences.

        :return: truncated_mse, mse over all validation rollouts
        and x, x_pred trajectories for best validation rollout
        """
        with torch.no_grad():
            simulation = self.netG.forward(self.reals)
            y = self.reals['Y']
            yprime = simulation['yn'][:, :-1, :]
            yprime = torch.nan_to_num(yprime, nan=200000., posinf=None, neginf=None)
            y = self.sys.denormalize(y, key='Y')
            yprime = self.sys.denormalize(yprime, key='Y')
            mses = self.mse(y, yprime)
        best = np.argmax(mses)
        return mses.mean(), yprime[best].detach().numpy(), y[best].detach().numpy()


def get_data(nsteps, sys, nsim, bs):
    """
    :param nsteps: (int) Number of timesteps for each batch of training data
    :param sys: (psl.ODE_NonAutonomous)

    """
    nsim = (nsim//nsteps)*nsteps
    sim = sys.simulate(nsim=nsim, x0=sys.get_x0(), U=sys.get_U(nsim+1))
    ny, nu, nd = sim['Y'].shape[-1], sim['U'].shape[-1], sim['D'].shape[-1]
    Y = sys.normalize(sim['Y'], key='Y')
    U = sys.normalize(sim['U'], key='U')
    D = sys.normalize(sim['D'], key='D')
    Y, U, D = Y.reshape(nsim//nsteps, nsteps, ny), U.reshape(nsim//nsteps, nsteps, nu), D.reshape(nsim//nsteps, nsteps, nd)
    sys.show({'Y': Y[0], 'U': U[0], 'D': D[0]}, figname=f'{sys}.png')
    Y = torch.tensor(Y, dtype=torch.float32)
    U = torch.tensor(U, dtype=torch.float32)
    D = torch.tensor(D, dtype=torch.float32)


    train_data = DictDataset({'Y': Y, 'U': U, 'D': D, 'yn': Y[:, 0:1, :]}, name='train')

    train_loader = DataLoader(train_data, batch_size=bs,
                              collate_fn=train_data.collate_fn, shuffle=True)
    dev_data = DictDataset({'Y': Y[0:1], 'U': U[0:1], 'D': D[0:1], 'yn': Y[0:1, 0:1, :]}, name='dev')
    dev_loader = DataLoader(dev_data, num_workers=1, batch_size=bs,
                            collate_fn=dev_data.collate_fn, shuffle=False)
    test_loader = dev_loader
    return ny, nu, nd, train_loader, dev_loader, test_loader


def get_node(ny, nu, nd, args, sys):
    integrator = EulerIntegrator(ny, nu, nd, args.hsize, args.nlayers, torch.tensor(args.ts))
    nodes = [Node(integrator, ['yn', 'Y', 'U', 'D'], ['yn', 'ystep'])]
    system = System(nodes, nstep_key='Y', init_func=init)
    return system


def get_blackbox_ssm(ny, nu, nd, args, sys):
    """
    y_{t+1} = f_y(f_x(x_t) + f_u(u_t) + f_d(d_t))
    y_{t+1} = f(x_t, u_t, d_t)

    """
    map = MLP(ny + nu + nd, ny, bias=True,
              linear_map=nn.Linear, nonlin=SoftExponential,
              hsizes=[args.hsize for h in range(args.nlayers)])
    ssm1 = Node(map, ['xn'], ['yn'])
    ssm2 = Node(map, ['xs'], ['ystep'])
    cat = Node(lambda yn, ys, u, d: (torch.cat([yn, u, d], dim=-1),
                                     torch.cat([ys, u, d], dim=-1)), ['yn', 'Y', 'U', 'D'], ['xn', 'xs'])
    system = System([cat, ssm1, ssm2], nstep_key='Y', init_func=init)
    return system


class BilinearMap(nn.Module):

    def __init__(self, sys):
        super().__init__()
        self._rho, self._time_reg, self._cp, self.n_mf, self.n_dT = sys.rho, sys.time_reg, sys.cp, sys.n_mf, sys.n_dT
        print(self.n_dT, self.n_mf)
        self.rho = torch.nn.Parameter(torch.randn(1))
        self.time_reg = torch.nn.Parameter(torch.randn(1))
        self.cp = torch.nn.Parameter(torch.randn(1))

    def forward(self, u):
        q = self.rho * self.cp * self.time_reg * u
        return q


class Bilinear(nn.Module):

    def __init__(self, fu, fq, fd, fx, fy, state_estimator):
        super().__init__()
        self.fu, self.fq, self.fd, self.fx, self.fy, self.state_estimator = fu, fq, fd, fx, fy, state_estimator

    def forward(self, y, u, d):
        fu = self.fu(self.fq(u))
        fd = self.fd(d)
        fx = self.fx(self.state_estimator(y))

        x = fu + fd + fx
        return self.fy(x)


def get_bilinear_ssm(ny, nu, nd, args, psl_sys):
    """
    y_{t+1} = f_y(f_x(x_t) + f_u(u_t) + f_d(d_t))
    y_{t+1} = f(x_t, u_t, d_t)

    """
    nx = psl_sys.nx
    fu = torch.nn.Linear(sys.n_mf, nx)
    fq = BilinearMap(psl_sys)
    fd = torch.nn.Linear(nd, nx)
    fx = torch.nn.Linear(nx, nx)
    fy = torch.nn.Linear(nx, ny)
    state_estimator = MLP(ny, nx, bias=True,
                          linear_map=nn.Linear, nonlin=SoftExponential,
                          hsizes=[args.hsize for h in range(args.nlayers)])
    map = Bilinear(fu, fq, fd, fx, fy, state_estimator)
    ssm1 = Node(map, ['yn', 'U', 'D'], ['yn'])
    ssm2 = Node(map, ['Y', 'U', 'D'], ['ystep'])
    system = System([ssm1, ssm2], nstep_key='Y', init_func=init)
    return system


class EulerIntegrator(nn.Module):
    """
    Simple black-box NODE
    """
    def __init__(self, ny, nu, nd, hsize, nlayers, ts):
        super().__init__()
        self.dy = MLP(ny + nu + nd, ny, bias=True, linear_map=nn.Linear, nonlin=SoftExponential,
                      hsizes=[hsize for h in range(nlayers)])
        interp_u = lambda tq, t, u: u
        self.integrator = Euler(self.dy, h=0.001, interp_u=interp_u)
        self.bias = nn.Parameter(torch.randn(ny), requires_grad=True)

    def forward(self, yn, ystep, u, d):
        """

        :param xn: (Tensor, shape=(batchsize, nx)) State
        :param u: (Tensor, shape=(batchsize, nu)) Control action
        :return: (Tensor, shape=(batchsize, nx)) xn+1
        """
        return self.integrator(yn, u=torch.cat([u, d], dim=-1)), self.integrator(ystep, u=torch.cat([u, d], dim=-1)) + self.bias


class LinearModel(nn.Module):
    def __init__(self, nx, nu, nd):
        super().__init__()
        self.fu = nn.Linear(nu, nx, bias=True)
        self.fx = nn.Linear(nx, nx, bias=True)
        self.fd = nn.Linear(nd, nx, bias=True)
        self.bias = nn.Parameter(torch.randn(nx), requires_grad=True)

    def forward(self, xn, xstep, u, d):
        """

        :param xn: (Tensor, shape=(batchsize, nx)) State
        :param u: (Tensor, shape=(batchsize, nu)) Control action
        :return: (Tensor, shape=(batchsize, nx)) xn+1
        """
        ud = self.fu(u) + self.fd(d)
        return self.fx(xn) + ud + self.bias, self.fx(xstep) + ud + self.bias


def get_linear(nx, nu, nd, args, sys):
    m = LinearModel(nx, nu, nd)
    nodes = [Node(m, ['yn', 'Y', 'U', 'D'], ['yn', 'ystep'])]
    system = System(nodes, nstep_key='Y', init_func=init)
    return system


def init(data):
    """
    Any nodes in the graph that are start nodes will need some data initialized.
    Here is an example of initializing an x0 entry in the input_dict.

    Provide in base class analysis of computational graph. Label the source nodes. Keys for source nodes have to
    be in the data.
    """
    data['yn'] = data['Y'][:, 0:1, :]
    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-system', default='LinearReno_full', choices=[k for k in systems],
                        help='You can use any of the systems from psl.nonautonomous with this script')
    parser.add_argument('-epochs', type=int, default=1000,
                        help='Number of epochs of training.')
    parser.add_argument('-lr', type=float, default=0.001,
                        help='Learning rate for gradient descent.')
    parser.add_argument('-nsteps', type=int, default=2,
                        help='Prediction horizon for optimization objective. During training will roll out for nsteps from and initial condition')
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-nsim', type=int, default=5000,
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
    parser.add_argument('-nlayers', type=int, default=3, help='Number of hidden layers for MLP')
    parser.add_argument('-iterations', type=int, default=5,
                        help='How many episodes of curriculum learning by doubling the prediction horizon and halving the learn rate each episode')
    parser.add_argument('-eval_metric', type=str, default='eval_mse')
    parser.add_argument('-model', default='node')
    parser.add_argument('-qstep', type=float, default=1.)
    parser.add_argument('-input', default='sines')
    parser.add_argument('-optimizer', default='adamw')
    parser.add_argument('-activation', default='softexp')


    args = parser.parse_args()
    os.makedirs(args.logdir, exist_ok=True)

    get_model = {'node': get_node, 'linear': get_linear, 'blackbox': get_blackbox_ssm,
                 'bilinear': get_bilinear_ssm}
    sys = systems[args.system]()
    args.ts = sys.ts
    x0 = sys.get_x0()
    test_sim = sys.simulate(nsim=1000, x0=x0,
                            U=psl.signals.sines(1000 + 1, sys.nu, min=0.0, max=(sys.stats['U']['max'] + sys.stats['U']['min'])/2.))
    ny, nu, nd, train_data, dev_data, test_data = get_data(args.nsteps, sys, args.nsim, args.batch_size)

    ssm = get_model[args.model](ny, nu, nd, args, sys)
    # opt = optimizers[args.optimizer](ssm.parameters, args.lr)
    # opt = optim.AdamW(ssm.parameters(), args.lr, betas=(0.0, 0.9))
    opt = optim.AdamW(ssm.parameters(), args.lr)

    validator = Validator(ssm, sys, args)
    callback = TSCallback(validator, args.logdir)
    objectives = []

    xpred = variable('yn')[:, :-1, :]
    xstep = variable('ystep')

    xtrue = variable('Y')
    loss1 = (xpred == xtrue) ^ 2
    loss1.update_name('nstep_mse')
    loss2 = args.qstep*(xstep[:, :-1, :] == xtrue[:, 1:, :]) ^ 2
    loss2.update_name('1step_mse')

    obj = PenaltyLoss([loss1, loss2], [])
    problem = Problem([ssm], obj)

    print(type(train_data.dataset.datadict), type(dev_data.dataset.datadict), type(test_data.dataset.datadict))
    logout = ['nstep_mse', '1step_mse', 'mse_test', 'mae_test', 'r2_test', 'eval_mse']
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
        # lr /= 2.
        nsteps *= 2
        ny, nu, nd, train_data, dev_data, test_data = get_data(nsteps, sys, args.nsim, args.batch_size)
        trainer.train_data, trainer.dev_data, trainer.test_data = train_data, dev_data, test_data
        # opt.param_groups[0]['lr'] = lr
        # exit()

    # Test set results

    sim = test_sim
    Y = torch.tensor(sim['Y'], dtype=torch.float32)
    Y = Y.view(1, *Y.shape)
    U = torch.tensor(sim['U'], dtype=torch.float32)
    U = U.view(1, *U.shape)
    D = torch.tensor(sim['D'], dtype=torch.float32)
    D = D.view(1, *D.shape)
    Y = sys.normalize(Y, key='Y')
    U = sys.normalize(U, key='U')
    D = sys.normalize(D, key='D')

    true_traj = Y
    with torch.no_grad():
        pred_traj = ssm.forward({'Y': Y, 'U': U, 'D': D})['yn'][:, 1:, :]
    pred_traj = sys.denormalize(pred_traj, key='Y')
    true_traj = sys.denormalize(true_traj, key='Y')
    pred_traj = pred_traj.detach().numpy().reshape(-1, ny)
    true_traj = true_traj.detach().numpy().reshape(-1, ny)

    mae = sklearn.metrics.mean_absolute_error(true_traj, pred_traj)
    mse = sklearn.metrics.mean_squared_error(true_traj, pred_traj, squared=False)
    r2 = sklearn.metrics.r2_score(true_traj, pred_traj)
    print(f'mae: {mae}\tmse: {mse}\tr2: {r2}')
    logger.log_metrics({f'mse_test': mse,
                        f'mae_test': mae,
                        f'r2_test': r2})
    textstr = f'MSE: {mse: .3f}\nMAE: {mae: .3f}\nR^2: {r2: .3f}'
    figsize = 25
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    plt.xticks(fontsize=figsize)
    fig, ax = plt.subplots(sys.ny + sys.nU, figsize=(figsize, figsize))
    labels = [f'$y_{k}$' for k in range(len(true_traj))]

    np.save(os.path.join(args.logdir, f'test_true_loop.npy'), true_traj)
    np.save(os.path.join(args.logdir, f'test_pred_loop.npy'), pred_traj)

    pred_traj, true_traj = pred_traj.transpose(1, 0), true_traj.transpose(1, 0)

    figsize = 25
    plt.xticks(fontsize=figsize)
    fig, ax = plt.subplots(ny, figsize=(figsize, figsize))
    labels = [f'$y_{k}$' for k in range(len(true_traj))]
    for row, (t1, t2, label) in enumerate(zip(true_traj, pred_traj, labels)):
        if ny > 1:
            axe = ax[row]
        else:
            axe = ax
        if row == 0:
            axe.text(0.05, 0.95, textstr, transform=axe.transAxes, fontsize=14,
                     verticalalignment='top', bbox=props)
        axe.set_ylabel(label, rotation=0, labelpad=20, fontsize=figsize)
        axe.plot(t1, label='True', c='c')
        axe.plot(t2, label='Pred', c='m')
        axe.tick_params(labelbottom=False, labelsize=figsize)
    axe.tick_params(labelbottom=True, labelsize=figsize)
    axe.legend(fontsize=figsize)
    plt.tight_layout()
    plt.savefig(os.path.join(args.logdir, '../open_loop.png'))
    torch.save(sys, os.path.join(args.logdir, 'sys.pth'), pickle_module=dill)
    logger.log_artifacts({})


