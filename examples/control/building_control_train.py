"""
Test and validation sets are step function references
Length of test and validation sets: 1000

What to do if not learning well:
+ Increase network capacity
+ Fiddle with optimization hyperparameters
+ Fiddle with nonlinearity
+ Fiddle with normalization
+ Forecast disturbances
+ Moving horizon for y
"""
import os
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from neuromancer.activations import SoftExponential
from neuromancer import blocks
from neuromancer.psl.building_envelope import systems
from neuromancer.problem import Problem
from neuromancer.loggers import MLFlowLogger
from neuromancer.trainer import Trainer
from neuromancer.constraint import Loss, variable
from neuromancer.loss import PenaltyLoss
from neuromancer.dataset import DictDataset

import numpy as np
from neuromancer.system import Node, System
import torch


def get_data(nsteps, sys, nsim, bs, normalize=False):
    """
    Gets a reference trajectory by simulating the system with random initial conditions.

    :param nsteps: (int) Number of timesteps for each batch of training data
    :param sys: (psl.ODE_NonAutonomous)
    :param normalize: (bool) Whether to normalize the data

    """
    R, D, Dhidden = [], [], []
    for _ in range(nsim//nsteps):
        sim = sys.simulate(nsim=nsteps, x0=sys.get_x0())
        R.append(sim['Y'])
        D.append(sim['D'])
        Dhidden.append(sim['Dhidden'])
    R, D, Dhidden = sys.B.core.stack(R), sys.B.core.stack(D), sys.B.core.stack(Dhidden)
    sim = sys.simulate(nsim=(nsim//nsteps) * nsteps, x0=sys.get_x0())

    ny, nu, nd, ndh = sys.ny, sys.nU, 1, sys.nD
    r, d, dhidden = (sim['Y'].reshape(nsim//nsteps, nsteps, ny),
                     sim['D'].reshape(nsim//nsteps, nsteps, nd),
                     sim['Dhidden'].reshape(nsim//nsteps, nsteps, ndh))
    R, D, Dhidden = (torch.cat([R, r], axis=0),
                     torch.cat([D, d], axis=0),
                     torch.cat([Dhidden, dhidden], axis=0))
    R, D, Dhidden = (torch.tensor(R, dtype=torch.float32, requires_grad=False),
                     torch.tensor(D, dtype=torch.float32, requires_grad=False),
                     torch.tensor(Dhidden, dtype=torch.float32, requires_grad=False))
    if normalize:
        R = sys.normalize(R, key='Y')
        Dhidden = sys.normalize(Dhidden, key='D')
        D = Dhidden[:, :, sys.d_idx]

    U_upper = torch.tensor(sys.umax, dtype=torch.float32).view(1, 1, -1).expand(*R.shape[:-1], -1)
    U_lower = torch.tensor(sys.umin, dtype=torch.float32).view(1, 1, -1).expand(*R.shape[:-1], -1)

    train_data = DictDataset({'R': R, 'D': D, 'Dhidden': Dhidden, 'U_upper': U_upper, 'U_lower': U_lower}, name='train')
    train_loader = DataLoader(train_data, batch_size=bs,
                              collate_fn=train_data.collate_fn, shuffle=True)
    dev_data = DictDataset({'R': R[0:1], 'D': D[0:1], 'Dhidden': Dhidden[0:1], 'U_upper': U_upper[0:1], 'U_lower': U_lower[0:1]}, name='dev')
    dev_loader = DataLoader(dev_data, num_workers=1, batch_size=bs,
                            collate_fn=dev_data.collate_fn, shuffle=False)
    test_loader = dev_loader
    return train_loader, dev_loader, test_loader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-system', default='LinearReno_full', choices=[k for k in systems],
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
    parser.add_argument('-forecast', type=int, default=2, help='Number of lookahead steps for reference.')
    args = parser.parse_args()
    os.makedirs(args.logdir, exist_ok=True)

    def get_system(sys, forecast):
        """
        ny = observable state size
        ny*forecast = reference preview size
        nd = disturbance dimension
        2*nu = Upper and lower bounds on controls

        :param ny:
        :param nu:
        :param nd:
        :param forecast:
        :return:
        """
        """
        Inputs to the control policy:
        System state at current time or current history
        Disturbance at current time
        Constraints at current time
        Reference with forecast
        """
        ny, nu, nd = sys.ny, sys.nU, 1

        class Policy(nn.Module):

            def __init__(self, insize, outsize):
                super().__init__()
                self.net = blocks.MLP(insize, outsize, bias=True, linear_map=nn.Linear, nonlin=SoftExponential,
                                      hsizes=[args.hsize for h in range(args.nlayers)])

            def forward(self, y, D, U_upper, U_lower, R):
                features = torch.cat([y, D, U_upper, U_lower, R.reshape(R.shape[0], -1)], dim=-1)
                return self.net(features)

        insize = ny + ny*forecast + nd + 2*nu
        policy = Policy(insize, nu)
        policy_node = Node(policy, ['yn', 'D', 'U_upper', 'U_lower', 'R'], ['U'],
                           horizon_data=['R'], horizon=forecast)
        system_node = Node(sys, ['xn', 'U', 'Dhidden'], ['xn', 'yn'])

        def init(data):
            bs = data['R'].shape[0]
            xy = [sys.get_xy() for i in range(bs)]
            x = [torch.tensor(v[0], dtype=torch.float32) for v in xy]
            y = [torch.tensor(v[1], dtype=torch.float32) for v in xy]
            data['xn'] = torch.stack(x, dim=0).reshape(bs, 1, -1)
            data['yn'] = torch.stack(y, dim=0).reshape(bs, 1, -1)
            return data
        system = System([policy_node, system_node], init_func=init, nsteps=args.nsteps)
        return system, policy


    sys = systems[args.system](backend='torch', requires_grad=True)
    simulator, policy = get_system(sys, args.forecast)

    train_data, dev_data, test_data = get_data(args.nsteps+args.forecast, sys, args.nsim, args.batch_size, normalize=args.normalize)

    opt = optim.Adam(policy.parameters(), args.lr, betas=(0.0, 0.9))

    tru = variable('yn')[:, 1:, :]
    ref = variable('R')[:, :-args.forecast, :]
    u_hi = variable('U_upper')[:, :-args.forecast, :]
    u_lo = variable('U_lower')[:, :-args.forecast, :]
    u = variable('U')
    loss = (ref == tru) ^ 2
    loss.update_name('loss')

    c_upper = u < u_hi
    c_lower = u > u_lo
    obj = PenaltyLoss([loss], [])#, [0.01*c_upper, 0.01*c_lower])
    problem = Problem([simulator], obj)

    logout = ['loss']
    logger = MLFlowLogger(args, savedir=args.logdir, stdout=['train_loss', 'dev_loss'], logout=logout)
    trainer = Trainer(problem, train_data, dev_data, test_data, opt, logger,
                      epochs=args.epochs,
                      patience=args.epochs*args.iterations,
                      train_metric='train_loss',
                      dev_metric='dev_loss',
                      test_metric='test_loss',
                      eval_metric='dev_loss')

    best_model = trainer.train()
    trainer.model.load_state_dict(best_model)
    # # Model training
    # lr = args.lr
    # nsteps = args.nsteps
    # for i in range(args.iterations):
    #     print(f'training {nsteps} objective, lr={lr}')
    #     best_model = trainer.train()
    #     trainer.model.load_state_dict(best_model)
    #     lr /= 2.0
    #     nsteps *= 2
    #     nx, nu, train_data, dev_data, test_data = get_data(nsteps, sys, args.nsim, args.batch_size)
    #     trainer.train_data, trainer.dev_data, trainer.test_data = train_data, dev_data, test_data
    #     opt.param_groups[0]['lr'] = lr
