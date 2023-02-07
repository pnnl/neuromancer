import os

import numpy as np

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
from train import get_data, Validator
import psl
from neuromancer.constraint import variable
from neuromancer.integrators import integrators
from neuromancer.problem import Problem
from neuromancer.loggers import MLFlowLogger
from neuromancer.trainer import Trainer
from neuromancer.constraint import Loss
from util import TSCallback
from neuromancer.loss import PenaltyLoss
from models import get_model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', type=int, default=1000,
                        help='Number of epochs of training.')
    parser.add_argument('-system', choices=[k for k in psl.nonautonomous.systems], default='LorenzControl')
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-nsteps', type=int, default=4)
    parser.add_argument('-stepper', default='Euler', choices=[k for k in integrators])
    parser.add_argument('-batch_size', type=int, default=100)
    parser.add_argument('-nsim', type=int, default=1000)
    parser.add_argument('-q_mse', type=float, default=2.0)
    parser.add_argument('-logdir', default='test')
    parser.add_argument("-exp", type=str, default="test",
           help="Will group all run under this experiment name.")
    parser.add_argument("-location", type=str, default="mlruns",
           help="Where to write mlflow experiment tracking stuff")
    parser.add_argument("-run", type=str, default="neuromancer",
           help="Some name to tell what the experiment run was about.")
    parser.add_argument('-hsize', type=int, default=128, help='Size of hiddens states')
    parser.add_argument('-nlayers', type=int, default=4, help='Number of hidden layers for MLP')

    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    os.makedirs(args.logdir, exist_ok=True)

    sys = psl.nonautonomous.systems[args.system]()
    args.ts = sys.ts
    nx, nu, train_data, dev_data, test_data, stats = get_data(args.nsteps, sys, args.nsim, args.batch_size)
    ssm = get_model['node'](nx, nu, args)
    opt = optim.Adam(ssm.parameters(), args.lr, betas=(0.0, 0.9))
    validator = Validator(ssm, sys, args)
    callback = TSCallback(validator, args.logdir)

    objectives = []
    # mse_loss = mse_loss if args.scaled_loss else F.mse_loss
    # yhat = variable(f"X_nstep")
    # y = variable("X")
    # finite_difference_loss = args.q_fd * ((yhat[:, 1:, :] - yhat[:, :-1, :] == y[:, 1:, :] - y[:, :-1, :]) ^ 2)
    # finite_difference_loss.update_name('fd')
    # objectives.append(finite_difference_loss)
    # objectives.append(Loss(['X', 'X_step'], mse_loss, weight=args.q_mse_xstep, name='msexstep'))
    objectives.append(Loss(['X', 'X_nstep'], F.mse_loss, weight=2.0, name='mse'))
    # objectives.append(Loss(['Z', 'Z_step'], F.mse_loss, weight=args.q_mse_zstep, name='msezstep'))
    # objectives.append(Loss(['Z', 'Z_nstep'], F.mse_loss, weight=args.q_mse_znstep, name='mseznstep'))
    loss = PenaltyLoss(objectives, [])
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


    # lr = args.lr
    # nsteps = args.nsteps
    # ssm.nsteps = nsteps
    # for i in range(5):
    #     print(f'training {nsteps} objective, lr={lr}')
    #     validator.figname = str(nsteps)
    #     best_model = trainer.train()
    #     trainer.model.load_state_dict(best_model)
    #     lr/= 2.0
    #     nsteps *= 2
    #     nx, nu, train_data, dev_data, test_data, stats = get_data(nsteps, sys, args.nsim, args.batch_size)
    #     trainer.train_data, trainer.dev_data, trainer.test_data = train_data, dev_data, test_data
    #     ssm.nsteps = nsteps
    #     opt.param_groups[0]['lr'] = lr

    lr = args.lr
    nsteps = args.nsteps
    for i in range(5):
        print(f'training {nsteps} objective, lr={lr}')
        validator.figname = str(nsteps)
        best_model = trainer.train()
        trainer.model.load_state_dict(best_model)
        lr /= 2.0
        nsteps *= 2
        nx, nu, train_data, dev_data, test_data, stats = get_data(nsteps, sys, args.nsim, args.batch_size, stats=stats)
        trainer.train_data, trainer.dev_data, trainer.test_data = train_data, dev_data, test_data
        opt.param_groups[0]['lr'] = lr






