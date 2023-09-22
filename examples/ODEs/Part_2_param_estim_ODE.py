"""
Parameter estimation for a 1D Brusselator system.
"""

import torch
from neuromancer.psl import plot
from neuromancer import psl
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader

from neuromancer.system import Node, System
from neuromancer.dynamics import integrators, ode
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.loggers import BasicLogger
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss


def get_data(sys, nsim, nsteps, bs):
    """
    :param nsteps: (int) Number of timesteps for each batch of training data
    :param sys: (psl.system)
    :param normalize: (bool) Whether to normalize the data

    """
    train_sim, dev_sim, test_sim = [sys.simulate(nsim=nsim, ts=0.05) for i in range(3)]
    nx = sys.nx
    nbatch = nsim//nsteps
    length = (nsim//nsteps) * nsteps

    trainX = train_sim['X'][:length].reshape(nbatch, nsteps, nx)
    trainX = torch.tensor(trainX, dtype=torch.float32)
    train_data = DictDataset({'X': trainX, 'xn': trainX[:, 0:1, :]}, name='train')
    train_loader = DataLoader(train_data, batch_size=bs,
                              collate_fn=train_data.collate_fn, shuffle=True)

    devX = dev_sim['X'][:length].reshape(nbatch, nsteps, nx)
    devX = torch.tensor(devX, dtype=torch.float32)
    dev_data = DictDataset({'X': devX, 'xn': devX[:, 0:1, :]}, name='dev')
    dev_loader = DataLoader(dev_data, batch_size=bs,
                            collate_fn=dev_data.collate_fn, shuffle=True)

    testX = test_sim['X'][:length].reshape(1, nsim, nx)
    testX = torch.tensor(testX, dtype=torch.float32)
    test_data = {'X': testX, 'xn': testX[:, 0:1, :]}

    return train_loader, dev_loader, test_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', type=int, default=100)

    args = parser.parse_args()

    torch.manual_seed(0)

    # %%  ground truth system
    system = psl.systems['Brusselator1D']
    ts = 0.05
    modelSystem = system()
    nx = modelSystem.nx
    raw = modelSystem.simulate(nsim=1000, ts=ts)
    plot.pltOL(Y=raw['Y'])
    plot.pltPhase(X=raw['Y'])

    # get datasets
    train_loader, dev_loader, test_data = get_data(modelSystem, 500, 2, 32)

    # construct ODE model in Neuromancer
    brussels = ode.BrusselatorParam()
    fxRK4 = integrators.RK4(brussels, h=ts)
    dynamics_model = System([Node(fxRK4, ['xn'], ['xn'])])

    # %% Constraints + losses:
    x = variable("X")
    xhat = variable('xn')[:, :-1, :]

    xFD = (x[:, 1:, :] - x[:, :-1, :])
    xhatFD = (xhat[:, 1:, :] - xhat[:, :-1, :])

    fd_loss = 2.0*((xFD == xhatFD)^2)
    fd_loss.name = 'FD_loss'

    reference_loss = ((xhat == x)^2)
    reference_loss.name = "ref_loss"

    # %%
    objectives = [reference_loss, fd_loss]
    constraints = []
    # create constrained optimization loss
    loss = PenaltyLoss(objectives, constraints)
    # construct constrained optimization problem
    problem = Problem([dynamics_model], loss)
    # plot computational graph

    # %%
    optimizer = torch.optim.Adam(problem.parameters(), lr=0.1)
    logger = BasicLogger(args=None, savedir='test', verbosity=1,
                         stdout=['dev_loss', 'train_loss'])

    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        test_data,
        optimizer,
        patience=10,
        warmup=10,
        epochs=args.epochs,
        eval_metric="dev_loss",
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="dev_loss",
        logger=logger,
    )
    # %%
    best_model = trainer.train()
    problem.load_state_dict(best_model)
    # %%
    # %%
    print('alpha = '+str(brussels.alpha.item()))
    print('beta = '+str(brussels.beta.item()))

    # Test set results
    test_outputs = dynamics_model(test_data)

    pred_traj = test_outputs['xn'][:, :-1, :]
    true_traj = test_data['X']
    pred_traj = pred_traj.detach().numpy().reshape(-1, nx)
    true_traj = true_traj.detach().numpy().reshape(-1, nx)
    pred_traj, true_traj = pred_traj.transpose(1, 0), true_traj.transpose(1, 0)

    figsize = 25
    fig, ax = plt.subplots(nx, figsize=(figsize, figsize))
    labels = [f'$y_{k}$' for k in range(len(true_traj))]
    for row, (t1, t2, label) in enumerate(zip(true_traj, pred_traj, labels)):
        if nx > 1:
            axe = ax[row]
        else:
            axe = ax
        axe.set_ylabel(label, rotation=0, labelpad=20, fontsize=figsize)
        axe.plot(t1, 'c', linewidth=4.0, label='True')
        axe.plot(t2, 'm--', linewidth=4.0, label='Pred')
        axe.tick_params(labelbottom=False, labelsize=figsize)
    axe.tick_params(labelbottom=True, labelsize=figsize)
    axe.legend(fontsize=figsize)
    axe.set_xlabel('$time$', fontsize=figsize)
    plt.tight_layout()
    # plt.savefig('open_loop.png')