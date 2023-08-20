"""
Learning neural Lyapunov functions for stability certificates of dynamical systems


TODO: two versions:

1, Neural Lyapunov dynamics: to certify stability of autonomous system
or closed-loop system from given rollouts of state trajectories

2, Neural Lyapunov DPC: to learn control policy and certify stability
of closed-loop system

"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import neuromancer.psl as psl
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.dynamics import ode, integrators
from neuromancer.plot import pltCL, pltPhase


def plot_Lyapunov(net, trajectory=None, xmin=-2, xmax=2, save_path=None):
    # sample state sapce and get function values
    x = torch.arange(xmin, xmax, 0.1)
    y = torch.arange(xmin, xmax, 0.1)
    xx, yy = torch.meshgrid(x, y)
    features = torch.stack([xx, yy]).transpose(0, 2)
    uu = net(features)
    plot_u = uu.detach().numpy()[:,:,0]
    plot_x = xx.detach().numpy()
    plot_y = yy.detach().numpy()

    # plot surface of the lyapunov function
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(plot_x, plot_y, plot_u,
                           cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    # plot contours
    ax.contour(plot_x, plot_y, plot_u, 20, offset=-1,
               cmap=cm.viridis, linestyles="solid")

    # plot minimum of the lyapunov function
    min_idx = np.where(plot_u == np.min(plot_u))
    point_u = plot_u[min_idx]
    point_x = plot_x[min_idx]
    point_y = plot_y[min_idx]
    ax.scatter(point_x, point_y, point_u, color='red', s=200,
               marker='o')

    # plot sample trajectory
    if trajectory is not None:
        u_traj = net(trajectory)
        # ax.plot(trajectory[:, 1].detach().numpy(),
        #         trajectory[:, 0].detach().numpy(),
        #         u_traj[:, 0].detach().numpy(),
        #         'r--', linewidth=5,
        #         label='sample trajectory')
        ax.plot(trajectory[:, 1].detach().numpy(),
                trajectory[:, 0].detach().numpy(),
                'k--', linewidth=4,
                label='sample trajectory')

    # set labels
    ax.set(ylabel='$x_1$')
    ax.set(xlabel='$x_2$')
    ax.set(zlabel='$u$')
    ax.set(title='Lyapunov function')

    if save_path is not None:
        plt.savefig(save_path+'/lyapunov.pdf')



if __name__ == "__main__":

    """
    # # #  Obtain trajectories from the system model
    """
    model = psl.autonomous.Brusselator1D(backend='numpy')

    # select parameters
    model.ts = 0.2
    model.a = 1.0
    model.b = 1.6

    # problem dimensions and number of samples
    nx = model.nx  # number of states
    nsteps = 100  # rollout horizon
    n_samples = 500  # number of sampled scenarios

    # sample random initial conditions
    init_cond = np.random.uniform(low=0.0, high=4.0, size=[n_samples, nx])
    # obtain rollouts
    Rollouts = []
    for k in range(n_samples):
        rollout = model.simulate(nsim=nsteps, x0=init_cond[k, :])
        Rollouts.append(rollout['X'])
    state_rollouts = np.stack(Rollouts)
    # visualize single rollout
    pltPhase(X=rollout['X'])

    """
    # # #  Datasets 
    """
    # train/dev set ratio
    train_ratio = 0.8
    n_samples_train = int(n_samples*train_ratio)
    X_train = torch.tensor(state_rollouts[:n_samples_train, :, :])
    X_dev = torch.tensor(state_rollouts[n_samples_train:, :, :])
    # Training dataset
    train_data = DictDataset({'x': X_train}, name='train')
    # Development dataset
    dev_data = DictDataset({'x': X_dev}, name='dev')
    # torch dataloaders
    batch_size = 10
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               collate_fn=train_data.collate_fn,
                                               shuffle=False)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size,
                                             collate_fn=dev_data.collate_fn,
                                             shuffle=False)

    """
    # # #  Neural Lyapunov function candidate 
    """
    # Input convex neural network (ICNN) from https://arxiv.org/abs/2001.06116
    g = blocks.InputConvexNN(nx, 1, hsizes=[20] * 4)
    # positive-definite layer if eps=0.0
    lyap_net = blocks.PosDef(g, eps=0.0)
    lyap_net.requires_grad_(True)
    # symbolic wrapper for Lyapunov neural net
    lyap_function = Node(lyap_net, ['x'], ['V'], name='Lyapunov')

    """
    # # #  Define objectives of the learning problem
    """
    # variable: Lyapunov function value
    V = variable("V")
    # objective: discrete-time Lyapunov condition
    Lyapunov_condition = 1.0 * (V[:, 1:, :] - V[:, :-1, :] < - 0.1)

    # create constrained optimization loss
    loss = PenaltyLoss(objectives=[Lyapunov_condition], constraints=[])
    # construct optimization problem
    problem = Problem(nodes=[lyap_function], loss=loss)
    # plot computational graph
    # problem.show()

    """
    # # #  Solving the problem 
    """
    optimizer = torch.optim.AdamW(problem.parameters(), lr=0.002)
    #  Neuromancer trainer
    trainer = Trainer(
        problem,
        train_loader, dev_loader,
        optimizer,
        epochs=200,
        train_metric='train_loss',
        eval_metric='dev_loss',
        warmup=200,
    )
    # Train control policy
    best_model = trainer.train()
    # load best trained model
    trainer.model.load_state_dict(best_model)

    """
    Visualize learned Lyapunov function with a sample trajectory
    """
    plot_Lyapunov(lyap_net, trajectory=X_train[0,:,:],
                  xmin=0, xmax=4, save_path=None)


