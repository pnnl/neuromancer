"""
Learning the parameters of Universal differential equations from time series data
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from neuromancer.system import Node, System
from neuromancer.dynamics import integrators, ode
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.modules import blocks
from neuromancer.psl import plot
from neuromancer import psl


def get_data(sys, nsim, nsteps, ts, bs):
    """
    :param nsteps: (int) Number of timesteps for each batch of training data
    :param sys: (psl.system)
    :param ts: (float) step size
    :param bs: (int) batch size

    """
    train_sim, dev_sim, test_sim = [sys.simulate(nsim=nsim, ts=ts) for i in range(3)]
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


class LotkaVolterraHybrid(ode.ODESystem):

    def __init__(self, block, insize=2, outsize=2):
        """

        :param block:
        :param insize:
        :param outsize:
        """
        super().__init__(insize=insize, outsize=outsize)
        self.block = block
        self.alpha = nn.Parameter(torch.tensor([.10]), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor([.10]), requires_grad=True)
        self.delta = nn.Parameter(torch.tensor([.10]), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor([.10]), requires_grad=True)
        assert self.block.in_features == 2
        assert self.block.out_features == 1

    def ode_equations(self, x):
        x1 = x[:, [0]]
        x2 = x[:, [-1]]
        dx1 = self.alpha*x1 - self.beta*self.block(x)
        dx2 = self.delta*self.block(x) - self.gamma*x2
        return torch.cat([dx1, dx2], dim=-1)


if __name__ == '__main__':
    torch.manual_seed(0)

    # %%  ground truth system
    system = psl.systems['LotkaVolterra']
    modelSystem = system()
    ts = modelSystem.ts
    nx = modelSystem.nx
    raw = modelSystem.simulate(nsim=1000, ts=ts)
    plot.pltOL(Y=raw['Y'])
    plot.pltPhase(X=raw['Y'])

    # get datasets
    nsim = 2000
    nsteps = 2
    bs = 10
    train_loader, dev_loader, test_data = \
        get_data(modelSystem, nsim, nsteps, ts, bs)

    # construct UDE model in Neuromancer
    net = blocks.MLP(2, 1, bias=True,
                     linear_map=torch.nn.Linear,
                     nonlin=torch.nn.GELU,
                     hsizes=4*[20])
    fx = LotkaVolterraHybrid(net)
    # integrate UDE model
    fxRK4 = integrators.RK4(fx, h=ts)
    # create symbolic UDE model
    ude = Node(fxRK4, ['xn'], ['xn'], name='UDE')
    dynamics_model = System([ude])

    # %% Constraints + losses:
    x = variable("X")
    xhat = variable('xn')[:, :-1, :]

    # trajectory tracking loss
    reference_loss = (xhat == x)^2
    reference_loss.name = "ref_loss"

    # finite difference variables
    xFD = (x[:, 1:, :] - x[:, :-1, :])
    xhatFD = (xhat[:, 1:, :] - xhat[:, :-1, :])

    # finite difference loss
    fd_loss = 2.0*((xFD == xhatFD)^2)
    fd_loss.name = 'FD_loss'

    # %%
    objectives = [reference_loss, fd_loss]
    constraints = []
    # create constrained optimization loss
    loss = PenaltyLoss(objectives, constraints)
    # construct constrained optimization problem
    problem = Problem([dynamics_model], loss)
    # plot computational graph
    problem.show()

    # %%
    optimizer = torch.optim.Adam(problem.parameters(), lr=0.001)

    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        test_data,
        optimizer,
        patience=50,
        warmup=100,
        epochs=500,
        eval_metric="dev_loss",
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="dev_loss",
    )
    # %%
    best_model = trainer.train()
    problem.load_state_dict(best_model)
    # %%

    # evaluate learned parameters
    print('Learned parameter a=', float(fx.alpha))
    print('Learned parameter b=', float(fx.beta))
    print('Learned parameter c=', float(fx.gamma))
    print('Learned parameter d=', float(fx.delta))

    print('True parameter a=', float(modelSystem.a))
    print('True parameter b=', float(modelSystem.b))
    print('True parameter c=', float(modelSystem.c))
    print('True parameter d=', float(modelSystem.d))

    # evaluate learned black box block
    x1 = torch.arange(0., 150., 0.5)
    x2 = torch.arange(0., 150., 0.5)
    true_block = x1*x2
    learned_block = net(torch.stack([x1, x2]).T).squeeze()
    plt.figure()
    plt.plot(true_block.detach().numpy(), 'c',
             linewidth=4.0, label='True')
    plt.plot(learned_block.detach().numpy(), 'm--',
             linewidth=4.0, label='Learned')
    plt.legend(fontsize=25)

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
