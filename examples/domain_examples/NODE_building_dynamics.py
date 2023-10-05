"""
Learning neural ODEs with exogenous inputs from time series data
"""

import torch
from neuromancer.psl import plot
from neuromancer import psl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from neuromancer.system import Node, System
from neuromancer.dynamics import integrators, ode
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.loggers import BasicLogger
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.modules import blocks


def normalize(x, mean, std):
    return (x - mean) / std

def denormalize(x, mean, std):
    return (x + mean) * std

def get_data(sys, nsim, nsteps, ts, bs):
    """
    :param nsteps: (int) Number of timesteps for each batch of training data
    :param sys: (psl.system)
    :param ts: (float) step size
    :param bs: (int) batch size

    """
    train_sim, dev_sim, test_sim = [sys.simulate(nsim=nsim, ts=ts) for i in range(3)]
    nx = sys.nx
    nu = sys.nu
    nd = sys.nd
    ny = sys.ny
    nbatch = nsim//nsteps
    length = (nsim//nsteps) * nsteps

    mean_x = modelSystem.stats['X']['mean']
    std_x = modelSystem.stats['X']['std']
    mean_y = modelSystem.stats['Y']['mean']
    std_y = modelSystem.stats['Y']['std']
    mean_u = modelSystem.stats['U']['mean']
    std_u = modelSystem.stats['U']['std']
    mean_d = modelSystem.stats['D']['mean']
    std_d = modelSystem.stats['D']['std']

    trainX = normalize(train_sim['X'][:length], mean_x, std_x)
    trainX = trainX.reshape(nbatch, nsteps, nx)
    trainX = torch.tensor(trainX, dtype=torch.float32)
    trainY = normalize(train_sim['Y'][:length], mean_y, std_y)
    trainY = trainY.reshape(nbatch, nsteps, ny)
    trainY = torch.tensor(trainY, dtype=torch.float32)
    trainU = normalize(train_sim['U'][:length], mean_u, std_u)
    trainU = trainU.reshape(nbatch, nsteps, nu)
    trainU = torch.tensor(trainU, dtype=torch.float32)
    trainD = normalize(train_sim['D'][:length], mean_d, std_d)
    trainD = trainD.reshape(nbatch, nsteps, nd)
    trainD = torch.tensor(trainD, dtype=torch.float32)
    train_data = DictDataset({'X': trainX, 'yn': trainY[:, 0:1, :],
                              'Y': trainY,
                              'U': trainU,
                              'D': trainD}, name='train')
    train_loader = DataLoader(train_data, batch_size=bs,
                              collate_fn=train_data.collate_fn, shuffle=True)

    devX = normalize(dev_sim['X'][:length], mean_x, std_x)
    devX = devX.reshape(nbatch, nsteps, nx)
    devX = torch.tensor(devX, dtype=torch.float32)
    devY = normalize(dev_sim['Y'][:length], mean_y, std_y)
    devY = devY.reshape(nbatch, nsteps, ny)
    devY = torch.tensor(devY, dtype=torch.float32)
    devU = normalize(dev_sim['U'][:length], mean_u, std_u)
    devU = devU[:length].reshape(nbatch, nsteps, nu)
    devU = torch.tensor(devU, dtype=torch.float32)
    devD = normalize(dev_sim['D'][:length], mean_d, std_d)
    devD = devD[:length].reshape(nbatch, nsteps, nd)
    devD = torch.tensor(devD, dtype=torch.float32)
    dev_data = DictDataset({'X': devX, 'yn': devY[:, 0:1, :],
                            'Y': devY,
                            'U': devU,
                            'D': devD}, name='dev')
    dev_loader = DataLoader(dev_data, batch_size=bs,
                            collate_fn=dev_data.collate_fn, shuffle=True)

    testX = normalize(test_sim['X'][:length], mean_x, std_x)
    testX = testX.reshape(1, nbatch*nsteps, nx)
    testX = torch.tensor(testX, dtype=torch.float32)
    testY = normalize(test_sim['Y'][:length], mean_y, std_y)
    testY = testY.reshape(1, nbatch*nsteps, ny)
    testY = torch.tensor(testY, dtype=torch.float32)
    testU = normalize(test_sim['U'][:length], mean_u, std_u)
    testU = testU.reshape(1, nbatch * nsteps, nu)
    testU = torch.tensor(testU, dtype=torch.float32)
    testD = normalize(test_sim['D'][:length], mean_d, std_d)
    testD = testD.reshape(1, nbatch*nsteps, nd)
    testD = torch.tensor(testD, dtype=torch.float32)
    test_data = {'X': testX, 'yn': testY[:, 0:1, :],
                 'Y': testY, 'U': testU, 'D': testD,
                 'name': 'test'}

    return train_loader, dev_loader, test_data


if __name__ == '__main__':
    torch.manual_seed(0)

    # select system:
    #   TwoTank, CSTR, SwingEquation,
    #   VanDerPolControl, IverSimple
    #   SEIR_population, LorenzControl

    # %%  ground truth system
    system = psl.systems['SimpleSingleZone']
    modelSystem = system()
    ts = modelSystem.ts
    nx = modelSystem.nx
    ny = modelSystem.ny
    nu = modelSystem.nu
    nd = modelSystem.nd

    raw = modelSystem.simulate(nsim=1000)
    plot.pltOL(Y=raw['Y'], U=raw['U'], D=raw['D'])

    # get datasets
    nsim = 2000
    nsteps = 50
    bs = 10
    train_loader, dev_loader, test_data = \
        get_data(modelSystem, nsim, nsteps, ts, bs)

    n_latent = 4  # latent state space dimension

    # latent state estimator
    encoder = blocks.MLP(ny, n_latent, bias=True,
                    linear_map=torch.nn.Linear,
                    nonlin=torch.nn.ReLU,
                    hsizes=[40])
    encode_sym = Node(encoder, ['yn'], ['xn'], name='encoder')

    # construct latent NODE model in Neuromancer
    fx = blocks.MLP(n_latent+nu+nd, n_latent, bias=True,
                     linear_map=torch.nn.Linear,
                     nonlin=torch.nn.Tanh,
                     hsizes=[40, 40])
    fxRK4 = integrators.RK4(fx, h=ts)
    model = Node(fxRK4, ['xn', 'U', 'D'], ['xn'], name='NODE')

    # latent output model
    decoder = blocks.MLP(n_latent, ny, bias=True,
                    linear_map=torch.nn.Linear,
                    nonlin=torch.nn.ReLU,
                    hsizes=[40])
    decode_sym = Node(decoder, ['xn'], ['y'], name='decoder')

    # latent NODE rollout
    dynamics_model = System([model, decode_sym], name='system', nsteps=nsteps)

    # %% Constraints + losses:
    y = variable("Y")                      # observed
    yhat = variable('y')                   # predicted output

    # trajectory tracking loss
    reference_loss = 5.*(yhat == y)^2
    reference_loss.name = "ref_loss"

    # one step tracking loss
    onestep_loss = 1.*(yhat[:, 1, :] == y[:, 1, :])^2
    onestep_loss.name = "onestep_loss"

    # finite difference variables
    xFD = (y[:, 1:, :] - y[:, :-1, :])
    xhatFD = (yhat[:, 1:, :] - yhat[:, :-1, :])

    # finite difference loss
    fd_loss = 10.*(xFD == xhatFD)^2
    fd_loss.name = 'FD_loss'

    # %%
    nodes = [encode_sym, dynamics_model]
    objectives = [reference_loss, onestep_loss]
    constraints = []
    # create constrained optimization loss
    loss = PenaltyLoss(objectives, constraints)
    # construct constrained optimization problem
    problem = Problem(nodes, loss)
    # plot computational graph
    problem.show()

    # %%
    optimizer = torch.optim.Adam(problem.parameters(),
                                 lr=0.003)
    logger = BasicLogger(args=None, savedir='test', verbosity=1,
                         stdout=['dev_loss', 'train_loss'])

    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        test_data,
        optimizer,
        patience=100,
        warmup=500,
        epochs=1000,
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

    # Test set results
    problem.nodes[1].nsteps = test_data['Y'].shape[1]
    test_outputs = problem(test_data)

    def denormalize(x, mean, std):
        return (x * std) + mean

    pred_traj = denormalize(test_outputs['test_y'], modelSystem.stats["Y"]["mean"],
                            modelSystem.stats["Y"]["std"]).reshape(-1, ny).T
    true_traj = denormalize(test_data['Y'], modelSystem.stats["Y"]["mean"],
                            modelSystem.stats["Y"]["std"]).reshape(-1,ny).T
    input_traj = denormalize(test_data['U'],
                             modelSystem.stats["U"]["mean"], modelSystem.stats["U"]["std"]).reshape(-1, nu).T
    dist_traj = denormalize(test_data['D'],
                            modelSystem.stats["D"]["mean"], modelSystem.stats["D"]["std"]).reshape(-1,nd).T

    plt_nsteps = 500

    # plot rollout
    figsize = 25
    fig, ax = plt.subplots(ny + nu + nd, figsize=(figsize, figsize))

    x_labels = [f'$y_{k}$' for k in range(len(true_traj))]
    for row, (t1, t2, label) in enumerate(zip(true_traj, pred_traj, x_labels)):
        axe = ax[row]
        axe.set_ylabel(label, rotation=0, labelpad=20, fontsize=figsize)
        axe.plot(t1[:plt_nsteps], 'c', linewidth=4.0, label='True')
        axe.plot(t2[:plt_nsteps], 'm--', linewidth=4.0, label='Pred')
        axe.tick_params(labelbottom=False, labelsize=figsize)
    axe.tick_params(labelbottom=True, labelsize=figsize)
    axe.legend(fontsize=figsize)

    u_labels = [f'$u_{k}$' for k in range(len(input_traj))]
    for row, (u, label) in enumerate(zip(input_traj, u_labels)):
        axe = ax[row+ny]
        axe.plot(u[:plt_nsteps], linewidth=4.0, label='inputs')
        axe.legend(fontsize=figsize)
        axe.set_ylabel(label, rotation=0, labelpad=20, fontsize=figsize)
        axe.tick_params(labelbottom=True, labelsize=figsize)

    d_labels = [f'$d_{k}$' for k in range(len(dist_traj))]
    for row, (d, label) in enumerate(zip(dist_traj, d_labels)):
        axe = ax[row+ny+nu]
        axe.plot(d[:plt_nsteps], linewidth=4.0, label='disturbances')
        axe.legend(fontsize=figsize)
        axe.set_ylabel(label, rotation=0, labelpad=20, fontsize=figsize)
        axe.tick_params(labelbottom=True, labelsize=figsize)

    ax[-1].set_xlabel('$time$', fontsize=figsize)
    plt.tight_layout()


