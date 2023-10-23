"""
Learning Deep Koopman model and control policiy from time series data

references Koopman control models:
[1] https://arxiv.org/abs/2202.08004
[2] https://iopscience.iop.org/article/10.1088/2632-2153/abf0f5
[3] https://ieeexplore.ieee.org/document/9799788
[4] https://ieeexplore.ieee.org/document/9022864
[5] https://github.com/HaojieSHI98/DeepKoopmanWithControl
[6] https://arxiv.org/abs/2202.08004

Control problem:
[7] http://apmonitor.com/do/index.php/Main/NonlinearControl

"""

import torch
import torch.nn as nn
import numpy as np

from neuromancer.psl import plot
from neuromancer import psl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from neuromancer.system import Node, System
from neuromancer.slim import slim
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer. modules import blocks
from neuromancer.plot import pltCL, pltPhase


def get_data(sys, nsim, nsteps, ts, bs):
    """
    Generate training data for system identification using Koopman model

    :param nsteps: (int) Number of timesteps for each batch of training data
    :param sys: (psl.system)
    :param ts: (float) step size
    :param bs: (int) batch size

    """
    train_sim, dev_sim, test_sim = [sys.simulate(nsim=nsim, ts=ts) for i in range(3)]
    nx = sys.nx
    nu = sys.nu
    nbatch = nsim//nsteps
    length = (nsim//nsteps) * nsteps

    mean_x = modelSystem.stats['Y']['mean']
    std_x = modelSystem.stats['Y']['std']
    mean_u = modelSystem.stats['U']['mean']
    std_u = modelSystem.stats['U']['std']
    def normalize(x, mean, std):
        return (x - mean) / std

    trainX = normalize(train_sim['Y'][:length], mean_x, std_x)
    trainX = trainX.reshape(nbatch, nsteps, nx)
    trainX = torch.tensor(trainX, dtype=torch.float32)
    trainU = normalize(train_sim['U'][:length], mean_u, std_u)
    trainU = trainU.reshape(nbatch, nsteps, nu)
    trainU = torch.tensor(trainU, dtype=torch.float32)
    train_data = DictDataset({'Y': trainX, 'Y0': trainX[:, 0:1, :],
                              'U': trainU}, name='train')
    train_loader = DataLoader(train_data, batch_size=bs,
                              collate_fn=train_data.collate_fn, shuffle=True)

    devX = normalize(dev_sim['Y'][:length], mean_x, std_x)
    devX = devX.reshape(nbatch, nsteps, nx)
    devX = torch.tensor(devX, dtype=torch.float32)
    devU = normalize(dev_sim['U'][:length], mean_u, std_u)
    devU = devU[:length].reshape(nbatch, nsteps, nu)
    devU = torch.tensor(devU, dtype=torch.float32)
    dev_data = DictDataset({'Y': devX, 'Y0': devX[:, 0:1, :],
                            'U': devU}, name='dev')
    dev_loader = DataLoader(dev_data, batch_size=bs,
                            collate_fn=dev_data.collate_fn, shuffle=True)

    testX = normalize(test_sim['Y'][:length], mean_x, std_x)
    testX = testX.reshape(1, nbatch*nsteps, nx)
    testX = torch.tensor(testX, dtype=torch.float32)
    testU = normalize(test_sim['U'][:length], mean_u, std_u)
    testU = testU.reshape(1, nbatch*nsteps, nu)
    testU = torch.tensor(testU, dtype=torch.float32)
    test_data = {'Y': testX, 'Y0': testX[:, 0:1, :],
                 'U': testU}

    return train_loader, dev_loader, test_data

class Koopman_control(nn.Module):
    """
    Baseline class for Koopman control model
    Implements discrete-time dynamical system:
        x_k+1 = K x_k + u_k
    with variables:
        x_k - latent states
        u_k - latent control inputs
    """

    def __init__(self, K):
        super().__init__()
        self.K = K

    def forward(self, x, u):
        """
        :param x: (torch.Tensor, shape=[batchsize, nx])
        :param u: (torch.Tensor, shape=[batchsize, nx])
        :return: (torch.Tensor, shape=[batchsize, nx])
        """
        x = self.K(x) + u
        return x


if __name__ == '__main__':
    torch.manual_seed(0)

    """
    # # # # # # # # # # # # # # # # # # # # # 
    #       Stage 1: data generation        #
    # # # # # # # # # # # # # # # # # # # # # 
    """

    # select system:
    #   TwoTank, CSTR, SwingEquation, IverSimple

    # %%  ground truth system
    system = psl.systems['CSTR']
    modelSystem = system()
    ts = modelSystem.ts
    nx = modelSystem.nx
    ny = modelSystem.ny
    nu = modelSystem.nu
    raw = modelSystem.simulate(nsim=1000, ts=ts)
    plot.pltOL(Y=raw['Y'], U=raw['U'])
    plot.pltPhase(X=raw['Y'])

    # get datasets
    nsim = 2000
    nsteps = 20
    bs = 100
    train_loader, dev_loader, test_data = get_data(modelSystem, nsim, nsteps, ts, bs)

    """
    # # # # # # # # # # # # # # # # # # # # # # # #
    #       Stage 2: system identification        #
    # # # # # # # # # # # # # # # # # # # # # # # #
    """

    nx_koopman = 50
    n_hidden = 60
    n_layers = 2

    # instantiate state encoder neural net
    f_y = blocks.MLP(ny, nx_koopman, bias=True,
                     linear_map=torch.nn.Linear,
                     nonlin=torch.nn.ELU,
                     hsizes=n_layers*[n_hidden])
    # initial condition encoder
    encode_Y0 = Node(f_y, ['Y0'], ['x'], name='encoder_Y0')
    # observed trajectory encoder
    encode_Y = Node(f_y, ['Y'], ['x_latent'], name='encoder_Y')

    # instantiate input encoder net
    f_u = blocks.MLP(nu, nx_koopman, bias=True,
                     linear_map=torch.nn.Linear,
                     nonlin=torch.nn.ELU,
                     hsizes=n_layers*[n_hidden])
    # control action encoder
    encode_U = Node(f_u, ['U'], ['u_latent'], name='encoder_U')

    # instantiate state decoder neural net
    f_y_inv = blocks.MLP(nx_koopman, ny, bias=True,
                    linear_map=torch.nn.Linear,
                    nonlin=torch.nn.ELU,
                    hsizes=n_layers*[n_hidden])
    # predicted trajectory decoder
    decode_y = Node(f_y_inv, ['x'], ['y'], name='decoder_y')

    # instantiate SVD factorized Koopman operator with bounded eigenvalues
    K = slim.linear.SVDLinear(nx_koopman, nx_koopman,
                          sigma_min=0.01, sigma_max=1.0,
                          bias=False)

    # SVD penalty variable
    K_reg_error = variable(K.reg_error())
    # SVD penalty loss term
    K_reg_loss = 1.*(K_reg_error == 0.0)
    K_reg_loss.name = 'SVD_loss'

    # symbolic Koopman model with control inputs
    Koopman = Node(Koopman_control(K), ['x', 'u_latent'], ['x'], name='K')

    # latent Koopmann rollout
    dynamics_model = System([Koopman], name='Koopman', nsteps=nsteps)
    dynamics_model.show()

    # variables
    Y = variable("Y")  # observed
    yhat = variable('y')  # predicted output
    x_latent = variable('x_latent')  # encoded output trajectory in the latent space
    u_latent = variable('u_latent')  # encoded input trajectory in the latent space
    x = variable('x')  # Koopman latent space trajectory
    xu_latent = x_latent + u_latent  # latent state trajectory

    # output trajectory tracking loss
    y_loss = 10. * (yhat[:, 1:-1, :] == Y[:, 1:, :]) ^ 2
    y_loss.name = "y_loss"

    # one-step tracking loss
    onestep_loss = 1.*(yhat[:, 1, :] == Y[:, 1, :])^2
    onestep_loss.name = "onestep_loss"

    # latent trajectory tracking loss
    x_loss = 1. * (x[:, 1:-1, :] == xu_latent[:, 1:, :]) ^ 2
    x_loss.name = "x_loss"

    # % objectives and constraints
    objectives = [y_loss, x_loss, onestep_loss, K_reg_loss]
    constraints = []

    # create constrained optimization loss
    loss = PenaltyLoss(objectives, constraints)
    # construct constrained optimization problem
    nodes = [encode_Y0, encode_Y, encode_U, dynamics_model, decode_y]
    problem = Problem(nodes, loss)
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
        epochs=1000,
        eval_metric="dev_loss",
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="dev_loss",
    )
    # %%
    best_model = trainer.train()
    problem.load_state_dict(best_model)
    # %%

    # Test set results
    problem.nodes[3].nsteps = test_data['Y'].shape[1]
    test_outputs = problem.step(test_data)

    pred_traj = test_outputs['y'][:, 1:-1, :].detach().numpy().reshape(-1, nx).T
    true_traj = test_data['Y'][:, 1:, :].detach().numpy().reshape(-1, nx).T
    input_traj = test_data['U'].detach().numpy().reshape(-1, nu).T

    # plot trajectories
    figsize = 25
    fig, ax = plt.subplots(nx + nu, figsize=(figsize, figsize))

    x_labels = [f'$y_{k}$' for k in range(len(true_traj))]
    for row, (t1, t2, label) in enumerate(zip(true_traj, pred_traj, x_labels)):
        axe = ax[row]
        axe.set_ylabel(label, rotation=0, labelpad=20, fontsize=figsize)
        axe.plot(t1, 'c', linewidth=4.0, label='True')
        axe.plot(t2, 'm--', linewidth=4.0, label='Pred')
        axe.tick_params(labelbottom=False, labelsize=figsize)
    axe.tick_params(labelbottom=True, labelsize=figsize)
    axe.legend(fontsize=figsize)

    u_labels = [f'$u_{k}$' for k in range(len(input_traj))]
    for row, (u, label) in enumerate(zip(input_traj, u_labels)):
        axe = ax[row+nx]
        axe.plot(u, linewidth=4.0, label='inputs')
        axe.legend(fontsize=figsize)
        axe.set_ylabel(label, rotation=0, labelpad=20, fontsize=figsize)
        axe.tick_params(labelbottom=True, labelsize=figsize)

    ax[-1].set_xlabel('$time$', fontsize=figsize)
    plt.tight_layout()

    """
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Stage 3: learning neural control policy         #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    """

    """
    # # #  Control Dataset 
    """
    nsteps = 50  # prediction horizon
    n_samples = 2000    # number of sampled scenarios
    nref = 1            # number of references

    # TODO: sample right data
    #  sampled references for training the policy
    list_refs = [torch.rand(1, 1)*torch.ones(nsteps+1, nref) for k in range(n_samples)]
    ref = torch.cat(list_refs)
    batched_ref = ref.reshape([n_samples, nsteps+1, nref])
    # Training dataset
    train_data = DictDataset({'y': torch.rand(n_samples, 1, ny),
                              'r': batched_ref}, name='train')

    # references for dev set
    list_refs = [torch.rand(1, 1)*torch.ones(nsteps+1, nref) for k in range(n_samples)]
    ref = torch.cat(list_refs)
    batched_ref = ref.reshape([n_samples, nsteps+1, nref])
    # Development dataset
    dev_data = DictDataset({'y': torch.rand(n_samples, 1, ny),
                            'r': batched_ref}, name='dev')

    # torch dataloaders
    batch_size = 200
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               collate_fn=train_data.collate_fn,
                                               shuffle=False)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size,
                                             collate_fn=dev_data.collate_fn,
                                             shuffle=False)

    """
    # # #  Deep Koopman DPC architecture
    """
    # state encoder
    encode_y = Node(f_y, ['y'], ['x'], name='encoder_y')

    # fix parameters of the Koopman model
    encode_y.requires_grad_(False)
    encode_U.requires_grad_(False)
    dynamics_model.requires_grad_(False)
    decode_y.requires_grad_(False)

    # neural net control policy
    net = blocks.MLP(insize=nx + nref, outsize=nu, hsizes=4*[32],
                            nonlin=torch.nn.ELU)
    policy = Node(net, ['y', 'r'], ['U'], name='policy')

    nodes = [encode_y, policy, encode_U, dynamics_model, decode_y]
    cl_system = System(nodes, name='cl_system', nsteps=nsteps)
    cl_system.show()

    """
    # # #  Differentiable Predictive Control objectives and constraints
    """
    # variables
    y = variable('y')
    ref = variable("r")
    # objectives
    regulation_loss = 5. * ((y == ref) ^ 2)  # target posistion
    # constraints
    terminal_lower_bound_penalty = 10.*(y[:, [-1], :] > ref-0.01)
    terminal_upper_bound_penalty = 10.*(y[:, [-1], :] < ref+0.01)
    # objectives and constraints names for nicer plot
    regulation_loss.name = 'ref_tracking'
    terminal_lower_bound_penalty.name = 'x_N_min'
    terminal_upper_bound_penalty.name = 'x_N_max'
    # list of constraints and objectives
    objectives = [regulation_loss]
    constraints = [
        terminal_lower_bound_penalty,
        terminal_upper_bound_penalty,
    ]

    """
    # # #  Differentiable optimal control problem 
    """
    # data (x_k, r_k) -> parameters (xi_k) -> policy (u_k) -> dynamics (x_k+1)
    nodes = [cl_system]
    # create constrained optimization loss
    loss = PenaltyLoss(objectives, constraints)
    # construct constrained optimization problem
    problem = Problem(nodes, loss)
    # plot computational graph
    problem.show()

    """
    # # #  Solving the problem 
    """
    optimizer = torch.optim.AdamW(problem.parameters(), lr=0.002)
    #  Neuromancer trainer
    trainer = Trainer(
        problem,
        train_loader, dev_loader,
        optimizer=optimizer,
        epochs=100,
        train_metric='train_loss',
        eval_metric='dev_loss',
        warmup=50,
    )
    # Train control policy
    best_model = trainer.train()
    # load best trained model
    trainer.model.load_state_dict(best_model)

    """
    Test Closed Loop System
    """
    print('\nTest Closed Loop System \n')
    nsteps = 750
    step_length = 150
    # generate reference
    np_refs = psl.signals.step(nsteps+1, 1, min=xmin, max=xmax, randsteps=5)
    R = torch.tensor(np_refs, dtype=torch.float32).reshape(1, nsteps+1, 1)
    torch_ref = torch.cat([R, R], dim=-1)
    # generate initial data for closed loop simulation
    data = {'x': torch.rand(1, 1, nx, dtype=torch.float32),
            'r': torch_ref}
    cl_system.nsteps = nsteps
    # perform closed-loop simulation
    trajectories = cl_system(data)

    # constraints bounds
    Umin = umin * np.ones([nsteps, nu])
    Umax = umax * np.ones([nsteps, nu])
    Xmin = xmin * np.ones([nsteps+1, nx])
    Xmax = xmax * np.ones([nsteps+1, nx])
    # plot closed loop trajectories
    pltCL(Y=trajectories['x'].detach().reshape(nsteps + 1, nx),
          R=trajectories['r'].detach().reshape(nsteps + 1, nref),
          U=trajectories['u'].detach().reshape(nsteps, nu),
          Umin=Umin, Umax=Umax, Ymin=Xmin, Ymax=Xmax)
    # plot phase portrait
    pltPhase(X=trajectories['x'].detach().reshape(nsteps + 1, nx))
