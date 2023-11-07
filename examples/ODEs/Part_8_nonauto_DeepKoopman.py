"""
Learning provably stable Deep Koopman model from time series data

references Koopman control models:
[1] https://arxiv.org/abs/2202.08004
[2] https://iopscience.iop.org/article/10.1088/2632-2153/abf0f5
[3] https://ieeexplore.ieee.org/document/9799788
[4] https://ieeexplore.ieee.org/document/9022864
[5] https://github.com/HaojieSHI98/DeepKoopmanWithControl
[6] https://pubs.aip.org/aip/cha/article-abstract/22/4/047510/341880/Applied-Koopmanisma
[7] https://arxiv.org/abs/1312.0041

references stability:
[8] https://ieeexplore.ieee.org/document/9482930

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
    decode_y = Node(f_y_inv, ['x'], ['yhat'], name='decoder_y')

    # instantiate Koopman matrix
    stable = True     # provably stable Koopman operator
    if stable:
        # SVD factorized Koopman operator with bounded eigenvalues
        K = slim.linear.SVDLinear(nx_koopman, nx_koopman,
                              sigma_min=0.01, sigma_max=1.0,
                              bias=False)
    else:
        # linear Koopman operator without guaranteed stability
        K = torch.nn.Linear(nx_koopman, nx_koopman, bias=False)

    # symbolic Koopman model with control inputs
    Koopman = Node(Koopman_control(K), ['x', 'u_latent'], ['x'], name='K')

    # latent Koopmann rollout
    dynamics_model = System([Koopman], name='Koopman', nsteps=nsteps)

    # variables
    Y = variable("Y")  # observed
    yhat = variable('yhat')  # predicted output
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
    objectives = [y_loss, x_loss, onestep_loss]
    constraints = []

    if stable:
        # SVD penalty variable
        K_reg_error = variable(K.reg_error())
        # SVD penalty loss term
        K_reg_loss = 1.*(K_reg_error == 0.0)
        K_reg_loss.name = 'SVD_loss'
        objectives.append(K_reg_loss)

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

    pred_traj = test_outputs['yhat'][:, 1:-1, :].detach().numpy().reshape(-1, nx).T
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

    # compute Koopman eigenvalues and eigenvectors
    if stable:
        eig, eig_vec = torch.linalg.eig(K.effective_W())
    else:
        eig, eig_vec = torch.linalg.eig(K.weight)
    # Koopman eigenvalues real and imaginary parts
    eReal = eig.real.detach().numpy()
    eImag = eig.imag.detach().numpy()
    # unit circle
    t = np.linspace(0.0, 2 * np.pi, 1000)
    x_circ = np.cos(t)
    y_circ = np.sin(t)

    # plot Koopman eigenvalues
    fig1, ax1 = plt.subplots()
    ax1.plot(x_circ, y_circ, 'c', linewidth=4)
    ax1.plot(eReal, eImag, 'wo')
    ax1.set_aspect('equal', 'box')
    ax1.set_facecolor("navy")
    ax1.set_xlabel("$Re(\lambda)$", fontsize=figsize)
    ax1.set_ylabel("$Im(\lambda)$", fontsize=figsize)
    fig1.suptitle('Koopman operator eigenvalues')

    # compute Koopman state eigenvectors
    y_min = 1.1*test_data['Y'].min()
    y_max = 1.1*test_data['Y'].max()
    y1 = torch.linspace(y_min, y_max, 1000)
    y2 = torch.linspace(y_min, y_max, 1000)
    yy1, yy2 = torch.meshgrid(y1, y1)
    plot_yy1 = yy1.detach().numpy()
    plot_yy2 = yy2.detach().numpy()
    # eigenvectors
    features = torch.stack([yy1, yy2]).transpose(0, 2)
    latent = f_y(features)
    phi = torch.matmul(latent, abs(eig_vec))
    # select first 6 eigenvectors
    phi_1 = phi.detach().numpy()[:,:,0]
    phi_2 = phi.detach().numpy()[:,:,1]
    phi_3 = phi.detach().numpy()[:,:,2]
    phi_4 = phi.detach().numpy()[:,:,3]
    phi_5 = phi.detach().numpy()[:,:,4]
    phi_6 = phi.detach().numpy()[:,:,6]
    # plot eigenvectors
    fig2, axs = plt.subplots(2, 3)
    im1 = axs[0,0].imshow(phi_1)
    im2 = axs[0,1].imshow(phi_2)
    im3 = axs[0,2].imshow(phi_3)
    im4 = axs[1,0].imshow(phi_4)
    im5 = axs[1,1].imshow(phi_5)
    im6 = axs[1,2].imshow(phi_6)
    fig2.colorbar(im1, ax=axs)
    fig2.suptitle('first six eigenfunctions')
