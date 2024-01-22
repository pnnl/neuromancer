"""
Learning provably stable Deep Koopman model from time series data

references Koopman models:
[1] https://www.nature.com/articles/s41467-018-07210-0
[2] https://ieeexplore.ieee.org/document/8815339
[3] https://arxiv.org/abs/1710.04340
[4] https://nicholasgeneva.com/deep-learning/koopman/dynamics/2020/05/30/intro-to-koopman.html
[5] https://pubs.aip.org/aip/cha/article-abstract/22/4/047510/341880/Applied-Koopmanisma
[6] https://arxiv.org/abs/1312.0041

references stability:
[7] https://ieeexplore.ieee.org/document/9482930

"""

import torch
import numpy as np

from neuromancer.psl import plot
from neuromancer import psl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from neuromancer.system import Node, System
from neuromancer.slim import slim
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.loggers import BasicLogger
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
    ny = sys.ny
    nbatch = nsim//nsteps
    length = (nsim//nsteps) * nsteps

    trainY = train_sim['Y'][:length].reshape(nbatch, nsteps, ny)
    trainY = torch.tensor(trainY, dtype=torch.float32)
    train_data = DictDataset({'Y': trainY, 'Y0': trainY[:, 0:1, :]}, name='train')
    train_loader = DataLoader(train_data, batch_size=bs,
                              collate_fn=train_data.collate_fn, shuffle=True)

    devY = dev_sim['Y'][:length].reshape(nbatch, nsteps, ny)
    devY = torch.tensor(devY, dtype=torch.float32)
    dev_data = DictDataset({'Y': devY, 'Y0': devY[:, 0:1, :]}, name='dev')
    dev_loader = DataLoader(dev_data, batch_size=bs,
                            collate_fn=dev_data.collate_fn, shuffle=True)

    testY = test_sim['Y'][:length].reshape(1, nsim, ny)
    testY = torch.tensor(testY, dtype=torch.float32)
    test_data = {'Y': testY, 'Y0': testY[:, 0:1, :], 'name': 'test'}

    return train_loader, dev_loader, test_data


if __name__ == '__main__':
    torch.manual_seed(0)

    # %%  ground truth system
    system = psl.systems['VanDerPol']
    modelSystem = system()
    ts = modelSystem.ts
    nx = modelSystem.nx
    ny = modelSystem.ny
    raw = modelSystem.simulate(nsim=1000, ts=ts)
    plot.pltOL(Y=raw['Y'])
    plot.pltPhase(X=raw['Y'])

    # get datasets
    nsim = 2000
    nsteps = 20
    bs = 100
    train_loader, dev_loader, test_data = get_data(modelSystem, nsim, nsteps, ts, bs)

    nx_koopman = 50
    n_hidden = 60
    n_layers = 2

    # instantiate encoder neural net
    encode = blocks.MLP(ny, nx_koopman, bias=True,
                     linear_map=torch.nn.Linear,
                     nonlin=torch.nn.ELU,
                     hsizes=n_layers*[n_hidden])
    # initial condition encoder
    encode_Y0 = Node(encode, ['Y0'], ['x'], name='encoder_Y0')
    # observed trajectory encoder
    encode_Y = Node(encode, ['Y'], ['x_traj'], name='encoder_Y')

    # instantiate decoder neural net
    decode = blocks.MLP(nx_koopman, ny, bias=True,
                    linear_map=torch.nn.Linear,
                    nonlin=torch.nn.ELU,
                    hsizes=n_layers*[n_hidden])
    # reconstruction decoder
    decode_y0 = Node(decode, ['x'], ['yhat0'], name='decoder_y0')
    # predicted trajectory decoder
    decode_y = Node(decode, ['x'], ['yhat'], name='decoder_y')

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

    # symbolic Koopman model
    Koopman = Node(K, ['x'], ['x'], name='K')

    # latent Koopmann rollout
    dynamics_model = System([Koopman], name='Koopman', nsteps=nsteps)

    # %% Constraints + losses:
    Y = variable("Y")                      # observed
    yhat = variable('yhat')                # predicted output
    Y0 = variable('Y0')                    # observed initial conditions
    yhat0 = variable('yhat0')              # reconstructed initial conditions
    x_traj = variable('x_traj')            # encoded trajectory in the latent space
    x = variable('x')                      # Koopman latent space trajectory

    # output trajectory tracking loss
    y_loss = 10. * (yhat[:, 1:-1, :] == Y[:, 1:, :]) ^ 2
    y_loss.name = "y_loss"

    # latent trajectory tracking loss
    x_loss = 1. * (x[:, 1:-1, :] == x_traj[:, 1:, :]) ^ 2
    x_loss.name = "x_loss"

    # one-step tracking loss
    onestep_loss = 1.*(yhat[:, 1, :] == Y[:, 1, :])^2
    onestep_loss.name = "onestep_loss"

    # encoder-decoder reconstruction loss
    reconstruct_loss = 1.*(Y0 == yhat0)^2
    reconstruct_loss.name = "reconstruct_loss"

    # %%
    objectives = [y_loss, x_loss, onestep_loss, reconstruct_loss]
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
    nodes = [encode_Y0, decode_y0, encode_Y,
             dynamics_model, decode_y]
    problem = Problem(nodes, loss)
    # plot computational graph
    problem.show()

    # %%
    optimizer = torch.optim.Adam(problem.parameters(), lr=0.001)
    logger = BasicLogger(args=None, savedir='test', verbosity=1,
                         stdout=['dev_loss', 'train_loss'])

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
        logger=logger,
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

    # plot trajectories
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

    # compute Koopman eigenvectors
    y1 = torch.linspace(-2.2, 2.2, 1000)
    y2 = torch.linspace(-2.2, 2.2, 1000)
    yy1, yy2 = torch.meshgrid(y1, y1)
    plot_yy1 = yy1.detach().numpy()
    plot_yy2 = yy2.detach().numpy()
    # eigenvectors
    features = torch.stack([yy1, yy2]).transpose(0, 2)
    latent = encode(features)
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
