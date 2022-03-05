"""
DPC Double integrator example
plots work with N = 1

    # TODO: generalize terminal penalty for N step ahead policy
    # TODO: generalize to plot N-step ahead policy
"""

import torch
import torch.nn.functional as F
import slim
import psl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import seaborn as sns
DENSITY_PALETTE = sns.color_palette("crest_r", as_cmap=True)
DENSITY_FACECLR = DENSITY_PALETTE(0.01)
sns.set_theme(style="white")
import copy

from neuromancer.activations import activations
from neuromancer import blocks, estimators, dynamics
from neuromancer.trainer import Trainer
from neuromancer.simulators import ClosedLoopSimulator
from neuromancer.problem import Problem, Objective
from neuromancer import policies
import neuromancer.arg as arg
from neuromancer.datasets import Dataset
from neuromancer.loggers import BasicLogger


def arg_dpc_problem(prefix=''):
    """
    Command line parser for DPC problem definition arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers
                         are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = arg.ArgParser(prefix=prefix, add_help=False)
    gp = parser.group("DPC")
    gp.add("-nsteps", type=int, default=1,
           help="prediction horizon.")
    gp.add("-Qx", type=float, default=1.0,
           help="state weight.")                # tuned value: 1.0
    gp.add("-Qu", type=float, default=1.0,
           help="control action weight.")       # tuned value: 1.0,  unstable for paper: 0.5
    gp.add("-Qn", type=float, default=1.0,
           help="terminal penalty weight.")     # tuned value: 1.0,  unstable for paper: 0.0
    gp.add("-Q_sub", type=float, default=0.0,
           help="regularization weight.")
    gp.add("-Q_con_x", type=float, default=10.0,
           help="state constraints penalty weight.")  # tuned value: 10.0
    gp.add("-Q_con_u", type=float, default=20.0,
           help="Input constraints penalty weight.")  # tuned value: 20.0
    gp.add("-nx_hidden", type=int, default=20,
           help="Number of hidden states")
    gp.add("-n_layers", type=int, default=4,
           help="Number of hidden layers")
    gp.add("-bias", action="store_true",
           help="Whether to use bias in the neural network block component models.")
    gp.add("-norm", nargs="+", default=[], choices=["U", "D", "Y", "X"],
               help="List of sequences to max-min normalize")
    gp.add("-data_seed", type=int, default=408,
           help="Random seed used for simulated data")
    gp.add("-epochs", type=int, default=1000,
           help='Number of training epochs')
    gp.add("-lr", type=float, default=0.001,
           help="Step size for gradient descent.")
    gp.add("-patience", type=int, default=100,
           help="How many epochs to allow for no improvement in eval metric before early stopping.")
    gp.add("-warmup", type=int, default=10,
           help="Number of epochs to wait before enacting early stopping policy.")
    return parser


def plot_loss(model, dataset, xmin=-5, xmax=5, save_path=None):
    x = torch.arange(xmin, xmax, 0.2)
    y = torch.arange(xmin, xmax, 0.2)
    xx, yy = torch.meshgrid(x, y)
    dataset_plt = copy.deepcopy(dataset)
    dataset_plt.dims['nsim'] = 1
    Loss = np.ones([x.shape[0], y.shape[0]])*np.nan
    # Alpha contraction coefficient: ||x_k+1|| = alpha * ||x_k||
    Alpha = np.ones([x.shape[0], y.shape[0]])*np.nan
    # ||A+B*Kx||
    Phi_norm = np.ones([x.shape[0], y.shape[0]])*np.nan
    policy = model.components[1].net
    A = model.components[2].fx.linear.weight
    B = model.components[2].fu.linear.weight
    Anp = A.detach().numpy()
    Bnp = B.detach().numpy()
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            # check loss
            X = torch.stack([x[[i]], y[[j]]]).reshape(1,1,-1)
            dataset_plt.train_data['Yp'] = X
            step = model(dataset_plt.train_data)
            Loss[i,j] = step['nstep_train_loss'].detach().numpy()
            # check contraction
            x0 = X.view(1, X.shape[-1])
            Astar, bstar, _, _, _ = lpv_batched(policy, x0)
            BKx = torch.mm(B, Astar[:, :, 0])
            phi = A + BKx
            Phi_norm[i,j] = torch.norm(phi, 2).detach().numpy()
            # print(torch.matmul(Astar[:, :, 0], x0.transpose(0, 1))+bstar)
            u = policy(x0).detach().numpy()
            xnp = x0.transpose(0, 1).detach().numpy()
            xnp_n = np.matmul(Anp, xnp) + np.matmul(Bnp, u)
            if not np.linalg.norm(xnp) == 0:
                Alpha[i,j] = np.linalg.norm(xnp_n)/np.linalg.norm(xnp)
            else:
                Alpha[i, j] = 0

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx.detach().numpy(), yy.detach().numpy(), Loss,
                           cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    ax.set(ylabel='$x_1$')
    ax.set(xlabel='$x_2$')
    ax.set(zlabel='$L$')
    # plt.colorbar(surf)
    if save_path is not None:
        plt.savefig(save_path+'/loss.pdf')

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx.detach().numpy(), yy.detach().numpy(), Phi_norm,
                           cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    ax.set(ylabel='$x_1$')
    ax.set(xlabel='$x_2$')
    ax.set(zlabel='$Phi$')
    if save_path is not None:
        plt.savefig(save_path+'/phi_norm.pdf')

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx.detach().numpy(), yy.detach().numpy(), Alpha,
                           cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    ax.set(ylabel='$x_1$')
    ax.set(xlabel='$x_2$')
    ax.set(zlabel='$alpha$')
    if save_path is not None:
        plt.savefig(save_path+'/contraction.pdf')

    fig1, ax1 = plt.subplots()
    cm_map = plt.cm.get_cmap('RdBu_r')
    im1 = ax1.imshow(Alpha, vmin=abs(Alpha).min(), vmax=abs(Alpha).max(),
                     cmap=cm_map, origin='lower',
                     extent=[xx.detach().numpy().min(), xx.detach().numpy().max(),
                             yy.detach().numpy().min(), yy.detach().numpy().max()],
                     interpolation="bilinear")
    fig1.colorbar(im1, ax=ax1)
    ax1.set(ylabel='$x_1$')
    ax1.set(xlabel='$x_2$')
    im1.set_clim(0., 2.)  #  color limit
    if save_path is not None:
        plt.savefig(save_path+'/contraction_regions.pdf')


def plot_policy(net, xmin=-5, xmax=5, save_path=None):
    x = torch.arange(xmin, xmax, 0.1)
    y = torch.arange(xmin, xmax, 0.1)
    xx, yy = torch.meshgrid(x, y)
    features = torch.stack([xx, yy]).transpose(0, 2)
    uu = net(features)
    plot_u = uu.detach().numpy()[:,:,0]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx.detach().numpy(), yy.detach().numpy(), plot_u,
                           cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    ax.set(ylabel='$x_1$')
    ax.set(xlabel='$x_2$')
    ax.set(zlabel='$u$')
    if save_path is not None:
        plt.savefig(save_path+'/policy.pdf')


def cl_simulate(A, B, net, nstep=50, x0=np.ones([2, 1]), save_path=None):
    """

    :param A:
    :param B:
    :param net:
    :param nstep:
    :param x0:
    :return:
    """
    Anp = A.detach().numpy()
    Bnp = B.detach().numpy()
    x = x0
    X = [x]
    U = []
    for k in range(nstep+1):
        x_torch = torch.tensor(x).float().transpose(0, 1)
        u = net(x_torch).detach().numpy()
        # compute feedback gain
        Astar, bstar, _, _, _ = lpv_batched(net, x_torch)
        Kx = torch.mm(B, Astar[:, :, 0])
        # closed loop dynamics
        x = np.matmul(Anp, x) + np.matmul(Bnp, u)
        X.append(x)
        U.append(u)
    Xnp = np.asarray(X)[:, :, 0]
    Unp = np.asarray(U)[:, :, 0]

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Xnp, label='x', linewidth=2)
    ax[0].set(ylabel='$x$')
    ax[0].set(xlabel='time')
    ax[0].grid()
    ax[0].set_xlim(0, nstep)
    ax[1].plot(Unp, label='u', drawstyle='steps',  linewidth=2)
    ax[1].set(ylabel='$u$')
    ax[1].set(xlabel='time')
    ax[1].grid()
    ax[1].set_xlim(0, nstep)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path+'/closed_loop_dpc.pdf')


def lpv_batched(fx, x):
    x_layer = x
    Aprime_mats = []
    activation_mats = []
    bprimes = []

    for nlin, lin in zip(fx.nonlin, fx.linear):
        A = lin.effective_W()  # layer weight
        b = lin.bias if lin.bias is not None else torch.zeros(A.shape[-1])
        Ax = torch.matmul(x_layer, A) + b  # affine transform
        zeros = Ax == 0
        lambda_h = nlin(Ax) / Ax  # activation scaling
        lambda_h[zeros] = 0.
        lambda_h_mats = [torch.diag(v) for v in lambda_h]
        activation_mats += lambda_h_mats
        lambda_h_mats = torch.stack(lambda_h_mats)
        x_layer = Ax * lambda_h
        Aprime = torch.matmul(A, lambda_h_mats)
        Aprime_mats += [Aprime]
        bprime = lambda_h * b
        bprimes += [bprime]

    # network-wise parameter varying linear map:  A* = A'_L ... A'_1
    Astar = Aprime_mats[0]
    bstar = bprimes[0] # b x nx
    for Aprime, bprime in zip(Aprime_mats[1:], bprimes[1:]):
        Astar = torch.bmm(Astar, Aprime)
        bstar = torch.bmm(bstar.unsqueeze(-2), Aprime).squeeze(-2) + bprime

    return Astar, bstar, Aprime_mats, bprimes, activation_mats


if __name__ == "__main__":

    """
    # # #  Arguments, dimensions, bounds
    """
    parser = arg.ArgParser(parents=[arg.log(),
                                    arg_dpc_problem()])
    args, grps = parser.parse_arg_groups()

    args.bias = True

    # problem dimensions
    nx = 2
    ny = 2
    nu = 1
    # number of datapoints
    nsim = 10000
    # constraints bounds
    umin = -1
    umax = 1
    xmin = -10
    xmax = 10
    xN_min = -0.1
    xN_max = 0.1

    """
    # # #  Dataset 
    """
    #  randomly sampled input output trajectories for training
    #  we treat states as observables, i.e. Y = X
    sequences = {
        "Y_max": xmax*np.ones([nsim, nx]),
        "Y_min": xmin*np.ones([nsim, nx]),
        "U_max": umax*np.ones([nsim, nu]),
        "U_min": umin*np.ones([nsim, nu]),
        "Y": 3*np.random.randn(nsim, nx),
        "U": np.random.randn(nsim, nu),
    }
    dataset = Dataset(nsim=nsim, ninit=0, norm=args.norm, nsteps=args.nsteps,
                      device='cpu', sequences=sequences, name='closedloop')

    """
    # # #  System model
    """
    # Fully observable estimator as identity map: x0 = Yp
    estimator = estimators.FullyObservable({**dataset.dims, "x0": (nx,)},
                                           nsteps=1,  # future window Nf
                                           window_size=1,  # past window Np <= Nf
                                           input_keys=["Yp"],
                                           name='estimator')
    # A, B, C linear maps
    fu = slim.maps['linear'](nu, nx)
    fx = slim.maps['linear'](nx, nx)
    fy = slim.maps['linear'](nx, ny)
    # LTI SSM
    dynamics_model = dynamics.BlockSSM(fx, fy, fu=fu, name='dynamics',
                                       input_keys={'x0': f'x0_{estimator.name}'})
    dynamics_model.input_keys[dynamics_model.input_keys.index('Uf')] = 'U_pred_policy'
    # model matrices vlues
    A = torch.tensor([[1.2, 1.0],
                      [0.0, 1.0]])
    B = torch.tensor([[1.0],
                      [0.5]])
    C = torch.tensor([[1.0, 0.0],
                      [0.0, 1.0]])
    dynamics_model.fx.linear.weight = torch.nn.Parameter(A)
    dynamics_model.fu.linear.weight = torch.nn.Parameter(B)
    dynamics_model.fy.linear.weight = torch.nn.Parameter(C)
    # fix model parameters
    dynamics_model.requires_grad_(False)

    """
    # # #  Control policy
    """
    activation = activations['relu']
    linmap = slim.maps['linear']
    block = blocks.MLP
    policy = policies.MLPPolicy(
        {"x0_estim": (dynamics_model.nx,), **dataset.dims},
        nsteps=args.nsteps,
        bias=args.bias,
        linear_map=linmap,
        nonlin=activation,
        hsizes=[args.nx_hidden] * args.n_layers,
        input_keys=['Yp'],
        name='policy',
    )

    """
    # # #  DPC objectives and constraints
    """
    # objectives
    regulation_loss = Objective(
        [f'Y_pred_{dynamics_model.name}'],
        lambda x: F.mse_loss(x, x),
        weight=args.Qx,
        name="x^T*Qx*x loss",
    )
    action_loss = Objective(
        [f"U_pred_{policy.name}"],
        lambda x: F.mse_loss(x, x),
        weight=args.Qu,
        name="u^T*Qu*u loss",
    )
    # regularization
    regularization = Objective(
        [f"reg_error_{policy.name}"], lambda reg: reg,
        weight=args.Q_sub, name="reg_loss",
    )
    # constraints
    state_lower_bound_penalty = Objective(
        [f'Y_pred_{dynamics_model.name}', "Y_minf"],
        lambda x, xmin: torch.mean(F.relu(-x + xmin)),
        weight=args.Q_con_x,
        name="state_lower_bound",
    )
    state_upper_bound_penalty = Objective(
        [f'Y_pred_{dynamics_model.name}', "Y_maxf"],
        lambda x, xmax: torch.mean(F.relu(x - xmax)),
        weight=args.Q_con_x,
        name="state_upper_bound",
    )
    terminal_lower_bound_penalty = Objective(
        [f'Y_pred_{dynamics_model.name}', "Y_minf"],
        lambda x, xmin: torch.mean(F.relu(-x + xN_min)),
        weight=args.Qn,
        name="terminl_lower_bound",
    )
    terminal_upper_bound_penalty = Objective(
        [f'Y_pred_{dynamics_model.name}', "Y_maxf"],
        lambda x, xmax: torch.mean(F.relu(x - xN_max)),
        weight=args.Qn,
        name="terminl_upper_bound",
    )
    inputs_lower_bound_penalty = Objective(
        [f"U_pred_{policy.name}", "U_minf"],
        lambda u, umin: torch.mean(F.relu(-u + umin)),
        weight=args.Q_con_u,
        name="input_lower_bound",
    )
    inputs_upper_bound_penalty = Objective(
        [f"U_pred_{policy.name}", "U_maxf"],
        lambda u, umax: torch.mean(F.relu(u - umax)),
        weight=args.Q_con_u,
        name="input_upper_bound",
    )

    objectives = [regularization, regulation_loss, action_loss]
    constraints = [
        state_lower_bound_penalty,
        state_upper_bound_penalty,
        inputs_lower_bound_penalty,
        inputs_upper_bound_penalty,
        terminal_lower_bound_penalty,
        terminal_upper_bound_penalty,
    ]

    """
    # # #  DPC problem = objectives + constraints + trainable components 
    """
    # data (y_k) -> estimator (x_k) -> policy (u_k) -> dynamics (x_k+1, y_k+1)
    components = [estimator, policy, dynamics_model]
    model = Problem(
        objectives,
        constraints,
        components,
    )

    """
    # # #  DPC trainer 
    """
    # logger and metrics
    args.savedir = 'test_control'
    args.verbosity = 1
    metrics = ["nstep_dev_loss", "loop_dev_loss", "best_loop_dev_loss",
               "nstep_dev_ref_loss", "loop_dev_ref_loss"]
    logger = BasicLogger(args=args, savedir=args.savedir, verbosity=args.verbosity, stdout=metrics)
    logger.args.system = 'integrator'
    # device and optimizer
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # simulator
    simulator = ClosedLoopSimulator(
        model=model, dataset=dataset, emulator=dynamics_model, policy=policy
    )
    # trainer
    trainer = Trainer(
        model,
        dataset,
        optimizer,
        logger=logger,
        simulator=simulator,
        epochs=args.epochs,
        patience=args.patience,
        warmup=args.warmup,
    )
    # Train control policy
    best_model = trainer.train()

    """
    # # #  Plots and Analysis
    """
    # plot closed loop trajectories
    cl_simulate(A, B, policy.net, nstep=40,
                x0=1.5*np.ones([2, 1]), save_path='test_control')
    # plot policy surface
    plot_policy(policy.net, save_path='test_control')
    # loss landscape and contraction regions
    plot_loss(model, dataset, xmin=-5, xmax=5, save_path='test_control')

