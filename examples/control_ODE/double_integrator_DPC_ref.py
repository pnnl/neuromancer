"""
Differentiable predictive control (DPC)

DPC double integrator example with given system dynamics model
fixed reference tracking problem
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
from neuromancer.problem import Problem
from neuromancer.constraint import Variable, Loss
from neuromancer import policies
import neuromancer.arg as arg
from neuromancer.dataset import normalize_data, split_sequence_data, SequenceDataset
from torch.utils.data import DataLoader
from neuromancer.loggers import BasicLogger
from neuromancer.loss import PenaltyLoss, BarrierLoss


def arg_dpc_problem(prefix=''):
    """
    Command line parser for DPC problem definition arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers
                         are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = arg.ArgParser(prefix=prefix, add_help=False)
    gp = parser.group("DPC")
    gp.add("-controlled_outputs", type=int, default=[0],
           help="Index of the controlled state.")
    gp.add("-nsteps", type=int, default=1,
           help="prediction horizon.")          # tuned values: 1, 2
    gp.add("-Qr", type=float, default=5.0,
           help="reference tracking weight.")   # tuned value: 5.0
    gp.add("-Qu", type=float, default=0.0,
           help="control action weight.")       # tuned value: 0.0
    gp.add("-Qn", type=float, default=1.0,
           help="terminal penalty weight.")     # tuned value: 1.0
    gp.add("-Q_sub", type=float, default=0.0,
           help="regularization weight.")
    gp.add("-Q_con_x", type=float, default=10.0,
           help="state constraints penalty weight.")  # tuned value: 10.0
    gp.add("-Q_con_u", type=float, default=50.0,
           help="Input constraints penalty weight.")  # tuned value: 50.0
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
    gp.add("-epochs", type=int, default=400,
           help='Number of training epochs')
    gp.add("-lr", type=float, default=0.001,
           help="Step size for gradient descent.")
    gp.add("-patience", type=int, default=100,
           help="How many epochs to allow for no improvement in eval metric before early stopping.")
    gp.add("-warmup", type=int, default=10,
           help="Number of epochs to wait before enacting early stopping policy.")
    gp.add("-loss", type=str, default='penalty',
           choices=['penalty', 'barrier'],
           help="type of the loss function.")
    gp.add("-barrier_type", type=str, default='log10',
           choices=['log', 'log10', 'inverse'],
           help="type of the barrier function in the barrier loss.")
    gp.add("-batch_second", default=True, choices=[True, False],
           help="whether the batch is a second dimension in the dataset.")
    return parser



def get_sequence_dataloaders(
    data, nsteps, moving_horizon=False, norm_type=None, split_ratio=None, num_workers=0,
):
    """This will generate dataloaders and open-loop sequence dictionaries for a given dictionary of
    data. Dataloaders are hard-coded for full-batch training to match NeuroMANCER's original
    training setup.

    :param data: (dict str: np.array or list[dict str: np.array]) data dictionary or list of data
        dictionaries; if latter is provided, multi-sequence datasets are created and splits are
        computed over the number of sequences rather than their lengths.
    :param nsteps: (int) length of windowed subsequences for N-step training.
    :param moving_horizon: (bool) whether to use moving horizon batching.
    :param norm_type: (str) type of normalization; see function `normalize_data` for more info.
    :param split_ratio: (list float) percentage of data in train and development splits; see
        function `split_sequence_data` for more info.
    """

    if norm_type is not None:
        data, _ = normalize_data(data, norm_type)
    train_data, dev_data, test_data = split_sequence_data(data, nsteps, moving_horizon, split_ratio)

    train_data = SequenceDataset(
        train_data,
        nsteps=nsteps,
        moving_horizon=moving_horizon,
        name="train",
    )
    dev_data = SequenceDataset(
        dev_data,
        nsteps=nsteps,
        moving_horizon=moving_horizon,
        name="dev",
    )
    test_data = SequenceDataset(
        test_data,
        nsteps=nsteps,
        moving_horizon=moving_horizon,
        name="test",
    )

    train_loop = train_data.get_full_sequence()
    dev_loop = dev_data.get_full_sequence()
    test_loop = test_data.get_full_sequence()

    train_data = DataLoader(
        train_data,
        batch_size=len(train_data),
        shuffle=False,
        collate_fn=train_data.collate_fn,
        num_workers=num_workers,
    )
    dev_data = DataLoader(
        dev_data,
        batch_size=len(dev_data),
        shuffle=False,
        collate_fn=dev_data.collate_fn,
        num_workers=num_workers,
    )
    test_data = DataLoader(
        test_data,
        batch_size=len(test_data),
        shuffle=False,
        collate_fn=test_data.collate_fn,
        num_workers=num_workers,
    )

    return (train_data, dev_data, test_data), (train_loop, dev_loop, test_loop), train_data.dataset.dims


def get_loss(objectives, constraints, args):
    if args.loss == 'penalty':
        loss = PenaltyLoss(objectives, constraints, batch_second=args.batch_second)
    elif args.loss == 'barrier':
        loss = BarrierLoss(objectives, constraints, barrier=args.barrier_type,
                           batch_second=args.batch_second)
    return loss


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
            if dataset.nsteps == 1:
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

    if dataset.nsteps == 1:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(xx.detach().numpy(), yy.detach().numpy(), Loss,
                               cmap=cm.viridis,
                               linewidth=0, antialiased=False)
        ax.set(ylabel='$x_1$')
        ax.set(xlabel='$x_2$')
        ax.set(zlabel='$L$')
        ax.set(title='Loss landscape')
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
    ax.set(title='CLS 2-norm')
    if save_path is not None:
        plt.savefig(save_path+'/phi_norm.pdf')

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx.detach().numpy(), yy.detach().numpy(), Alpha,
                           cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    ax.set(ylabel='$x_1$')
    ax.set(xlabel='$x_2$')
    ax.set(zlabel='$alpha$')
    ax.set(title='CLS contraction')
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
    ax1.set(title='CLS contraction regions')
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
    ax.set(title='Policy landscape')
    if save_path is not None:
        plt.savefig(save_path+'/policy.pdf')


def cl_simulate(A, B, policy, nstep=50, x0=np.ones([2, 1]), ref=None, save_path=None):
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
        # taking a first control action based on RHC principle
        uout = policy({'x0_estimator': x_torch})
        u = uout['U_pred_policy'][0,:,:].detach().numpy().transpose()
        # closed loop dynamics
        x = np.matmul(Anp, x) + np.matmul(Bnp, u)
        X.append(x)
        U.append(u)
    Xnp = np.asarray(X)[:, :, 0]
    Unp = np.asarray(U)[:, :, 0]

    if ref is None:
        ref = np.zeros(Xnp.shape)
    else:
        ref = ref[0:Xnp.shape[0], :]

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ref, 'k--', label='r', linewidth=2)
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
    # terminal constraints and reference for the controlled state
    xN_min = 1.9
    xN_max = 2.1
    ref = 2.0

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
        "R": ref*np.ones([nsim, 1]),
    }
    nstep_data, loop_data, dims = get_sequence_dataloaders(sequences, args.nsteps)
    train_data, dev_data, test_data = nstep_data
    train_loop, dev_loop, test_loop = loop_data

    """
    # # #  System model and Control policy
    """
    # Fully observable estimator as identity map: x0 = Yp[-1]
    # x_0 = Yp
    # Yp = [y_-N, ..., y_0]
    estimator = estimators.FullyObservable({**dims, "x0": (nx,)},
                                           nsteps=args.nsteps,  # future window Nf
                                           window_size=1,  # past window Np <= Nf
                                           input_keys=["Yp"],
                                           name='estimator')
    # full state feedback control policy
    # Uf = p(x_0)
    # Uf = [u_0, ..., u_N]
    activation = activations['relu']
    linmap = slim.maps['linear']
    block = blocks.MLP
    policy = policies.MLPPolicy(
        {f'x0_{estimator.name}': (nx,), **dims},
        nsteps=args.nsteps,
        bias=args.bias,
        linear_map=linmap,
        nonlin=activation,
        hsizes=[args.nx_hidden] * args.n_layers,
        input_keys=[f'x0_{estimator.name}'],
        name='policy',
    )

    # A, B, C linear maps
    fu = slim.maps['linear'](nu, nx)
    fx = slim.maps['linear'](nx, nx)
    fy = slim.maps['linear'](nx, ny)
    # LTI SSM
    # x_k+1 = Ax_k + Bu_k
    # y_k+1 = Cx_k+1
    dynamics_model = dynamics.BlockSSM(fx, fy, fu=fu, name='dynamics',
                                       input_key_map={'x0': f'x0_{estimator.name}',
                                                   'Uf': 'U_pred_policy'})
    # model matrices values
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
    # # #  DPC objectives and constraints
    """
    u = Variable(f"U_pred_{policy.name}", name='u')
    y = Variable(f"Y_pred_{dynamics_model.name}", name='y')
    r = Variable("Rf", name='r')

    # constraints bounds variables
    umin = Variable("U_minf")
    umax = Variable("U_maxf")
    ymin = Variable("Y_minf")
    ymax = Variable("Y_maxf")

    # objectives
    action_loss = args.Qu * ((u == 0) ^ 2)  # control penalty
    reference_loss = args.Qr * ((y[:, :, args.controlled_outputs] == r) ^ 2)  # target posistion
    # constraints
    state_lower_bound_penalty = args.Q_con_x*(y > ymin)
    state_upper_bound_penalty = args.Q_con_x*(y < ymax)
    inputs_lower_bound_penalty = args.Q_con_u*(u > umin)
    inputs_upper_bound_penalty = args.Q_con_u*(u < umax)
    terminal_lower_bound_penalty = args.Qn*(y[[-1], :, :] > xN_min)
    terminal_upper_bound_penalty = args.Qn*(y[[-1], :, :] < xN_max)
    # objectives and constraints names for nicer plot
    action_loss.name = "action_loss"
    reference_loss.name = 'control_loss'
    state_lower_bound_penalty.name = 'x_min'
    state_upper_bound_penalty.name = 'x_max'
    inputs_lower_bound_penalty.name = 'u_min'
    inputs_upper_bound_penalty.name = 'u_max'
    terminal_lower_bound_penalty.name = 'y_N_min'
    terminal_upper_bound_penalty.name = 'y_N_max'
    # regularization
    regularization = Loss(
        [f"reg_error_{policy.name}"], lambda reg: reg,
        weight=args.Q_sub, name="reg_loss",
    )

    objectives = [regularization, reference_loss, action_loss]
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
    # create constrained optimization loss
    loss = get_loss(objectives, constraints, args)
    # construct constrained optimization problem
    problem = Problem(components, loss)
    # plot computational graph
    problem.plot_graph()

    """
    # # #  DPC trainer 
    """
    # logger and metrics
    args.savedir = 'test_control'
    args.verbosity = 1
    metrics = ["nstep_dev_loss"]
    logger = BasicLogger(args=args, savedir=args.savedir, verbosity=args.verbosity, stdout=metrics)
    logger.args.system = 'dpc_ref'
    # device and optimizer
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
    problem = problem.to(device)
    optimizer = torch.optim.AdamW(problem.parameters(), lr=args.lr)

    # trainer
    trainer = Trainer(
        problem,
        train_data,
        dev_data,
        test_data,
        optimizer,
        logger=logger,
        epochs=args.epochs,
        patience=args.patience,
        train_metric="nstep_train_loss",
        dev_metric="nstep_dev_loss",
        test_metric="nstep_test_loss",
        eval_metric='nstep_dev_loss',
        warmup=args.warmup,
    )
    # Train control policy
    best_model = trainer.train()
    best_outputs = trainer.test(best_model)

    """
    # # #  Plots and Analysis
    """
    # plot closed loop trajectories
    cl_simulate(A, B, policy, nstep=40,
                x0=1.5*np.ones([2, 1]), ref=sequences['R'], save_path='test_control')
    # plot policy surface
    plot_policy(policy.net, save_path='test_control')
