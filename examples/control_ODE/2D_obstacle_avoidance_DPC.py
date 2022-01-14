"""
Differentiable predictive control (DPC)

DPC double integrator example with given system dynamics model
time varying reference tracking problem

observation: with increasing predicion horizon we need to retune the relative weights,
            otherwise the performance deteriorates
open problem: automatic retuning based on prediction horizon length

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
from neuromancer.constraint import Variable, Objective, Loss
from neuromancer import policies
import neuromancer.arg as arg
from neuromancer.dataset import normalize_data, split_sequence_data, SequenceDataset
from torch.utils.data import DataLoader
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
    gp.add("-controlled_outputs", type=int, default=[0],
           help="Index of the controlled state.")
    gp.add("-nsteps", type=int, default=10,
           help="prediction horizon.")          # tuned values: 1, 2
    gp.add("-Qr", type=float, default=5.0,
           help="reference tracking weight.")   # tuned value: 5.0
    gp.add("-Qu", type=float, default=0.0,
           help="control action weight.")       # tuned value: 0.0
    gp.add("-Qdu", type=float, default=0.1,
           help="control action difference weight.")       # tuned value: 0.0
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
    gp.add("-epochs", type=int, default=1000,
           help='Number of training epochs')
    gp.add("-lr", type=float, default=0.001,
           help="Step size for gradient descent.")
    gp.add("-patience", type=int, default=100,
           help="How many epochs to allow for no improvement in eval metric before early stopping.")
    gp.add("-warmup", type=int, default=10,
           help="Number of epochs to wait before enacting early stopping policy.")
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


def cl_simulate(A, B, policy, args, nstep=50, x0=np.ones([2, 1]), ref=None, save_path=None):
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
    N = args.nsteps
    if ref is None:
        ref = np.zeros([nstep, 1])

    for k in range(nstep+1-N):
        x_torch = torch.tensor(x).float().transpose(0, 1)
        # taking a first control action based on RHC principle
        r_k = torch.tensor([ref[k:k+N, :]]).float().transpose(0, 1)
        uout = policy({'x0_estimator': x_torch, 'Rf': r_k})
        u = uout['U_pred_policy'][0,:,:].detach().numpy().transpose()
        # closed loop dynamics
        x = np.matmul(Anp, x) + np.matmul(Bnp, u)
        X.append(x)
        U.append(u)
    Xnp = np.asarray(X)[:, :, 0]
    Unp = np.asarray(U)[:, :, 0]

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
    nu = 2
    # number of datapoints
    nsim = 10000        # rule of thumb: more data samples -> improved control performance
    # constraints bounds
    umin = -1
    umax = 1
    xmin = -10
    xmax = 10


    """
    # # #  Dataset 
    """
    #  randomly sampled input output trajectories for training
    #  we treat states as observables, i.e. Y = X
    sequences = {
        "X_max": xmax*np.ones([nsim, nx]),
        "X_min": xmin*np.ones([nsim, nx]),
        "U_max": umax*np.ones([nsim, nu]),
        "U_min": umin*np.ones([nsim, nu]),
        "Y": np.random.uniform(low=-0.1, high=0.1, size=(nsim, nx)),
        # "x_init": np.random.uniform(low=-0.1, high=0.1, size=(nsim,nx)),
        # "x_final": 1+np.random.uniform(low=-0.1, high=0.1, size=(nsim,nx)),
        "x_final": np.random.uniform(low=[0.8, -0.2], high=[1.2, 0.2], size=(nsim, nx)),
        'params': np.random.uniform(low=[1.0, 0.3, 0.0], high=[3.0, 0.7, 0.4], size=(nsim,3)),
        # "b": np.random.uniform(low=1.0, high=3.0) * np.ones([nsim, 1]),
        # "c": np.random.uniform(low=0.3, high=0.7) * np.ones([nsim, 1]),
        # "d": np.random.uniform(low=0.0, high=0.4) * np.ones([nsim, 1]),
        "U": np.random.randn(nsim, nu),
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
    # mapping reading parameters from the dataset
    parameters = estimators.FullyObservable({**dims, "x0": (nx+3,)},
                                           nsteps=args.nsteps,  # future window Nf
                                           window_size=1,  # past window Np <= Nf
                                           input_keys=["paramsp"],
                                           name='parameters')
    # mapping reading references from the dataset
    refs = estimators.FullyObservable({**dims, "x0": (nx+3,)},
                                           nsteps=args.nsteps,  # future window Nf
                                           window_size=1,  # past window Np <= Nf
                                           input_keys=["x_finalp"],
                                           name='refs')

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
    A = torch.tensor([[1.0, 0.1],
                      [0.0, 1.0]])
    B = torch.tensor([[1.0, 0.0],
                      [0.0, 1.0]])
    C = torch.tensor([[1.0, 0.0],
                      [0.0, 1.0]])
    dynamics_model.fx.linear.weight = torch.nn.Parameter(A)
    dynamics_model.fu.linear.weight = torch.nn.Parameter(B)
    dynamics_model.fy.linear.weight = torch.nn.Parameter(C)
    # fix model parameters
    dynamics_model.requires_grad_(False)

    # full state feedback control policy with reference preview Rf
    # U_policy = p(x_0, Rf)
    # U_policy = [u_0, ..., u_N]
    # Rf = [r_0, ..., r_N]
    activation = activations['relu']
    linmap = slim.maps['linear']
    block = blocks.MLP
    policy = policies.MLPPolicy(
        {f'x0_{estimator.name}': (nx,),
         f'x0_{refs.name}': (nx,),
         f'x0_{parameters.name}': (3,), **dims},
        nsteps=args.nsteps,
        bias=args.bias,
        linear_map=linmap,
        nonlin=activation,
        hsizes=[args.nx_hidden] * args.n_layers,
        input_keys=[f'x0_{estimator.name}', f'x0_{parameters.name}', f'x0_{refs.name}'],
        name='policy',
    )

    """
    # # #  DPC objectives and constraints
    """
    u = Variable(f"U_pred_{policy.name}")
    x = Variable(f"X_pred_{dynamics_model.name}")
    x1 = Variable(f"X_pred_{dynamics_model.name}")[:, :, [0]]
    x2 = Variable(f"X_pred_{dynamics_model.name}")[:, :, [1]]
    u = Variable(f"U_pred_{policy.name}")
    r = Variable('x_finalp')[[-1], :, :]
    # sampled parameters
    p = 0.5
    b = Variable('paramsp')[[-1], :, [0]]
    c = Variable('paramsp')[[-1], :, [1]]
    d = Variable('paramsp')[[-1], :, [2]]

    Q_con = 1.0
    obstacle = Q_con * ((p / 2) ** 2 <= b * (x1 - c) ** 2 + (x2 - d) ** 2)      # eliptic obstacle
    Q_u = 1.0
    action_loss = Q_u*((u==0)^2)                       # control penalty
    Q_r = 1.0
    reference_loss = Q_r*((r==x[[-1], :, :])^2)         # target posistion
    Q_r_mean = 1.0
    # reference_loss_mean = Q_r_mean*(r==x)         # target
    reference_loss_mean = Q_r_mean*((r==x)^2)         # target
    Q_dx = 1.0
    state_smoothing = Q_dx*((x[1:] == x[:-1])^2)           # delta x penalty
    Q_du = 1.0
    control_smoothing = Q_du*((u[1:] == u[:-1])^2)          # delta u penalty

    # dx bounded with some constant

    # constraints bounds variables
    umin = Variable("U_minf")
    umax = Variable("U_maxf")
    xmin = Variable("X_minf")
    xmax = Variable("X_maxf")
    Q_con_u = 1.0
    Q_con_x = 1.0
    state_lower_bound_penalty = Q_con_x*(x > xmin)
    state_upper_bound_penalty = Q_con_x*(x < xmax)
    input_lower_bound_penalty = Q_con_u*(u > umin)
    input_upper_bound_penalty = Q_con_u*(u < umax)

    objectives = [reference_loss, action_loss, reference_loss_mean,
                  state_smoothing, control_smoothing]
    constraints = [
        obstacle,
        state_lower_bound_penalty,
        state_upper_bound_penalty,
        input_lower_bound_penalty,
        input_upper_bound_penalty,
    ]

    """
    # # #  DPC problem = objectives + constraints + trainable components 
    """
    # data (y_k) -> estimator (x_k) -> policy (u_k) -> dynamics (x_k+1, y_k+1)
    components = [estimator, refs, parameters, policy, dynamics_model]
    # components = [estimator, b, c, d, policy, dynamics_model]

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
    metrics = ["nstep_dev_loss"]

    logger = BasicLogger(args=args, savedir=args.savedir, verbosity=args.verbosity, stdout=metrics)
    logger.args.system = 'dpc_var_ref'
    # device and optimizer
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # trainer
    trainer = Trainer(
        model,
        train_data,
        dev_data,
        test_data,
        optimizer,
        logger=logger,
        epochs=args.epochs,
        patience=args.patience,
        eval_metric='nstep_dev_loss',
        warmup=args.warmup,
    )
    # Train control policy
    best_model = trainer.train()
    best_outputs = trainer.test(best_model)

    """
    # # #  Plots and Analysis
    """
    # # plot closed loop trajectories with time varying reference
    # ref_step = 40
    # R = np.concatenate([0.5*np.ones([ref_step, 1]),
    #                     1*np.ones([ref_step, 1]), 0*np.ones([ref_step, 1])])
    # cl_simulate(A, B, policy, args=args, nstep=R.shape[0],
    #             x0=1.5*np.ones([2, 1]), ref=R, save_path='test_control')

    b = 3.0
    c = 0.3
    d = 0.2
    # 0.3 <= c < 0.7
    # 0.0 <= d <= 0.4
    # 1.0 <= b <= 3.0
    x_init = np.asarray([[0.0], [0.0]])
    x_final = np.asarray([[1.0], [0.2]])
    params = np.asarray([[b], [c], [d]])

    x1 = np.arange(-0.5, 1.5, 0.02)
    x2 = np.arange(-0.5, 1.5, 0.02)
    xx, yy = np.meshgrid(x1, x2)
    # eval objective and constraints
    c2 = b*(xx -c)**2+(yy-d)**2 - (p/2)**2

    # Plot
    fig, ax = plt.subplots(1,1)
    cg2 = ax.contour(xx, yy, c2, [0], colors='mediumblue', alpha=0.5)
    plt.setp(cg2.collections, facecolor='mediumblue')
    ax.plot(x_final[0], x_final[1], 'r*', markersize=10)
    ax.plot(x_init[0], x_init[1], 'g*', markersize=10)

    x_torch = torch.tensor(x_init).float().transpose(0, 1)
    r_torch = torch.tensor(x_final).float().transpose(0, 1)
    params_torch = torch.tensor(params).float().transpose(0, 1)
    Yf = torch.ones([args.nsteps, 1])

    uout = policy({'x0_estimator': x_torch, f'x0_{parameters.name}': params_torch,
                   f'x0_{refs.name}': r_torch})
    xout = dynamics_model({**uout, f'x0_{estimator.name}':x_torch, 'Yf': Yf})
    X = xout['X_pred_dynamics'][:, 0, :].detach().numpy()
    U = uout['U_pred_policy'][:, 0, :].detach().numpy()

    # plot trajectory
    ax.plot(X[:,0], X[:,1], '*--')

    fig, ax = plt.subplots(2,1)
    ax[0].plot(X)
    ax[1].plot(U)