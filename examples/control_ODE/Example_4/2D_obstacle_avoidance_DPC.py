"""
Solve obstacle avoidance control problem in Neuromancer
minimize     u_k^2
subject to   (p/2)^2 <= b(x[0]-c)^2 + (x[1]-d)^2
             x_k+1 = Ax_k + Bu_k

problem parameters:             p, b, c, d
problem decition variables:     x, u

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
import time
from casadi import *
import numpy as np

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
    gp.add("-nsteps", type=int, default=20,
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
    gp.add("-nx_hidden", type=int, default=100,
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
    gp.add("-patience", type=int, default=200,
           help="How many epochs to allow for no improvement in eval metric before early stopping.")
    gp.add("-warmup", type=int, default=100,
           help="Number of epochs to wait before enacting early stopping policy.")
    return parser


def get_sequence_dataloaders(
    data, nsteps, moving_horizon=False, norm_type=None, split_ratio=None, num_workers=0, batch_size=None,
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

    if batch_size is None:
        batch_size = len(train_data)

    train_data = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=train_data.collate_fn,
        num_workers=num_workers,
    )
    dev_data = DataLoader(
        dev_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dev_data.collate_fn,
        num_workers=num_workers,
    )
    test_data = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_data.collate_fn,
        num_workers=num_workers,
    )

    return (train_data, dev_data, test_data), (train_loop, dev_loop, test_loop), train_data.dataset.dims


def plot_obstacle(model, args, b, c, d, show_plot=True):
    """
       # # #  Benchmark solution in CasADi using IPOPT
    """
    # instantiate casadi optimizaiton problem class
    opti = casadi.Opti()
    N = args.nsteps
    X = opti.variable(2, N + 1)  # state trajectory
    x1 = X[0, :]
    x2 = X[1, :]
    U = opti.variable(2, N)  # control trajectory
    # system dynamics
    A = MX(np.array([[1.0, 0.1],
                     [0.0, 1.0]]))
    B = MX(np.array([[1.0, 0.0],
                     [0.0, 1.0]]))
    x_init = [0.0, 0.0]
    x_final = [1.0, 0.2]
    # initial conditions
    opti.subject_to(x1[:, 0] == x_init[0])
    opti.subject_to(x2[:, 0] == x_init[1])
    # terminal condition
    opti.subject_to(x1[:, N] == x_final[0])
    opti.subject_to(x2[:, N] == x_final[1])
    for k in range(N):
        opti.subject_to((p / 2) ** 2 <= b * (x1[:, k] - c) ** 2 + (x2[:, k] - d) ** 2)
        opti.subject_to(X[:, k + 1] == A @ X[:, k] + B @ U[:, k])
    opti.subject_to(opti.bounded(-1.0, U, 1.0))
    # define objective
    opti.minimize(sumsqr(U))
    opti.solver('ipopt')

    """
    Plot trajectories
    """
    x_init = np.asarray([[0.0], [0.0]])
    x_final = np.asarray([[1.0], [0.2]])
    params = np.asarray([[b], [c], [d]])

    x1 = np.arange(-0.5, 1.5, 0.02)
    x2 = np.arange(-0.5, 1.5, 0.02)
    xx, yy = np.meshgrid(x1, x2)
    # eval objective and constraints
    c2 = b*(xx -c)**2+(yy-d)**2 - (p/2)**2

    # define problem parameters
    x_torch = torch.tensor(x_init).float().transpose(0, 1)
    r_torch = torch.tensor(x_final).float().transpose(0, 1)
    params_torch = torch.tensor(params).float().transpose(0, 1)
    Yf = torch.ones([args.nsteps, 1])

    # DPC policy
    policy = model.components[3]
    dynamics_model = model.components[4]

    start_time = time.time()
    uout = policy({'x0_estimator': x_torch, 'x0_parameters': params_torch,
                   'x0_refs': r_torch})
    xout = dynamics_model({**uout, 'x0_estimator': x_torch, 'Yf': Yf})
    sol_time = time.time() - start_time
    print(f'DPC solution time: {sol_time}')

    # IPOPT
    start_time = time.time()
    sol = opti.solve()
    sol_time_casadi = time.time() - start_time
    print(f'IPOPT solution time: {sol_time_casadi}')

    X_model = xout['X_pred_dynamics'][:, 0, :].detach().numpy()
    U_dpc = uout['U_pred_policy'][:, 0, :].detach().numpy()
    # overall state trajectory
    X_traj = np.concatenate((x_init.transpose(), X_model), axis=0)
    X_ipopt = sol.value(X).transpose()
    U_ipopt = sol.value(U).transpose()

    if show_plot:
        # Plot
        fig, ax = plt.subplots(1,1)
        # plot obstacle
        cg2 = ax.contour(xx, yy, c2, [0], colors='mediumblue', alpha=0.6)
        plt.setp(cg2.collections, facecolor='mediumblue')
        # plot initial and target state
        ax.plot(x_final[0], x_final[1], 'r*', markersize=10)
        ax.plot(x_init[0], x_init[1], 'g*', markersize=10)
        # plot IPOPT trajectory
        ax.plot(X_ipopt[:, 0], X_ipopt[:, 1], '-', color='orange', label='IPOPT', linewidth=2)
        # plot DPC state trajectory
        ax.plot(X_traj[:, 0], X_traj[:, 1], '--', color='purple', label='DPC', linewidth=2)
        plt.legend(fontsize=18)
        ax.set_xlabel('$x_2$', fontsize=18)
        ax.set_ylabel('$x_1$', fontsize=18)
        plt.xlim(-0.25, 1.25)
        plt.ylim(-0.5, 1.0)
        plt.grid()
        # plot DPC states and control actions
        fig, ax = plt.subplots(2,1)
        ax[0].plot(X_traj)
        ax[1].plot(U_dpc)
        # IPOPT trajectories
        ax[0].plot(X_ipopt)
        ax[1].plot(U_ipopt)
    print(f'DPC energy use: {np.mean(U_dpc**2)}')
    print(f'IPOPT energy use: {np.mean(sol.value(U) ** 2)}')

    """
    Plot loss function
    """
    if show_plot:
        Loss = np.ones([x1.shape[0], x2.shape[0]])*np.nan
        for i in range(x1.shape[0]):
            for j in range(x2.shape[0]):
                ref_penalty = torch.mean((torch.tensor([x1[i], x2[j]])-r_torch)**2).detach().numpy()
                con = b * (x1[i] - c) ** 2 + (x2[j] - d) ** 2 - (p / 2) ** 2
                con_penalty = np.maximum(-con, 0)
                Loss[i, j] = ref_penalty+100*con_penalty

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(xx, yy, Loss.transpose(),
                               cmap=cm.viridis,
                               linewidth=0, antialiased=False)
        ax.set(ylabel='$x_1$')
        ax.set(xlabel='$x_2$')
        ax.set(zlabel='$L$')
        # ax.set(title='Loss landscape')
        fig, ax = plt.subplots()
        cp = ax.contourf(xx, yy, Loss.transpose(), cmap=cm.viridis, alpha=0.7,
                         levels=[0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
                                 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        ax.set_xlabel('$x_2$', fontsize=18)
        ax.set_ylabel('$x_1$', fontsize=18)
        plt.xlim(-0.25, 1.25)
        plt.ylim(-0.5, 1.0)
        ax.set(title='State loss contours')

    return sol_time, sol_time_casadi


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
    nsim = 30000        # rule of thumb: more data samples -> improved control performance
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
        "x_final": np.random.uniform(low=[0.8, -0.2], high=[1.2, 0.2], size=(nsim, nx)),
        'params': np.random.uniform(low=[1.0, 0.3, 0.0], high=[3.0, 0.7, 0.4], size=(nsim,3)),
        # "b": np.random.uniform(low=1.0, high=3.0) * np.ones([nsim, 1]),
        # "c": np.random.uniform(low=0.3, high=0.7) * np.ones([nsim, 1]),
        # "d": np.random.uniform(low=0.0, high=0.4) * np.ones([nsim, 1]),
        "U": np.random.randn(nsim, nu),
    }
    nstep_data, loop_data, dims = get_sequence_dataloaders(sequences, args.nsteps, batch_size=None)
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
        input_keys=[f'x0_{estimator.name}', f'x0_{parameters.name}',
                    f'x0_{refs.name}'],
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
    # r = Variable('x_finalp')[[-1], :, :]
    r = Variable(f'x0_{refs.name}')
    # sampled parameters
    p = 0.5
    # b = Variable('paramsp')[[-1], :, [0]]
    # c = Variable('paramsp')[[-1], :, [1]]
    # d = Variable('paramsp')[[-1], :, [2]]

    b = Variable(f'x0_{parameters.name}')[:, [0]]
    c = Variable(f'x0_{parameters.name}')[:, [1]]
    d = Variable(f'x0_{parameters.name}')[:, [2]]

    Q_con = 100.0
    obstacle = Q_con * (((p / 2) ** 2 <= b * (x1 - c) ** 2 + (x2 - d) ** 2)^2)      # eliptic obstacle
    Q_u = 10.0
    action_loss = Q_u*((u==0)^2)                       # control penalty
    Q_r = 1.0
    reference_loss = Q_r*((r==x[[-1], :, :])^2)         # target posistion
    Q_r_mean = 0.0
    # reference_loss_mean = Q_r_mean*(r==x)         # target
    reference_loss_mean = Q_r_mean*((r==x)^2)         # target
    Q_dx = 1.0
    state_smoothing = Q_dx*((x[1:] == x[:-1])^2)           # delta x penalty
    Q_du = 10.0
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
    # plotting a range of obstacles
    # 1.0 <= b <= 3.0
    # 0.3 <= c < 0.7
    # 0.0 <= d <= 0.4
    p, b, c, d = 0.5, 2.0, 0.7, 0.4
    sol_time, _ = plot_obstacle(model, args, b, c, d)
    p, b, c, d = 0.5, 2.0, 0.3, 0.4
    sol_time, _ = plot_obstacle(model, args, b, c, d)
    p, b, c, d = 0.5, 2.0, 0.3, 0.0
    sol_time, _ = plot_obstacle(model, args, b, c, d)
    p, b, c, d = 0.5, 2.0, 0.7, 0.0
    sol_time, _ = plot_obstacle(model, args, b, c, d)
    p, b, c, d = 0.5, 2.0, 0.7, 0.2
    sol_time, _ = plot_obstacle(model, args, b, c, d)

    times = []
    times_ipopt = []
    for k in range(50):
        sol_time, sol_time_casadi = plot_obstacle(model, args,
                                                  b, c, d, show_plot=False)
        times.append(sol_time)
        times_ipopt.append(sol_time_casadi)
    print(f'DPC mean solution time: {np.mean(times)}')
    print(f'DPC max solution time: {np.max(times)}')
    print(f'IPOPT mean solution time: {np.mean(sol_time_casadi)}')
    print(f'IPOPT max solution time: {np.max(sol_time_casadi)}')

    print(f'Q_r {Q_r}')
    print(f'Q_u {Q_u}')
    print(f'Q_du {Q_du}')
    print(f'Q_dx {Q_dx}')
    print(f'Q_con {Q_con}')
    print(f'Q_r_mean {Q_r_mean}')
