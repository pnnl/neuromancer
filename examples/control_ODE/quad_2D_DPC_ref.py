"""
Differentiable predictive control (DPC)

Learning to control 2D quadcopter model

based on 2D quadcopter LQR example with LTI model from:
https://github.com/charlestytler/QuadcopterSim/blob/master/quad2D_lqr.py
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
from scipy.signal import cont2discrete, lti, dlti, dstep
from pylab import *
import numpy as np
import scipy.linalg as splin

from neuromancer.activations import activations
from neuromancer import blocks, estimators, dynamics
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.constraint import Loss
from neuromancer import policies
import neuromancer.arg as arg
from neuromancer.dataset import normalize_data, split_sequence_data, SequenceDataset
from torch.utils.data import DataLoader
from neuromancer.loggers import BasicLogger
from neuromancer.constraint import Variable


def arg_dpc_problem(prefix=''):
    """
    Command line parser for DPC problem definition arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers
                         are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = arg.ArgParser(prefix=prefix, add_help=False)
    gp = parser.group("DPC")
    gp.add("-nsteps", type=int, default=10,
           help="prediction horizon.")          # tuned values: 1, 2
    gp.add("-nx_hidden", type=int, default=20,
           help="Number of hidden states")
    gp.add("-n_layers", type=int, default=4,
           help="Number of hidden layers")
    gp.add("-bias", action="store_true",
           help="Whether to use bias in the neural network block component models.")
    gp.add("-epochs", type=int, default=2000,
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


def cl_simulate(A, B, C, policy, nstep=50, x0=np.ones([2, 1]), ref=None, save_path=None):
    """

    :param A:
    :param B:
    :param net:
    :param nstep:
    :param x0:
    :return:
    """
    x = x0
    X = [x]
    U = []
    Y = []
    for k in range(nstep+1):
        x_torch = torch.tensor(x).float().transpose(0, 1)
        # taking a first control action based on RHC principle
        uout = policy({'x0_estimator': x_torch})
        u = uout['U_pred_policy'][0,:,:].detach().numpy().transpose()
        # closed loop dynamics
        x = np.matmul(A, x) + np.matmul(B, u)
        y = np.matmul(C, x)
        X.append(x)
        U.append(u)
        Y.append(y)
    Xnp = np.asarray(X)[:, :, 0]
    Unp = np.asarray(U)[:, :, 0]
    Ynp = np.asarray(Y)[:, :, 0]

    if ref is None:
        ref = np.zeros(Ynp.shape)
    else:
        ref = ref[0:Ynp.shape[0], :]

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Xnp, label='x', linewidth=2)
    ax[0].plot(ref, 'k--', label='r', linewidth=2)
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


if __name__ == "__main__":

    """
    # # #  Arguments
    """
    parser = arg.ArgParser(parents=[arg.log(),
                                    arg_dpc_problem()])
    args, grps = parser.parse_arg_groups()
    args.bias = True

    """
    # # # 2D quadcopter model 
    """
    # Constants
    m = 2
    I = 1
    d = 0.2
    g = 9.8  # m/s/s
    DTR = 1 / 57.3
    RTD = 57.3

    # Nonlinear Dynamics Equations of Motion of a 2D quadcopter
    def f(x, u):
        # idx  0,1,2,3,4,5
        # states = [u,v,q,x,y,theta]
        # u = longitudal velocity
        # v = lateral velocity
        # q = pitch rate
        # x = longitudal position
        # y = lateral position
        # theta = pitch attitude
        xnew = zeros(6)
        xnew[0] = -1 * x[2] * x[1] + 1 / m * (u[0] + u[1]) * math.sin(x[5])
        xnew[1] = x[2] * x[0] + 1 / m * (u[0] + u[1]) * math.cos(x[5]) - g
        xnew[2] = 1 / I * (u[0] - u[1]) * d
        xnew[3] = x[0]
        xnew[4] = x[1]
        xnew[5] = x[2]
        return xnew

    # 4th Order Runge Kutta integrator
    def RK4(x, u, dt):
        K1 = f(x, u)
        K2 = f(x + K1 * dt / 2, u)
        K3 = f(x + K2 * dt / 2, u)
        K4 = f(x + K3 * dt, u)
        xest = x + 1 / 6 * (K1 + 2 * K2 + 2 * K3 + K4) * dt
        return xest

    # Initial Conditions
    x_4 = 20  # y 20m
    x_5 = 20 * DTR  # theta 20deg
    # Calculate equilibrium values
    ue = 0
    ve = 0
    qe = 0
    theta_e = 0
    T1e = g * m / 2 / math.cos(theta_e)
    T2e = g * m / 2 / math.cos(theta_e)
    # Initial inputs
    u_0 = T1e
    u_1 = T2e

    # Create Jacobian matrix
    A = np.array([[0, -qe, -ve, 0, 0, g],
                  [qe, 0, ue, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0]])
    # Create linear Input matrix
    B = np.array([[0, 0],
                  [1 / m, 1 / m],
                  [d / I, -d / I],
                  [0, 0],
                  [0, 0],
                  [0, 0]])
    # output matrix
    C = np.eye(A.shape[0])
    # problem dimensions
    nx = A.shape[0]
    ny = C.shape[0]
    nu = B.shape[1]
    # D matrix
    D = np.zeros([ny, nu])

    # discretize model with sampling time dt
    l_system = lti(A, B, C, D)
    tstep = .01  # sec
    d_system = cont2discrete((A, B, C, D), dt=tstep, method='euler')
    A, B, C, D, dt = d_system

    # number of datapoints
    nsim = 12000
    # constraints bounds - TODO: apply indivodual constraints on states
    umin = -50
    umax = 50
    xmin = -10
    xmax = 30

    """
    # # #  Dataset 
    """

    # state distributions:
    X_1 = np.random.uniform(low=-6, high=6, size=(1, nsim))
    X_2 = np.random.uniform(low=-6, high=6, size=(1, nsim))
    X_3 = np.random.uniform(low=-1, high=1, size=(1, nsim))
    X_4 = np.random.uniform(low=-5, high=20, size=(1, nsim))
    X_5 = np.random.uniform(low=10, high=25, size=(1, nsim))
    X_6 = np.random.uniform(low=-1, high=1, size=(1, nsim))
    X = np.concatenate([X_1, X_2, X_3, X_4, X_5, X_6])

    #  randomly sampled input output trajectories for training
    #  we treat states as observables, i.e. Y = X
    sequences = {
        "X_max": xmax*np.ones([nsim, nx]),
        "X_min": xmin*np.ones([nsim, nx]),
        "U_max": umax*np.ones([nsim, nu]),
        "U_min": umin*np.ones([nsim, nu]),
        # "X": np.random.uniform(low=-10, high=30, size=(nsim,nx)),
        "X": X.T,
        "Y": np.random.uniform(low=-10, high=30, size=(nsim,ny)),
        "R": 10*np.ones([nsim, 2]),
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
                                           input_keys=["Xp"],
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
    dynamics_model.fx.linear.weight = torch.nn.Parameter(torch.tensor(A, dtype=torch.float32))
    dynamics_model.fu.linear.weight = torch.nn.Parameter(torch.tensor(B, dtype=torch.float32))
    dynamics_model.fy.linear.weight = torch.nn.Parameter(torch.tensor(C, dtype=torch.float32))
    # fix model parameters
    dynamics_model.requires_grad_(False)

    """
    # # #  DPC objectives and constraints
    """
    y_c = Variable(f"X_pred_{dynamics_model.name}")[:,:,3:5]       # controller system outputs (positions)
    y_s = Variable(f"X_pred_{dynamics_model.name}")[:,:,[0,1,2,5]]       # stabilized system outputs (velocities, angles)
    x = Variable(f"X_pred_{dynamics_model.name}")       # system states
    r = Variable("Rf")                                  # references
    u = Variable(f"U_pred_{policy.name}")
    # constraints bounds variables
    umin = Variable("U_minf")
    umax = Variable("U_maxf")
    xmin = Variable("X_minf")
    xmax = Variable("X_maxf")
    # weight factors of loss function terms and constraints
    Q_r = 1.0
    Q_s = 10.0
    Q_u = 0.0
    Q_dx = 0.0
    Q_du = 0.0
    Q_con_u = 0.0
    Q_con_x = 0.0
    # define loss function terms and constraints via neuromancer constraints syntax
    reference_loss = Q_r*((r == y_c)^2)                       # track reference
    stabilize_loss = Q_s*((0 == y_s)^2)                       # stabilize system
    action_loss = Q_u*((u==0)^2)                       # control penalty

    control_smoothing = Q_du*((u[1:] == u[:-1])^2)          # delta u penalty
    state_smoothing = Q_dx*((x[1:] == x[:-1])^2)           # delta x penalty

    state_lower_bound_penalty = Q_con_x*(x > xmin)
    state_upper_bound_penalty = Q_con_x*(x < xmax)
    input_lower_bound_penalty = Q_con_u*(u > umin)
    input_upper_bound_penalty = Q_con_u*(u < umax)

    objectives = [reference_loss, stabilize_loss, action_loss]
    constraints = [
        # state_lower_bound_penalty,
        # state_upper_bound_penalty,
        # input_lower_bound_penalty,
        # input_upper_bound_penalty,
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
    metrics = ["nstep_dev_loss"]
    logger = BasicLogger(args=args, savedir=args.savedir, verbosity=args.verbosity, stdout=metrics)
    logger.args.system = 'dpc_ref'
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
    # plot closed loop trajectories from different initial conditions
    # TODO: initialize states in their correct distributions
    cl_simulate(A, B, C, policy, nstep=100,
                x0=10*np.ones([nx, 1]), ref=sequences['R'], save_path='test_control')
    cl_simulate(A, B, C, policy, nstep=100,
                x0=5*np.ones([nx, 1]), ref=sequences['R'], save_path='test_control')
    cl_simulate(A, B, C, policy, nstep=100,
                x0=1.0*np.ones([nx, 1]), ref=sequences['R'], save_path='test_control')