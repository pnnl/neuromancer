"""
approximate model predictive control (aMPC)

Learning to stabilize VTOL aircraft model from sampled MPC examples

aircraft model from the text Feedback Systems by Astrom and Murray
Example 3.12 Vectored thrust aircraft
http://www.cds.caltech.edu/~murray/books/AM08/pdf/fbs-statefbk_24Jul2020.pdf

LQR code example at
https://python-control.readthedocs.io/en/0.8.3/pvtol-lqr.html
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
from scipy.io import loadmat

from neuromancer.activations import activations
from neuromancer import blocks, estimators, dynamics
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer import policies
import neuromancer.arg as arg
from neuromancer.dataset import normalize_data, split_sequence_data, \
    SequenceDataset, split_static_data, StaticDataset
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


def get_dataloaders(data, norm_type=None, split_ratio=None, num_workers=0):
    """This will generate dataloaders for a given dictionary of data.
    Dataloaders are hard-coded for full-batch training to match NeuroMANCER's training setup.

    :param data: (dict str: np.array or list[dict str: np.array]) data dictionary or list of data
        dictionaries; if latter is provided, multi-sequence datasets are created and splits are
        computed over the number of sequences rather than their lengths.
    :param norm_type: (str) type of normalization; see function `normalize_data` for more info.
    :param split_ratio: (list float) percentage of data in train and development splits; see
        function `split_sequence_data` for more info.
    """

    if norm_type is not None:
        data, _ = normalize_data(data, norm_type)
    train_data, dev_data, test_data = split_static_data(data, split_ratio)

    train_data = StaticDataset(
        train_data,
        name="train",
    )
    dev_data = StaticDataset(
        dev_data,
        name="dev",
    )
    test_data = StaticDataset(
        test_data,
        name="test",
    )

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

    return (train_data, dev_data, test_data), train_data.dataset.dims


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


def cl_simulate(A, B, C, policy, nstep=50,
                umin=-5, umax=5, xmin=-5, xmax=5,
                x0=np.ones([6, 1]), ref=None, save_path=None):
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
        uout = policy({'X': x_torch})
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

    line = np.ones([Ynp.shape[0], 1])

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Xnp, label='x', linewidth=2)
    # ax[0].plot(ref, 'k--', label='r', linewidth=2)
    ax[0].plot(xmin*line, 'k--', linewidth=2)
    ax[0].plot(xmax*line, 'k--', linewidth=2)
    ax[0].set(ylabel='$x$')
    ax[0].set(xlabel='time')
    ax[0].grid()
    ax[0].set_xlim(0, nstep)
    ax[1].plot(Unp, label='u', drawstyle='steps',  linewidth=2)
    ax[1].plot(umin*line, 'k--',  linewidth=2)
    ax[1].plot(umax*line, 'k--',  linewidth=2)
    ax[1].set(ylabel='$u$')
    ax[1].set(xlabel='time')
    ax[1].grid()
    ax[1].set_xlim(0, nstep)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path+'/closed_loop_dpc.pdf')


def cl_validate(A, B, C, policy, nstep=50,
                umin=-5, umax=5, xmin=-5, xmax=5, xmin_f=-0.5, xmax_f=0.5,
                X0=np.ones([6, 100]), alpha=0.5, beta=0.5, delta=0.9, epsilon=0.1):
    """

    :param A:
    :param B:
    :param net:
    :param nstep:
    :param x0:
    :return:
    """
    samples = X0.shape[1]
    I_s = []        # stability indicator
    I_c = []        # constraints indicator
    X_c_mean = []
    U_c_mean = []

    for sample in range(samples):
        x = X0[:,[sample]]
        X = []
        U = []
        for k in range(nstep):
            x_torch = torch.tensor(x).float().transpose(0, 1)
            # taking a first control action based on RHC principle
            uout = policy({'X': x_torch})
            u = uout['U_pred_policy'][0,:,:].detach().numpy().transpose()
            # closed loop dynamics
            x = np.matmul(A, x) + np.matmul(B, u)
            X.append(x)
            U.append(u)
        Xnp = np.asarray(X)[:, :, 0]
        Unp = np.asarray(U)[:, :, 0]

        # ||x||_2 + ||u||_2
        X_norm = np.linalg.norm(Xnp, 2)
        U_norm = np.linalg.norm(Unp, 2)

        # terminal constraint satisfaction for stability
        stability_flag = int((x > xmin_f).all() and (x < xmax_f).all())
        I_s.append(stability_flag)

        # state and input constraints mean violations
        x_violations = np.mean(np.maximum(Xnp - xmax, 0) + np.maximum(-Xnp + xmin, 0))
        u_violations = np.mean(np.maximum(Unp - umax, 0) + np.maximum(-Unp + umin, 0))
        X_c_mean.append(x_violations)
        U_c_mean.append(u_violations)

        # state and input constraints satisfaction
        x_con_flag = int((Xnp > xmin).all() and (Xnp < xmax).all())
        u_con_flag = int((Unp > umin).all() and (Unp < umax).all())
        constraints_flag = x_con_flag and u_con_flag
        I_c.append(constraints_flag)

    # evaluate empirical risk
    mu_tilde = np.mean(alpha*np.asarray(I_s) + beta*np.asarray(I_c))
    # evaluate conficence level given risk tolerance epsilon
    confidence = 1-2*np.exp(-2*samples*epsilon**2)
    mu = mu_tilde-epsilon
    # evaluate risk lower bound given number of samples, and desired confidence
    mu_lbound = mu_tilde - np.sqrt(-np.log(delta/2)/(2*samples))

    return mu, mu_tilde, confidence, mu_lbound, \
           I_s, I_c, X_c_mean, U_c_mean, X_norm, U_norm


if __name__ == "__main__":

    """
    # # #  Arguments
    """
    parser = arg.ArgParser(parents=[arg.log(),
                                    arg_dpc_problem()])
    args, grps = parser.parse_arg_groups()
    args.bias = True

    """
    # # #  VTOL aircraft model 
    """
    # System parameters
    m = 4  # mass of aircraft
    J = 0.0475  # inertia around pitch axis
    r = 0.25  # distance to center of force
    g = 9.8  # gravitational constant
    c = 0.05  # damping factor (estimated)
    # State space dynamics
    xe = [0, 0, 0, 0, 0, 0]  # equilibrium point of interest
    ue = [0, m * g]
    # model matrices
    A = np.array(
        [[0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1],
         [0, 0, (-ue[0] * np.sin(xe[2]) - ue[1] * np.cos(xe[2])) / m, -c / m, 0, 0],
         [0, 0, (ue[0] * np.cos(xe[2]) - ue[1] * np.sin(xe[2])) / m, 0, -c / m, 0],
         [0, 0, 0, 0, 0, 0]]
    )
    # Input matrix
    B = np.array(
        [[0, 0], [0, 0], [0, 0],
         [np.cos(xe[2]) / m, -np.sin(xe[2]) / m],
         [np.sin(xe[2]) / m, np.cos(xe[2]) / m],
         [r / J, 0]]
    )
    # Output matrix
    C = np.array([[1., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0.]])
    D = np.array([[0, 0], [0, 0]])
    # reference
    x_ref = np.array([[0], [0], [0], [0], [0], [0]])
    # control equilibria
    u_ss = np.array([[0], [m * g]])
    # problem dimensions
    nx = A.shape[0]
    ny = C.shape[0]
    nu = B.shape[1]

    # discretize model with sampling time dt
    l_system = lti(A, B, C, D)
    d_system = cont2discrete((A, B, C, D), dt=0.2, method='euler')
    A, B, C, D, dt = d_system

    # constraints bounds
    umin = -5
    umax = 5
    xmin = -5
    xmax = 5

    """
    # # #  Dataset - examples from MPC
    """
    # STEP 1: run ./Benchmarks/pvtol_aircraft_iMPC.m to generate training dataset: aMPC_dataset.mat

    # supervised dataset generated by sampling initial conditions and solving the MPC problem
    f = loadmat("./Benchmarks/aMPC_dataset.mat")
    U = f.get("samples_u", None)  # inputs
    X = f.get("samples_x", None)  # states

    # number of total datapoints: split into 1/3 for train, dev, and test set
    # nsim = U.shape[1]
    nsim = 9000

    #  randomly sampled input output trajectories for training
    #  we treat states as observables, i.e. Y = X
    samples = {
        # "X_max": xmax*np.ones([nsim, nx]),
        # "X_min": xmin*np.ones([nsim, nx]),
        # "U_max": umax*np.ones([nsim, nu]),
        # "U_min": umin*np.ones([nsim, nu]),
        "U": U[:,0:nsim].T,
        "X": X[:,0:nsim].T,
    }

    nstep_data, dims = get_dataloaders(samples)
    train_data, dev_data, test_data = nstep_data

    """
    # # #  System model and Control policy
    """

    # full state feedback control policy
    # Uf = p(x_0)
    # Uf = [u_0, ..., u_N]
    activation = activations['relu']
    linmap = slim.maps['linear']
    block = blocks.MLP
    policy = policies.MLPPolicy(
        {**dims},
        bias=args.bias,
        linear_map=linmap,
        nonlin=activation,
        hsizes=[args.nx_hidden] * args.n_layers,
        input_keys=["X"],
        name='policy',
    )

    """
    # # #  DPC objectives and constraints
    """
    u_ref = Variable("U")                                  # references
    u = Variable(f"U_pred_{policy.name}")

    # weight factors of loss function terms and constraints
    Q = 1.0
    Q_con_u = 1.0
    # define loss function terms and constraints via neuromancer constraints syntax
    imitation_loss = Q*((u == u_ref)^2)                       # imitate MPC
    # constraints on u
    input_lower_bound_penalty = Q_con_u*(u > umin)
    input_upper_bound_penalty = Q_con_u*(u < umax)

    objectives = [imitation_loss]
    constraints = []

    """
    # # #  DPC problem = objectives + constraints + trainable components 
    """
    # data (x_k) -> policy (u_k) -> loss (u_k - u*_k)
    components = [policy]
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
    metrics = ["train_loss", "dev_loss"]
    logger = BasicLogger(args=args, savedir=args.savedir, verbosity=args.verbosity, stdout=metrics)
    logger.args.system = 'ampc_ref'
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
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="test_loss",
        eval_metric="dev_loss",
        warmup=args.warmup,
    )
    # Train control policy
    best_model = trainer.train()
    best_outputs = trainer.test(best_model)

    """
    # # #  Plots and Analysis
    """
    nstep = 60
    ref = np.ones([nstep, 1]) * x_ref.T
    # plot closed loop trajectories from different initial conditions
    cl_simulate(A, B, C, policy, nstep=nstep, umin=-5, umax=5, xmin=-5, xmax=5,
                x0=0.5*np.ones([nx, 1]), ref=ref, save_path='test_control')
    cl_simulate(A, B, C, policy, nstep=nstep, umin=-5, umax=5, xmin=-5, xmax=5,
                x0=-0.5*np.ones([nx, 1]), ref=ref, save_path='test_control')
    cl_simulate(A, B, C, policy, nstep=nstep, umin=-5, umax=5, xmin=-5, xmax=5,
                x0=0.7*np.ones([nx, 1]), ref=ref, save_path='test_control')
    cl_simulate(A, B, C, policy, nstep=nstep, umin=-5, umax=5, xmin=-5, xmax=5,
                x0=-0.7*np.ones([nx, 1]), ref=ref, save_path='test_control')

    # validate empirical risk of the trained policy
    m = 6000         # number of randomly sampled initial conditions
    delta = 0.1     # confidence level: 1-delta
    epsilon = 0.02   # risk tolerance
    # X0 = np.random.uniform(-1.0, 1.0, [nx, m])
    X0 = 0.25*np.random.randn(nx, m)
    mu, mu_tilde, confidence, mu_lbound, \
    I_s, I_c, X_c_mean, U_c_mean, X_norm, U_norm = \
        cl_validate(A, B, C, policy, nstep=50,
                umin=-5, umax=5, xmin=-5, xmax=5, xmin_f=-0.5, xmax_f=0.5,
                X0=X0, alpha=0.5, beta=0.5,
                delta=delta, epsilon=epsilon)
    print(f'empirical risk mu = {mu}')
    print(f'confidence level  = {confidence}')