"""
Stochastic Differentiable predictive control (S-DPC)
Learning to stabilize unstable linear double integrator
system with given system dynamics model subject to additive uncertainties
using scenario stochastic MPC approach

"""

import torch
import torch.nn.functional as F
import slim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import seaborn as sns

DENSITY_PALETTE = sns.color_palette("crest_r", as_cmap=True)
DENSITY_FACECLR = DENSITY_PALETTE(0.01)
sns.set_theme(style="white")
import time

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
from neuromancer.visuals import VisualizerDobleIntegrator
from neuromancer.callbacks import DoubleIntegratorCallback


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
           help="prediction horizon.")          # tuned values: 1, 2
    gp.add("-Qx", type=float, default=5.0,
           help="state weight.")                # tuned value: 5.0
    gp.add("-Qu", type=float, default=0.2,
           help="control action weight.")       # tuned value: 0.2
    gp.add("-Qn", type=float, default=1.0,
           help="terminal penalty weight.")     # tuned value: 1.0
    gp.add("-Q_sub", type=float, default=0.0,
           help="regularization weight.")
    gp.add("-Q_con_x", type=float, default=10.0,
           help="state constraints penalty weight.")  # tuned value: 10.0
    gp.add("-Q_con_u", type=float, default=100.0,
           help="Input constraints penalty weight.")  # tuned value: 100.0
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


def cl_simulate_stoch(A, B, policy, args, umin, umax, w_sigma=0.1,
                      nstep=50, x0=np.ones([2, 1]), ref=None, save_path=None):
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
    Umin = umin*np.ones(nstep+1)
    Umax = umax*np.ones(nstep+1)
    N = args.nsteps
    if ref is None:
        ref = np.zeros([nstep, 1])

    X_trajs = []
    U_trajs = []
    w_runs = 20
    for j in range(w_runs):
        x = x0
        X = [x]
        U = []
        for k in range(nstep+1-N):
            wf = w_sigma * np.random.randn(2, 1)  # additive uncertainty
            x_torch = torch.tensor(x).float().transpose(0, 1)
            # taking a first control action based on RHC principle
            r_k = torch.tensor([ref[k:k+N, :]]).float().transpose(0, 1)
            uout = policy({'x0_estimator': x_torch, 'Rf': r_k})
            u = uout['U_pred_policy'][0,:,:].detach().numpy().transpose()
            # closed loop dynamics
            x = np.matmul(Anp, x) + np.matmul(Bnp, u) + wf
            X.append(x)
            U.append(u)
        Xnp = np.asarray(X)[:, :, 0]
        Unp = np.asarray(U)[:, :, 0]
        X_trajs.append(Xnp)
        U_trajs.append(Unp)

    fig, ax = plt.subplots(2, 1)
    for j in range(w_runs):
        ax[0].plot(X_trajs[j], linewidth=1.0)
    ax[0].plot(ref, 'k--', label='r', linewidth=2)
    ax[0].set(ylabel='$x$')
    ax[0].set(xlabel='time')
    ax[0].grid()
    ax[0].set_xlim(0, nstep)
    for j in range(w_runs):
        ax[1].plot(U_trajs[j], drawstyle='steps',  linewidth=1.0)
    ax[1].plot(Umin, linestyle='--', color='k', linewidth=2)
    ax[1].plot(Umax,  linestyle='--', color='k', linewidth=2)
    ax[1].set(ylabel='$u$')
    ax[1].set(xlabel='time')
    ax[1].grid()
    ax[1].set_xlim(0, nstep)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path+'/closed_loop_sdpc.pdf')


def cl_validate(A, B, C, policy, nstep=50, x0=None, w_sigma=0.1,
                xmin=-10, xmax=10,  umin=-1, umax=1, xmin_f=-0.4, xmax_f=0.4,
                n_sim_p=100, n_sim_w=10, epsilon=0.1, delta=0.99):
    """

    :param A:
    :param B:
    :param net:
    :param nstep:
    :param x0:
    :return:
    """
    A = A.detach().numpy()
    B = B.detach().numpy()

    X_trajs = []
    U_trajs = []
    times = []
    I_s = []        # stability indicator
    I_c = []        # constraints indicator
    X_c_mean = []
    U_c_mean = []

    samples = n_sim_w*n_sim_p
    for j in range(samples):
        if x0 is None:
            # x0 = np.zeros([A.shape[0],1])
            X_dist_bound=0.2
            x0 = np.random.uniform(low=-X_dist_bound, high=X_dist_bound, size=(A.shape[0],1))
        x = x0
        X = [x]
        U = []
        Y = []
        for k in range(nstep+1):
            wf = w_sigma * np.random.randn(A.shape[0], 1)  # additive uncertainty
            x_torch = torch.tensor(x).float().transpose(0, 1)
            # taking a first control action based on RHC principle
            start_time = time.time()
            uout = policy({'x0_estimator': x_torch})
            sol_time = time.time() - start_time
            times.append(sol_time)
            u = uout['U_pred_policy'][0,:,:].detach().numpy().transpose()
            u = np.clip(u, umin, umax)
            # closed loop dynamics
            x = np.matmul(A, x) + np.matmul(B, u) + wf
            y = np.matmul(C, x)
            X.append(x)
            U.append(u)
            Y.append(y)
        Xnp = np.asarray(X)[:, :, 0]
        Unp = np.asarray(U)[:, :, 0]
        X_trajs.append(Xnp)
        U_trajs.append(Unp)

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
    mu_tilde = np.mean(0.5*np.asarray(I_s) + 0.5*np.asarray(I_c))
    # evaluate conficence level given risk tolerance epsilon
    confidence = 1 - 2 * np.exp(-2 * samples * epsilon ** 2)
    mu = mu_tilde - epsilon
    # evaluate risk lower bound given number of samples, and desired confidence
    mu_lbound = mu_tilde - np.sqrt(-np.log(delta / 2) / (2 * samples))

    print(f'empirical risk {mu_tilde}')
    print(f'risk lower bound beta {mu_lbound}')
    print(f'choosen condifence delta {delta}')

    return mu, mu_tilde, confidence, mu_lbound, \
               I_s, I_c, X_c_mean, U_c_mean


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
    n_sim_p = 10000  # number of parametric samples
    n_sim_w = 10  # number of disturbance samples per parameter
    nsim = n_sim_p * n_sim_w  # rule of thumb: more data samples -> improved control performance
    # constraints bounds
    umin = -1
    umax = 1
    xmin = -10
    xmax = 10
    xN_min = -0.1
    xN_max = 0.1
    # uncertainties
    w_sigma = 0.1

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
        # "Y": 3*np.random.randn(nsim, nx),
        "U": np.random.randn(nsim, nu),
        "Y": np.repeat(np.random.uniform(low=-10., high=10., size=(n_sim_p, nx)),
                       n_sim_w, axis=0),
        "w": w_sigma * np.random.randn(nsim, nx),  # additive uncertainties
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
    fd = slim.maps['linear'](nx, nx)

    # LTI SSM
    # x_k+1 = Ax_k + Bu_k
    # y_k+1 = Cx_k+1
    dynamics_model = dynamics.BlockSSM(fx, fy, fu=fu, fd=fd, name='dynamics',
                                       input_key_map={'x0': f'x0_{estimator.name}',
                                                      'Uf': 'U_pred_policy', 'Df': 'wf'})

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
    dynamics_model.fd.linear.weight = torch.nn.Parameter(torch.eye(nx, nx))
    # fix model parameters
    dynamics_model.requires_grad_(False)


    """
    # # #  DPC objectives and constraints
    """
    u = Variable(f"U_pred_{policy.name}")
    y = Variable(f"Y_pred_{dynamics_model.name}")
    # constraints bounds variables
    uminb = Variable("U_minf")
    umaxb = Variable("U_maxf")
    yminb = Variable("Y_minf")
    ymaxb = Variable("Y_maxf")

    action_loss = args.Qu * ((u == 0) ^ 2)  # control penalty
    regulation_loss = args.Qx * ((y == 0) ^ 2)  # target posistion
    # constraints
    state_lower_bound_penalty = args.Q_con_x*(y > yminb)
    state_upper_bound_penalty = args.Q_con_x*(y < ymaxb)
    inputs_lower_bound_penalty = args.Q_con_u*(u > uminb)
    inputs_upper_bound_penalty = args.Q_con_u*(u < umaxb)
    terminal_lower_bound_penalty = args.Qn*(y[[-1], :, :] > xN_min)
    terminal_upper_bound_penalty = args.Qn*(y[[-1], :, :] < xN_max)
    # regularization
    regularization = Loss(
        [f"reg_error_{policy.name}"], lambda reg: reg,
        weight=args.Q_sub, name="reg_loss",
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
    metrics = ["nstep_dev_loss"]
    logger = BasicLogger(args=args, savedir=args.savedir, verbosity=args.verbosity,
                         stdout=metrics)
    logger.args.system = 'dpc_stabilize'
    # device and optimizer
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # visualizer object to be called in callback for plotting
    visualizer = VisualizerDobleIntegrator(train_data, model,
                     args.verbosity, savedir=args.savedir,
                     nstep=40, x0=1.5 * np.ones([2, 1]),
                     training_visuals=False, trace_movie=False)

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
        callback=DoubleIntegratorCallback(visualizer),
        warmup=args.warmup,
    )
    # Train control policy
    best_model = trainer.train()
    best_outputs = trainer.test(best_model)


    """
    # # #  Plots and Analysis
    """
    cl_simulate_stoch(A, B, policy, umin=umin, umax=umax, args=args, nstep=50,
                x0=1.5*np.ones([2, 1]), w_sigma=w_sigma, save_path='test_control')

    mu, mu_tilde, confidence, mu_lbound, I_s, I_c, X_c_mean, U_c_mean \
        = cl_validate(A, B, C, policy, n_sim_p=3333, n_sim_w=10, w_sigma=w_sigma)