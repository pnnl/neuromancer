"""
DPC double integrator example with given system dynamics model
time varying reference tracking problem
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
from neuromancer.visuals import VisualizerClosedLoop
from neuromancer.callbacks import SysIDCallback


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
    gp.add("-nsteps", type=int, default=2,
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
    gp.add("-epochs", type=int, default=400,
           help='Number of training epochs')
    gp.add("-lr", type=float, default=0.001,
           help="Step size for gradient descent.")
    gp.add("-patience", type=int, default=100,
           help="How many epochs to allow for no improvement in eval metric before early stopping.")
    gp.add("-warmup", type=int, default=10,
           help="Number of epochs to wait before enacting early stopping policy.")
    return parser


def cl_simulate(A, B, policy, compensator, args,
                nstep=50, x0=np.ones([2, 1]), ref=None, save_path=None):
    """

    :param A:
    :param B:
    :param net:
    :param nstep:
    :param x0:
    :return:
    """
    N = args.nsteps
    Anp = A.detach().numpy()
    Bnp = B.detach().numpy()
    x = x0
    X = N*[x]
    U = []
    if ref is None:
        ref = np.zeros([nstep+N, 1])
    for k in range(N, nstep+1-N):
        x_torch = torch.tensor(x).float().transpose(0, 1)
        # taking a first control action based on RHC principle
        Rf = torch.tensor([ref[k:k+N, :]]).float().transpose(0, 1)
        u_nominal = policy({'x0_estimator': x_torch, 'Rf': Rf})['U_pred_policy']
        Rp = torch.tensor([ref[k-N:k, :]]).float().transpose(0, 1)
        Yp_np = np.stack(X[k-N:k])[:, args.controlled_outputs, 0]
        Yp = torch.tensor([Yp_np]).float().transpose(0, 1)
        if k == N:
            Ep = Rp - Yp  # past tracking error signal
        else:
            Ep = 0.2*Ep + Rp - Yp  # integrating tracking error signal
        uout = compensator({f'{policy.output_keys}': u_nominal, 'Ep': Ep})
        u = uout['U_pred_compensator'][0, :, :].detach().numpy().transpose()
        # closed loop dynamics
        x = np.matmul(Anp, x) + np.matmul(Bnp, u)
        X.append(x)
        U.append(u)
    Xnp = np.asarray(X[N:])[:, :, 0]
    Unp = np.asarray(U)[:, :, 0]

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ref[N:], 'k--', label='r', linewidth=2)
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
        plt.savefig(save_path+'/closed_loop_dpc_osf.pdf')


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
    # terminal constraints as deviations from desired reference
    xN_min = -0.1
    xN_max = 0.1
    # reference bounds for the controlled state
    ref_min = 0.0
    ref_max = 2.0

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
        "R": np.concatenate([np.random.uniform(low=ref_min, high=ref_max)*np.ones([args.nsteps, 1])
                             for i in range(int(nsim/args.nsteps))]),
    }
    sequences['E'] = sequences['R'] - sequences['Y'][:, args.controlled_outputs]
    dataset = Dataset(nsim=nsim, ninit=0, norm=args.norm, nsteps=args.nsteps,
                      device='cpu', sequences=sequences, name='closedloop')

    """
    # # #  System model
    """
    # Fully observable estimator as identity map:
    # x_0 = Yp
    # Yp = [y_-N, ..., y_0]
    estimator = estimators.FullyObservable({**dataset.dims, "x0": (nx,)},
                                           nsteps=args.nsteps,  # future window Nf
                                           window_size=1,  # past window Np <= Nf
                                           input_keys=["Yp"],
                                           name='estimator')
    # A, B, C linear maps
    fu = slim.maps['linear'](nu, nx)
    fx = slim.maps['linear'](nx, nx)
    fy = slim.maps['linear'](nx, ny)
    # LTI SSM
    # x_k+1 = Ax_k + Bu_k
    # y_k+1 = Cx_k+1
    dynamics_model = dynamics.BlockSSM(fx, fy, fu=fu, name='dynamics',
                                       input_keys={'x0': f'x0_{estimator.name}'})
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
    # # #  Control policy + offset-free (OSF) feedback compensator
    """
    # full state feedback control policy with reference preview Rf
    # U_policy = p(x_0, Rf)
    # U_policy = [u_0, ..., u_N]
    # Rf = [r_0, ..., r_N]
    activation = activations['relu']
    linmap = slim.maps['linear']
    block = blocks.MLP
    policy = policies.MLPPolicy(
        {f'x0_{estimator.name}': (dynamics_model.nx,), **dataset.dims},
        nsteps=args.nsteps,
        bias=args.bias,
        linear_map=linmap,
        nonlin=activation,
        hsizes=[args.nx_hidden] * args.n_layers,
        input_keys=[f'x0_{estimator.name}', 'Rf'],
        name='policy',
    )
    policy.output_keys = f'U_pred_{policy.name}'
    # feedback compensator to mitigate offset from past observations: Ep = Rp-Yp
    # by modulating the policy output (policy.output_keys)
    # compensator plays similar role than integrator in PID
    # U_compensator = c(Ep)
    # Uf = U_policy + U_compensator
    linmap_c = slim.maps['pf']      # using Perron-Frobenius positive operator
    # min and max eigenvalues of the compensator: modulate the integrator signal gain in the closed loop
    linargs = {"sigma_min": 0.0, "sigma_max": 0.5}
    compensator = policies.LinearCompensator(dataset.dims,
                                             policy.output_keys,
                                             nsteps=args.nsteps,
                                             bias=False,
                                             linear_map=linmap_c,
                                             linargs=linargs,
                                             input_keys=['Ep'],
                                             name='compensator')
    # link policy with the model through the input keys
    dynamics_model.input_keys[dynamics_model.input_keys.index('Uf')] = 'U_pred_compensator'

    """
    # # #  DPC objectives and constraints
    """
    # objectives
    reference_loss = Objective(
        [f'Y_pred_{dynamics_model.name}', "Rf"],
        lambda pred, ref: F.mse_loss(pred[:, :, args.controlled_outputs], ref),
        weight=args.Qr,
        name="ref_loss",
    )
    action_loss = Objective(
        [f"U_pred_{compensator.name}"],
        lambda x:  torch.norm(x, 2),
        weight=args.Qu,
        name="u^T*Qu*u",
    )
    du_loss = Objective(
        [f"U_pred_{compensator.name}"],
        lambda x:  F.mse_loss(x[1:], x[:-1]),
        weight=args.Qdu,
        name="control_smoothing",
    )
    # regularization
    regularization1 = Objective(
        [f"reg_error_{policy.name}"], lambda reg: reg,
        weight=args.Q_sub, name="reg_loss1",
    )
    # regularization
    regularization2 = Objective(
        [f"reg_error_{compensator.name}"], lambda reg: reg,
        weight=args.Q_sub, name="reg_loss2",
    )
    # constraints
    state_lower_bound_penalty = Objective(
        [f'Y_pred_{dynamics_model.name}', "Y_minf"],
        lambda x, xmin: torch.norm(F.relu(-x + xmin), 1),
        weight=args.Q_con_x,
        name="state_lower_bound",
    )
    state_upper_bound_penalty = Objective(
        [f'Y_pred_{dynamics_model.name}', "Y_maxf"],
        lambda x, xmax: torch.norm(F.relu(x - xmax), 1),
        weight=args.Q_con_x,
        name="state_upper_bound",
    )
    terminal_lower_bound_penalty = Objective(
        [f'Y_pred_{dynamics_model.name}', "Rf"],
        lambda x, ref: torch.norm(F.relu(-x[:, :, args.controlled_outputs] + ref + xN_min), 1),
        weight=args.Qn,
        name="terminl_lower_bound",
    )
    terminal_upper_bound_penalty = Objective(
        [f'Y_pred_{dynamics_model.name}', "Rf"],
        lambda x, ref: torch.norm(F.relu(x[:, :, args.controlled_outputs] - ref - xN_max), 1),
        weight=args.Qn,
        name="terminl_upper_bound",
    )
    # alternative definition: args.nsteps*torch.mean(F.relu(-u + umin))
    inputs_lower_bound_penalty = Objective(
        [f"U_pred_{compensator.name}", "U_minf"],
        lambda u, umin: torch.norm(F.relu(-u + umin), 1),
        weight=args.Q_con_u,
        name="input_lower_bound",
    )
    # alternative definition: args.nsteps*torch.mean(F.relu(u - umax))
    inputs_upper_bound_penalty = Objective(
        [f"U_pred_{compensator.name}", "U_maxf"],
        lambda u, umax: torch.norm(F.relu(u - umax), 1),
        weight=args.Q_con_u,
        name="input_upper_bound",
    )

    objectives = [regularization1, regularization2, reference_loss, du_loss]
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
    components = [estimator, policy, compensator, dynamics_model]
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
    # visualizer
    plot_keys = ["Y_pred", "U_pred"]  # variables to be plotted
    visualizer = VisualizerClosedLoop(
        dataset, policy, plot_keys, args.verbosity, savedir=args.savedir
    )
    # trainer
    trainer = Trainer(
        model,
        dataset,
        optimizer,
        logger=logger,
        callback=SysIDCallback(simulator, visualizer),
        epochs=args.epochs,
        patience=args.patience,
        warmup=args.warmup,
    )
    # Train control policy
    best_model = trainer.train()

    """
    # # #  Plots and Analysis
    """
    # plot closed loop trajectories with time varying reference
    ref_step = 40
    R = np.concatenate([0.5*np.ones([ref_step, 1]),
                        1*np.ones([ref_step, 1]), 0*np.ones([ref_step, 1])])
    cl_simulate(A, B, policy, compensator, args=args, nstep=R.shape[0],
                x0=1.5*np.ones([2, 1]), ref=R, save_path='test_control')
