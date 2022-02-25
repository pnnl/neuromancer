"""
Differentiable predictive control (DPC)

DPC double integrator example with given system dynamics model
time varying reference tracking problem
Augmented state space model with offset-free tracking penalty

background theory references:
    https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.690490213
    https://www.sciencedirect.com/science/article/pii/S0959152401000518
    https://arxiv.org/abs/2011.14006

MODEL architectre:
    # U_policy = p(x_k, Rf)
    # u_k = U_policy[0]
    # e_k+1 = e_k + r_k - y_k
    # [x_k+1; e_k+1] = [A, 0; -C, I] [x_k; e_k] + [B; 0] u_k + [0; I] r_k
    # y_k+1 = [C, 0] [x_k+1; e_k+1]
    # DPC losses + loss(||e_k||^2)
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
from neuromancer.constraint import Loss, Variable
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
    gp.add("-Q_osf", type=float, default=5.0,
           help="offset-free disturbance rejection weight.")   # tuned value: 5.0
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
    gp.add("-Q_Ki", type=float, default=1000.0,
           help="Integrator form penalty.")            # tuned value: 1000.0
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
    gp.add("-warmup", type=int, default=100,
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


def cl_simulate(A, B, policy, args, K_i=None, err_add=0.0, err_param=1.0,
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
    d_int = torch.zeros([1, 1])
    X = N*[x]
    U = []
    if ref is None:
        ref = np.zeros([nstep+N, 1])
    for k in range(N, nstep+1-N):
        x_torch = torch.tensor(x).float().transpose(0, 1)
        d_k = torch.tensor(ref[k]-x[args.controlled_outputs]).float().transpose(0, 1)
        d_int = d_k + d_int  # integrating tracking error
        d_int = torch.clamp(d_int, min=-1.0, max=1.0)
        x_aug = torch.cat([x_torch, d_int], 1)
        # taking a first control action based on RHC principle
        Rf = torch.tensor([ref[k:k+N, :]]).float().transpose(0, 1)
        u_nominal = policy({'x0_estimator_ctrl': x_torch, 'Rf': Rf})['U_pred_policy'][0, :, :]
        # u_nominal = policy({'x0_estimator': x_aug, 'Rf': Rf})['U_pred_policy'][0, :, :]
        # integrator gain
        if K_i is not None:
            # pick only the compensator row for controlled states
            u_int = K_i(x_aug)[:, args.controlled_outputs]
            u_nominal = u_nominal + u_int
            u_nominal = torch.clamp(u_nominal, min=args.umin, max=args.umax)
        u = u_nominal.detach().numpy().transpose()
        # closed loop dynamics
        x = err_param*np.matmul(Anp, x) + np.matmul(Bnp, u) + err_add
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
        plt.savefig(save_path+'/closed_loop_dpc_osf_int.pdf')


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
    nd = 1
    nr = 1
    # number of datapoints
    nsim = 10000    # increase sample density for more robust results
    # constraints bounds
    umin = -1
    umax = 1
    xmin = -10
    xmax = 10
    args.umin = umin
    args.umax = umax
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
                             for i in range(int(np.ceil(nsim/args.nsteps)))])[:nsim, :],
    }
    nstep_data, loop_data, dims = get_sequence_dataloaders(sequences, args.nsteps)
    train_data, dev_data, test_data = nstep_data
    train_loop, dev_loop, test_loop = loop_data

    """
    # # #  System model and Control policy
    """
    # Fully observable estimator as identity map:
    # x_0 = [Yp[-1], 0]
    # Yp = [y_-N, ..., y_0]
    estimator = estimators.FullyObservableAugmented({**dims, "x0": (nx+nd,)},
                                           nsteps=args.nsteps,  # future window Nf
                                           window_size=1,  # past window Np <= Nf
                                           nd=1, d0=0.0,   # dimensions and initial values of augmented states
                                           input_keys=["Yp"],
                                           name='estimator')
    estimator_ctrl = estimators.FullyObservable({**dims, "x0": (nx,)},
                                           nsteps=args.nsteps,  # future window Nf
                                           window_size=1,  # past window Np <= Nf
                                           input_keys=["Yp"],
                                           name='estimator_ctrl')

    # full state feedback control policy with reference preview Rf
    # U_policy = p(x_0, Rf)
    # U_policy = [u_0, ..., u_N]
    # Rf = [r_0, ..., r_N]
    activation = activations['relu']
    linmap = slim.maps['linear']
    block = blocks.MLP
    policy = policies.MLPPolicy(
        {f'x0_{estimator_ctrl.name}': (nx,), **dims},
        nsteps=args.nsteps,
        bias=args.bias,
        linear_map=linmap,
        nonlin=activation,
        hsizes=[args.nx_hidden] * args.n_layers,
        input_keys=[f'x0_{estimator_ctrl.name}', 'Rf'],
        name='policy',
    )

    # LTI SSM
    fu = slim.maps['linear'](nu, nx+nd)     #  [B; 0] u_k
    fx = slim.maps['linear'](nx+nd, nx+nd)  #  [x_k+1; e_k+1] = [A, 0; -C, I] [x_k; e_k]
    fy = slim.maps['linear'](nx+nd, ny)     #  [C, 0] [x_k+1; e_k+1]
    fd = slim.maps['linear'](nr, nx + nd)   #  [0; I] r_k   - we treat reference signal as additive disturbance
    fe = slim.maps['linear'](nx + nd, nx + nd)   #  [0, K_i; 0, 0] [x_k+1; e_k+1]   -  learnable integrator feedback K_i
    # x_k+1 = Ax_k + Bu_k + Ee_k
    # y_k+1 = Cx_k+1
    dynamics_model = dynamics.BlockSSM(fx, fy, fu=fu, fd=fd, fe=fe, name='dynamics',
                                       input_key_map={'x0': f'x0_{estimator.name}',
                                                      'Uf': 'U_pred_policy', 'Df': 'Rf'})
    # model matrices values
    A = torch.tensor([[1.2, 1.0],
                     [0.0, 1.0]])
    B = torch.tensor([[1.0],
                      [0.5]])
    C = torch.tensor([[1.0, 0.0],
                      [0.0, 1.0]])
    # augmented system matrix with disturbances
    # [x_k+1; e_k+1] = [A, 0; -C, I] [x_k; e_k] + [B; 0] u_k + [0; I] r_k
    # y_k+1 = [C, 0] [x_k+1; e_k+1]
    A_up = torch.cat((A, torch.zeros(nx, nd)), 1)
    A_low = torch.cat((torch.zeros(nd, nx), torch.eye(nd, nd)), 1)
    A_low[args.controlled_outputs, args.controlled_outputs] =\
        -C[args.controlled_outputs, args.controlled_outputs]
    A_aug = torch.cat((A_up, A_low), 0)
    B_aug = torch.cat((B, torch.zeros(nd, nu)), 0)
    C_aug = torch.cat((C, torch.zeros(nx, nd)), 1)
    E_aug = torch.cat((torch.zeros(nx, nr), torch.eye(nd, nr)), 0)
    # A_aug = torch.tensor([[1.2, 1.0, 0.0],
    #                   [0.0, 1.0, 0.0],
    #                   [-1.0, 0.0, 1.0]])
    K_i = 0.1       # integrator gain initial value
    F_aug = torch.zeros(nx+nd, nx+nd)
    F_aug[args.controlled_outputs, nx:] = K_i
    # set of model parameters
    dynamics_model.fx.linear.weight = torch.nn.Parameter(A_aug)
    dynamics_model.fu.linear.weight = torch.nn.Parameter(B_aug)
    dynamics_model.fy.linear.weight = torch.nn.Parameter(C_aug)
    dynamics_model.fd.linear.weight = torch.nn.Parameter(E_aug)
    dynamics_model.fe.linear.weight = torch.nn.Parameter(F_aug)
    # freeze model parameters
    dynamics_model.requires_grad_(False)
    # unfreeze model compensator - the desired zeros would need to be penalized as constraints
    dynamics_model.fe.linear.weight.requires_grad_(True)

    """
    # # #  DPC objectives and constraints
    """
    u = Variable(f"U_pred_{policy.name}", name='u')
    y = Variable(f"Y_pred_{dynamics_model.name}", name='y')
    x = Variable(f"X_pred_{dynamics_model.name}", name='x')
    r = Variable("Rf", name='r')
    # constraints bounds variables
    umin = Variable("U_minf")
    umax = Variable("U_maxf")
    ymin = Variable("Y_minf")
    ymax = Variable("Y_maxf")

    # objectives
    action_loss = args.Qu * ((u == 0) ^ 2)  # control penalty
    reference_loss = args.Qr * ((y[:, :, args.controlled_outputs] == r) ^ 2)  # target posistion
    du_loss = args.Qdu*((u[1:] == u[:-1]) ^ 2)
    osf_loss = args.Q_osf*((x[:, :, nx:] == 0) ^ 2)
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
    du_loss.name = "control_smoothing"
    state_lower_bound_penalty.name = 'x_min'
    state_upper_bound_penalty.name = 'x_max'
    inputs_lower_bound_penalty.name = 'u_min'
    inputs_upper_bound_penalty.name = 'u_max'
    terminal_lower_bound_penalty.name = 'y_N_min'
    terminal_upper_bound_penalty.name = 'y_N_max'

    # note: using Loss class requests to be included in the list of objectives

    # regularization
    regularization = Loss(
        [f"reg_error_{policy.name}"], lambda reg: reg,
        weight=args.Q_sub, name="reg_loss",
    )
    # integrator feedback penalty
    mask = torch.ones(dynamics_model.fe.linear.weight.shape, dtype=torch.bool)
    mask[args.controlled_outputs, nx:] = False
    # dummy callable: argument x is not used
    Ki_form_penalty = Loss(["Rf"],
        lambda x: torch.norm(torch.masked_select(dynamics_model.fe.linear.weight, mask), 1),
        weight=args.Q_Ki*nsim,
        name="Ki_form_penalty")
    Ki_min = 0
    Ki_max = 1.0
    Ki_upper_boud_penalty = Loss([],
        lambda: torch.norm(F.relu(dynamics_model.fe.linear.weight - Ki_max), 1),
        weight=args.Q_Ki*nsim,
        name="Ki_upper_bound_penalty")
    Ki_lower_boud_penalty = Loss([],
        lambda: torch.norm(F.relu(-dynamics_model.fe.linear.weight + Ki_min), 1),
        weight=args.Q_Ki*nsim,
        name="Ki_lower_bound_penalty")

    objectives = [regularization, reference_loss, du_loss, osf_loss,
                  Ki_form_penalty,
                  Ki_upper_boud_penalty,
                  Ki_lower_boud_penalty]
    constraints = [
        state_lower_bound_penalty,
        state_upper_bound_penalty,
        inputs_lower_bound_penalty,
        inputs_upper_bound_penalty,
        terminal_lower_bound_penalty,
        terminal_upper_bound_penalty]

    """
    # # #  DPC problem = objectives + constraints + trainable components 
    """
    # data (y_k) -> estimator (x_k) -> policy (u_k) -> dynamics (x_k+1, y_k+1)
    components = [estimator, estimator_ctrl, policy, dynamics_model]
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
    logger.args.system = 'integrator'
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
    # plot closed loop trajectories with time varying reference
    ref_step = 40
    R = np.concatenate([0.5*np.ones([ref_step, 1]),
                        1*np.ones([ref_step, 1]), 0*np.ones([ref_step, 1])])
    cl_simulate(A, B, policy, K_i=dynamics_model.fe, args=args, nstep=R.shape[0],
                x0=1.5*np.ones([2, 1]), ref=R, save_path='test_control')
