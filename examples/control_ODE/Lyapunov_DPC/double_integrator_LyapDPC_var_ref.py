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
from neuromancer.constraint import Variable, Loss
from neuromancer import policies
import neuromancer.arg as arg
from neuromancer.dataset import normalize_data, split_sequence_data, SequenceDataset
from torch.utils.data import DataLoader
from neuromancer.loggers import BasicLogger
from neuromancer.loss import PenaltyLoss, BarrierLoss
from neuromancer.bounds import HardMinMaxScale, HardMinMaxBound
from neuromancer.component import Component
from neuromancer.blocks import InputConvexNN, PosDef, MLP


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
    gp.add("-nsteps", type=int, default=5,
           help="prediction horizon.")          # tuned values: 1, 2
    gp.add("-Qr", type=float, default=5.0,
           help="reference tracking weight.")   # tuned value: 1.0
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



class LyapunovTrajectoryRef(Component):

    DEFAULT_INPUT_KEYS = ["x0", "Xf", "Rf"]
    DEFAULT_OUTPUT_KEYS = ["reg_error", "V_pred"]

    def __init__(
        self,
        func,
        input_key_map={},
        output_keys=[],
        name=None,
    ):
        """

        :param func: (nn.Module) scalar Lyapunov function R^n -> R
        :param input_key_map:
        :param output_keys:
        :param name:
        """

        self.update_input_keys(input_key_map=input_key_map)
        output_keys = output_keys if isinstance(output_keys, list) else [output_keys]
        output_keys = output_keys if output_keys else self.DEFAULT_OUTPUT_KEYS
        output_keys = [f"{k}_{name}" if name is not None else k for k in output_keys]
        super().__init__(input_keys=self.input_keys, output_keys=output_keys, name=name)
        self.func = func

    def update_input_keys(self, input_key_map={}):
        assert isinstance(input_key_map, dict), \
            f"{type(self).__name__} input_key_map must be dict for remapping input variable names; "
        self.input_key_map = {
            **{k: k for k in self.DEFAULT_INPUT_KEYS if k not in input_key_map.keys()},
            **input_key_map
        }
        self.input_keys = list(self.input_key_map.values())
        assert len(self.input_keys) == len(self.DEFAULT_INPUT_KEYS), \
            "Length of given input keys must equal the length of default input keys"

    def reg_error(self):
        return sum([k.reg_error() for k in self.children() if hasattr(k, 'reg_error')])

    def forward(self, data):
        """

        :param data: (dict: {str: Tensor})
        :return: output (dict: {str: Tensor})
        """
        Energy = []
        nsteps = data[self.input_key_map['Xf']].shape[0]
        r = data[self.input_key_map['Rf']][0]
        x = data[self.input_key_map['x0']]
        # coordiante shift with fixed point at the reference x-r
        Vx = self.func(x-r)
        Energy.append(Vx)
        for i in range(nsteps):
            x = data[self.input_key_map['Xf']][i]
            r = data[self.input_key_map['Rf']][i]
            Vx = self.func(x-r)
            Energy.append(Vx)
        output = {}
        output[self.output_keys[1]] = torch.stack(Energy)
        output[self.output_keys[0]] = self.reg_error()
        return output


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
    nsim = 12000        # rule of thumb: more data samples -> improved control performance
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
                             for i in range(int(np.ceil(nsim/args.nsteps)))])[:nsim, :],
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
    # full state feedback control policy with reference preview Rf
    # U_policy = p(x_0, Rf)
    # U_policy = [u_0, ..., u_N]
    # Rf = [r_0, ..., r_N]
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
        input_keys=[f'x0_{estimator.name}', 'Rf'],
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
    # # #  Lyapunov candiates
    """

    # Input convex neural network (ICNN) from https://arxiv.org/abs/2001.06116
    g = InputConvexNN(nx, 1,
                bias=args.bias,
                linear_map=linmap,
                nonlin=activation,
                hsizes=[args.nx_hidden] * args.n_layers)
    # positive-definitene of lyapunov function based on ICNN
    V_1 = PosDef(g)
    lyapunov = LyapunovTrajectoryRef(V_1, input_key_map={'x0': f'x0_{estimator.name}',
                                        'Xf': f"X_pred_{dynamics_model.name}",
                                        'Rf': 'Rf'},
                                        name='Lyapunov')

    """
    # # #  DPC objectives and constraints
    """
    u = Variable(f"U_pred_{policy.name}", name='u')
    y = Variable(f"Y_pred_{dynamics_model.name}", name='y')
    r = Variable("Rf", name='r')
    # lyapunov variables
    V = Variable(lyapunov.output_keys[1])
    # constraints bounds variables
    umin = Variable("U_minf")
    umax = Variable("U_maxf")
    ymin = Variable("Y_minf")
    ymax = Variable("Y_maxf")

    # objectives
    action_loss = args.Qu * ((u == 0) ^ 2)  # control penalty
    reference_loss = args.Qr * ((y[:, :, args.controlled_outputs] == r) ^ 2)  # target posistion
    du_loss = args.Qdu*((u[1:] == u[:-1])^2)
    # Lyanpunov constraint
    Lyap_weight = 1.
    Lyapunov_con_traj = Lyap_weight*(V[1:,:,:] - V[:-1,:,:] < 0)
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
    # regularization
    regularization = Loss(
        [f"reg_error_{policy.name}"], lambda reg: reg,
        weight=args.Q_sub, name="reg_loss",
    )

    objectives = [regularization, reference_loss, action_loss]
    constraints = [Lyapunov_con_traj,
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
    # data (y_k) -> estimator (x_k) -> policy (u_k) -> dynamics (x_k+1, y_k+1) - > V_trajectory(x_k, x_k+1)
    components = [estimator, policy, dynamics_model, lyapunov]
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
    logger.args.system = 'dpc_var_ref'
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
    # load best model dict
    problem.load_state_dict(best_model)
    policy = problem.components[1]

    """
    # # #  Plots and Analysis
    """
    # plot closed loop trajectories with time varying reference
    ref_step = 40
    R = np.concatenate([0.5*np.ones([ref_step, 1]),
                        1*np.ones([ref_step, 1]), 0*np.ones([ref_step, 1])])
    cl_simulate(A, B, policy, args=args, nstep=R.shape[0],
                x0=1.5*np.ones([2, 1]), ref=R, save_path='test_control')
