"""
Differentiable predictive control (DPC)

Learning to stabilize VTOL aircraft model
aircraft model from the text Feedback Systems by Astrom and Murray
Example 3.12 Vectored thrust aircraft
http://www.cds.caltech.edu/~murray/books/AM08/pdf/fbs-statefbk_24Jul2020.pdf
LQR code example at
https://python-control.readthedocs.io/en/0.8.3/pvtol-lqr.html

"""

import torch
import neuromancer.slim as slim
import numpy as np
import matplotlib.pyplot as plt
import neuromancer.psl as psl
from scipy.signal import cont2discrete, lti, dlti, dstep

from neuromancer.activations import activations
from neuromancer import estimators, dynamics
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.constraint import variable
from neuromancer import policies
import neuromancer.arg as arg
from neuromancer.dataset import get_sequence_dataloaders
from neuromancer.loss import get_loss
import neuromancer.simulator as sim


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
           help="prediction horizon.")
    gp.add("-nx_hidden", type=int, default=20,
           help="Number of hidden states")
    gp.add("-n_layers", type=int, default=4,
           help="Number of hidden layers")
    gp.add("-data_seed", type=int, default=408,
           help="Random seed used for simulated data")
    gp.add("-epochs", type=int, default=2000,
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
    return parser


if __name__ == "__main__":

    """
    # # #  Arguments, dimensions, bounds
    """
    parser = arg.ArgParser(parents=[arg.log(),
                                    arg_dpc_problem()])
    args, grps = parser.parse_arg_groups()
    torch.manual_seed(args.data_seed)

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
    A = torch.tensor(A).float()
    B = torch.tensor(B).float()
    C = torch.tensor(C).float()
    # constraints bounds
    umin = -5.
    umax = 5.
    xmin = -5.
    xmax = 5.

    """
    # # #  Dataset 
    """
    #  randomly sampled input output trajectories for training
    nsim = 10000   # number of datapoints
    sequences = {
        "X": np.random.randn(nsim, nx),       # sampled initial conditions
        "Y": 0.5*np.random.randn(nsim, ny),   # required for inference of nstep by dynamics class
        "R": np.ones([nsim, 1]) * x_ref.T,    # sampled reference trajectory
    }
    nstep_data, loop_data, dims = get_sequence_dataloaders(sequences, args.nsteps)
    train_data, dev_data, test_data = nstep_data
    train_loop, dev_loop, test_loop = loop_data

    """
    # # #  System model and Control policy
    """
    # Fully observable estimator as identity map: x0 = Yp[-1]
    # x_0 = e(Yp)
    # Yp = [y_-N, ..., y_0]
    estimator = estimators.FullyObservable(
                   {**dims, "x0": (nx,)},
                   input_keys=["Xp"], name='est')
    # full state feedback control policy
    # [u_0, ..., u_N] = p(x_0)
    policy = policies.MLP_boundsPolicy(
        {estimator.output_keys[0]: (nx,), 'U': (nsim, nu)},
        nsteps=args.nsteps,
        linear_map=slim.maps['linear'],
        nonlin=activations['relu'],
        hsizes=[args.nx_hidden] * args.n_layers,
        input_keys=[estimator.output_keys[0]],
        bias=True,
        min=umin,
        max=umax,
        name='pol',
    )
    # LTI SSM
    # x_k+1 = Ax_k + Bu_k
    # y_k+1 = Cx_k+1
    dynamics_model = dynamics.LinearSSM(A, B, C, name='mod',
                              input_key_map={'x0': estimator.output_keys[0],
                                             'Uf': policy.output_keys[0]})
    dynamics_model.requires_grad_(False)      # fix model parameters

    """
    # # #  DPC objectives and constraints
    """
    # variables
    u = variable(policy.output_keys[0])               # control actions
    y = variable(dynamics_model.output_keys[2])       # system outputs
    x = variable(dynamics_model.output_keys[1])       # system states
    r = variable("Rf")                                  # references
    # weight factors of loss function terms and constraints
    Q_r = 2.0
    Q_u = 0.1
    Q_dx = 0.0
    Q_du = 0.0
    Q_con_u = 2.0
    Q_con_x = 2.0
    # loss function terms and constraints via neuromancer constraints syntax
    reference_loss = Q_r*((r == x)^2)                       # track reference
    control_smoothing = Q_du*((u[:, 1:, :] == u[:, :-1, :])^2)          # delta u penalty
    state_smoothing = Q_dx*((x[:, 1:, :] == x[:, :-1, :])^2)           # delta x penalty
    action_loss = Q_u*((u == 0.)^2)                       # control penalty
    state_lower_bound_penalty = Q_con_x*(x > xmin)
    state_upper_bound_penalty = Q_con_x*(x < xmax)
    # objectives and constraints names for nicer plot
    action_loss.name = "action_loss"
    reference_loss.name = 'ref_loss'
    control_smoothing.name = 'du_loss'
    state_smoothing.name = 'dx_loss'
    state_lower_bound_penalty.name = 'x_min'
    state_upper_bound_penalty.name = 'x_max'
    # constructing problem
    objectives = [reference_loss, action_loss,
                  control_smoothing, state_smoothing]
    constraints = [
        state_lower_bound_penalty,
        state_upper_bound_penalty,
    ]
    # data (y_k) -> estimator (x_k) -> policy (u_k) -> dynamics (x_k+1, y_k+1)
    components = [estimator, policy, dynamics_model]
    # create constrained optimization loss
    loss = get_loss(objectives, constraints, train_data, args)
    # construct constrained optimization problem
    problem = Problem(components, loss)
    # plot computational graph
    problem.plot_graph()

    """
    # # #  Neuromancer trainer 
    """
    # logger and metrics
    args.savedir = 'test_control'
    args.verbosity = 1
    metrics = ["nstep_dev_loss"]
    # device and optimizer
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
    problem = problem.to(device)
    optimizer = torch.optim.AdamW(problem.parameters(), lr=args.lr)

    trainer = Trainer(
        problem,
        train_data,
        dev_data,
        test_data,
        optimizer,
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
    Test Closed Loop System
    """
    print('\nTest Closed Loop System \n')
    policy_cl = sim.ControllerPytorch(policy=policy.net,
                                      nsteps=args.nsteps, input_keys=['x'])
    system_cl = sim.DynamicsLinSSM(A.numpy(), B.numpy(), C.numpy())
    components = [policy_cl, system_cl]
    cl_sim = sim.SystemSimulator(components)
    plt.figure()
    cl_sim.plot_graph()
    data = {'x': np.ones(nx)}
    trajectories = cl_sim.simulate(nsim=60, data_init=data)
    # plot closed loop trajectories
    Umin = umin * np.ones([trajectories['y'].shape[0], 1])
    Umax = umax * np.ones([trajectories['y'].shape[0], 1])
    Ymin = xmin * np.ones([trajectories['y'].shape[0], 1])
    Ymax = xmax * np.ones([trajectories['y'].shape[0], 1])
    R = np.zeros([trajectories['y'].shape[0], 1])
    psl.plot.pltCL(Y=trajectories['x'], U=trajectories['u'], R=R,
                   Umin=Umin, Umax=Umax, Ymin=Ymin, Ymax=Ymax)

