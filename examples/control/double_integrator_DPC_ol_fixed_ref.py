"""
Differentiable predictive control (DPC)
Learning to stabilize unstable linear double integrator system with given system dynamics model

"""

import torch
import neuromancer.slim as slim
import numpy as np
import matplotlib.pyplot as plt
import neuromancer.psl as psl
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
    gp.add("-nsteps", type=int, default=1,
           help="prediction horizon.")          # tuned values: 1, 2
    gp.add("-Qx", type=float, default=5.0,
           help="state weight.")                # tuned value: 5.0
    gp.add("-Qu", type=float, default=0.1,
           help="control action weight.")       # tuned value: 0.2
    gp.add("-Qn", type=float, default=1.0,
           help="terminal penalty weight.")     # tuned value: 1.0
    gp.add("-Q_con_x", type=float, default=10.0,
           help="state constraints penalty weight.")  # tuned value: 10.0
    gp.add("-Q_con_u", type=float, default=100.0,
           help="Input constraints penalty weight.")  # tuned value: 100.0
    gp.add("-nx_hidden", type=int, default=20,
           help="Number of hidden states")
    gp.add("-n_layers", type=int, default=4,
           help="Number of hidden layers")
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
    return parser


if __name__ == "__main__":

    """
    # # #  Arguments, dimensions, bounds
    """
    parser = arg.ArgParser(parents=[arg.log(),
                                    arg_dpc_problem()])
    args, grps = parser.parse_arg_groups()
    torch.manual_seed(args.data_seed)
    # problem dimensions
    nx = 2
    ny = 2
    nu = 1
    # number of datapoints
    nsim = 10000
    # constraints bounds
    umin = -1.
    umax = 1.
    xmin = -2.
    xmax = 2.
    xN_min = -0.1
    xN_max = 0.1

    """
    # # #  Dataset 
    """
    #  randomly sampled input output trajectories for training
    #  we treat states as observables, i.e. Y = X
    sequences = {"Y": 3*np.random.randn(nsim, nx)}
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
                   input_keys=["Yp"], name='est')
    # full state feedback control policy
    # [u_0, ..., u_N] = p(x_0)
    policy = policies.MLPPolicy(
        {estimator.output_keys[0]: (nx,), 'U': (nsim, nu)},
        nsteps=args.nsteps,
        linear_map=slim.maps['linear'],
        nonlin=activations['relu'],
        hsizes=[args.nx_hidden] * args.n_layers,
        input_keys=[estimator.output_keys[0]],
        name='pol',
    )
    # LTI SSM
    # x_k+1 = Ax_k + Bu_k
    # y_k+1 = Cx_k+1
    A = torch.tensor([[1.2, 1.0],
                      [0.0, 1.0]])
    B = torch.tensor([[1.0],
                      [0.5]])
    C = torch.tensor([[1.0, 0.0],
                      [0.0, 1.0]])
    dynamics_model = dynamics.LinearSSM(A, B, C, name='mod',
                              input_key_map={'x0': estimator.output_keys[0],
                                             'Uf': policy.output_keys[0]})
    dynamics_model.requires_grad_(False)  # fix model parameters

    """
    # # #  DPC objectives and constraints
    """
    u = variable(policy.output_keys[0])
    y = variable(dynamics_model.output_keys[2])
    # objectives
    action_loss = args.Qu * ((u == 0.) ^ 2)  # control penalty
    regulation_loss = args.Qx * ((y == 0.) ^ 2)  # target posistion
    # constraints
    state_lower_bound_penalty = args.Q_con_x*(y > xmin)
    state_upper_bound_penalty = args.Q_con_x*(y < xmax)
    inputs_lower_bound_penalty = args.Q_con_u*(u > umin)
    inputs_upper_bound_penalty = args.Q_con_u*(u < umax)
    terminal_lower_bound_penalty = args.Qn*(y[:, [-1], :] > xN_min)
    terminal_upper_bound_penalty = args.Qn*(y[:, [-1], :] < xN_max)
    # objectives and constraints names for nicer plot
    action_loss.name = "action_loss"
    regulation_loss.name = 'state_loss'
    state_lower_bound_penalty.name = 'x_min'
    state_upper_bound_penalty.name = 'x_max'
    inputs_lower_bound_penalty.name = 'u_min'
    inputs_upper_bound_penalty.name = 'u_max'
    terminal_lower_bound_penalty.name = 'y_N_min'
    terminal_upper_bound_penalty.name = 'y_N_max'
    # list of objectives and constraints
    objectives = [regulation_loss, action_loss]
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
    # construct trainer
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
    trajectories = cl_sim.simulate(nsim=50, data_init=data)
    # plot closed loop trajectories
    Umin = umin * np.ones([trajectories['y'].shape[0], 1])
    Umax = umax * np.ones([trajectories['y'].shape[0], 1])
    Ymin = xmin * np.ones([trajectories['y'].shape[0], 1])
    Ymax = xmax * np.ones([trajectories['y'].shape[0], 1])
    R = np.zeros([trajectories['y'].shape[0], 1])
    psl.plot.pltCL(Y=trajectories['x'], U=trajectories['u'], R=R,
                   Umin=Umin, Umax=Umax, Ymin=Ymin, Ymax=Ymax)
    psl.plot.pltPhase(X=trajectories['y'])