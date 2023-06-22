"""
Neural Ordinary Differentiable predictive control (NO-DPC)

Controlling nonlinear ODE system with explicit neural control policy via DPC algorithm

system: Van der Pol oscillator
see section V.A in for system equations: https://arxiv.org/abs/2203.14114
objective: stabilize towards origin

"""

import torch
import torch.nn as nn

import neuromancer.slim as slim
import numpy as np
import matplotlib.pyplot as plt
import neuromancer.psl as psl

from neuromancer.activations import activations
from neuromancer import blocks, estimators, dynamics, ode, integrators
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.constraint import variable, Loss
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
    gp.add("-nsteps", type=int, default=50,
           help="prediction horizon.")
    gp.add("-Qx", type=float, default=20.0,
           help="state weight.")
    gp.add("-Qu", type=float, default=0.0,
           help="control action weight.")
    gp.add("-Q_sub", type=float, default=1.0,
           help="regularization weight.")
    gp.add("-Qn", type=float, default=10.0,
           help="terminal penalty weight.")
    gp.add("-Q_du", type=float, default=1.0,
           help="delta control action penalty weight.")
    gp.add("-Q_con_x", type=float, default=10.0,
           help="state constraints penalty weight.")
    gp.add("-Q_con_u", type=float, default=100.0,
           help="Input constraints penalty weight.")
    gp.add("-nx_hidden", type=int, default=20,
           help="Number of hidden states")
    gp.add("-n_layers", type=int, default=4,
           help="Number of hidden layers")
    gp.add("-data_seed", type=int, default=408,
           help="Random seed used for simulated data")
    gp.add("-epochs", type=int, default=250,
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
    # ground truth system model
    gt_model = psl.nonautonomous.VanDerPolControl(ts=0.02)
    # problem dimensions
    nx = gt_model.nx
    ny = gt_model.nx
    nu = gt_model.nu
    nref = nx    # number of references
    # constraints bounds
    umin = -5.
    umax = 5.
    xmin = -4.
    xmax = 4.

    """
    # # #  Dataset 
    """
    # number of datapoints
    nsim = 10000
    #  randomly sampled output trajectories and control parameters for training
    # repeats = int(np.ceil(nsim/args.nsteps))
    # list_refs = [np.random.randn(1, nref)*np.ones([args.nsteps, 1]) for k in range(repeats)]
    # Ref = np.concatenate(list_refs)
    sequences = {
        # "R": Ref[:nsim, :],
        'R': np.zeros([nsim, nx]),
        "Y": np.random.randn(nsim, nx),
    }
    nstep_data, loop_data, dims = get_sequence_dataloaders(sequences,
                                       args.nsteps, batch_size=32)
    train_data, dev_data, test_data = nstep_data
    train_loop, dev_loop, test_loop = loop_data

    """
    # # #  System model and Control policy
    """
    # Fully observable estimator as identity map: x0 = Yp[-1]
    estimator = estimators.FullyObservable(
                   {**dims, "x0": (nx,)},
                   input_keys=["Yp"], name='est')
    # ODE model
    two_tank_ode = ode.VanDerPolControl()
    two_tank_ode.mu = nn.Parameter(torch.tensor([gt_model.mu]), requires_grad=False)
    # control policy
    policy = blocks.MLP_bounds(insize=nx + nref, outsize=nu, hsizes=[32, 32],
                        nonlin=activations['gelu'], min=umin, max=umax)
    # control ODE
    control_ode = ode.ControlODE(policy=policy, ode=two_tank_ode,
                                 nx=nx, nu=nu, np=nref, u_con=[])
    # closed loop simulated via ODE integration of ControlODE class
    interp_u = lambda tq, t, u: u   #  Interpolant for offline mode
    fx_int = integrators.RK4(control_ode, interp_u=interp_u, h=gt_model.ts)
    fy = slim.maps['identity'](nx, nx)
    cl_model = dynamics.ODENonAuto(fx_int, fy, extra_inputs=['Rf'],
                    input_key_map={"x0": estimator.output_keys[0]},
                    name='closed_loop', online_flag=False)

    # regularization to log the control constraints in cl_model
    regularization = Loss(
        [f"reg_error_{cl_model.name}"], lambda reg: reg,
        weight=args.Q_sub, name="reg_loss")

    """
    # # #  DPC objectives and constraints
    """
    # variables
    y = variable(cl_model.output_keys[2])
    ref = variable("Rf")
    # objectives
    regulation_loss = args.Qx * ((y == ref) ^ 2)  # target posistion
    # constraints
    state_lower_bound_penalty = args.Q_con_x*(y > xmin)
    state_upper_bound_penalty = args.Q_con_x*(y < xmax)
    terminal_lower_bound_penalty = args.Qn*(y[:, [-1], :] > ref-0.01)
    terminal_upper_bound_penalty = args.Qn*(y[:, [-1], :] < ref+0.01)
    # objectives and constraints names for nicer plot
    regulation_loss.name = 'state_loss'
    state_lower_bound_penalty.name = 'x_min'
    state_upper_bound_penalty.name = 'x_max'
    terminal_lower_bound_penalty.name = 'y_N_min'
    terminal_upper_bound_penalty.name = 'y_N_max'
    # list of constraints and objectives
    objectives = [regulation_loss, regularization]
    constraints = [
        state_lower_bound_penalty,
        state_upper_bound_penalty,
        terminal_lower_bound_penalty,
        terminal_upper_bound_penalty,
    ]

    """
    # # #  DPC problem = objectives + constraints + trainable components 
    """
    # data (y_k) -> estimator (x_k) -> policy (u_k) -> dynamics (x_k+1, y_k+1)
    components = [estimator, cl_model]
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
    print('\nTest Closed Loop System with PSL model \n')
    np.random.seed(20)
    sim_steps = 500
    # construct system simulator
    policy_cl = sim.ControllerPytorch(policy=cl_model.fx.block.policy,
                                     input_keys=['x', 'r'])
    system_psl = sim.DynamicsPSL(gt_model, input_key_map={'x': 'x', 'u': 'u'})
    policy_cl.name = 'policy'
    system_psl.name = 'dynamics'
    components = [policy_cl, system_psl]
    cl_sim = sim.SystemSimulator(components)
    # plot system simulator computational graph
    plt.figure()
    cl_sim.plot_graph()
    # generate initial data for closed loop simulation
    x0 = np.random.rand(nx)
    data_init = {'x': x0}
    R = np.zeros([sim_steps+1, 2])
    data_traj = {'r': R}
    # simulate closed loop
    trajectories = cl_sim.simulate(nsim=sim_steps, data_init=data_init,
                                   data_traj=data_traj)
    # plot closed loop trajectories
    Umin = umin * np.ones([trajectories['y'].shape[0], 1])
    Umax = umax * np.ones([trajectories['y'].shape[0], 1])
    Ymin = xmin * np.ones([trajectories['y'].shape[0], 1])
    Ymax = xmax * np.ones([trajectories['y'].shape[0], 1])
    psl.plot.pltCL(Y=trajectories['y'], U=trajectories['u'], R=R,
                   Umin=Umin, Umax=Umax, Ymin=Ymin, Ymax=Ymax)

    U0 = np.zeros([sim_steps, 1])
    gt_model.x0 = x0
    out_uncontrolled = gt_model.simulate(U=U0, nsim=sim_steps)
    psl.plot.pltPhase(X=trajectories['y'], Xtrain=out_uncontrolled['Y'])
