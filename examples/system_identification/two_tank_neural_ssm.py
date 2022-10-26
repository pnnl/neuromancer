"""
Modeling an aerodynamic body using neural state space models.
Script for training block dynamics models for system identification using the new Constraints and variables
instead of Objectives.

Basic model options are:
    + prior on the linear maps of the neural network
    + state estimator
    + non-linear map type
    + hidden state dimension
    + Whether to use affine or linear maps (bias term)
Basic data options are:
    + Load from a variety of premade data sequences
    + Load from a variety of emulators
    + Normalize input, output, or disturbance data
    + Nstep prediction horizon
Basic optimization options are:
    + Number of epochs to train on
    + Learn rate
Basic logging options are:
    + print to stdout
    + mlflow
    + weights and bias

More detailed description of options in the `get_base_parser()` function in common.py.
"""

import torch
import slim
import psl

from neuromancer import blocks, estimators, dynamics, arg
from neuromancer.activations import activations
from neuromancer.visuals import VisualizerOpen
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.simulators import OpenLoopSimulator, MultiSequenceOpenLoopSimulator
from neuromancer.dataset import read_file, get_sequence_dataloaders
from neuromancer.loss import get_loss
from neuromancer.callbacks import SysIDCallback
from neuromancer.loggers import BasicLogger, MLFlowLogger
from neuromancer.constraint import variable


def arg_sys_id_problem(prefix='', system='aero'):
    """
    Command line parser for system identification problem arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers
                         are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = arg.ArgParser(prefix=prefix, add_help=False)
    gp = parser.group("system_id")
    #  DATA
    gp.add("-dataset", type=str, default=system,
           help="select particular dataset with keyword")
    gp.add("-nsteps", type=int, default=32,
           help="prediction horizon.")
    gp.add("-nsim", type=int, default=10000,
           help="Number of time steps for full dataset. (ntrain + ndev + ntest)"
                "train, dev, and test will be split evenly from contiguous, sequential, "
                "non-overlapping chunks of nsim datapoints, e.g. first nsim/3 art train,"
                "next nsim/3 are dev and next nsim/3 simulation steps are test points."
                "None will use a default nsim from the selected dataset or emulator")
    gp.add("-norm", nargs="+", default=["U", "D", "Y"], choices=["U", "D", "Y", "X"],
           help="List of sequences to max-min normalize")
    gp.add("-data_seed", type=int, default=408,
           help="Random seed used for simulated data")
    #  MODEL
    gp.add("-ssm_type", type=str, choices=["blackbox", "hw", "hammerstein", "blocknlin", "linear"],
           default="hammerstein",
           help='Choice of block structure for system identification model')
    gp.add("-nx_hidden", type=int, default=32,
           help="Number of hidden states per output")
    gp.add("-n_layers", type=int, default=2,
           help="Number of hidden layers of single time-step state transition")
    gp.add("-state_estimator", type=str, choices=["rnn", "mlp", "linear",
                      "residual_mlp", "fully_observable"], default="mlp",
           help='Choice of model architecture for state estimator.')
    gp.add("-estimator_input_window", type=int, default=8,
           help="Number of previous time steps measurements to include in state estimator input")
    gp.add("-nonlinear_map", type=str, default="mlp", choices=["mlp", "rnn", "pytorch_rnn", "linear", "residual_mlp"],
           help='Choice of architecture for component blocks in state space model.')
    gp.add("-linear_map", type=str, choices=["linear", "softSVD", "pf"], default="linear",
           help='Choice of map from SLiM package')
    gp.add("-sigma_min", type=float, default=0.1,
           help='Minimum singular value (for maps with singular value constraints)')
    gp.add("-sigma_max", type=float, default=1.0,
           help='Maximum singular value (for maps with singular value constraints)')
    gp.add("-bias", action="store_true",
           help="Whether to use bias in the neural network block component models.")
    gp.add("-activation", choices=activations.keys(), default="gelu",
           help="Activation function for component block neural networks")
    #  LOSS
    gp.add("-Q_con_x", type=float, default=1.0,
           help="Hidden state constraints penalty weight.")
    gp.add("-Q_dx", type=float, default=0.1,
           help="Penalty weight on hidden state difference in one time step.")
    gp.add("-Q_sub", type=float, default=0.1,
           help="Linear maps regularization weight.")
    gp.add("-Q_y", type=float, default=1.0,
           help="Output tracking penalty weight")
    gp.add("-Q_e", type=float, default=1.0,
           help="State estimator hidden prediction penalty weight")
    gp.add("-Q_con_fdu", type=float, default=0.0,
           help="Penalty weight on control actions and disturbances.")
    #  LOG
    gp.add("-logger", type=str, choices=["mlflow", "stdout"], default="stdout",
           help="Logging setup to use")
    gp.add("-savedir", type=str, default="test_system_id_v1",
           help="Where should your trained model and plots be saved (temp)")
    gp.add("-verbosity", type=int, default=1,
           help="How many epochs in between status updates")
    gp.add("-metrics", nargs="+", default=["nstep_dev_loss", "loop_dev_loss", "best_loop_dev_loss",
               "nstep_dev_ref_loss", "loop_dev_ref_loss"],
           help="Metrics to be logged")
    #  OPTIMIZATION
    gp.add("-eval_metric", type=str, default="loop_dev_ref_loss",
            help="Metric for model selection and early stopping.")
    gp.add("-epochs", type=int, default=200,
           help='Number of training epochs')
    gp.add("-lr", type=float, default=0.001,
           help="Step size for gradient descent.")
    gp.add("-patience", type=int, default=100,
           help="How many epochs to allow for no improvement in eval metric before early stopping.")
    gp.add("-warmup", type=int, default=10,
           help="Number of epochs to wait before enacting early stopping policy.")
    gp.add("-skip_eval_sim", action="store_true",
           help="Whether to run simulator during evaluation phase of training.")
    gp.add("-seed", type=int, default=408, help="Random seed used for weight initialization.")
    gp.add("-gpu", type=int, help="GPU to use")
    gp.add("-loss", type=str, default='penalty',
           choices=['penalty', 'barrier'],
           help="type of the loss function.")
    gp.add("-barrier_type", type=str, default='log10',
           choices=['log', 'log10', 'inverse'],
           help="type of the barrier function in the barrier loss.")
    gp.add("-batch_second", default=True, choices=[True, False],
           help="whether the batch is a second dimension in the dataset.")
    return parser


def get_model_components(args, dims, estim_name="estim", dynamics_name="dynamics"):
    torch.manual_seed(args.seed)
    if not args.state_estimator == 'fully_observable':
        nx = dims["Y"][-1] * args.nx_hidden
    else:
        nx = dims["Y"][-1]
    print('dims', dims)
    print('nx', nx)
    activation = activations[args.activation]
    linmap = slim.maps[args.linear_map]
    linargs = {"sigma_min": args.sigma_min, "sigma_max": args.sigma_max}

    nonlinmap = {
        "linear": linmap,
        "mlp": blocks.MLP,
        "rnn": blocks.RNN,
        "pytorch_rnn": blocks.PytorchRNN,
        "residual_mlp": blocks.ResMLP,
    }[args.nonlinear_map]

    estimator = {
        "linear": estimators.LinearEstimator,
        "mlp": estimators.MLPEstimator,
        "rnn": estimators.RNNEstimator,
        "residual_mlp": estimators.ResMLPEstimator,
        "fully_observable": estimators.FullyObservable,
    }[args.state_estimator](
        {**dims, "x0": (nx,)},
        nsteps=args.nsteps,
        window_size=args.estimator_input_window,
        bias=args.bias,
        linear_map=linmap,
        nonlin=activation,
        hsizes=[nx] * args.n_layers,
        input_keys=["Yp"],
        linargs=linargs,
        name=estim_name,
    )

    dynamics_model = (
        dynamics.blackbox_model(
            {**dims, "x0": (nx,)},
            linmap,
            nonlinmap,
            bias=args.bias,
            n_layers=args.n_layers,
            activation=activation,
            name=dynamics_name,
            input_key_map={"x0": f"x0_{estimator.name}"},
            linargs=linargs
        ) if args.ssm_type == "blackbox"
        else dynamics.block_model(
            args.ssm_type,
            {**dims, "x0": (nx,)},
            linmap,
            nonlinmap,
            bias=args.bias,
            n_layers=args.n_layers,
            activation=activation,
            name=dynamics_name,
            input_key_map={"x0": f"x0_{estimator.name}"},
            linargs=linargs
        )
    )
    return estimator, dynamics_model


def get_objective_terms(args, dims, estimator, dynamics_model):
    xmin = -0.2
    xmax = 1.2
    dxudmin = -0.05
    dxudmax = 0.05

    xhat = variable(f"X_pred_{dynamics_model.name}")
    state_smoothing = args.Q_dx*((xhat[:, 1:, :] == xhat[:,:-1, :])^2)

    est_reg = variable(f"reg_error_{estimator.name}")
    dyn_reg = variable(f"reg_error_{estimator.name}")
    regularization = args.Q_sub*((est_reg + dyn_reg == 0)^2)

    yhat = variable(f"Y_pred_{dynamics_model.name}")
    y = variable("Yf")
    reference_loss = args.Q_y*((yhat == y)^2)

    observation_lower_bound_penalty = args.Q_con_x*(yhat > xmin)
    observation_upper_bound_penalty = args.Q_con_x*(yhat < xmax)

    objectives = [reference_loss, regularization]
    constraints = [
        state_smoothing,
        observation_lower_bound_penalty,
        observation_upper_bound_penalty,
    ]

    if args.ssm_type != "blackbox":
        if "U" in dims:
            fu = variable(f"fU_{dynamics_model.name}")
            inputs_max_influence_lb = args.Q_con_fdu*(fu > dxudmin)
            inputs_max_influence_ub = args.Q_con_fdu*(fu < dxudmax)
            constraints += [inputs_max_influence_lb, inputs_max_influence_ub]
        if "D" in dims:
            fd = variable(f"fD_{dynamics_model.name}")
            disturbances_max_influence_lb = args.Q_con_fdu * (fd > dxudmin)
            disturbances_max_influence_ub = args.Q_con_fdu * (fd < dxudmax)
            constraints += [disturbances_max_influence_lb, disturbances_max_influence_ub]

    return objectives, constraints


if __name__ == "__main__":

    # for available systems and datasets in PSL library check: psl.systems.keys() and psl.datasets.keys()
    # load argument parser
    parser = arg.ArgParser(parents=[arg_sys_id_problem(system='TwoTank')])
    args, grps = parser.parse_arg_groups()
    log_constructor = MLFlowLogger if args.logger == 'mlflow' else BasicLogger
    logger = log_constructor(args=args, savedir=args.savedir,
                             verbosity=args.verbosity, stdout=args.metrics)
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"

    # load raw dataset
    if args.dataset in psl.emulators:
        data = psl.emulators[args.dataset](nsim=args.nsim, ninit=0, seed=args.data_seed).simulate()
    elif args.dataset in psl.datasets:
        data = read_file(psl.datasets[args.dataset])
    else:
        data = read_file(args.dataset)
    # get dataloader
    nstep_data, loop_data, dims = get_sequence_dataloaders(data, args.nsteps, norm_type=None)
    train_data, dev_data, test_data = nstep_data
    train_loop, dev_loop, test_loop = loop_data

    # get component models, objectives, and constraints
    estimator, dynamics_model = get_model_components(args, dims)
    components = [estimator, dynamics_model]
    objectives, constraints = get_objective_terms(args, dims, estimator, dynamics_model)
    # create constrained optimization loss
    loss = get_loss(objectives, constraints, train_data, args)
    # construct constrained optimization problem
    problem = Problem(components, loss)
    # plot computational graph
    problem.plot_graph()

    # define callback
    simulator = OpenLoopSimulator(
        problem, train_loop, dev_loop, test_loop, eval_sim=not args.skip_eval_sim, device=device,
    ) if isinstance(train_loop, dict) else MultiSequenceOpenLoopSimulator(
        problem, train_loop, dev_loop, test_loop, eval_sim=not args.skip_eval_sim, device=device,
    )
    visualizer = VisualizerOpen(
        dynamics_model,
        args.verbosity,
        args.savedir,
        training_visuals=False,
        figname='two_tank_neural_ssm.png',
        trace_movie=False,
    )
    callback = SysIDCallback(simulator, visualizer)
    # select optimizer
    optimizer = torch.optim.AdamW(problem.parameters(), lr=args.lr)

    # get trainer
    trainer = Trainer(
        problem,
        train_data,
        dev_data,
        test_data,
        optimizer,
        logger=logger,
        callback=callback,
        epochs=args.epochs,
        eval_metric=f"{dev_loop['name']}_{objectives[0].name}",
        train_metric="nstep_train_loss",
        dev_metric="nstep_dev_loss",
        test_metric="nstep_test_loss",
        patience=args.patience,
        warmup=args.warmup,
        device=device,
    )
    # train
    best_model = trainer.train()
    best_outputs = trainer.test(best_model)
    logger.clean_up()
