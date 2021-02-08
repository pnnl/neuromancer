import argparse

import torch
import torch.nn.functional as F
from torch import nn

import slim
from neuromancer import loggers
from neuromancer.datasets import EmulatorDataset, FileDataset, systems
from neuromancer import blocks
from neuromancer import dynamics
from neuromancer import estimators
from neuromancer.activations import activations
from neuromancer.problem import Problem, Objective
from neuromancer.activations import BLU, SoftExponential
from common import get_base_parser


def get_parser(parser=None):
    if parser is None:
        parser = get_base_parser()

    # optimization parameters
    opt_group = parser.add_argument_group("OPTIMIZATION PARAMETERS")
    opt_group.add_argument("-epochs", type=int, default=2000)
    opt_group.add_argument(
        "-lr", type=float, default=0.001, help="Step size for gradient descent."
    )
    opt_group.add_argument(
        "-eval_metric",
        type=str,
        default="loop_dev_ref_loss",
        help="Metric for model selection and early stopping.",
    )
    opt_group.add_argument(
        "-patience",
        type=int,
        default=50,
        help="How many epochs to allow for no improvement in eval metric before early stopping.",
    )
    opt_group.add_argument(
        "-warmup",
        type=int,
        default=100,
        help="Number of epochs to wait before enacting early stopping policy.",
    )
    opt_group.add_argument(
        "-skip_eval_sim",
        action="store_true",
        help="Whether to run simulator during evaluation phase of training.",
    )

    # data parameters
    data_group = parser.add_argument_group("DATA PARAMETERS")
    data_group.add_argument(
        "-system",
        type=str,
        default='Reno_full',
        choices=list(systems.keys()),
        help="select particular dataset with keyword",
    )
    data_group.add_argument(
        "-nsim",
        type=int,
        default=18000,
        help="Number of time steps for full dataset. (ntrain + ndev + ntest) "
        "train, dev, and test will be split evenly from contiguous, sequential, "
        "non-overlapping chunks of nsim datapoints, e.g. first nsim/3 art train, "
        "next nsim/3 are dev and next nsim/3 simulation steps are test points. "
        "None will use a default nsim from the selected dataset or emulator",
    )
    data_group.add_argument(
        "-nsteps",
        type=int,
        default=32,
        help="Number of steps for open loop during training.",
    )
    data_group.add_argument(
        "-norm",
        nargs="+",
        default=["U", "D", "Y"],
        choices=["U", "D", "Y"],
        help="List of sequences to max-min normalize",
    )
    data_group.add_argument(
        "-data_seed",
        type=int,
        default=408,
        help="Random seed used for simulated data"
    )

    # model parameters
    model_group = parser.add_argument_group("MODEL PARAMETERS")
    model_group.add_argument(
        "-ssm_type",
        type=str,
        choices=["blackbox", "hw", "hammerstein", "blocknlin", "linear"],
        default="hammerstein",
    )
    model_group.add_argument(
        "-nx_hidden", type=int, default=5, help="Number of hidden states per output"
    )
    model_group.add_argument(
        "-n_layers",
        type=int,
        default=2,
        help="Number of hidden layers of single time-step state transition",
    )
    model_group.add_argument(
        "-state_estimator",
        type=str,
        choices=["rnn", "mlp", "linear", "residual_mlp", "fully_observable"],
        default="mlp",
    )
    model_group.add_argument(
        "-estimator_input_window",
        type=int,
        default=32,
        help="Number of previous time steps measurements to include in state estimator input",
    )
    model_group.add_argument(
        "-nonlinear_map",
        type=str,
        default="mlp",
        choices=["mlp", "rnn", "pytorch_rnn", "linear", "residual_mlp"],
    )
    model_group.add_argument(
        "-bias",
        action="store_true",
        help="Whether to use bias in the neural network models.",
    )
    model_group.add_argument(
        "-activation",
        choices=activations.keys(),
        default="gelu",
        help="Activation function for neural networks",
    )
    model_group.add_argument(
        "-seed",
        type=int,
        default=408,
        help="Random seed used for weight initialization."
    )

    # linear parameters
    linear_group = parser.add_argument_group("LINEAR PARAMETERS")
    linear_group.add_argument(
        "-linear_map", type=str, choices=list(slim.maps.keys()), default="linear"
    )
    linear_group.add_argument("-sigma_min", type=float, default=0.9)
    linear_group.add_argument("-sigma_max", type=float, default=1.0)

    # weight parameters
    weight_group = parser.add_argument_group("WEIGHT PARAMETERS")
    weight_group.add_argument(
        "-Q_con_x",
        type=float,
        default=1.0,
        help="Hidden state constraints penalty weight.",
    )
    weight_group.add_argument(
        "-Q_dx",
        type=float,
        default=0.0,
        help="Penalty weight on hidden state difference in one time step.",
    )
    weight_group.add_argument(
        "-Q_sub", type=float, default=0.1, help="Linear maps regularization weight."
    )
    weight_group.add_argument(
        "-Q_y", type=float, default=1.0, help="Output tracking penalty weight"
    )
    weight_group.add_argument(
        "-Q_e",
        type=float,
        default=0.0,
        help="State estimator hidden prediction penalty weight",
    )
    weight_group.add_argument(
        "-Q_con_fdu",
        type=float,
        default=0.2,
        help="Penalty weight on control actions and disturbances.",
    )

    return parser


def get_model_components(args, dataset, estim_name="estim", dynamics_name="dynamics"):
    torch.manual_seed(args.seed)
    nx = dataset.dims["Y"][-1] * args.nx_hidden
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
        {**dataset.dims, "x0": (nx,)},
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
            {**dataset.dims, "x0_estim": (nx,)},
            linmap,
            nonlinmap,
            bias=args.bias,
            n_layers=args.n_layers,
            activation=activation,
            name=dynamics_name,
            input_keys={'x0': f'x0_{estimator.name}'},
            linargs=linargs
        ) if args.ssm_type == "blackbox"
        else dynamics.block_model(
            args.ssm_type,
            {**dataset.dims, "x0_estim": (nx,)},
            linmap,
            nonlinmap,
            bias=args.bias,
            n_layers=args.n_layers,
            activation=activation,
            name=dynamics_name,
            input_keys={'x0': f'x0_{estimator.name}'},
            linargs=linargs
        )
    )
    return estimator, dynamics_model


def get_objective_terms(args, dataset, estimator, dynamics_model):
    xmin = -0.2
    xmax = 1.2
    dxudmin = -0.1
    dxudmax = 0.1
    estimator_loss = Objective(
        [f"X_pred_{dynamics_model.name}", f"x0_{estimator.name}"],
        lambda X_pred, x0: F.mse_loss(X_pred[-1, :-1, :], x0[1:]),
        weight=args.Q_e,
        name="arrival_cost",
    )
    regularization = Objective(
        [f"reg_error_{estimator.name}", f"reg_error_{dynamics_model.name}"],
        lambda reg1, reg2: reg1 + reg2,
        weight=args.Q_sub,
        name="reg_error",
    )
    reference_loss = Objective(
        [f"Y_pred_{dynamics_model.name}", "Yf"], F.mse_loss, weight=args.Q_y, name="ref_loss"
    )
    state_smoothing = Objective(
        [f"X_pred_{dynamics_model.name}"],
        lambda x: F.mse_loss(x[1:], x[:-1]),
        weight=args.Q_dx,
        name="state_smoothing",
    )
    observation_lower_bound_penalty = Objective(
        [f"Y_pred_{dynamics_model.name}"],
        lambda x: torch.mean(F.relu(-x + xmin)),
        weight=args.Q_con_x,
        name="y_low_bound_error",
    )
    observation_upper_bound_penalty = Objective(
        [f"Y_pred_{dynamics_model.name}"],
        lambda x: torch.mean(F.relu(x - xmax)),
        weight=args.Q_con_x,
        name="y_up_bound_error",
    )

    objectives = [regularization, reference_loss, estimator_loss]
    constraints = [
        state_smoothing,
        observation_lower_bound_penalty,
        observation_upper_bound_penalty,
    ]

    if args.ssm_type != "blackbox":
        if "U" in dataset.data:
            inputs_max_influence_lb = Objective(
                [f"fU_{dynamics_model.name}"],
                lambda x: torch.mean(F.relu(-x + dxudmin)),
                weight=args.Q_con_fdu,
                name="input_influence_lb",
            )
            inputs_max_influence_ub = Objective(
                [f"fU_{dynamics_model.name}"],
                lambda x: torch.mean(F.relu(x - dxudmax)),
                weight=args.Q_con_fdu,
                name="input_influence_ub",
            )
            constraints += [inputs_max_influence_lb, inputs_max_influence_ub]
        if "D" in dataset.data:
            disturbances_max_influence_lb = Objective(
                [f"fD_{dynamics_model.name}"],
                lambda x: torch.mean(F.relu(-x + dxudmin)),
                weight=args.Q_con_fdu,
                name="dist_influence_lb",
            )
            disturbances_max_influence_ub = Objective(
                [f"fD_{dynamics_model.name}"],
                lambda x: torch.mean(F.relu(x - dxudmax)),
                weight=args.Q_con_fdu,
                name="dist_influence_ub",
            )
            constraints += [
                disturbances_max_influence_lb,
                disturbances_max_influence_ub,
            ]

    return objectives, constraints
