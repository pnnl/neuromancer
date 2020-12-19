import argparse

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

import psl
import slim
from neuromancer import loggers
from neuromancer.datasets import EmulatorDataset, FileDataset, systems
from neuromancer import blocks
from neuromancer import dynamics
from neuromancer import estimators
from neuromancer.problem import Problem, Objective
from neuromancer.activations import BLU, SoftExponential
from neuromancer import policies

from .common import get_base_parser


def get_parser(parser=None, add_prefix=False):
    if parser is None:
        parser = get_base_parser()

    # maybe prefix arg with "ctrl_"
    pfx = (lambda x: f"-ctrl_{x.strip('-')}") if add_prefix else (lambda x: x)

    # optimization parameters
    opt_group = parser.add_argument_group("OPTIMIZATION PARAMETERS")
    opt_group.add_argument(pfx("-epochs"), type=int, default=100)
    opt_group.add_argument(
        pfx("-lr"), type=float, default=0.001, help="Step size for gradient descent."
    )
    opt_group.add_argument(
        pfx("-patience"),
        type=int,
        default=100,
        help="How many epochs to allow for no improvement in eval metric before early stopping.",
    )
    opt_group.add_argument(
        pfx("-warmup"),
        type=int,
        default=100,
        help="Number of epochs to wait before enacting early stopping policy.",
    )
    opt_group.add_argument(
        pfx("-skip_eval_sim"),
        action="store_true",
        help="Whether to run simulator during evaluation phase of training.",
    )

    # data parameters
    data_group = parser.add_argument_group("DATA PARAMETERS")
    data_group.add_argument(
        pfx("-nsteps"),
        type=int,
        default=32,
        help="Number of steps for open loop during training.",
    )
    data_group.add_argument(
        pfx("-system"),
        type=str,
        default="flexy_air",
        help="select particular dataset with keyword",
    )
    data_group.add_argument(
        pfx("-nsim"),
        type=int,
        default=10000,
        help="Number of time steps for full dataset. (ntrain + ndev + ntest)"
        "train, dev, and test will be split evenly from contiguous, sequential, "
        "non-overlapping chunks of nsim datapoints, e.g. first nsim/3 art train,"
        "next nsim/3 are dev and next nsim/3 simulation steps are test points."
        "None will use a default nsim from the selected dataset or emulator",
    )
    data_group.add_argument(
        pfx("-norm"),
        nargs="+",
        default=["U", "D", "Y"],
        choices=["U", "D", "Y", "X"],
        help="List of sequences to max-min normalize",
    )
    data_group.add_argument(
        pfx("-data_seed"), type=int, default=408, help="Random seed used for simulated data"
    )

    # TODO: option with loading trained model
    # mfiles = ['models/best_model_flexy1.pth',
    #           'models/best_model_flexy2.pth',
    #           'ape_models/best_model_blocknlin.pth']
    # data_group.add_argument('-model_file', type=str, default=mfiles[0])

    ##################
    # POLICY PARAMETERS
    policy_group = parser.add_argument_group("POLICY PARAMETERS")
    policy_group.add_argument(
        pfx("-policy"), type=str, choices=["mlp", "linear"], default="mlp"
    )
    policy_group.add_argument(
        "-controlled_outputs", nargs='+', default=[0],
        help="list of indices of controlled outputs len(default)<=ny"
    )
    policy_group.add_argument(
        pfx("-n_hidden"), type=int, default=20, help="Number of hidden states"
    )
    policy_group.add_argument(
        pfx("-n_layers"),
        type=int,
        default=3,
        help="Number of hidden layers of single time-step state transition",
    )
    policy_group.add_argument(
        pfx("-bias"),
        action="store_true",
        help="Whether to use bias in the neural network models.",
    )
    policy_group.add_argument(
        pfx("-policy_features"),
        nargs="+",
        default=['Y_ctrl_p', 'Rf', 'Y_maxf', 'Y_minf'],
        help="Policy features",
    )  # reference tracking option
    policy_group.add_argument(
        pfx("-activation"),
        choices=["gelu", "softexp"],
        default="gelu",
        help="Activation function for neural networks",
    )
    policy_group.add_argument(
        pfx("-perturbation"),
        choices=["white_noise_sine_wave", "white_noise"],
        default="white_noise",
    )
    policy_group.add_argument(
        pfx("-seed"),
        type=int,
        default=408,
        help="Random seed used for weight initialization.",
    )

    # linear parameters
    linear_group = parser.add_argument_group("LINEAR PARAMETERS")
    linear_group.add_argument(
        pfx("-linear_map"), type=str, choices=["linear", "softSVD", "pf"], default="linear"
    )
    linear_group.add_argument(pfx("-sigma_min"), type=float, default=0.1)
    linear_group.add_argument(pfx("-sigma_max"), type=float, default=1.0)

    # layers
    layers_group = parser.add_argument_group("LAYERS PARAMETERS")
    # TODO: generalize freeze unfreeze - we want to unfreeze only policy network
    layers_group.add_argument(
        "-freeze", nargs="+", default=[""], help="sets requires grad to False"
    )
    layers_group.add_argument(
        "-unfreeze", default=["components.2"], help="sets requires grad to True"
    )

    # weight parameters
    weight_group = parser.add_argument_group("WEIGHT PARAMETERS")
    weight_group.add_argument(
        pfx("-Q_con_x"),
        type=float,
        default=1.0,
        help="Hidden state constraints penalty weight.",
    )
    weight_group.add_argument(
        pfx("-Q_con_y"),
        type=float,
        default=2.0,
        help="Observable constraints penalty weight.",
    )
    weight_group.add_argument(
        pfx("-Q_dx"),
        type=float,
        default=0.1,
        help="Penalty weight on hidden state difference in one time step.",
    )
    weight_group.add_argument(
        pfx("-Q_sub"), type=float, default=0.1, help="Linear maps regularization weight."
    )
    weight_group.add_argument(
        pfx("-Q_y"), type=float, default=1.0, help="Output tracking penalty weight"
    )
    weight_group.add_argument(
        pfx("-Q_e"),
        type=float,
        default=1.0,
        help="State estimator hidden prediction penalty weight",
    )
    weight_group.add_argument(
        pfx("-Q_con_fdu"),
        type=float,
        default=0.0,
        help="Penalty weight on control actions and disturbances.",
    )
    weight_group.add_argument(
        pfx("-Q_con_u"), type=float, default=10.0, help="Input constraints penalty weight."
    )
    weight_group.add_argument(
        pfx("-Q_r"), type=float, default=1.0, help="Reference tracking penalty weight"
    )
    weight_group.add_argument(
        pfx("-Q_du"),
        type=float,
        default=0.1,
        help="control action difference penalty weight",
    )

    # objective and constraints variations
    weight_group.add_argument(pfx("-con_tighten"), action="store_true")
    weight_group.add_argument(
        pfx("-tighten"),
        type=float,
        default=0.05,
        help="control action difference penalty weight",
    )
    weight_group.add_argument(pfx("-loss_clip"), action="store_true")
    weight_group.add_argument(pfx("-noise"), action="store_true")

    return parser


def update_system_id_inputs(args, dataset, estimator, dynamics_model):
    dynamics_model.input_keys[dynamics_model.input_keys.index('Uf')] = 'U_pred_policy'
    dynamics_model.fe = None
    dynamics_model.fyu = None

    estimator.input_keys[0] = 'Y_ctrl_p'
    estimator.data_dims = dataset.dims
    estimator.data_dims['Y_ctrl_p'] = dataset.dims['Yp']
    # estimator.data_dims = {**dataset.dims, 'Y_ctrl_p': estimator.data_dims['Yp']}
    estimator.nsteps = args.nsteps

    return estimator, dynamics_model


def get_policy_components(args, dataset, dynamics_model, policy_name="policy"):
    torch.manual_seed(args.seed)

    # control policy setup
    activation = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "blu": BLU,
        "softexp": SoftExponential,
    }[args.activation]

    linmap = slim.maps[args.linear_map]
    linargs = {"sigma_min": args.sigma_min, "sigma_max": args.sigma_max}
    nh_policy = args.n_hidden

    policy = {
        "linear": policies.LinearPolicy,
        "mlp": policies.MLPPolicy,
        "rnn": policies.RNNPolicy,
    }[args.policy](
        {"x0_estim": (dynamics_model.nx,), **dataset.dims},
        nsteps=args.nsteps,
        bias=args.bias,
        linear_map=linmap,
        nonlin=activation,
        hsizes=[nh_policy] * args.n_layers,
        input_keys=args.policy_features,
        linargs=linargs,
        name=policy_name,
    )
    return policy


def get_objective_terms(args, policy):
    if args.noise:
        output_key = "Y_pred_dynamics_noise"
    else:
        output_key = "Y_pred_dynamics"

    reference_loss = Objective(
        [output_key, "Rf"],
        lambda pred, ref: F.mse_loss(pred[:, :, args.controlled_outputs], ref),
        weight=args.Q_r,
        name="ref_loss",
    )
    regularization = Objective(
        [f"reg_error_{policy.name}"], lambda reg: reg, weight=args.Q_sub, name="reg_loss",
    )
    control_smoothing = Objective(
        [f"U_pred_{policy.name}"],
        lambda x: F.mse_loss(x[1:], x[:-1]),
        weight=args.Q_du,
        name="control_smoothing",
    )
    observation_lower_bound_penalty = Objective(
        [output_key, "Y_minf"],
        lambda x, xmin: torch.mean(F.relu(-x[:, :, args.controlled_outputs] + xmin)),
        weight=args.Q_con_y,
        name="observation_lower_bound",
    )
    observation_upper_bound_penalty = Objective(
        [output_key, "Y_maxf"],
        lambda x, xmax: torch.mean(F.relu(x[:, :, args.controlled_outputs] - xmax)),
        weight=args.Q_con_y,
        name="observation_upper_bound",
    )
    inputs_lower_bound_penalty = Objective(
        [f"U_pred_{policy.name}", "U_minf"],
        lambda x, xmin: torch.mean(F.relu(-x + xmin)),
        weight=args.Q_con_u,
        name="input_lower_bound",
    )
    inputs_upper_bound_penalty = Objective(
        [f"U_pred_{policy.name}", "U_maxf"],
        lambda x, xmax: torch.mean(F.relu(x - xmax)),
        weight=args.Q_con_u,
        name="input_upper_bound",
    )

    # Constraints tightening
    if args.con_tighten:
        observation_lower_bound_penalty = Objective(
            [output_key, "Y_minf"],
            lambda x, xmin: torch.mean(F.relu(-x[:, :, args.controlled_outputs] + xmin + args.tighten)),
            weight=args.Q_con_y,
            name="observation_lower_bound",
        )
        observation_upper_bound_penalty = Objective(
            [output_key, "Y_maxf"],
            lambda x, xmax: torch.mean(F.relu(x[:, :, args.controlled_outputs] - xmax + args.tighten)),
            weight=args.Q_con_y,
            name="observation_upper_bound",
        )
        inputs_lower_bound_penalty = Objective(
            [f"U_pred_{policy.name}", "U_minf"],
            lambda x, xmin: torch.mean(F.relu(-x + xmin + args.tighten)),
            weight=args.Q_con_u,
            name="input_lower_bound",
        )
        inputs_upper_bound_penalty = Objective(
            [f"U_pred_{policy.name}", "U_maxf"],
            lambda x, xmax: torch.mean(F.relu(x - xmax + args.tighten)),
            weight=args.Q_con_u,
            name="input_upper_bound",
        )

    # Loss clipping
    if args.loss_clip:
        reference_loss = Objective(
            [output_key, "Rf", "Y_minf", "Y_maxf"],
            lambda pred, ref, xmin, xmax: F.mse_loss(
                pred[:, :, args.controlled_outputs] * torch.gt(ref, xmin).int() * torch.lt(ref, xmax).int(),
                ref * torch.gt(ref, xmin).int() * torch.lt(ref, xmax).int(),
            ),
            weight=args.Q_r,
            name="ref_loss",
        )

    objectives = [regularization, reference_loss]
    constraints = [
        observation_lower_bound_penalty,
        observation_upper_bound_penalty,
        inputs_lower_bound_penalty,
        inputs_upper_bound_penalty,
    ]

    return objectives, constraints


def add_reference_features(args, dataset, dynamics_model):
    """
    """
    ny = dynamics_model.fy.out_features
    if ny != dataset.data["Y"].shape[1]:
        new_sequences = {"Y": dataset.data["Y"][:, :1]}
        dataset.add_data(new_sequences, overwrite=True)
    dataset.min_max_norms["Ymin"] = dataset.min_max_norms["Ymin"][0]
    dataset.min_max_norms["Ymax"] = dataset.min_max_norms["Ymax"][0]

    nsim = dataset.data["Y"].shape[0]
    nu = dataset.data["U"].shape[1]
    dataset.add_data({
        "Y_max": psl.Periodic(nx=1, nsim=nsim, numPeriods=30, xmax=0.9, xmin=0.6),
        "Y_min": psl.Periodic(nx=1, nsim=nsim, numPeriods=25, xmax=0.4, xmin=0.1),
        "U_max": np.ones([nsim, nu]),
        "U_min": np.zeros([nsim, nu]),
        "R": psl.Periodic(nx=1, nsim=nsim, numPeriods=20, xmax=0.8, xmin=0.2)
        # 'Y_ctrl_': psl.WhiteNoise(nx=ny, nsim=nsim, xmax=[1.0] * ny, xmin=[0.0] * ny)
    })
    # indices of controlled states, e.g. [0, 1, 3] out of 5 outputs
    dataset.ctrl_outputs = args.controlled_outputs
    return dataset