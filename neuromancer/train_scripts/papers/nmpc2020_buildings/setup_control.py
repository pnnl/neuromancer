"""
Argument parser script for control.py

File for setting:
    hyperparameters of the model and optimizer via get_parser()
    closed-loop system model architecture via get_policy_components()
    define objective terms and penalty constraints via get_objective_terms()
    add new synthetic data features in the dataset via add_reference_features()
"""


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
from common import get_base_parser


def get_parser(parser=None, add_prefix=False):
    if parser is None:
        parser = get_base_parser()

    # maybe prefix arg with "ctrl_"
    pfx = (lambda x: f"-ctrl_{x.strip('-')}") if add_prefix else (lambda x: x)

    # optimization parameters
    opt_group = parser.add_argument_group("OPTIMIZATION PARAMETERS")
    opt_group.add_argument(pfx("-epochs"), type=int, default=1000)
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
        default="Reno_full",
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

    # TODO:update path after training a new model using system_id.py
    # path = f"./test/Reno_full_best_model.pth"
    path = f"./sys_ID_models/model3/Reno_full_best_model.pth"
    # path = f"./sys_ID_models/model8/Reno_full_best_model.pth"

    data_group.add_argument('-model_file', type=str, default=path)

    ##################
    # POLICY PARAMETERS
    policy_group = parser.add_argument_group("POLICY PARAMETERS")
    policy_group.add_argument(
        pfx("-policy"), type=str, choices=["mlp", "linear"], default="mlp"
    )
    policy_group.add_argument(
        "-controlled_outputs", nargs='+', default=[0, 1, 2, 3, 4, 5],
        help="list of indices of controlled outputs len(default)<=ny"
    )
    policy_group.add_argument(
        pfx("-n_hidden"), type=int, default=100, help="Number of hidden states"
    )
    policy_group.add_argument(
        pfx("-n_layers"),
        type=int,
        default=4,
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
        # default=['Y_ctrl_p', 'Rf', 'Df'],
        # default=['Y_ctrl_p', 'Rf', 'Df', 'Y_maxf', 'Y_minf'],
        # default=['Y_ctrl_p', 'Df', 'Y_maxf', 'Y_minf'],
        default=['Y_ctrl_p', 'Df', 'Y_minf'],
        help="Policy features",
    )
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
        pfx("-linear_map"), type=str,
        choices=["linear", "softSVD", "pf", "nneg"], default="linear"
    )
    linear_group.add_argument(pfx("-sigma_min"), type=float, default=0.1)
    linear_group.add_argument(pfx("-sigma_max"), type=float, default=0.9)

    # layers
    layers_group = parser.add_argument_group("LAYERS PARAMETERS")
    layers_group.add_argument(
        "-freeze", nargs="+", default=[""], help="sets requires grad to False"
    )
    layers_group.add_argument(
        "-unfreeze", default=["components.2"], help="sets requires grad to True"
    )

    # weight parameters
    weight_group = parser.add_argument_group("WEIGHT PARAMETERS")
    weight_group.add_argument(
        pfx("-Q_con_y"),
        type=float,
        default=1.0,
        help="Observable constraints penalty weight.",
    )
    weight_group.add_argument(
        pfx("-Q_sub"), type=float, default=0.2, help="Linear maps regularization weight."
    )
    weight_group.add_argument(
        pfx("-Q_umin"), type=float, default=0.5, help="Input minimization weight."
    )
    weight_group.add_argument(
        pfx("-Q_con_u"), type=float, default=1.0, help="Input constraints penalty weight."
    )
    weight_group.add_argument(
        pfx("-Q_r"), type=float, default=0.0, help="Reference tracking penalty weight"
    )
    weight_group.add_argument(
        pfx("-Q_du"),
        type=float,
        default=1.0,
        help="control action difference penalty weight",
    )

    # weight_group.add_argument(
    #     pfx("-Q_umin"), type=float, default=2.0, help="Input minimization weight."
    # )
    # weight_group.add_argument(
    #     pfx("-Q_dT_ref"), type=float, default=1.0, help="dT static reference weight."
    # )


    # objective and constraints variations
    weight_group.add_argument(pfx("-con_tighten"), action="store_true")
    weight_group.add_argument(
        pfx("-tighten"),
        type=float,
        default=0.0,
        help="control action difference penalty weight",
    )
    weight_group.add_argument(pfx("-loss_clip"), action="store_true")
    weight_group.add_argument(pfx("-noise"), action="store_true")

    return parser


def update_system_id_inputs(args, dataset, estimator, dynamics_model):
    dynamics_model.input_keys[dynamics_model.input_keys.index('Uf')] = 'U_pred_policy'
    if isinstance(dynamics_model, dynamics.DecoupSISO_BlockSSM_building):
        for k in range(len(dynamics_model.SISO_models)):
            dynamics_model.SISO_models[k].input_keys[
                dynamics_model.SISO_models[k].input_keys.index('Uf')] = 'U_pred_policy'
            dynamics_model.SISO_models[k].fe = None
            dynamics_model.SISO_models[k].fyu = None
    else:
        dynamics_model.fe = None
        dynamics_model.fyu = None

    estimator.input_keys[0] = 'Y_ctrl_p'
    estimator.data_dims = dataset.dims
    estimator.data_dims['Y_ctrl_p'] = dataset.dims['Yp']
    estimator.nsteps = args.nsteps

    return estimator, dynamics_model


def get_policy_components(args, dataset, dynamics_model, policy_name="policy"):
    torch.manual_seed(args.seed)

    args.bias = False

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
    # # minimize only temperatures
    # control_min = Objective(
    #     [f"U_pred_{policy.name}"],
    #     lambda x: F.mse_loss(x[:,:,-1], torch.zeros(x[:,:,-1].shape)),
    #     weight=args.Q_umin,
    #     name="control_min",
    # )
    control_min = Objective(
        [f"U_pred_{policy.name}"],
        lambda x: F.mse_loss(x, torch.zeros(x.shape)),
        weight=args.Q_umin,
        name="control_min",
    )
    control_smoothing = Objective(
        [f"U_pred_{policy.name}"],
        lambda x: F.mse_loss(x[1:], x[:-1]),
        weight=args.Q_du,
        name="control_smoothing",
    )
    # # dT spt fixed ref
    # control_dT_ref = Objective(
    #     [f"U_pred_{policy.name}"],
    #     lambda x: F.mse_loss(x[:,:,-1], 1.0*torch.ones(x[:,:,-1].shape)),
    #     weight=args.Q_dT_ref,
    #     name="control_dT_ref",
    # )
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

    objectives = [regularization, reference_loss,
                  control_min, control_smoothing]
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

    if isinstance(dynamics_model, dynamics.DecoupSISO_BlockSSM_building):
        ny = dynamics_model.out_features
    else:
        ny = dynamics_model.fy.out_features

    if ny != dataset.data["Y"].shape[1]:
        new_sequences = {"Y": dataset.data["Y"][:, :1]}
        dataset.add_data(new_sequences, overwrite=True)

    nsim = dataset.dims['nsim']
    nu = dataset.data["U"].shape[1]
    ny = len(args.controlled_outputs)
    
    ymax = 30*[1.0, 0.8]
    ymin = 30*[0.4, 0.6]
    dataset.add_data({
        "Y_max": psl.Steps(nx=ny, nsim=nsim, values=ymax, xmax=1.0, xmin=0.0)[:nsim, :],
        "Y_min": psl.Steps(nx=ny, nsim=nsim, values=ymin, xmax=1.0, xmin=0.0)[:nsim, :],
        # "Y_max": psl.Periodic(nx=ny, nsim=nsim, numPeriods=30, xmax=1.0, xmin=0.9)[:nsim,:],
        # "Y_min": psl.Periodic(nx=ny, nsim=nsim, numPeriods=24, xmax=0.4, xmin=0.3)[:nsim,:],
        "U_max": np.ones([nsim, nu]),
        "U_min": np.zeros([nsim, nu]),
        "R": psl.Periodic(nx=ny, nsim=nsim, numPeriods=20, xmax=0.8, xmin=0.6)[:nsim,:]
        # 'Y_ctrl_': psl.WhiteNoise(nx=ny, nsim=nsim, xmax=[1.0] * ny, xmin=[0.0] * ny)
    })
    # indices of controlled states, e.g. [0, 1, 3] out of 5 outputs
    dataset.ctrl_outputs = args.controlled_outputs

    for k in ('Umin', 'Umax'):
        dataset.min_max_norms[f'{k}min'] = dataset.min_max_norms['Umin']
        dataset.min_max_norms[f'{k}max'] = dataset.min_max_norms['Umax']
    for k in ('Ymin', 'Ymax', 'R'):
        dataset.min_max_norms[f'{k}min'] = dataset.min_max_norms['Ymin'][1]
        dataset.min_max_norms[f'{k}max'] = dataset.min_max_norms['Ymax'][1]
    dataset.min_max_norms["Ymin"] = dataset.min_max_norms["Ymin"][0]
    dataset.min_max_norms["Ymax"] = dataset.min_max_norms["Ymax"][0]
    dataset.norms = dataset.min_max_norms
    return dataset