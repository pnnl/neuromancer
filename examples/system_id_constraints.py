"""
Script for training block dynamics models for system identification using the new Constraints and Variables
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
import torch.nn.functional as F
import slim

from neuromancer import blocks, estimators, dynamics, arg
from neuromancer.activations import activations
from neuromancer.visuals import VisualizerOpen
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem, Objective
from neuromancer.simulators import OpenLoopSimulator
from neuromancer.datasets import load_dataset
from neuromancer.callbacks import SysIDCallback
from neuromancer.loggers import BasicLogger, MLFlowLogger
from neuromancer.constraint import Variable


def get_model_components(args, dataset, estim_name="estim", dynamics_name="dynamics"):
    torch.manual_seed(args.seed)
    if not args.state_estimator == "fully_observable":
        nx = dataset.dims["Y"][-1] * args.nx_hidden
    else:
        nx = dataset.dims["Y"][-1]
    print("dims", dataset.dims)
    print("nx", nx)
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
            {**dataset.dims, "x0": (nx,)},
            linmap,
            nonlinmap,
            bias=args.bias,
            n_layers=args.n_layers,
            activation=activation,
            name=dynamics_name,
            input_keys={"x0": f"x0_{estimator.name}", "Uf": "Uf"},
            linargs=linargs
        ) if args.ssm_type == "blackbox"
        else dynamics.block_model(
            args.ssm_type,
            {**dataset.dims, "x0": (nx,)},
            linmap,
            nonlinmap,
            bias=args.bias,
            n_layers=args.n_layers,
            activation=activation,
            name=dynamics_name,
            input_keys={"x0": f"x0_{estimator.name}", "Uf": "Uf"},
            linargs=linargs
        )
    )
    return estimator, dynamics_model


def get_objective_terms(args, dataset, estimator, dynamics_model):
    xmin = -0.2
    xmax = 1.2
    dxudmin = -0.05
    dxudmax = 0.05

    x0 = Variable(f"x0_{estimator.name}")

    xhat = Variable(f"X_pred_{dynamics_model.name}")
    estimator_loss = args.Q_e*((x0[1:] == xhat[-1, :-1, :])^2)
    state_smoothing = args.Q_dx*((xhat[1:] == xhat[:-1])^2)

    est_reg = Variable(f"reg_error_{estimator.name}")
    dyn_reg = Variable(f"reg_error_{estimator.name}")
    regularization = args.Q_sub*((est_reg + dyn_reg == 0)^2)

    yhat = Variable(f"Y_pred_{dynamics_model.name}")
    y = Variable("Yf")
    reference_loss = args.Q_y*((yhat == y)^2)

    observation_lower_bound_penalty = args.Q_con_x*(yhat > xmin)
    observation_upper_bound_penalty = args.Q_con_x*(yhat < xmax)

    objectives = [reference_loss, regularization, estimator_loss]
    constraints = [
        state_smoothing,
        observation_lower_bound_penalty,
        observation_upper_bound_penalty,
    ]

    if args.ssm_type != "blackbox":
        if "U" in dataset.data:
            fu = Variable(f"fU_{dynamics_model.name}")
            inputs_max_influence_lb = args.Q_con_fdu*(fu > dxudmin)
            inputs_max_influence_ub = args.Q_con_fdu*(fu < dxudmax)
            constraints += [inputs_max_influence_lb, inputs_max_influence_ub]
        if "D" in dataset.data:
            fd = Variable(f"fD_{dynamics_model.name}")
            disturbances_max_influence_lb = args.Q_con_fdu * (fd > dxudmin)
            disturbances_max_influence_ub = args.Q_con_fdu * (fd < dxudmax)
            constraints += [disturbances_max_influence_lb, disturbances_max_influence_ub]

    return objectives, constraints


if __name__ == "__main__":

    # for available systems in PSL library check: psl.systems.keys()
    # for available datasets in PSL library check: psl.datasets.keys()
    system = "aero"         # keyword of selected system
    parser = arg.ArgParser(parents=[arg.log(), arg.opt(), arg.data(system=system),
                                    arg.loss(), arg.lin(), arg.ssm()])

    grp = parser.group("OPTIMIZATION")
    args, grps = parser.parse_arg_groups()
    print({k: str(getattr(args, k)) for k in vars(args) if getattr(args, k)})

    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"

    log_constructor = MLFlowLogger if args.logger == "mlflow" else BasicLogger
    metrics = ["nstep_dev_loss", "loop_dev_loss", "best_loop_dev_loss",
               "nstep_dev_ref_loss", "loop_dev_ref_loss"]
    logger = log_constructor(args=args, savedir=args.savedir, verbosity=args.verbosity, stdout=metrics)

    dataset = load_dataset(args, device, "openloop")
    print(dataset.dims)

    estimator, dynamics_model = get_model_components(args, dataset)
    objectives, constraints = get_objective_terms(args, dataset, estimator, dynamics_model)

    model = Problem(objectives, constraints, [estimator, dynamics_model])
    model = model.to(device)

    simulator = OpenLoopSimulator(model=model, dataset=dataset, eval_sim=not args.skip_eval_sim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    visualizer = VisualizerOpen(
        dataset,
        dynamics_model,
        args.verbosity,
        args.savedir,
        training_visuals=False,
        trace_movie=False,
    )

    trainer = Trainer(
        model,
        dataset,
        optimizer,
        logger=logger,
        callback=SysIDCallback(simulator, visualizer),
        epochs=args.epochs,
        eval_metric=f"{dataset.dev_loop.name}_{objectives[0].name}",
        patience=args.patience,
        warmup=args.warmup,
    )

    best_model = trainer.train()
    best_outputs = trainer.test(best_model)
    logger.clean_up()
