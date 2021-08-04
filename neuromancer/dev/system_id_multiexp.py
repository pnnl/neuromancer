import torch
import torch.nn.functional as F
import slim
import psl

from neuromancer import blocks, estimators, dynamics, arg
from neuromancer.activations import activations
from neuromancer.visuals import VisualizerOpen
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem, Objective
from neuromancer.simulators import MultiSequenceOpenLoopSimulator
from neuromancer.dataset import read_file
from neuromancer.callbacks import SysIDCallback
from neuromancer.loggers import BasicLogger, MLFlowLogger


import sys
import os
sys.path.append(os.path.abspath("../examples"))

from system_id import get_sequence_dataloaders


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
            {**dims, "x0_estim": (nx,)},
            linmap,
            nonlinmap,
            bias=args.bias,
            n_layers=args.n_layers,
            activation=activation,
            name=dynamics_name,
            input_keys={f"x0_{estimator.name}": "x0", "Uf": "Uf"},
            linargs=linargs
        ) if args.ssm_type == "blackbox"
        else dynamics.block_model(
            args.ssm_type,
            {**dims, "x0_estim": (nx,)},
            linmap,
            nonlinmap,
            bias=args.bias,
            n_layers=args.n_layers,
            activation=activation,
            name=dynamics_name,
            input_keys={f"x0_{estimator.name}": "x0", "Uf": "Uf"},
            linargs=linargs
        )
    )
    return estimator, dynamics_model


def get_objective_terms(args, dims, estimator, dynamics_model):
    xmin = -0.2
    xmax = 1.2
    dxudmin = -0.05
    dxudmax = 0.05
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
        if "U" in dims:
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
        if "D" in dims:
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


if __name__ == "__main__":
    # for available systems in PSL library check: psl.systems.keys()
    # for available datasets in PSL library check: psl.datasets.keys()
    system = "aero"         # keyword of selected system
    parser = arg.ArgParser(parents=[arg.log(), arg.opt(), arg.data(system=system),
                                    arg.loss(), arg.lin(), arg.ssm()])

    grp = parser.group('OPTIMIZATION')
    grp.add("-eval_metric", type=str, default="loop_dev_ref_loss",
            help="Metric for model selection and early stopping.")
    args, grps = parser.parse_arg_groups()
    print({k: str(getattr(args, k)) for k in vars(args) if getattr(args, k)})

    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"

    log_constructor = MLFlowLogger if args.logger == 'mlflow' else BasicLogger
    metrics = ["nstep_dev_loss", "loop_dev_loss", "best_loop_dev_loss",
               "nstep_dev_ref_loss", "loop_dev_ref_loss"]
    logger = log_constructor(args=args, savedir=args.savedir, verbosity=args.verbosity, stdout=metrics)

    # TODO: this can probably be moved to dataset.py
    if args.dataset in psl.emulators:
        data = psl.emulators[args.dataset](nsim=args.nsim, ninit=0, seed=args.data_seed).simulate()
    elif args.dataset in psl.datasets:
        data = read_file(psl.datasets[args.dataset])
    else:
        data = read_file(args.dataset)

    nstep_data, loop_data, dims = get_sequence_dataloaders(data, args.nsteps, device=device)
    train_data, dev_data, test_data = nstep_data
    train_loop, dev_loop, test_loop = loop_data

    estimator, dynamics_model = get_model_components(args, dims)
    objectives, constraints = get_objective_terms(args, dims, estimator, dynamics_model)

    model = Problem(objectives, constraints, [estimator, dynamics_model])
    model = model.to(device)

    simulator = MultiSequenceOpenLoopSimulator(
        model, train_loop, dev_loop, test_loop, eval_sim=not args.skip_eval_sim
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    visualizer = VisualizerOpen(
        dynamics_model,
        args.verbosity,
        args.savedir,
        training_visuals=False,
        trace_movie=False,
    )

    callback = SysIDCallback(simulator, visualizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    trainer = Trainer(
        model,
        train_data,
        dev_data,
        test_data,
        optimizer,
        logger=logger,
        callback=callback,
        epochs=args.epochs,
        eval_metric=args.eval_metric,
        patience=args.patience,
        warmup=args.warmup,
    )

    best_model = trainer.train()
    best_outputs = trainer.test(best_model)
    logger.clean_up()
