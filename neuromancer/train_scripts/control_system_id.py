"""
Script for training neural control policies based on system identification.
"""

import torch
from neuromancer.visuals import (
    VisualizerOpen,
    VisualizerTrajectories,
    VisualizerClosedLoop,
)
from neuromancer.trainer import Trainer, freeze_weight, unfreeze_weight
from neuromancer.problem import Problem
from neuromancer.simulators import OpenLoopSimulator, ClosedLoopSimulator
from neuromancer.signals import (
    NoiseGenerator,
    SignalGenerator,
    WhiteNoisePeriodicGenerator,
    PeriodicGenerator,
    WhiteNoiseGenerator,
    AddGenerator,
)

from common import load_dataset, get_logger
import common.system_id as sys_id
import common.control as ctl

if __name__ == "__main__":
    args_id = sys_id.get_parser().parse_args()
    print({k: str(getattr(args_id, k)) for k in vars(args_id) if getattr(args_id, k)})
    device = f"cuda:{args_id.gpu}" if args_id.gpu is not None else "cpu"

    # Load Dataset System ID
    logger_id = get_logger(args_id)
    dataset_id = load_dataset(args_id, device)
    print(dataset_id.dims)

    # System ID Model
    estimator, dynamics_model = sys_id.get_model_components(args_id, dataset_id)
    objectives_id, constraints_id = sys_id.get_objective_terms(
        args_id, dataset_id, estimator, dynamics_model
    )

    model_id = Problem(objectives_id, constraints_id, [estimator, dynamics_model])
    model_id = model_id.to(device)

    simulator_ol = OpenLoopSimulator(
        model=model_id, dataset=dataset_id, eval_sim=not args_id.skip_eval_sim
    )
    optimizer = torch.optim.AdamW(model_id.parameters(), lr=args_id.lr)
    trainer = Trainer(
        model_id,
        dataset_id,
        optimizer,
        logger=logger_id,
        simulator=simulator_ol,
        epochs=args_id.epochs,
        eval_metric=args_id.eval_metric,
        patience=args_id.patience,
        warmup=args_id.warmup,
    )

    # System ID train
    best_model = trainer.train()
    best_outputs = trainer.evaluate(best_model)

    visualizer_system_ID = VisualizerOpen(
        dataset_id,
        dynamics_model,
        args_id.verbosity,
        args_id.savedir,
        training_visuals=args_id.train_visuals,
        trace_movie=args_id.trace_movie,
    )
    plots_id = visualizer_system_ID.eval(best_outputs)

    # Logger
    logger_id.log_artifacts(plots_id)
    logger_id.clean_up()

    # Control Dataset
    args_ctrl = ctl.get_parser().parse_args()
    print({k: str(getattr(args_id, k)) for k in vars(args_id) if getattr(args_id, k)})
    device = f"cuda:{args_ctrl.gpu}" if args_ctrl.gpu is not None else "cpu"
    logger_ctrl = get_logger(args_ctrl)
    dataset_ctrl = load_dataset(args_ctrl, device)
    dataset_ctrl = ctl.add_reference_features(dataset_ctrl, dynamics_model)

    # Control Problem Definition
    estimator, dynamics_model = ctl.update_system_id_inputs(
        args_ctrl, dataset_ctrl, estimator, dynamics_model
    )
    policy = ctl.get_policy_components(
        args_ctrl, dataset_ctrl, dynamics_model, policy_name="policy"
    )
    signal_generator = WhiteNoisePeriodicGenerator(
        args_ctrl.nsteps,
        dynamics_model.fy.out_features,
        xmax=(0.8, 0.7),
        xmin=0.2,
        min_period=1,
        max_period=20,
        name="Y_ctrl_",
    )
    noise_generator = NoiseGenerator(
        ratio=0.05, keys=["Y_pred_dynamics"], name="_noise"
    )

    objectives, constraints = ctl.get_objective_terms(args_ctrl, policy)
    model_ctrl = Problem(
        objectives,
        constraints,
        [signal_generator, estimator, policy, dynamics_model, noise_generator],
    )
    model_ctrl = model_ctrl.to(device)

    # train only policy component
    freeze_weight(model_ctrl, module_names=args_ctrl.freeze)
    unfreeze_weight(model_ctrl, module_names=args_ctrl.unfreeze)
    optimizer = torch.optim.AdamW(model_ctrl.parameters(), lr=args_ctrl.lr)

    plot_keys = ["Y_pred", "U_pred"]  # variables to be plotted
    visualizer = VisualizerClosedLoop(
        dataset_ctrl, policy, plot_keys, args_ctrl.verbosity, savedir=args_ctrl.savedir
    )

    policy.input_keys[0] = "Yp"  # hack for policy input key compatibility w/ simulator
    simulator = ClosedLoopSimulator(
        model=model_ctrl, dataset=dataset_ctrl, emulator=dynamics_model, policy=policy
    )
    trainer = Trainer(
        model_ctrl,
        dataset_ctrl,
        optimizer,
        logger=logger_ctrl,
        visualizer=visualizer,
        simulator=simulator,
        epochs=args_ctrl.epochs,
        patience=args_ctrl.patience,
        warmup=args_ctrl.warmup,
    )

    # Train control policy
    best_model = trainer.train()
    best_outputs = trainer.evaluate(best_model)
    plots_ctrl = visualizer.eval(best_outputs)

    # Logger
    logger_ctrl.log_artifacts(plots_ctrl)
    logger_ctrl.log_metrics({"alive": 0.0})
    logger_ctrl.clean_up()
