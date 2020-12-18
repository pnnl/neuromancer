"""
Script for training neural control policies based on system identification.
"""
import argparse

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
import common.control as ctrl

if __name__ == "__main__":
    parser = sys_id.get_parser()
    parser = ctrl.get_parser(parser=parser, add_prefix=True)
    args = parser.parse_args()

    print({k: str(getattr(args, k)) for k in vars(args) if getattr(args, k)})
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"

    # Load Dataset System ID
    logger = get_logger(args)
    dataset = load_dataset(args, device)
    print(dataset.dims)

    # System ID Model
    estimator, dynamics_model = sys_id.get_model_components(args, dataset)
    objectives, constraints = sys_id.get_objective_terms(
        args, dataset, estimator, dynamics_model
    )

    model = Problem(objectives, constraints, [estimator, dynamics_model])
    model = model.to(device)

    simulator = OpenLoopSimulator(
        model=model, dataset=dataset, eval_sim=not args.skip_eval_sim
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    trainer = Trainer(
        model,
        dataset,
        optimizer,
        logger=logger,
        simulator=simulator,
        epochs=args.epochs,
        eval_metric=args.eval_metric,
        patience=args.patience,
        warmup=args.warmup,
    )

    # System ID train
    best_model = trainer.train()
    best_outputs = trainer.evaluate(best_model)

    visualizer_system_ID = VisualizerOpen(
        dataset,
        dynamics_model,
        args.verbosity,
        args.savedir,
        training_visuals=args.train_visuals,
        trace_movie=args.trace_movie,
    )
    plots = visualizer_system_ID.eval(best_outputs)

    logger.log_artifacts(plots)
    logger.clean_up()

    ## Control Policy Learning:

    ctrl_args = argparse.Namespace(
        **{
            **{k: v for k, v in vars(args).items() if not k.startswith("ctrl_")},
            **{
                k.replace("ctrl_", ""): v
                for k, v in vars(args).items()
                if k.startswith("ctrl_")
            },
        }
    )

    ###################
    #### Control   ####
    ###################

    # Control Dataset
    logger = get_logger(ctrl_args)
    dataset = load_dataset(ctrl_args, device, name="closedloop")
    dataset = ctrl.add_reference_features(dataset, dynamics_model)

    # Control Problem Definition
    estimator, dynamics_model = ctrl.update_system_id_inputs(
        ctrl_args, dataset, estimator, dynamics_model
    )
    policy = ctrl.get_policy_components(
        ctrl_args, dataset, dynamics_model, policy_name="policy"
    )
    signal_generator = WhiteNoisePeriodicGenerator(
        ctrl_args.nsteps,
        dynamics_model.fy.out_features,
        xmax=(0.8, 0.7),
        xmin=0.2,
        min_period=1,
        max_period=20,
        name="Y_ctrl_",
    )
    noise_generator = NoiseGenerator(ratio=0.05, keys=["Y_pred_dynamics"], name="_noise")

    objectives, constraints = ctrl.get_objective_terms(ctrl_args, policy)
    model = Problem(
        objectives,
        constraints,
        [signal_generator, estimator, policy, dynamics_model, noise_generator],
    )
    model = model.to(device)

    # train only policy component
    freeze_weight(model, module_names=ctrl_args.freeze)
    unfreeze_weight(model, module_names=ctrl_args.unfreeze)
    optimizer = torch.optim.AdamW(model.parameters(), lr=ctrl_args.lr)

    plot_keys = ["Y_pred", "U_pred"]  # variables to be plotted
    visualizer = VisualizerClosedLoop(
        dataset, policy, plot_keys, args.verbosity, savedir=args.savedir
    )

    policy.input_keys[0] = "Yp"  # hack for policy input key compatibility w/ simulator
    simulator = ClosedLoopSimulator(
        model=model, dataset=dataset, emulator=dynamics_model, policy=policy
    )
    trainer = Trainer(
        model,
        dataset,
        optimizer,
        logger=logger,
        visualizer=visualizer,
        simulator=simulator,
        epochs=ctrl_args.epochs,
        patience=ctrl_args.patience,
        warmup=ctrl_args.warmup,
    )

    # Train control policy
    best_model = trainer.train()
    best_outputs = trainer.evaluate(best_model)
    plots = visualizer.eval(best_outputs)

    logger.log_artifacts(plots)
    logger.clean_up()