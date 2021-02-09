"""
Script for training block dynamics models for system identification.

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
import argparse

import torch
import slim
import neuromancer.blocks as blocks
from neuromancer.visuals import VisualizerOpen, VisualizerTrajectories
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.simulators import OpenLoopSimulator
from common import load_dataset, get_logger

from setup_system_id import (
    get_model_components,
    get_objective_terms,
    get_parser
)

# TODO: bilinear for fu
# https://pytorch.org/docs/stable/generated/torch.nn.Bilinear.html
# TODO: tetrain model

if __name__ == "__main__":
    args = get_parser().parse_args()
    args.bias = False
    print({k: str(getattr(args, k)) for k in vars(args) if getattr(args, k)})
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"

    logger = get_logger(args)
    dataset = load_dataset(args, device, "openloop")
    print(dataset.dims)

    estimator, dynamics_model = get_model_components(args, dataset)
    objectives, constraints = get_objective_terms(args, dataset, estimator, dynamics_model)

    model = Problem(objectives, constraints, [estimator, dynamics_model])
    model = model.to(device)

    simulator = OpenLoopSimulator(model=model, dataset=dataset, eval_sim=not args.skip_eval_sim)
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

    best_model = trainer.train()
    best_outputs = trainer.evaluate(best_model)

    visualizer = VisualizerOpen(
        dataset,
        dynamics_model,
        args.verbosity,
        args.savedir,
        training_visuals=args.train_visuals,
        trace_movie=args.trace_movie,
    )
    plots = visualizer.eval(best_outputs)

    logger.log_artifacts(plots)
    logger.clean_up()
