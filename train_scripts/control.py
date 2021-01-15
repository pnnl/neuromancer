import torch
import dill
from neuromancer.problem import Problem
from neuromancer.signals import WhiteNoisePeriodicGenerator, NoiseGenerator
from neuromancer.simulators import ClosedLoopSimulator
from neuromancer.trainer import Trainer, freeze_weight, unfreeze_weight
from neuromancer.visuals import VisualizerClosedLoop

import common.control as ctrl
from common import load_dataset, get_logger


if __name__ == "__main__":
    parser = ctrl.get_parser()
    args = parser.parse_args()
    args.savedir = 'test_control'
    logger = get_logger(args)

    print({k: str(getattr(args, k)) for k in vars(args) if getattr(args, k)})
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"

    sysid_model = torch.load(args.model_file, pickle_module=dill, map_location=torch.device(device))
    dynamics_model = sysid_model.components[1]
    estimator = sysid_model.components[0]

    dataset = load_dataset(args, device, 'closedloop')
    dataset = ctrl.add_reference_features(args, dataset, dynamics_model)

    # Control Problem Definition
    estimator, dynamics_model = ctrl.update_system_id_inputs(
        args, dataset, estimator, dynamics_model
    )
    policy = ctrl.get_policy_components(
        args, dataset, dynamics_model, policy_name="policy"
    )
    signal_generator = WhiteNoisePeriodicGenerator(
        args.nsteps,
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

    objectives, constraints = ctrl.get_objective_terms(args, policy)
    model = Problem(
        objectives,
        constraints,
        [signal_generator, estimator, policy, dynamics_model, noise_generator],
    )
    model = model.to(device)

    # train only policy component
    freeze_weight(model, module_names=args.freeze)
    unfreeze_weight(model, module_names=args.unfreeze)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

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
        epochs=args.epochs,
        patience=args.patience,
        warmup=args.warmup,
    )

    # Train control policy
    best_model = trainer.train()
    best_outputs = trainer.evaluate(best_model)
    plots = visualizer.eval(best_outputs)

    # Logger
    logger.log_artifacts(plots)
    logger.clean_up()