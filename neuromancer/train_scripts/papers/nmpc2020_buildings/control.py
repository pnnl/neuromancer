import torch
import dill
from neuromancer.problem import Problem
from neuromancer.signals import WhiteNoisePeriodicGenerator, NoiseGenerator
from neuromancer.simulators import ClosedLoopSimulator, CLSimulator
from neuromancer.trainer import Trainer, freeze_weight, unfreeze_weight
from neuromancer.visuals import VisualizerClosedLoop
from neuromancer.nmpc_visuals import VisualizerClosedLoop2
from common import load_dataset, get_logger
import setup_control as ctrl
import psl
import numpy as np


# TODO: online updates based on system ID and control optimization in the closed loop
# TODO: losses:  control loss based on the model + systemID loss
# TODO: sequential update
# 1, systemID
# 2, policy update
# 3, iterate until convergence in the closed loop


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

    emul = psl.emulators[args.system]()
    umin = np.concatenate([emul.mf_min, emul.dT_min[0]])
    umax = np.concatenate([emul.mf_max, emul.dT_max[0]])
    norm_bounds = {"U": {'min': umin, 'max': umax}}
    dataset = load_dataset(args, device, 'closedloop',
                           reduce_d=True, norm_bounds=norm_bounds)
    dataset = ctrl.add_reference_features(args, dataset, dynamics_model)

    dataset.dims['Y_ctrl_minf'] = dataset.dims['Y_minf']
    dataset.dims['Y_ctrl_maxf'] = dataset.dims['Y_maxf']

    # TODO: for standard BlockSSM
    # ny = dynamics_model.fy.out_features
    # TODO: for DecoupSISO_BlockSSM_building
    ny = dynamics_model.out_features

    # Control Problem Definition
    estimator, dynamics_model = ctrl.update_system_id_inputs(
        args, dataset, estimator, dynamics_model
    )
    policy = ctrl.get_policy_components(
        args, dataset, dynamics_model, policy_name="policy"
    )
    signal_generator = WhiteNoisePeriodicGenerator(
        args.nsteps,
        ny,
        xmax=(0.8, 0.7),
        xmin=0.2,
        min_period=1,
        max_period=20,
        name="Y_ctrl_",
    )
    disturb_generator = WhiteNoisePeriodicGenerator(
        args.nsteps,
        ny,
        xmax=(0.9, 0.7),
        xmin=0.2,
        min_period=1,
        max_period=20,
        name="D_ctrl_",
    )
    ymin_generator = WhiteNoisePeriodicGenerator(
        args.nsteps,
        ny,
        xmax=(1.0, 0.8),
        xmin=0.7,
        min_period=1,
        max_period=20,
        name="Y_ctrl_min",
    )
    ymax_generator = WhiteNoisePeriodicGenerator(
        args.nsteps,
        ny,
        xmax=(0.6, 0.3),
        xmin=0.2,
        min_period=1,
        max_period=20,
        name="Y_ctrl_max",
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
    # model = Problem(
    #     objectives,
    #     constraints,
    #     [signal_generator, disturb_generator,
    #      estimator, policy, dynamics_model, noise_generator],
    # )
    model = model.to(device)

    # train only policy component
    freeze_weight(model, module_names=args.freeze)
    unfreeze_weight(model, module_names=args.unfreeze)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    plot_keys = ["Y_pred", "U_pred"]  # variables to be plotted
    visualizer = VisualizerClosedLoop2(
        dataset, policy, plot_keys, args.verbosity, savedir=args.savedir
    )
    policy.input_keys[0] = "Yp"  # hack for policy input key compatibility w/ simulator
    simulator = CLSimulator(
        model=model, dataset=dataset, emulator=dynamics_model, policy=policy,
        gt_emulator=psl.emulators[args.system](),
        diff=False, K_r=5.0, Ki_r=0.1, Ki_con=0.1, integrator_steps=30,
    )
    # eval_metric = 'dev_sim_error',

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
        eval_metric="loop_dev_loss",
    )

    # Train control policy
    best_model = trainer.train()
    best_outputs = trainer.evaluate(best_model)
    # plots = visualizer.eval(best_outputs)
    #
    # # Logger
    # logger.log_artifacts(plots)
    logger.clean_up()
