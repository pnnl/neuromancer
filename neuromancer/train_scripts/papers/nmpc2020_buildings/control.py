"""
Main run file for training constrained differentiable predictive control (DPC) policies
Hyperparameter setup, model architecture design,
and loss function definitions are defined in file ./setup_control.py

DPC algorithm
    1, Obtain dataset of observations of the dynamical system
        (in our case from psl library)
    2, System identification
        2a, setup system identification via ./setup_system_id.py
        2b, perform system identification via ./system_id.py
    3, Learn DPC policies offline
        3a, setup control problem via ./setup_control.py
        3b, learn control policy via this script ./control.py
        4c, evaluate closed-loop control via trainer.evaluate() using CLSimulator()
"""


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
import psl      # library for generating training datasets by simulating dynamical systems
import numpy as np





if __name__ == "__main__":
    # # # # # # # # # # # # # # # # #
    # # # ARGS and DATASET  # # # # #
    # # # # # # # # # # # # # # # # #
    parser = ctrl.get_parser()     # for argument choices see ./setup_control.py
    args = parser.parse_args()
    args.savedir = 'test_control'  # directory for saving results
    logger = get_logger(args)
    print({k: str(getattr(args, k)) for k in vars(args) if getattr(args, k)})
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
    # load trained dynamics model using file: system_id.py
    sysid_model = torch.load(args.model_file, pickle_module=dill,
                             map_location=torch.device(device))
    # separate state space dynamics model from the estimator component
    dynamics_model = sysid_model.components[1]
    estimator = sysid_model.components[0]

    # emulator of the real system using psl library
    emul = psl.emulators[args.system]()
    # control actions limits defined by the emulator model
    umin = np.concatenate([emul.mf_min, emul.dT_min[0]])
    umax = np.concatenate([emul.mf_max, emul.dT_max[0]])
    norm_bounds = {"U": {'min': umin, 'max': umax}}

    # Load the dataset
    dataset = load_dataset(args, device, 'closedloop',
                           reduce_d=True, norm_bounds=norm_bounds)
    # add synthetically generated features in the dataset such as
    # time-varying references,  constraints bounds, and sampled initial conditions
    dataset = ctrl.add_reference_features(args, dataset, dynamics_model)
    # updating dimensions dictionary in the dataset
    dataset.dims['Y_ctrl_minf'] = dataset.dims['Y_minf']
    dataset.dims['Y_ctrl_maxf'] = dataset.dims['Y_maxf']
    # Number of output features, depends on the model architecture type
        # # USE for standard BlockSSM:
        # ny = dynamics_model.fy.out_features
        # # USE for DecoupSISO_BlockSSM_building
    ny = dynamics_model.out_features


    # # # # # # # # # # # # # # # # # # # # #
    # # #  DPC Architecture components  # # #
    # # # # # # # # # # # # # # # # # # # # #

    #  state estimator and dynamics model obtained from system identification
    estimator, dynamics_model = ctrl.update_system_id_inputs(
        args, dataset, estimator, dynamics_model
    )
    # control policy to be optimizer
    policy = ctrl.get_policy_components(
        args, dataset, dynamics_model, policy_name="policy"
    )
    # additional auxiliary signal generators for noise and perturbations during training
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

    # # # # # # # # # # # # # # # # #
    # # # PROBLEM DEFINITION  # # # #
    # # # # # # # # # # # # # # # # #

    #  min_W objectives + constraints
    #  s.t. components_W
    #  W = model parameters

    # objectives and penalty constraints of the loss function
    objectives, constraints = ctrl.get_objective_terms(args, policy)
    # neural components defining the model architecture as directed acyclic graph:
    # signal_generator -> estimator -> policy -> dynamics_model -> noise_generator
    components = [signal_generator, estimator, policy, dynamics_model, noise_generator]
    # Parametric Optimal Control Problem Construction
    model = Problem(
        objectives,
        constraints,
        components,
    )
    model = model.to(device)
    # for details see problem.py in the main package folder

    # # # # # # # # # # # # # # # # # # # # # #
    # # #  UTILITIES FOR PEFROMANCE EVAL  # # #
    # # # # # # # # # # # # # # # # # # # # # #

    # Visualizer object calling plotting functions
    # for details see visuals.py in the main package folder
    plot_keys = ["Y_pred", "U_pred"]  # variables to be plotted
    visualizer = VisualizerClosedLoop2(
        dataset, policy, plot_keys, args.verbosity, savedir=args.savedir
    )
    # Simulator object for performing closed-loop simulation
    # for details see simulators.py in the main package folder
    policy.input_keys[0] = "Yp"  # update policy input key for compatibility with simulator
    simulator = CLSimulator(
        model=model, dataset=dataset, emulator=dynamics_model, policy=policy,
        gt_emulator=psl.emulators[args.system](),
        diff=True, K_r=5.0, Ki_r=0.1, Ki_con=0.1, integrator_steps=30,
    )

    # # # # # # # # # # # #
    # # #  TRAINING   # # #
    # # # # # # # # # # # #

    # train only policy component by freezing model parameters
    freeze_weight(model, module_names=args.freeze)
    unfreeze_weight(model, module_names=args.unfreeze)
    # select optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # define trainer which will optimize the model given the dataset using optimizer
    # for details see trainer.py in the main package folder
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
    logger.clean_up()
