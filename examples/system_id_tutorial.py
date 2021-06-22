
import torch
import torch.nn.functional as F
import slim
import psl

from neuromancer.activations import activations
from neuromancer import blocks, estimators, dynamics, arg
from neuromancer.visuals import VisualizerOpen
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem, Objective
from neuromancer.simulators import OpenLoopSimulator
from neuromancer.datasets import load_dataset
from neuromancer.loggers import BasicLogger


if __name__ == "__main__":

    """
    # # # # # # # # # # # # # # # # # # #
    # # #  DATASET LOAD   # # # # # # # #
    # # # # # # # # # # # # # # # # # # #
    """

    # for available systems in PSL library check: psl.systems.keys()
    # for available datasets in PSL library check: psl.datasets.keys()
    system = 'aero'         # keyword of selected system

    # load argument parser
    parser = arg.ArgParser(parents=[arg.log(), arg.opt(), arg.data(system=system),
                                    arg.loss(), arg.lin(), arg.ssm()])
    grp = parser.group('OPTIMIZATION')
    grp.add("-eval_metric", type=str, default="loop_dev_ref_loss",
            help="Metric for model selection and early stopping.")
    args, grps = parser.parse_arg_groups()
    device = "cpu"
    # metrics to be logged
    metrics = ["nstep_dev_loss", "loop_dev_loss", "best_loop_dev_loss",
               "nstep_dev_ref_loss", "loop_dev_ref_loss"]
    logger = BasicLogger(args=args, savedir=args.savedir, verbosity=args.verbosity, stdout=metrics)

    args.nsteps = 32  # define prediction horizon length

    #  load the dataset
    dataset = load_dataset(args, device, 'openloop')
    print(dataset.dims)
    # nsim = number of time steps in the dataset time series
    # nsteps = legth of the prediction horizon
    # Y = observed outputs
    # U = inputs
    # D = disturbances
    # Yp = past trajectories generated as Y[0:-nsteps]
    # Yf = future trajectories generated as Y[nesteps:]

    #  Train, Development, Test sets
    dataset.train_data['Yp'].shape
    dataset.dev_data['Yp'].shape
    dataset.test_data['Yp'].shape
    # out: torch.Size([batch size (prediction horizon),
    #                  number of batches, number of variables])

    """
    # # # # # # # # # # # # # # # # # # #
    # # #  MODEL CONSTRUCTION   # # # # #
    # # # # # # # # # # # # # # # # # # #
    
    # Model = directed acyclic graph of neural components
    # neural components = structured neural networks composed of blocks
    # blocks = standard neural architectures such as MLP, ResNet, RNN,
    #         composed of linear layers and activations
    # linear layer = possibly using constrained/factorized matrices from slim
    # activations = possibly using learnable activation functions
    """

    activation = activations['relu']
    linmap = slim.maps['softSVD']
    linargs = {"sigma_min": 0.5, "sigma_max": 1.0}
    block = blocks.MLP

    nx = 90  # size of the latent variables
    estimator = estimators.MLPEstimator(
        {**dataset.dims, "x0": (nx,)},
        nsteps=args.nsteps,  # future window Nf
        window_size=args.nsteps,  # past window Np <= Nf
        bias=True,
        linear_map=linmap,
        nonlin=activation,
        hsizes=[90, 120, 90],
        input_keys=["Yp"],
        linargs=linargs,
        name='estimator',
    )
    # x0 = f_estim(Yp)
    # x0: initial values of latent variables estimated from time lagged outputs Yp

    ny = dataset.dims['Y'][1]
    nu = dataset.dims['U'][1]

    # neural network blocks
    fx = blocks.RNN(nx, nx)
    fy = slim.maps['pf'](nx, ny, linargs=linargs)
    fu = block(nu, nx) if nu != 0 else None

    dynamics_model = dynamics.BlockSSM(fx, fy, fu=fu, name='dynamics',
                                       input_keys={'x0': f'x0_{estimator.name}'})


    """    
    # # # # # # # # # # # # # # # # # # #
    # # #  POBLEM DEFINITION    # # # # #
    # # # # # # # # # # # # # # # # # # #

    # Problem = model components + constraints + objectives
    """

    xmin = -0.2
    xmax = 1.2
    dxudmin = -0.5
    dxudmax = 0.5

    #  L_r = Q_y*|| Y_pred_dynamics - Yf ||_2^2  or L_r = Q_y*mse(Y_pred_dynamics - Yf)
    reference_loss = Objective(
        [f"Y_pred_{dynamics_model.name}", "Yf"], F.mse_loss, weight=args.Q_y, name="ref_loss"
    )

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
    state_smoothing = Objective(
        [f"X_pred_{dynamics_model.name}"],
        lambda x: F.mse_loss(x[1:], x[:-1]),
        weight=args.Q_dx,
        name="state_smoothing",
    )
    # inequality constraints
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


    """    
    # # # # # # # # # # # # # # # # # # #
    # # #   Problem Definition    # # # # 
    # # # # # # # # # # # # # # # # # # #

    #  problem = objectives + constraints + components
    """
    # estimator -> dynamics_model
    components = [estimator, dynamics_model]
    model = Problem(objectives, constraints, components)
    model = model.to(device)


    """    
    # # # # # # # # # # # # # # # # # # #
    # # #        Training         # # # # 
    # # # # # # # # # # # # # # # # # # #

    #  trainer = problem + dataset + optimizer
    """
    args.epochs = 1000
    simulator = OpenLoopSimulator(model=model, dataset=dataset, eval_sim=not args.skip_eval_sim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    trainer = Trainer(
        model,
        dataset,
        optimizer,
        simulator=simulator,
        logger=logger,
        epochs=args.epochs,
        eval_metric=args.eval_metric,
        patience=args.patience,
        warmup=args.warmup,
    )

    best_model = trainer.train()

    """    
    # # # # # # # # # # # # # # # # # # #
    # # #        RESULTS        # # # # #
    # # # # # # # # # # # # # # # # # # #

    # visualize and log results
    """
    best_outputs = trainer.evaluate(best_model)
    visualizer = VisualizerOpen(
        dataset,
        dynamics_model,
        args.verbosity,
        args.savedir,
        training_visuals=False,
        trace_movie=False,
    )
    plots = visualizer.eval(best_outputs)

    logger.log_artifacts(plots)
    logger.clean_up()
