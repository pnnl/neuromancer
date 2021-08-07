
import torch
import torch.nn.functional as F
import slim
import psl

from neuromancer import blocks, estimators, dynamics, arg
from neuromancer.activations import activations
from neuromancer.visuals import VisualizerOpen
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.constraint import Objective
from neuromancer.simulators import OpenLoopSimulator, MultiSequenceOpenLoopSimulator
from neuromancer.callbacks import SysIDCallback
from neuromancer.loggers import BasicLogger, MLFlowLogger
from neuromancer.dataset import read_file, normalize_data, split_sequence_data, SequenceDataset
from torch.utils.data import DataLoader
from neuromancer.constraint import Variable


def get_sequence_dataloaders(
    data, nsteps, moving_horizon=False, norm_type="zero-one", split_ratio=None, num_workers=0,
):
    """This will generate dataloaders and open-loop sequence dictionaries for a given dictionary of
    data. Dataloaders are hard-coded for full-batch training to match NeuroMANCER's original
    training setup.

    :param data: (dict str: np.array or list[dict str: np.array]) data dictionary or list of data
        dictionaries; if latter is provided, multi-sequence datasets are created and splits are
        computed over the number of sequences rather than their lengths.
    :param nsteps: (int) length of windowed subsequences for N-step training.
    :param moving_horizon: (bool) whether to use moving horizon batching.
    :param norm_type: (str) type of normalization; see function `normalize_data` for more info.
    :param split_ratio: (list float) percentage of data in train and development splits; see
        function `split_sequence_data` for more info.
    """

    data, _ = normalize_data(data, norm_type)
    train_data, dev_data, test_data = split_sequence_data(data, nsteps, moving_horizon, split_ratio)

    train_data = SequenceDataset(
        train_data,
        nsteps=nsteps,
        moving_horizon=moving_horizon,
        name="train",
    )
    dev_data = SequenceDataset(
        dev_data,
        nsteps=nsteps,
        moving_horizon=moving_horizon,
        name="dev",
    )
    test_data = SequenceDataset(
        test_data,
        nsteps=nsteps,
        moving_horizon=moving_horizon,
        name="test",
    )

    train_loop = train_data.get_full_sequence()
    dev_loop = dev_data.get_full_sequence()
    test_loop = test_data.get_full_sequence()

    train_data = DataLoader(
        train_data,
        batch_size=len(train_data),
        shuffle=False,
        collate_fn=train_data.collate_fn,
        num_workers=num_workers,
    )
    dev_data = DataLoader(
        dev_data,
        batch_size=len(dev_data),
        shuffle=False,
        collate_fn=dev_data.collate_fn,
        num_workers=num_workers,
    )
    test_data = DataLoader(
        test_data,
        batch_size=len(test_data),
        shuffle=False,
        collate_fn=test_data.collate_fn,
        num_workers=num_workers,
    )

    return (train_data, dev_data, test_data), (train_loop, dev_loop, test_loop), train_data.dataset.dims


if __name__ == "__main__":

    """
    # # # # # # # # # # # # # # # # # # #
    # # #  ARGS and LOGGER  # # # # # # # 
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
    args.nsteps = 32  # define prediction horizon length

    device = "cpu"
    # metrics to be logged
    metrics = ["nstep_dev_loss", "loop_dev_loss", "best_loop_dev_loss",
               "nstep_dev_ref_loss", "loop_dev_ref_loss"]
    logger = BasicLogger(args=args, savedir=args.savedir, verbosity=args.verbosity, stdout=metrics)

    """
    # # # # # # # # # # # # # # # # # # #
    # # #  DATASET LOAD   # # # # # # # #
    # # # # # # # # # # # # # # # # # # #
    
    # nsim = number of time steps in the dataset time series
    # nsteps = legth of the prediction horizon
    # Y = observed outputs
    # U = inputs
    # D = disturbances
    # Yp = past trajectories generated as Y[0:-nsteps]
    # Yf = future trajectories generated as Y[nesteps:]
    
    # data format in dataset dictionaries:
    # train_data['key']: torch.Size([batch size (prediction horizon),
    #                  number of batches, number of variables])
    """

    #  load and split the dataset
    if args.dataset in psl.emulators:
        data = psl.emulators[args.dataset](nsim=args.nsim, ninit=0, seed=args.data_seed).simulate()
    elif args.dataset in psl.datasets:
        data = read_file(psl.datasets[args.dataset])
    else:
        data = read_file(args.dataset)

    #  Train, Development, Test sets - nstep and loop format
    nstep_data, loop_data, dims = get_sequence_dataloaders(data, args.nsteps)
    train_data, dev_data, test_data = nstep_data
    train_loop, dev_loop, test_loop = loop_data


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
    linargs = {"sigma_min": 0.95, "sigma_max": 1.0}
    block = blocks.MLP

    nx = 90  # size of the latent variables
    estimator = estimators.MLPEstimator(
        {**dims, "x0": (nx,)},
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
    # x0 = estimator(Yp)
    # x0: initial values of latent variables estimated from time lagged outputs Yp

    ny = dims['Y'][1]
    nu = dims['U'][1]

    # neural network blocks
    fx = blocks.RNN(nx, nx)
    fy = slim.maps['softSVD'](nx, ny, linargs=linargs)
    fu = block(nu, nx) if nu != 0 else None

    dynamics_model = dynamics.BlockSSM(fx, fy, fu=fu, name='dynamics',
                                       input_keys={f"x0_{estimator.name}": "x0"})
    # Yf = dynamics_model(x0, Uf)
    # Uf: future control actions
    # Yf: predicted outputs

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

    # # # Loss terms and constraitns definition via Objective class:
    # # # this approach can be used for more user flexibility as it supports any callable

    # reference_loss = Objective(
    #     [f"Y_pred_{dynamics_model.name}", "Yf"], F.mse_loss, weight=args.Q_y, name="ref_loss"
    # )
    #
    # estimator_loss = Objective(
    #     [f"X_pred_{dynamics_model.name}", f"x0_{estimator.name}"],
    #     lambda X_pred, x0: F.mse_loss(X_pred[-1, :-1, :], x0[1:]),
    #     weight=args.Q_e,
    #     name="arrival_cost",
    # )
    # regularization = Objective(
    #     [f"reg_error_{estimator.name}", f"reg_error_{dynamics_model.name}"],
    #     lambda reg1, reg2: reg1 + reg2,
    #     weight=args.Q_sub,
    #     name="reg_error",
    # )
    # state_smoothing = Objective(
    #     [f"X_pred_{dynamics_model.name}"],
    #     lambda x: F.mse_loss(x[1:], x[:-1]),
    #     weight=args.Q_dx,
    #     name="state_smoothing",
    # )
    # # inequality constraints
    # observation_lower_bound_penalty = Objective(
    #     [f"Y_pred_{dynamics_model.name}"],
    #     lambda x: torch.mean(F.relu(-x + xmin)),
    #     weight=args.Q_con_x,
    #     name="y_low_bound_error",
    # )
    # observation_upper_bound_penalty = Objective(
    #     [f"Y_pred_{dynamics_model.name}"],
    #     lambda x: torch.mean(F.relu(x - xmax)),
    #     weight=args.Q_con_x,
    #     name="y_up_bound_error",
    # )

    # # #  alternative loss terms and constraints definition via variable class:

    # neuromancer variable declaration
    yhat = Variable(f"Y_pred_{dynamics_model.name}")
    y = Variable("Yf")
    x0 = Variable(f"x0_{estimator.name}")
    xhat = Variable(f"X_pred_{dynamics_model.name}")
    est_reg = Variable(f"reg_error_{estimator.name}")
    dyn_reg = Variable(f"reg_error_{dynamics_model.name}")

    # define loss function terms and constraints via operator overlad
    reference_loss = args.Q_y*((yhat == y)^2)
    estimator_loss = args.Q_e*((x0[1:] == xhat[-1, :-1, :])^2)
    state_smoothing = args.Q_dx*((xhat[1:] == xhat[:-1])^2)
    regularization = args.Q_sub*((est_reg + dyn_reg == 0)^2)
    observation_lower_bound_penalty = args.Q_con_x*(yhat > xmin)
    observation_upper_bound_penalty = args.Q_con_x*(yhat < xmax)

    # custom loss and constraints names
    reference_loss.name = "ref_loss"
    estimator_loss.name = "arrival_cost"
    regularization.name = "reg_error"
    observation_lower_bound_penalty.name = "y_low_bound_error"
    observation_upper_bound_penalty.name = "y_up_bound_error"

    # list of objectives and constraints
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

    #  trainer = problem + datasets + optimizer + callback
    """
    args.epochs = 20

    simulator = OpenLoopSimulator(
        model, train_loop, dev_loop, test_loop, eval_sim=not args.skip_eval_sim, device=device,
    ) if isinstance(train_loop, dict) else MultiSequenceOpenLoopSimulator(
        model, train_loop, dev_loop, test_loop, eval_sim=not args.skip_eval_sim, device=device,
    )
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
        device=device,
    )

    best_model = trainer.train()

    """    
    # # # # # # # # # # # # # # # # # # #
    # # #        RESULTS        # # # # #
    # # # # # # # # # # # # # # # # # # #

    # simulate, visualize and log results
    """
    best_outputs = trainer.test(best_model)
    # best_outputs: dictionary of test call results
    # we can just call visualizer on the results dictionary:
    # visualizer.eval(best_outputs)
    logger.clean_up()
