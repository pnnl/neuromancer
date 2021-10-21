
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
from neuromancer.constraint import Variable


def arg_sys_id_problem(prefix='', system='CSTR'):
    """
    Command line parser for system identification problem arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers
                         are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = arg.ArgParser(prefix=prefix, add_help=False)
    gp = parser.group("system_id")
    #  DATA
    gp.add("-dataset", type=str, default=system,
           help="select particular dataset with keyword")
    gp.add("-nsteps", type=int, default=32,
           help="prediction horizon.")
    gp.add("-nsim", type=int, default=10000,
           help="Number of time steps for full dataset. (ntrain + ndev + ntest)"
                "train, dev, and test will be split evenly from contiguous, sequential, "
                "non-overlapping chunks of nsim datapoints, e.g. first nsim/3 art train,"
                "next nsim/3 are dev and next nsim/3 simulation steps are test points."
                "None will use a default nsim from the selected dataset or emulator")
    gp.add("-norm", nargs="+", default=["U", "D", "Y"], choices=["U", "D", "Y", "X"],
           help="List of sequences to max-min normalize")
    gp.add("-data_seed", type=int, default=408,
           help="Random seed used for simulated data")
    #  LOSS
    gp.add("-Q_con_x", type=float, default=1.0,
           help="Hidden state constraints penalty weight.")
    gp.add("-Q_dx", type=float, default=0.1,
           help="Penalty weight on hidden state difference in one time step.")
    gp.add("-Q_sub", type=float, default=0.1,
           help="Linear maps regularization weight.")
    gp.add("-Q_y", type=float, default=1.0,
           help="Output tracking penalty weight")
    gp.add("-Q_e", type=float, default=1.0,
           help="State estimator hidden prediction penalty weight")
    #  LOG
    gp.add("-savedir", type=str, default="test",
           help="Where should your trained model and plots be saved (temp)")
    gp.add("-verbosity", type=int, default=1,
           help="How many epochs in between status updates")
    gp.add("-metrics", nargs="+", default=["nstep_dev_loss", "loop_dev_loss", "best_loop_dev_loss",
               "nstep_dev_ref_loss", "loop_dev_ref_loss"],
           help="Metrics to be logged")
    #  OPTIMIZATION
    gp.add("-eval_metric", type=str, default="loop_dev_ref_loss",
            help="Metric for model selection and early stopping.")
    gp.add("-epochs", type=int, default=300,
           help='Number of training epochs')
    gp.add("-lr", type=float, default=0.001,
           help="Step size for gradient descent.")
    gp.add("-patience", type=int, default=100,
           help="How many epochs to allow for no improvement in eval metric before early stopping.")
    gp.add("-warmup", type=int, default=10,
           help="Number of epochs to wait before enacting early stopping policy.")
    gp.add("-skip_eval_sim", action="store_true",
           help="Whether to run simulator during evaluation phase of training.")
    gp.add("-seed", type=int, default=408, help="Random seed used for weight initialization.")
    gp.add("-device", type=str, default="cpu", choices=["cpu", "gpu"],
           help="select device")
    return parser


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

    # for available systems and datasets in PSL library check: psl.systems.keys() and psl.datasets.keys()
    system = 'aero'         # keyword of selected system
    # load argument parser
    parser = arg.ArgParser(parents=[arg_sys_id_problem(system=system)])
    args, grps = parser.parse_arg_groups()
    logger = BasicLogger(args=args, savedir=args.savedir,
                         verbosity=args.verbosity, stdout=args.metrics)

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

    ny = dims['Y'][1]
    nu = dims['U'][1]

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
    linmap = slim.maps['linear']

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
        linargs={},
        name='estimator',
    )
    # x0 = estimator(Yp)
    # x0: initial values of latent variables estimated from time lagged outputs Yp

    # neural network blocks
    fx = blocks.RNN(nx, nx, linear_map=linmap,
                    nonlin=activations['softexp'], hsizes=[60, 60])
    linargs = {"sigma_min": 0.5, "sigma_max": 1.0}
    fy = slim.maps['softSVD'](nx, ny, linargs=linargs)
    fu = blocks.MLP(nu, nx, hsizes=[60, 60]) if nu != 0 else None

    dynamics_model = dynamics.BlockSSM(fx, fy, fu=fu, name='dynamics', xou=torch.add,
                                       input_key_map={"x0": f"x0_{estimator.name}"})
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

    # # #  loss terms and constraints definition via variable class:
    # neuromancer variable declaration
    yhat = Variable(f"Y_pred_{dynamics_model.name}")
    y = Variable("Yf")
    x0 = Variable(f"x0_{estimator.name}")
    xhat = Variable(f"X_pred_{dynamics_model.name}")
    est_reg = Variable(f"reg_error_{estimator.name}")
    dyn_reg = Variable(f"reg_error_{dynamics_model.name}")

    # define loss function terms and constraints via operator overload
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
    model = model.to(args.device)


    """    
    # # # # # # # # # # # # # # # # # # #
    # # #        Training         # # # # 
    # # # # # # # # # # # # # # # # # # #

    #  trainer = problem + datasets + optimizer + callback
    """
    # define callback
    simulator = OpenLoopSimulator(
        model, train_loop, dev_loop, test_loop, eval_sim=not args.skip_eval_sim, device=args.device,
    ) if isinstance(train_loop, dict) else MultiSequenceOpenLoopSimulator(
        model, train_loop, dev_loop, test_loop, eval_sim=not args.skip_eval_sim, device=args.device,
    )
    visualizer = VisualizerOpen(
        dynamics_model,
        args.verbosity,
        args.savedir,
        training_visuals=False,
        trace_movie=False,
    )
    callback = SysIDCallback(simulator, visualizer)
    # select optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # define trainer
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
        device=args.device,
    )
    # train the model
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

