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

import torch
import torch.nn.functional as F
import slim
import psl

from neuromancer import blocks, estimators, dynamics, arg
from neuromancer.activations import activations
from neuromancer.visuals import VisualizerOpen
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.constraint import Loss
from neuromancer.simulators import OpenLoopSimulator, MultiSequenceOpenLoopSimulator
from neuromancer.callbacks import SysIDCallback
from neuromancer.loggers import BasicLogger, MLFlowLogger
from neuromancer.dataset import read_file, normalize_data, split_sequence_data, SequenceDataset
from torch.utils.data import DataLoader



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
    #  COMPONENTS
    gp.add("-ssm_type", type=str, choices=["blackbox", "hw", "hammerstein",
                                "blocknlin", "linear"], default="hammerstein",
           help='Choice of block structure for system identification model')
    gp.add("-state_estimator", type=str, choices=["rnn", "mlp", "linear",
                            "residual_mlp", "fully_observable"], default="mlp",
           help='Choice of model architecture for state estimator.')
    gp.add("-estimator_input_window", type=int, default=8,
           help="Number of previous time steps measurements to include in state estimator input")
    gp.add("-nx_hidden", type=int, default=32,
           help="Number of hidden states per output")
    gp.add("-n_layers", type=int, default=2,
           help="Number of hidden layers of single time-step state transition")
    gp.add("-nonlinear_map", type=str, default="mlp", choices=["mlp", "rnn", "pytorch_rnn",
                                                               "linear", "residual_mlp"],
           help='Choice of architecture for component blocks in state space model.')
    gp.add("-bias", action="store_true",
           help="Whether to use bias in the neural network block component models.")
    gp.add("-activation", choices=activations.keys(), default="gelu",
           help="Activation function for component block neural networks")
    gp.add("-linear_map", type=str, choices=["linear", "softSVD", "pf"], default="linear",
           help='Choice of map from SLiM package')
    gp.add("-sigma_min", type=float, default=0.1,
           help='Minimum singular value (for maps with singular value constraints)')
    gp.add("-sigma_max", type=float, default=1.0,
           help='Maximum singular value (for maps with singular value constraints)')
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
    gp.add("-Q_con_fdu", type=float, default=0.0,
           help="Penalty weight on control actions and disturbances.")
    #  LOG
    gp.add("-logger", type=str, choices=["mlflow", "stdout"], default="stdout",
           help="Logging setup to use")
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
    parser.add("-gpu", type=int, help="GPU to use")
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
            {**dims, "x0": (nx,)},
            linmap,
            nonlinmap,
            bias=args.bias,
            n_layers=args.n_layers,
            activation=activation,
            name=dynamics_name,
            input_key_map={"x0": f"x0_{estimator.name}"},
            linargs=linargs
        ) if args.ssm_type == "blackbox"
        else dynamics.block_model(
            args.ssm_type,
            {**dims, "x0": (nx,)},
            linmap,
            nonlinmap,
            bias=args.bias,
            n_layers=args.n_layers,
            activation=activation,
            name=dynamics_name,
            input_key_map={"x0": f"x0_{estimator.name}"},
            linargs=linargs
        )
    )
    return estimator, dynamics_model


def get_objective_terms(args, dims, estimator, dynamics_model):
    xmin = -0.2
    xmax = 1.2
    dxudmin = -0.05
    dxudmax = 0.05
    estimator_loss = Loss(
        [f"X_pred_{dynamics_model.name}", f"x0_{estimator.name}"],
        lambda X_pred, x0: F.mse_loss(X_pred[-1, :-1, :], x0[1:]),
        weight=args.Q_e,
        name="arrival_cost",
    )
    regularization = Loss(
        [f"reg_error_{estimator.name}", f"reg_error_{dynamics_model.name}"],
        lambda reg1, reg2: reg1 + reg2,
        weight=args.Q_sub,
        name="reg_error",
    )
    reference_loss = Loss(
        [f"Y_pred_{dynamics_model.name}", "Yf"], F.mse_loss, weight=args.Q_y,
        name="ref_loss"
    )
    state_smoothing = Loss(
        [f"X_pred_{dynamics_model.name}"],
        lambda x: F.mse_loss(x[1:], x[:-1]),
        weight=args.Q_dx,
        name="state_smoothing",
    )
    observation_lower_bound_penalty = Loss(
        [f"Y_pred_{dynamics_model.name}"],
        lambda x: torch.mean(F.relu(-x + xmin)),
        weight=args.Q_con_x,
        name="y_low_bound_error",
    )
    observation_upper_bound_penalty = Loss(
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
            inputs_max_influence_lb = Loss(
                [f"fU_{dynamics_model.name}"],
                lambda x: torch.mean(F.relu(-x + dxudmin)),
                weight=args.Q_con_fdu,
                name="input_influence_lb",
            )
            inputs_max_influence_ub = Loss(
                [f"fU_{dynamics_model.name}"],
                lambda x: torch.mean(F.relu(x - dxudmax)),
                weight=args.Q_con_fdu,
                name="input_influence_ub",
            )
            constraints += [inputs_max_influence_lb, inputs_max_influence_ub]
        if "D" in dims:
            disturbances_max_influence_lb = Loss(
                [f"fD_{dynamics_model.name}"],
                lambda x: torch.mean(F.relu(-x + dxudmin)),
                weight=args.Q_con_fdu,
                name="dist_influence_lb",
            )
            disturbances_max_influence_ub = Loss(
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
    system = 'aero'         # keyword of selected system
    parser = arg.ArgParser(parents=[arg_sys_id_problem(system=system)])
    args, grps = parser.parse_arg_groups()
    print({k: str(getattr(args, k)) for k in vars(args) if getattr(args, k)})
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
    log_constructor = MLFlowLogger if args.logger == 'mlflow' else BasicLogger
    logger = log_constructor(args=args, savedir=args.savedir, verbosity=args.verbosity, stdout=args.metrics)

    # load raw data
    if args.dataset in psl.emulators:
        data = psl.emulators[args.dataset](nsim=args.nsim, ninit=0, seed=args.data_seed).simulate()
    elif args.dataset in psl.datasets:
        data = read_file(psl.datasets[args.dataset])
    else:
        data = read_file(args.dataset)

    # create dataloaders from raw data
    nstep_data, loop_data, dims = get_sequence_dataloaders(data, args.nsteps)
    train_data, dev_data, test_data = nstep_data
    train_loop, dev_loop, test_loop = loop_data

    # create component models, objectives, and constraints
    estimator, dynamics_model = get_model_components(args, dims)
    objectives, constraints = get_objective_terms(args, dims, estimator, dynamics_model)

    # create problem
    model = Problem(objectives, constraints, [estimator, dynamics_model])
    model = model.to(device)
    print(model)

    # create callback
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
    # select optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # create trainer
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
    # train
    best_model = trainer.train()
    best_outputs = trainer.test(best_model)
    logger.clean_up()
