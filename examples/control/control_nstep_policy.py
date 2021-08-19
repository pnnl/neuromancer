import dill
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import psl
import slim

from neuromancer.activations import BLU, SoftExponential
from neuromancer import policies, arg
from neuromancer.activations import activations
from neuromancer.problem import Problem, Objective
from torch.utils.data import DataLoader
from neuromancer.signals import WhiteNoisePeriodicGenerator, DataNoiseGenerator
from neuromancer.simulators import ClosedLoopSimulator
from neuromancer.trainer import Trainer, freeze_weight, unfreeze_weight
from neuromancer.visuals import VisualizerClosedLoop, VisualizerOpen
from neuromancer.dataset import normalize_data, split_sequence_data, SequenceDataset
from neuromancer.loggers import BasicLogger, MLFlowLogger
from neuromancer.callbacks import ControlCallback, SysIDCallback

# TODO adhoc imports for testing plots
import numpy as np
from neuromancer.plot import pltCL


def arg_control_problem(prefix='', system='TwoTank', path='./test/best_model.pth'):
    """
    Command line parser for differentiable predictive (DPC) problem arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers
                         are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = arg.ArgParser(prefix=prefix, add_help=False)
    gp = parser.group("control")
    #  DATA
    gp.add("-dataset", type=str, default=system,
           help="select particular dataset with keyword")
    parser.add('-model_file', type=str, default=path,
               help='Path to pytorch pickled model.')
    gp.add("-nsteps", type=int, default=8,
           help="prediction horizon.")
    gp.add("-nsim", type=int, default=10000,
           help="Number of time steps for full dataset. (ntrain + ndev + ntest)"
                "train, dev, and test will be split evenly from contiguous, sequential, "
                "non-overlapping chunks of nsim datapoints, e.g. first nsim/3 art train,"
                "next nsim/3 are dev and next nsim/3 simulation steps are test points."
                "None will use a default nsim from the selected dataset or emulator")
    gp.add("-norm", nargs="+", default=[], choices=["U", "D", "Y", "X"],
           help="List of sequences to max-min normalize")
    gp.add("-data_seed", type=int, default=408,
           help="Random seed used for simulated data")
    #  SYSTEM MODEL
    gp.add("-ssm_type", type=str, choices=["blackbox", "hw", "hammerstein", "blocknlin", "linear"],
           default="hammerstein",
           help='Choice of block structure for system identification model')
    gp.add("-nx_hidden", type=int, default=32,
           help="Number of hidden states per output")
    gp.add("-n_hidden", type=int, default=32,
           help="Number of hidden nodes in policy")
    gp.add("-state_estimator", type=str, choices=["rnn", "mlp", "linear",
                      "residual_mlp", "fully_observable"], default="mlp",
           help='Choice of model architecture for state estimator.')
    gp.add("-estimator_input_window", type=int, default=8,
           help="Number of previous time steps measurements to include in state estimator input")
    gp.add("-nonlinear_map", type=str, default="mlp", choices=["mlp", "rnn", "pytorch_rnn", "linear", "residual_mlp"],
           help='Choice of architecture for component blocks in state space model.')
    gp.add("-linear_map", type=str, choices=["linear", "softSVD", "pf"], default="linear",
           help='Choice of map from SLiM package')
    gp.add("-sigma_min", type=float, default=0.1,
           help='Minimum singular value (for maps with singular value constraints)')
    gp.add("-sigma_max", type=float, default=1.0,
           help='Maximum singular value (for maps with singular value constraints)')
    gp.add("-bias", action="store_true",
           help="Whether to use bias in the neural network block component models.")
    gp.add("-activation", choices=activations.keys(), default="gelu",
           help="Activation function for component block neural networks")
    gp.add("-noise", action="store_true",
           help='Whether to add noise to control actions during training.')
    gp.add("-loss_clip", action="store_true",
           help='Clip loss terms to avoid terms taking over at beginning of training')
    # POLICY
    gp.add("-policy", type=str, choices=["mlp", "linear"], default="mlp",
           help='Choice of architecture for modeling control policy.')
    gp.add("-policy_features", nargs="+", default=['Y_ctrl_p', 'Rf', 'Y_maxf', 'Y_minf'],
           help="Policy features")  # reference tracking option
    gp.add("-n_layers", type=int, default=2,
           help="Number of hidden layers of single time-step state transition")
    gp.add("-perturbation", choices=["white_noise_sine_wave", "white_noise"], default="white_noise",
           help='System perturbation method.')
    #  LOSS
    gp.add("-Q_sub", type=float, default=0.0,
           help="Linear maps regularization weight.")
    gp.add("-Q_r", type=float, default=1.0,
           help="Reference tracking penalty weight")
    gp.add("-Q_du", type=float, default=0.0,
           help="control action difference penalty weight")
    gp.add("-Q_con_u", type=float, default=0.0,
           help="Input constraints penalty weight.")
    gp.add("-Q_con_y", type=float, default=0.0,
           help="Output constraints penalty weight.")
    gp.add("-con_tighten", action="store_true",
           help='Tighten constraints')
    gp.add("-tighten", type=float, default=0.0,
           help="control action difference penalty weight")
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
    # SELECT LEARNABLE COMPONENTS
    gp.add("-freeze", nargs="+", default=[""],
           help="sets requires grad to False")
    gp.add("-unfreeze", default=["components.1"],
           help="sets requires grad to True")
    # gp.add("-unfreeze", default=["components.2"],
    #        help="sets requires grad to True")
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
    gp.add("-gpu", type=int, help="GPU to use")
    return parser


def get_sequence_dataloaders(
    data, nsteps, moving_horizon=False, norm_type=None, split_ratio=None, num_workers=0,
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

    if norm_type is not None:
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


def get_policy_components(args, dims, dynamics_model, policy_name="policy"):
    torch.manual_seed(args.seed)

    activation = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "blu": BLU,
        "softexp": SoftExponential,
    }[args.activation]

    linmap = slim.maps[args.linear_map]
    linargs = {"sigma_min": args.sigma_min, "sigma_max": args.sigma_max}
    nh_policy = args.n_hidden

    policy = {
        "linear": policies.LinearPolicy,
        "mlp": policies.MLPPolicy,
        "rnn": policies.RNNPolicy,
    }[args.policy](
        {"x0_estim": (dynamics_model.nx,), **dims},
        nsteps=args.nsteps,
        bias=args.bias,
        linear_map=linmap,
        nonlin=activation,
        hsizes=[nh_policy] * args.n_layers,
        input_keys=args.policy_features,
        linargs=linargs,
        name=policy_name,
    )
    return policy


def get_objective_terms(args, policy):
    if args.noise:
        output_key = "Y_pred_dynamics_noise"
    else:
        output_key = "Y_pred_dynamics"

    reference_loss = Objective(
        [output_key, "Rf"],
        lambda pred, ref: F.mse_loss(pred[:,:,1], ref[:,:,1]),
        weight=args.Q_r,
        name="ref_loss",
    )
    regularization = Objective(
        [f"reg_error_{policy.name}"], lambda reg: reg, weight=args.Q_sub, name="reg_loss",
    )
    control_smoothing = Objective(
        [f"U_pred_{policy.name}"],
        lambda x: F.mse_loss(x[1:], x[:-1]),
        weight=args.Q_du,
        name="control_smoothing",
    )
    observation_lower_bound_penalty = Objective(
        [output_key, "Y_minf"],
        lambda x, xmin: torch.mean(F.relu(-x + xmin)),
        weight=args.Q_con_y,
        name="observation_lower_bound",
    )
    observation_upper_bound_penalty = Objective(
        [output_key, "Y_maxf"],
        lambda x, xmax: torch.mean(F.relu(x - xmax)),
        weight=args.Q_con_y,
        name="observation_upper_bound",
    )
    inputs_lower_bound_penalty = Objective(
        [f"U_pred_{policy.name}", "U_minf"],
        lambda x, xmin: torch.mean(F.relu(-x + xmin)),
        weight=args.Q_con_u,
        name="input_lower_bound",
    )
    inputs_upper_bound_penalty = Objective(
        [f"U_pred_{policy.name}", "U_maxf"],
        lambda x, xmax: torch.mean(F.relu(x - xmax)),
        weight=args.Q_con_u,
        name="input_upper_bound",
    )

    # Constraints tightening
    if args.con_tighten:
        observation_lower_bound_penalty = Objective(
            [output_key, "Y_minf"],
            lambda x, xmin: torch.mean(F.relu(-x + xmin + args.tighten)),
            weight=args.Q_con_y,
            name="observation_lower_bound",
        )
        observation_upper_bound_penalty = Objective(
            [output_key, "Y_maxf"],
            lambda x, xmax: torch.mean(F.relu(x - xmax + args.tighten)),
            weight=args.Q_con_y,
            name="observation_upper_bound",
        )
        inputs_lower_bound_penalty = Objective(
            [f"U_pred_{policy.name}", "U_minf"],
            lambda x, xmin: torch.mean(F.relu(-x + xmin + args.tighten)),
            weight=args.Q_con_u,
            name="input_lower_bound",
        )
        inputs_upper_bound_penalty = Objective(
            [f"U_pred_{policy.name}", "U_maxf"],
            lambda x, xmax: torch.mean(F.relu(x - xmax + args.tighten)),
            weight=args.Q_con_u,
            name="input_upper_bound",
        )

    # Loss clipping
    if args.loss_clip:
        reference_loss = Objective(
            [output_key, "Rf", "Y_minf", "Y_maxf"],
            lambda pred, ref, xmin, xmax: F.mse_loss(
                pred * torch.gt(ref, xmin).int() * torch.lt(ref, xmax).int(),
                ref * torch.gt(ref, xmin).int() * torch.lt(ref, xmax).int(),
            ),
            weight=args.Q_r,
            name="ref_loss",
        )

    objectives = [regularization, reference_loss, control_smoothing]
    constraints = [
        observation_lower_bound_penalty,
        observation_upper_bound_penalty,
        inputs_lower_bound_penalty,
        inputs_upper_bound_penalty,
    ]

    return objectives, constraints


if __name__ == "__main__":
    """
    # # #  Arguments, logger
    """
    # for available systems and datasets in PSL library check: psl.systems.keys() and psl.datasets.keys()
    system = "TwoTank"  # keyword of selected system to control
    # path with saved model parameters obtained from system identification
    path = f'./trained_models/{system}_best_model.pth'
    parser = arg.ArgParser(parents=[arg_control_problem(system=system, path=path)])
    args, grps = parser.parse_arg_groups()
    log_constructor = MLFlowLogger if args.logger == 'mlflow' else BasicLogger
    logger = log_constructor(args=args, savedir=args.savedir,
                             verbosity=args.verbosity, stdout=args.metrics)
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"

    """
    # # #  Load trained dynamics model
    """
    # TODO: fix issue that args.nsteps must be larger than estimator.window_size
    sysid_model = torch.load(args.model_file, pickle_module=dill,
                             map_location=torch.device(device))
    dynamics_model = sysid_model.components[1]
    estimator = sysid_model.components[0]

    """
    # # #  Problem Dimensions
    """
    # problem dimensions
    nx = dynamics_model.nx
    ny = dynamics_model.ny
    nu = dynamics_model.nu
    # number of datapoints
    nsim = 10000
    # # constraints bounds
    umin = 0
    umax = 5
    xmin = 0
    xmax = 5

    """
    # # #  Dataset
    """
    # sample raw dataset
    data = {
        "Y_max": xmax*np.ones([nsim, ny]),
        "Y_min": xmin*np.ones([nsim, ny]),
        "U_max": umax*np.ones([nsim, nu]),
        "U_min": umin*np.ones([nsim, nu]),
        "R": 0.9 * np.ones([nsim, nu]),
        # "R": psl.Periodic(nx=ny, nsim=nsim, numPeriods=60, xmax=0.9, xmin=0.8)[:nsim, :],
        "Y_ctrl_": 0.5 * np.random.randn(nsim, ny),
        # "Y_ctrl_": np.random.uniform(low=0.0, high=1.5, size=(nsim, ny)),
        "U": np.random.randn(nsim, nu),
    }
    # note: sampling of the past trajectories "Y_ctrl_" has a significant effect on learned control performance

    # get dataloaders
    nstep_data, loop_data, dims = get_sequence_dataloaders(data, args.nsteps)
    train_data, dev_data, test_data = nstep_data
    train_loop, dev_loop, test_loop = loop_data

    """
    # # #  Component Models
    """
    # update model dimensions and input output keys
    dynamics_model._input_keys[2] = ('U_pred_policy', 'Uf')     # this key matching links policy output with control inputs to the model
    dynamics_model._input_keys[0] = ('Rf', 'Yf')        # this key is needed to infer the prediction horizon from the dataset
    estimator._input_keys[0] = ('Y_ctrl_p', 'Yp')
    estimator.data_dims = dims
    estimator.nsteps = args.nsteps
    # define policy
    policy = get_policy_components(args, dims, dynamics_model, policy_name="policy")

    # TODO: test signal generators and add docstrings
    # # get feedback signal sampler
    # signal_generator = WhiteNoisePeriodicGenerator(
    #     args.nsteps,
    #     dynamics_model.fy.out_features,
    #     xmax=(0.8, 0.7),
    #     xmin=0.2,
    #     min_period=1,
    #     max_period=20,
    #     name="Y_ctrl_",
    # )
    # # add output noise for robustness
    # noise_generator = NoiseGenerator(
    #     ratio=0.05, keys=["Y_pred_dynamics"], name="_noise"
    # )

    """
    # # #  Differentiable Predictive Control Problem Definition
    """
    # get objectives and constraints
    objectives, constraints = get_objective_terms(args, policy)
    # define component models
    components = [estimator, policy, dynamics_model]
    # components = [signal_generator, estimator, policy, dynamics_model]
    # define constrained optimal control problem
    model = Problem(
        objectives,
        constraints,
        components,
    )
    model = model.to(device)

    # train only policy component
    freeze_weight(model, module_names=args.freeze)
    unfreeze_weight(model, module_names=args.unfreeze)

    """
    # # # Training
    """
    # select optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # TODO: fix generic callbacks for closed loop control
    # define callback
    # plot_keys = ["Y_pred_dynamics", "U_pred_policy"]  # variables to be plotted
    # visualizer = VisualizerClosedLoop(
    #     policy,
    #     plot_keys,
    #     args.verbosity,
    #     savedir=args.savedir,
    # )
    simulator = ClosedLoopSimulator(sim_data=train_loop, policy=policy,
                                    system_model=dynamics_model, estimator=estimator,
                                    emulator=psl.systems[system](),
                                    emulator_output_keys=["Y_pred_dynamics", "X_pred_dynamics"],
                                    emulator_input_keys=["U_pred_policy"])
    # sim_out_model = simulator.simulate_model(nsim=200)
    # sim_out_emul = simulator.simulate_emulator(nsim=200)

    # define trainer
    trainer = Trainer(
        model,
        train_data,
        dev_data,
        test_data,
        optimizer,
        logger=logger,
        # callback=ControlCallback(simulator, visualizer),
        eval_metric="nstep_dev_loss",
        epochs=args.epochs,
        patience=args.patience,
        warmup=args.warmup,
    )

    # Train control policy
    best_model = trainer.train()
    best_outputs = trainer.test(best_model)
    logger.clean_up()


    # TODO: ad hoc plots to verify simulator
    sim_out_model = simulator.simulate_model(nsim=200)
    Y = sim_out_model['Y_pred_dynamics'][:,0,:].detach().numpy()
    U = sim_out_model['U_pred_policy'][:,0,:].detach().numpy()
    R = sim_out_model['Rf'][::args.nsteps,0,:].detach().numpy()
    pltCL(Y=Y, U=U, R=R)
    sim_out_emul = simulator.simulate_emulator(nsim=200)
    Y = sim_out_emul['Y_pred_dynamics'][:,0,:].detach().numpy()
    U = sim_out_emul['U_pred_policy'][:,0,:].detach().numpy()
    R = sim_out_emul['Rf'][::args.nsteps,0,:].detach().numpy()
    pltCL(Y=Y, U=U, R=R)