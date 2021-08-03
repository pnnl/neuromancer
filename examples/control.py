import dill
from neuromancer.signals import WhiteNoisePeriodicGenerator, NoiseGenerator
from neuromancer.simulators import ClosedLoopSimulator
from neuromancer.trainer import Trainer, freeze_weight, unfreeze_weight
from neuromancer.visuals import VisualizerClosedLoop
from neuromancer.dataset import read_file, normalize_data, split_sequence_data, SequenceDataset
from neuromancer.loggers import BasicLogger, MLFlowLogger
from neuromancer.callbacks import SysIDCallback

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

import psl
import slim

from neuromancer.problem import Problem, Objective
from neuromancer.activations import BLU, SoftExponential
from neuromancer import policies
from neuromancer import arg
from torch.utils.data import DataLoader


def add_reference_features(args, data, dynamics_model):
    """
    """
    ny = dynamics_model.fy.out_features
    if ny != data["Y"].shape[1]:
        data["Y"] = data["Y"][:, :1]

    nsim = data["Y"].shape[0]
    nu = data["U"].shape[1]
    ny = len(args.controlled_outputs)
    data = {
        **data,
        # "Y_max": psl.Periodic(nx=ny, nsim=nsim, numPeriods=30, xmax=1.0, xmin=0.9)[:nsim, :],
        # "Y_min": psl.Periodic(nx=ny, nsim=nsim, numPeriods=24, xmax=0.8, xmin=0.7)[:nsim, :],
        "Y_max": np.ones([nsim, ny]),
        "Y_min": 0.7 * np.ones([nsim, ny]),
        "U_max": np.ones([nsim, nu]),
        "U_min": np.zeros([nsim, nu]),
        "R": psl.Periodic(nx=ny, nsim=nsim, numPeriods=20, xmax=0.9, xmin=0.8)[:nsim, :]
        # 'Y_ctrl_': psl.WhiteNoise(nx=ny, nsim=nsim, xmax=[1.0] * ny, xmin=[0.0] * ny)
    }
    # indices of controlled states, e.g. [0, 1, 3] out of 5 outputs
    # data.ctrl_outputs = args.controlled_outputs
    return data


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


def update_system_id_inputs(args, dataset, estimator, dynamics_model):
    dynamics_model.input_keys[dynamics_model.input_keys.index('Uf')] = 'U_pred_policy'
    dynamics_model.fe = None
    dynamics_model.fyu = None

    estimator.input_keys[0] = 'Y_ctrl_p'
    estimator.data_dims = dataset.dims
    estimator.data_dims['Y_ctrl_p'] = dataset.dims['Yp']
    estimator.nsteps = args.nsteps

    return estimator, dynamics_model


def get_policy_components(args, dataset, dynamics_model, policy_name="policy"):
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
        {"x0_estim": (dynamics_model.nx,), **dataset.dims},
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
        lambda pred, ref: F.mse_loss(pred[:, :, args.controlled_outputs], ref),
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
        lambda x, xmin: torch.mean(F.relu(-x[:, :, args.controlled_outputs] + xmin)),
        weight=args.Q_con_y,
        name="observation_lower_bound",
    )
    observation_upper_bound_penalty = Objective(
        [output_key, "Y_maxf"],
        lambda x, xmax: torch.mean(F.relu(x[:, :, args.controlled_outputs] - xmax)),
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
            lambda x, xmin: torch.mean(F.relu(-x[:, :, args.controlled_outputs] + xmin + args.tighten)),
            weight=args.Q_con_y,
            name="observation_lower_bound",
        )
        observation_upper_bound_penalty = Objective(
            [output_key, "Y_maxf"],
            lambda x, xmax: torch.mean(F.relu(x[:, :, args.controlled_outputs] - xmax + args.tighten)),
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
                pred[:, :, args.controlled_outputs] * torch.gt(ref, xmin).int() * torch.lt(ref, xmax).int(),
                ref * torch.gt(ref, xmin).int() * torch.lt(ref, xmax).int(),
            ),
            weight=args.Q_r,
            name="ref_loss",
        )

    objectives = [regularization, reference_loss]
    constraints = [
        observation_lower_bound_penalty,
        observation_upper_bound_penalty,
        inputs_lower_bound_penalty,
        inputs_upper_bound_penalty,
    ]

    return objectives, constraints


if __name__ == "__main__":
    # for available systems in PSL library check: psl.systems.keys()
    system = 'CSTR'         # keyword of selected system
    parser = arg.ArgParser(parents=[arg.log(), arg.log(prefix='sysid_'),
                                    arg.opt(), arg.data(system=system),
                                    arg.loss(), arg.lin(), arg.policy(),
                                    arg.ctrl_loss(), arg.freeze()])
    path = './test/best_model.pth'
    parser.add('-model_file', type=str, default=path,
               help='Path to pytorch pickled model.')

    args = parser.parse_args()

    log_constructor = MLFlowLogger if args.logger == 'mlflow' else BasicLogger
    metrics = ["nstep_dev_loss", "loop_dev_loss", "best_loop_dev_loss",
               "nstep_dev_ref_loss", "loop_dev_ref_loss"]
    logger = log_constructor(args=args, savedir=args.savedir, verbosity=args.verbosity, stdout=metrics)

    print({k: str(getattr(args, k)) for k in vars(args) if getattr(args, k)})
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"

    sysid_model = torch.load(args.model_file, pickle_module=dill, map_location=torch.device(device))
    dynamics_model = sysid_model.components[1]
    estimator = sysid_model.components[0]

    if args.dataset in psl.emulators:
        data = psl.emulators[args.dataset](nsim=args.nsim, ninit=0, seed=args.data_seed).simulate()
    elif args.dataset in psl.datasets:
        data = read_file(psl.datasets[args.dataset])
    else:
        data = read_file(args.dataset)

    data = add_reference_features(args, data, dynamics_model)
    nstep_data, loop_data, dims = get_sequence_dataloaders(data, args.nsteps)
    train_data, dev_data, test_data = nstep_data
    train_loop, dev_loop, test_loop = loop_data

    # Control Problem Definition
    estimator, dynamics_model = update_system_id_inputs(
        args, train_data.dataset, estimator, dynamics_model
    )
    policy = get_policy_components(
        args, train_data.dataset, dynamics_model, policy_name="policy"
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

    signal_generator = signal_generator.to(device)
    estimator = estimator.to(device)
    policy = policy.to(device)
    dynamics_model = dynamics_model.to(device)

    objectives, constraints = get_objective_terms(args, policy)
    model = Problem(
        objectives,
        constraints,
        [signal_generator, estimator, policy, dynamics_model],
    )
    model = model.to(device)

    # train only policy component
    freeze_weight(model, module_names=args.freeze)
    unfreeze_weight(model, module_names=args.unfreeze)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    plot_keys = ["Y_pred", "U_pred"]  # variables to be plotted
    visualizer = VisualizerClosedLoop(
        policy,
        plot_keys,
        args.controlled_outputs,
        args.verbosity,
        savedir=args.savedir,
    )

    policy.input_keys[0] = "Yp"  # hack for policy input key compatibility w/ simulator
    simulator = ClosedLoopSimulator(
        model,
        policy,
        dynamics_model,
        train_data.dataset,
        dev_data.dataset,
        test_data.dataset,
        eval_sim=not args.skip_eval_sim,
        device=device,
    )

    trainer = Trainer(
        model,
        train_data,
        dev_data,
        test_data,
        optimizer,
        logger=logger,
        callback=SysIDCallback(simulator, visualizer),
        eval_metric="nstep_dev_loss",
        epochs=args.epochs,
        patience=args.patience,
        warmup=args.warmup,
    )

    # Train control policy
    best_model = trainer.train()
    best_outputs = trainer.test(best_model)

    logger.clean_up()
