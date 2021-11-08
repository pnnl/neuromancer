"""
Differentiable predictive control (DPC) tutorial

Learning predictive control policies based on learned neural state space model
of unknown nonlinear dynamical system


"""

import numpy as np
import dill
import torch
import psl
import slim

from neuromancer import policies, arg
from neuromancer.activations import activations
from neuromancer.problem import Problem
from torch.utils.data import DataLoader
from neuromancer.simulators import ClosedLoopSimulator
from neuromancer.trainer import Trainer, freeze_weight, unfreeze_weight
from neuromancer.visuals import VisualizerClosedLoop
from neuromancer.dataset import normalize_data, split_sequence_data, SequenceDataset
from neuromancer.loggers import BasicLogger, MLFlowLogger
from neuromancer.callbacks import ControlCallback
from neuromancer.constraint import Variable
from neuromancer.plot import pltCL
from neuromancer import blocks, estimators, dynamics
from neuromancer.dynamics import BlockSSM


def arg_control_problem(prefix='', system='Reno_ROM40'):
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
    gp.add("-data_seed", type=int, default=408,
           help="Random seed used for simulated data")
    gp.add("-nsteps", type=int, default=8,
           help="prediction horizon.")          # tuned values:
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
    gp.add("-downsample", type=int, default=10, help="Number of timesteps to downsample")
    #  OPTIMIZATION
    gp.add("-eval_metric", type=str, default="loop_dev_ref_loss",
            help="Metric for model selection and early stopping.")
    gp.add("-epochs", type=int, default=2000,
           help='Number of training epochs')
    gp.add("-lr", type=float, default=0.001,
           help="Step size for gradient descent.")
    gp.add("-patience", type=int, default=100,
           help="How many epochs to allow for no improvement in eval metric before early stopping.")
    gp.add("-warmup", type=int, default=100,
           help="Number of epochs to wait before enacting early stopping policy.")
    gp.add("-skip_eval_sim", action="store_true",
           help="Whether to run simulator during evaluation phase of training.")
    gp.add("-seed", type=int, default=408, help="Random seed used for weight initialization.")
    gp.add("-gpu", type=int, help="GPU to use")
    return parser


class BuildingSSM(BlockSSM):
    def __init__(self, fx, fy, fu=None, fd=None, fe=None, F=None, G=None, x_ss=20.0, y_ss=273.15,
                 name='block_ssm', input_key_map={}):
        super().__init__(fx, fy, fu=fu, fd=fd, fe=fe,
                         input_key_map=input_key_map, name=name)
        self.F = F
        self.G = G
        self.x_ss = x_ss
        self.y_ss = y_ss

    def forward(self, data):
        """

        :param data: (dict: {str: Tensor})
        :return: output (dict: {str: Tensor})
        """
        nsteps = data[self.input_key_map['Yf']].shape[0]
        X, Y, FD, FU, FE = [], [], [], [], []
        x = data[self.input_key_map['x0']]
        for i in range(nsteps):
            x = self.fx(x)
            fu = self.fu(data[self.input_key_map['Uf']][i])
            x = x + fu
            FU.append(fu)
            fd = self.fd(data[self.input_key_map['Df']][i])
            x = x + fd
            FD.append(fd)
            x = x + self.G
            y = self.fy(x) + self.F
            X.append(x+self.x_ss)
            Y.append(y-self.y_ss)

        output = {name: torch.stack(tensor_list) for tensor_list, name
                  in zip([X, Y, FU, FD, FE], ['X_pred', 'Y_pred', 'fU', 'fD', 'fE'])
                  if tensor_list}
        output['reg_error'] = self.reg_error()
        return output



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


if __name__ == "__main__":
    """
    # # #  Arguments, logger
    """
    # for available systems and datasets in PSL library check: psl.systems.keys() and psl.datasets.keys()
    system = "Reno_ROM40"  # keyword of selected system to control
    parser = arg.ArgParser(parents=[arg_control_problem(system=system)])
    args, grps = parser.parse_arg_groups()
    log_constructor = MLFlowLogger if args.logger == 'mlflow' else BasicLogger
    logger = log_constructor(args=args, savedir=args.savedir,
                             verbosity=args.verbosity, stdout=args.metrics)
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
    # TODO: turn on seed
    # torch.manual_seed(args.seed)

    """
    # # #  Load system dynamics model from psl
    """
    # psl system model
    system_model = psl.systems[system](system=system)
    # problem dimensions
    nx = system_model.nx
    ny = system_model.ny
    nu = system_model.nu
    nd = system_model.nd
    # constraints bounds
    umin = 0
    umax = 10000
    xmin = 0
    xmax = 30
    # prediction horizon
    nsteps = args.nsteps
    # number of sampled datapoints
    nsimulate = 100000

    # TODO: debug model scales

    """
    # # #  Dataset
    """
    #  load and split the dataset
    data_sim = psl.emulators[args.dataset](nsim=nsimulate, ninit=0, seed=args.data_seed).simulate()
    # subsampling
    if args.downsample != 0:
        data_sim['Y'] = data_sim['Y'][::args.downsample, :]
        data_sim['U'] = data_sim['U'][::args.downsample, :]
        data_sim['D'] = data_sim['D'][::args.downsample, :]
        data_sim['X'] = data_sim['X'][::args.downsample, :]

    # control samples
    nsim = data_sim['Y'].shape[0]

    # sample raw dataset
    data = {
        "Y_max": xmax*np.ones([nsim, ny]),
        "Y_min": xmin*np.ones([nsim, ny]),
        "U_max": umax*np.ones([nsim, nu]),
        "U_min": umin*np.ones([nsim, nu]),
        # "R": psl.Periodic(nx=ny, nsim=nsim, numPeriods=60, xmax=0.6, xmin=0.4)[:nsim, :],
        # "Y": np.random.uniform(low=-1.5, high=1.5, size=(nsim, ny)),
        "R": psl.Periodic(nx=ny, nsim=nsim, numPeriods=60, xmax=24, xmin=18)[:nsim, :],
        "Y": np.random.uniform(low=16.0, high=26.0, size=(nsim, ny)),
        "X": np.random.uniform(low=-4.0, high=4.0, size=(nsim, nx)),
        "U": data_sim['U'],
        "D": data_sim['D'],
    }
    # note: sampling of the past trajectories "Y_ctrl_" has a significant effect on learned control performance

    # get torch dataloaders
    # norm_type = "zero-one"
    norm_type = None
    nstep_data, loop_data, dims = get_sequence_dataloaders(data, nsteps, norm_type=norm_type)
    train_data, dev_data, test_data = nstep_data
    train_loop, dev_loop, test_loop = loop_data

    """
    # # #  Component Models
    """
    # Fully observable estimator as identity map: x0 = Xp[-1]
    # Xp = [x_-N, ..., y_0]
    estimator = estimators.FullyObservable({**dims, "x0": (nx,)},
                                           nsteps=args.nsteps,  # future window Nf
                                           window_size=1,  # past window Np <= Nf
                                           input_keys=["Xp"],
                                           name='estimator')
    # A, B, C, E, F, G linear maps
    fu = slim.maps['linear'](nu, nx)
    fx = slim.maps['linear'](nx, nx)
    fy = slim.maps['linear'](nx, ny)
    fd = slim.maps['linear'](nd, nx)
    F = torch.tensor(system_model.F, dtype=torch.float32)[0]
    G = torch.tensor(system_model.G, dtype=torch.float32)[0]
    x_ss = torch.tensor(system_model.x_ss, dtype=torch.float32)[0][0]
    y_ss = torch.tensor(system_model.y_ss, dtype=torch.float32)[0][0]
    # LTI SSM
    # x_k+1 = Ax_k + Bu_k
    # y_k+1 = Cx_k+1
    dynamics_model = BuildingSSM(fx, fy, fu=fu, fd=fd, name='dynamics', F=F, G=G,
                                 x_ss=x_ss, y_ss=y_ss,
                                       input_key_map={'x0': f'x0_{estimator.name}',
                                                      'Uf': 'U_pred_policy', 'Yf': 'Rf'})

    # model matrices values
    A = torch.tensor(system_model.A, dtype=torch.float32)
    B = torch.tensor(system_model.B, dtype=torch.float32)
    C = torch.tensor(system_model.C, dtype=torch.float32)
    E = torch.tensor(system_model.E, dtype=torch.float32)
    dynamics_model.fx.linear.weight = torch.nn.Parameter(A)
    dynamics_model.fu.linear.weight = torch.nn.Parameter(B)
    dynamics_model.fy.linear.weight = torch.nn.Parameter(C)
    dynamics_model.fd.linear.weight = torch.nn.Parameter(E)
    # fix model parameters
    dynamics_model.requires_grad_(False)

    # construct policy
    activation = activations['gelu']
    linmap = slim.maps['linear']       # type of weights
    n_layers = 4        # number of layers
    nh_policy = 32      # number of hidden states
    policy = policies.MLPPolicy(
        dims,
        nsteps=nsteps,
        bias=True,
        linear_map=linmap,
        nonlin=activation,
        hsizes=[nh_policy] * n_layers,
        input_keys=['Xp', 'Rf', 'Df'],
        name="policy",
    )

    """
    # # #  Differentiable Predictive Control Problem Definition
    """
    # neuromancer variable declaration
    # variables refer to 3D tensors: [nsteps, nbatches, var_dim]
    # nsteps = prediction horizon, nbatches = number of batches, var_dim = variable dimension
    y = Variable(f"Y_pred_{dynamics_model.name}")       # system outputs
    r = Variable("Rf")                                  # references
    u = Variable(f"U_pred_{policy.name}")
    # constraints bounds variables
    umin = Variable("U_minf")
    umax = Variable("U_maxf")
    ymin = Variable("Y_minf")
    ymax = Variable("Y_maxf")

    # weight factors of loss function terms and constraints
    Q_r = 1.0
    Q_du = 0.1
    Q_con_u = 1.0
    Q_con_y = 1.0
    # define loss function terms and constraints via neuromancer constraints syntax
    reference_loss = Q_r*((r == y)^2)                       # track reference
    control_smoothing = Q_du*((u[1:] == u[:-1])^2)          # delta u penalty
    output_lower_bound_penalty = Q_con_y*(y > ymin)
    output_upper_bound_penalty = Q_con_y*(y < ymax)
    input_lower_bound_penalty = Q_con_u*(u > umin)
    input_upper_bound_penalty = Q_con_u*(u < umax)

    # list of objectives and constraints
    objectives = [reference_loss, control_smoothing]
    constraints = [
        output_lower_bound_penalty,
        output_upper_bound_penalty,
        input_lower_bound_penalty,
        input_upper_bound_penalty,
    ]
    # define component models
    components = [estimator, policy, dynamics_model]

    # define constrained optimal control problem
    model = Problem(
        objectives,
        constraints,
        components,
    )
    model = model.to(device)

    """
    # # # Training
    """
    # train only policy component
    freeze_weight(model, module_names=[""])                  # freeze all components
    unfreeze_weight(model, module_names=["components.1"])    # unfreeze policy component

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    visualizer = VisualizerClosedLoop(
        u_key=u.name,
        y_key=y.name,
        r_key=r.name,
        policy=policy,
        savedir=args.savedir,
    )
    simulator = ClosedLoopSimulator(
        sim_data=train_loop, policy=policy,
        system_model=dynamics_model, estimator=estimator,
        emulator=psl.systems[system](),
        emulator_output_keys=[y.name, "X_pred_dynamics"],
        emulator_input_keys=[u.name],
        nsim=200)

    trainer = Trainer(
        model,
        train_data,
        dev_data,
        test_data,
        optimizer,
        logger=logger,
        callback=ControlCallback(simulator, visualizer),
        eval_metric="nstep_dev_loss",
        epochs=args.epochs,
        patience=args.patience,
        warmup=args.warmup,
    )
    # Train control policy
    best_model = trainer.train()
    best_outputs = trainer.test(best_model)
    logger.clean_up()

    # TODO: test forward pass building model


    # TODO: debug
    # # simulate and plot outside of the callback
    # sim_out_model = simulator.simulate(nsim=400)
    # Y = sim_out_model['Y_pred_dynamics'].detach().numpy()
    # U = sim_out_model['U_pred_policy'].detach().numpy()
    # R = sim_out_model['Rf'].detach().numpy()
    # pltCL(Y=Y, U=U, R=R)
