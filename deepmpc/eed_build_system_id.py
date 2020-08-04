"""

Script for training block dynamics models for system identification.
Current block structure supported are black_box, hammerstein, hammerstein-weiner,
and block models with non-linear main transition dynamics.

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

More detailed description of options in the parse_args()
"""
# import matplotlib
# matplotlib.use("Agg")
import argparse
import torch
from datasets import EmulatorDataset, FileDataset
import dynamics
import estimators
import linear
import blocks
import logger
from visuals import VisualizerOpen, VisualizerTrajectories
from trainer import Trainer
from problem import Problem, Objective
import torch.nn.functional as F
from simulators import OpenLoopSimulator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default=None,
                        help="Gpu to use")
    # OPTIMIZATION PARAMETERS
    opt_group = parser.add_argument_group('OPTIMIZATION PARAMETERS')
    opt_group.add_argument('-epochs', type=int, default=500)
    opt_group.add_argument('-lr', type=float, default=0.003,
                           help='Step size for gradient descent.')

    #################
    # DATA PARAMETERS
    data_group = parser.add_argument_group('DATA PARAMETERS')
    data_group.add_argument('-nsteps', type=int, default=32,
                            help='Number of steps for open loop during training.')
    data_group.add_argument('-system_data', type=str, choices=['emulator', 'datafile'],
                            default='datafile',
                            help='source type of the dataset')
    data_group.add_argument('-system', default='EED_building',
                            help='select particular dataset with keyword')
    data_group.add_argument('-nsim', type=int, default=None,
                            help='Number of time steps for full dataset. (ntrain + ndev + ntest)'
                                 'train, dev, and test will be split evenly from contiguous, sequential, '
                                 'non-overlapping chunks of nsim datapoints, e.g. first nsim/3 art train,'
                                 'next nsim/3 are dev and next nsim/3 simulation steps are test points.'
                                 'None will use a default nsim from the selected dataset or emulator')
    data_group.add_argument('-norm', choices=['UDY', 'U', 'Y', None], type=str, default='UDY')
    data_group.add_argument('-dataset_name', type=str, choices=['openloop', 'closedloop'],
                            default='openloop',
                            help='name of the dataset')

    ##################
    # MODEL PARAMETERS
    model_group = parser.add_argument_group('MODEL PARAMETERS')
    model_group.add_argument('-ssm_type', type=str, choices=['blackbox', 'hw', 'hammerstein', 'blocknlin'],
                             default='blocknlin')
    model_group.add_argument('-nx_hidden', type=int, default=10, help='Number of hidden states per output')
    model_group.add_argument('-n_layers', type=int, default=2, help='Number of hidden layers of single time-step state transition')
    model_group.add_argument('-state_estimator', type=str,
                             choices=['rnn', 'mlp', 'linear', 'residual_mlp'], default='mlp')
    model_group.add_argument('-linear_map', type=str, choices=list(linear.maps.keys()),
                             default='linear')
    model_group.add_argument('-nonlinear_map', type=str, default='mlp',
                             choices=['mlp', 'rnn', 'pytorch_rnn', 'linear', 'residual_mlp'])
    model_group.add_argument('-bias', action='store_true', help='Whether to use bias in the neural network models.')

    ##################
    # Weight PARAMETERS
    weight_group = parser.add_argument_group('WEIGHT PARAMETERS')
    weight_group.add_argument('-Q_con_x', type=float,  default=1.0, help='Hidden state constraints penalty weight.')
    weight_group.add_argument('-Q_dx', type=float,  default=0.2,
                              help='Penalty weight on hidden state difference in one time step.')
    weight_group.add_argument('-Q_sub', type=float,  default=0.2, help='Linear maps regularization weight.')
    weight_group.add_argument('-Q_y', type=float,  default=1.0, help='Output tracking penalty weight')
    weight_group.add_argument('-Q_e', type=float,  default=1.0, help='State estimator hidden prediction penalty weight')
    weight_group.add_argument('-Q_con_fdu', type=float,  default=0.2, help='Penalty weight on control actions and disturbances.')

    ####################
    # LOGGING PARAMETERS
    log_group = parser.add_argument_group('LOGGING PARAMETERS')
    log_group.add_argument('-savedir', type=str, default='test',
                           help="Where should your trained model and plots be saved (temp)")
    log_group.add_argument('-verbosity', type=int, default=100,
                           help="How many epochs in between status updates")
    log_group.add_argument('-exp', default='test',
                           help='Will group all run under this experiment name.')
    log_group.add_argument('-location', default='mlruns',
                           help='Where to write mlflow experiment tracking stuff')
    log_group.add_argument('-run', default='deepmpc',
                           help='Some name to tell what the experiment run was about.')
    log_group.add_argument('-logger', choices=['mlflow', 'stdout'], default='stdout',
                           help='Logging setup to use')
    return parser.parse_args()


def logging(args):
    if args.logger == 'mlflow':
        Logger = logger.MLFlowLogger(args, args.savedir, args.verbosity)
    else:
        Logger = logger.BasicLogger(savedir=args.savedir, verbosity=args.verbosity)
    device = f'cuda:{args.gpu}' if (args.gpu is not None) else 'cpu'
    return Logger, device


def dataset_load(args, device):
    if args.system_data == 'emulator':
        dataset = EmulatorDataset(system=args.system, nsim=args.nsim,
                                  norm=args.norm, nsteps=args.nsteps, device=device, savedir=args.savedir)
    else:
        dataset = FileDataset(system=args.system, nsim=args.nsim,
                              norm=args.norm, nsteps=args.nsteps, device=device, savedir=args.savedir)
    return dataset


if __name__ == '__main__':
    ###############################
    ########## LOGGING ############
    ###############################
    args = parse_args()
    logger, device = logging(args)

    ###############################
    ########## DATA ###############
    ###############################
    dataset = dataset_load(args, device)

    ##########################################
    ########## PROBLEM COMPONENTS ############
    ##########################################
    nx = dataset.dims['Y']*args.nx_hidden
    nu = dataset.dims['U'] if 'U' in dataset.dims else 0
    nd = dataset.dims['D'] if 'D' in dataset.dims else 0
    ny = dataset.dims['Y']
    linmap = linear.maps[args.linear_map]
    nonlinmap = {'linear': linmap,
                 'mlp': blocks.MLP,
                 'rnn': blocks.RNN,
                 'pytorch_rnn': blocks.PytorchRNN,
                 'residual_mlp': blocks.ResMLP}[args.nonlinear_map]
    # state space model setup
    dynamics_model = {'blackbox': dynamics.blackbox,
                      'blocknlin': dynamics.blocknlin,
                      'hammerstein': dynamics.hammerstein,
                      'hw': dynamics.hw}[args.ssm_type](args.bias, linmap, nonlinmap, nx, nu, nd, ny,
                                                        n_layers=args.n_layers,
                                                        input_keys=['Yf', 'x0', 'Uf', 'Df'],
                                                        name='dynamics')

    # state estimator setup
    estimator = {'linear': estimators.LinearEstimator,
                 'mlp': estimators.MLPEstimator,
                 'rnn': estimators.RNNEstimator,
                 'residual_mlp': estimators.ResMLPEstimator
                 }[args.state_estimator]({**dataset.dims, 'x0': nx},
                   nsteps=args.nsteps,
                   bias=args.bias,
                   Linear=linmap,
                   nonlin=F.gelu,
                   hsizes=[nx]*args.n_layers,
                   input_keys=['Yp'],
                   output_keys=['x0'],
                   linargs=dict(),
                   name='estim')

    components = [estimator, dynamics_model]

    # component variables
    input_keys = list(set.union(*[set(comp.input_keys) for comp in components]))
    output_keys = list(set.union(*[set(comp.output_keys) for comp in components]))
    dataset_keys = list(set(dataset.train_data.keys()))
    plot_keys = ['Yf', 'Y_pred', 'X_pred', 'fU', 'fD']   # variables to be plotted

    ##########################################
    ########## MULTI-OBJECTIVE LOSS ##########
    ##########################################
    estimator_loss = Objective(['X_pred', 'x0'],
                                lambda X_pred, x0: F.mse_loss(X_pred[-1, :-1, :], x0[1:]),  # arrival cost
                                weight=args.Q_e)
    regularization = Objective(['estim_reg_error', 'dynamics_reg_error'], lambda reg1, reg2: reg1 + reg2, weight=args.Q_sub)
    reference_loss = Objective(['Y_pred', 'Yf'], F.mse_loss, weight=args.Q_y)
    state_smoothing = Objective(['X_pred'], lambda x: F.mse_loss(x[1:], x[:-1]), weight=args.Q_dx)
    observation_lower_bound_penalty = Objective(['Y_pred'], lambda x: torch.mean(F.relu(-x + -0.2)), weight=args.Q_con_x)
    observation_upper_bound_penalty = Objective(['Y_pred'], lambda x: torch.mean(F.relu(x - 1.2)), weight=args.Q_con_x)

    objectives = [regularization, reference_loss]
    constraints = [state_smoothing, observation_lower_bound_penalty, observation_upper_bound_penalty]

    if 'fU' in output_keys:
        inputs_max_influence_lb = Objective(['fU'], lambda x: torch.mean(F.relu(-x + 0.05)),
                                              weight=args.Q_con_fdu)
        inputs_max_influence_ub = Objective(['fU'], lambda x: torch.mean(F.relu(x - 0.05)),
                                            weight=args.Q_con_fdu)
        constraints.append(inputs_max_influence_lb)
        constraints.append(inputs_max_influence_ub)
    if 'fD' in output_keys:
        disturbances_max_influence_lb = Objective(['fD'], lambda x: torch.mean(F.relu(-x + 0.05)),
                                            weight=args.Q_con_fdu)
        disturbances_max_influence_ub = Objective(['fD'], lambda x: torch.mean(F.relu(x - 0.05)),
                                            weight=args.Q_con_fdu)
        constraints.append(disturbances_max_influence_lb)
        constraints.append(disturbances_max_influence_ub)

    ##########################################
    ########## OPTIMIZE SOLUTION ############
    ##########################################
    model = Problem(objectives, constraints, components).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    visualizer = VisualizerOpen(dataset, dynamics_model, args.verbosity, args.savedir)
    simulator = OpenLoopSimulator(model=model, dataset=dataset)
    trainer = Trainer(model, dataset, optimizer, logger=logger, visualizer=visualizer,
                      simulator=simulator, epochs=args.epochs)
    best_model = trainer.train()
    trainer.evaluate(best_model)
    logger.clean_up()
