"""
Script for training block dynamics models for system identification.
Current block structure supported are black_box, hammerstein, hammerstein-weiner, and general block non-linear

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
# python base imports
import argparse
import dill

# machine learning data science imports
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

# code ecosystem imports
import slim

# local imports
from neuromancer.datasets import EmulatorDataset, FileDataset, systems
import neuromancer.loggers as loggers
from neuromancer.visuals import VisualizerClosedLoop
from neuromancer.activations import BLU, SoftExponential
from neuromancer.simulators import ClosedLoopSimulator
import neuromancer.policies as policies
from neuromancer.problem import Objective, Problem
from neuromancer.trainer import Trainer
import psl


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=int, default=None,
                        help="Gpu to use")
    # OPTIMIZATION PARAMETERS
    opt_group = parser.add_argument_group('OPTIMIZATION PARAMETERS')
    opt_group.add_argument('-epochs', type=int, default=500)
    opt_group.add_argument('-lr', type=float, default=0.001,
                           help='Step size for gradient descent.')

    #################
    # DATA PARAMETERS
    data_group = parser.add_argument_group('DATA PARAMETERS')
    data_group.add_argument('-nsteps', type=int, default=32,
                            help='Number of steps for open loop during training.')
    data_group.add_argument('-system', type=str, default='flexy_air', choices=['flexy_air'],
                            help='select particular dataset with keyword')
    data_group.add_argument('-nsim', type=int, default=8640,
                            help='Number of time steps for full dataset. (ntrain + ndev + ntest)'
                                 'train, dev, and test will be split evenly from contiguous, sequential, '
                                 'non-overlapping chunks of nsim datapoints, e.g. first nsim/3 art train,'
                                 'next nsim/3 are dev and next nsim/3 simulation steps are test points.'
                                 'None will use a default nsim from the selected dataset or emulator')
    data_group.add_argument('-norm', nargs='+', default=['U', 'D', 'Y'], choices=['U', 'D', 'Y', 'X'],
                            help='List of sequences to max-min normalize')
    data_group.add_argument('-model_file', type=str, default='../datasets/Flexy_air/best_model.pth')

    ##################
    # POLICY PARAMETERS
    policy_group = parser.add_argument_group('POLICY PARAMETERS')
    policy_group.add_argument('-policy', type=str,
                              choices=['rnn', 'mlp', 'linear'], default='mlp')
    policy_group.add_argument('-n_hidden', type=int, default=10, help='Number of hidden states')
    policy_group.add_argument('-n_layers', type=int, default=4,
                              help='Number of hidden layers of single time-step state transition')
    policy_group.add_argument('-linear_map', type=str, choices=list(slim.maps.keys()),
                              default='linear')
    policy_group.add_argument('-bias', action='store_true', help='Whether to use bias in the neural network models.')
    policy_group.add_argument('-policy_features', nargs='+', default=['x0_estim', 'Rf', 'Df'], help='Policy features')
    policy_group.add_argument('-activation', choices=['relu', 'gelu', 'blu', 'softexp'], default='gelu',
                              help='Activation function for neural networks')

    ##################
    # LAYERS
    layers_group = parser.add_argument_group('LATERS PARAMETERS')
    layers_group.add_argument('-freeze', nargs='+', default=[''], help='sets requires grad to False')
    layers_group.add_argument('-unfreeze', default=['components.1'],
                              help='sets requires grat to True')

    ##################
    # WEIGHT PARAMETERS
    weight_group = parser.add_argument_group('WEIGHT PARAMETERS')
    weight_group.add_argument('-Q_con_y', type=float, default=10.0, help='Output constraints penalty weight.')
    weight_group.add_argument('-Q_con_u', type=float, default=10.0, help='Input constraints penalty weight.')
    weight_group.add_argument('-Q_sub', type=float, default=0.2, help='Linear maps regularization weight.')
    weight_group.add_argument('-Q_r', type=float, default=1.0, help='Reference tracking penalty weight')
    weight_group.add_argument('-Q_du', type=float, default=1.0, help='Reference tracking penalty weight')

    ####################
    # LOGGING PARAMETERS
    log_group = parser.add_argument_group('LOGGING PARAMETERS')
    log_group.add_argument('-savedir', type=str, default='test_control',
                           help="Where should your trained model and plots be saved (temp)")
    log_group.add_argument('-verbosity', type=int, default=1,
                           help="How many epochs in between status updates")
    log_group.add_argument('-exp', type=str, default='test',
                           help='Will group all run under this experiment name.')
    log_group.add_argument('-location', type=str, default='mlruns',
                           help='Where to write mlflow experiment tracking stuff')
    log_group.add_argument('-run', type=str, default='neuromancer',
                           help='Some name to tell what the experiment run was about.')
    log_group.add_argument('-logger', type=str, choices=['mlflow', 'stdout'], default='stdout',
                           help='Logging setup to use')
    return parser


def logging(args):
    if args.logger == 'mlflow':
        Logger = loggers.MLFlowLogger(args=args, savedir=args.savedir, verbosity=args.verbosity,
                                      stdout=('nstep_dev_loss', 'loop_dev_loss', 'best_loop_dev_loss',
                                              'nstep_dev_ref_loss', 'loop_dev_ref_loss'))
    else:
        Logger = loggers.BasicLogger(args=args, savedir=args.savedir, verbosity=args.verbosity,
                                     stdout=('nstep_dev_loss', 'loop_dev_loss', 'best_loop_dev_loss',
                                             'nstep_dev_ref_loss', 'loop_dev_ref_loss'))
    device = f'cuda:{args.gpu}' if (args.gpu is not None) else 'cpu'
    return Logger, device


def dataset_load(args, device):
    if systems[args.system] == 'emulator':
        dataset = EmulatorDataset(system=args.system, nsim=args.nsim,
                                  norm=args.norm, nsteps=args.nsteps, device=device, savedir=args.savedir,
                                  name='closedloop')
    else:
        dataset = FileDataset(system=args.system, nsim=args.nsim,
                              norm=args.norm, nsteps=args.nsteps, device=device, savedir=args.savedir,
                              name='closedloop')
    nsim, ny = dataset.data['Y'].shape
    nu = dataset.data['U'].shape[1]
    new_sequences = {'Y_max': 0.8 * np.ones([nsim, ny]), 'Y_min': 0.2 * np.ones([nsim, ny]),
                     'U_max': np.ones([nsim, nu]), 'U_min': np.zeros([nsim, nu]),
                     'R': psl.Steps(nx=1, nsim=nsim, randsteps=30, xmax=1, xmin=0),
                     # 'R': psl.Periodic(nx=1, nsim=nsim, numPeriods=12, xmax=1, xmin=0),
                     'Y_ctrl_': psl.WhiteNoise(nx=ny, nsim=nsim, xmax=[1.0] * ny, xmin=[0.0] * ny)}
    dataset.add_data(new_sequences)
    return dataset


def freeze_weight(model, module_names=['']):
    """
    ['parent->child->child']
    :param component:
    :param module_names:
    :return:
    """
    modules = dict(model.named_modules())
    for name in module_names:
        freeze_path = name.split('->')
        if len(freeze_path) == 1:
            modules[name].requires_grad_(False)
        else:
            parent = modules[freeze_path[0]]
            freeze_weight(parent, ['->'.join(freeze_path[1:])])

def unfreeze_weight(model, module_names=['']):
    """
    ['parent->child->child']
    :param component:
    :param module_names:
    :return:
    """
    modules = dict(model.named_modules())
    for name in module_names:
        freeze_path = name.split('->')
        if len(freeze_path) == 1:
            modules[name].requires_grad_(True)
        else:
            parent = modules[freeze_path[0]]
            freeze_weight(parent, ['->'.join(freeze_path[1:])])



if __name__ == '__main__':
    ###############################
    ########## LOGGING ############
    ###############################
    args = parse().parse_args()
    logger, device = logging(args)

    ###############################
    ########## DATA ###############
    ###############################
    dataset = dataset_load(args, device)
    print(dataset.dims)

    ##########################################
    ########## PROBLEM COMPONENTS ############
    ##########################################
    # Learned dynamics system ID model setup
    best_model = torch.load(args.model_file, pickle_module=dill)
    for k in range(len(best_model.components)):
        if best_model.components[k].name == 'dynamics':
            dynamics_model = best_model.components[k]
            dynamics_model.input_keys[2] = 'U_pred_policy'
            dynamics_model.fe = None  # TODO: ALARM! lazy fix - retrain the model with new dynamics module and get rid of this line
        if best_model.components[k].name == 'estim':
            estimator = best_model.components[k]
            estimator.input_keys[0] = 'Y_ctrl_p'
            estimator.data_dims = dataset.dims

    # control policy setup
    activation = {'gelu': nn.GELU,
                  'relu': nn.ReLU,
                  'blu': BLU,
                  'softexp': SoftExponential}[args.activation]
    linmap = slim.maps[args.linear_map]
    nh_policy = args.n_hidden
    policy = {'linear': policies.LinearPolicy,
              'mlp': policies.MLPPolicy,
              'rnn': policies.RNNPolicy
              }[args.policy]({'x0_estim': (dynamics_model.nx,), **dataset.dims},
                             nsteps=args.nsteps,
                             bias=args.bias,
                             Linear=linmap,
                             nonlin=activation,
                             hsizes=[nh_policy] * args.n_layers,
                             input_keys=args.policy_features,
                             linargs=dict(),
                             name='policy')

    components = [estimator, policy, dynamics_model]

    ##########################################
    ########## MULTI-OBJECTIVE LOSS ##########
    ##########################################
    regularization = Objective(['reg_error_policy'], lambda reg: reg,
                               weight=args.Q_sub)
    reference_loss = Objective(['Y_pred_dynamics', 'Rf'], lambda pred, ref: F.mse_loss(pred[:, :, :1], ref),
                               weight=args.Q_r, name='ref_loss')
    control_smoothing = Objective(['U_pred_policy'], lambda x: F.mse_loss(x[1:], x[:-1]),
                                  weight=args.Q_du, name='control_smoothing')
    observation_lower_bound_penalty = Objective(['Y_pred_dynamics', 'Y_minf'],
                                                lambda x, xmin: torch.mean(F.relu(-x + xmin)),
                                                weight=args.Q_con_y, name='observation_lower_bound')
    observation_upper_bound_penalty = Objective(['Y_pred_dynamics', 'Y_maxf'],
                                                lambda x, xmax: torch.mean(F.relu(x - xmax)),
                                                weight=args.Q_con_y, name='observation_upper_bound')
    inputs_lower_bound_penalty = Objective(['U_pred_policy', 'U_minf'], lambda x, xmin: torch.mean(F.relu(-x + xmin)),
                                           weight=args.Q_con_u, name='input_lower_bound')
    inputs_upper_bound_penalty = Objective(['U_pred_policy', 'U_maxf'], lambda x, xmax: torch.mean(F.relu(x - xmax)),
                                           weight=args.Q_con_u, name='input_upper_bound')

    objectives = [regularization, reference_loss]
    constraints = [observation_lower_bound_penalty, observation_upper_bound_penalty,
                   inputs_lower_bound_penalty, inputs_upper_bound_penalty]

    ##########################################
    ########## OPTIMIZE SOLUTION ############
    ##########################################
    model = Problem(objectives, constraints, components).to(device)
    freeze_weight(model, module_names=args.freeze)
    unfreeze_weight(model, module_names=args.unfreeze)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    plot_keys = ['Y_pred', 'U_pred']  # variables to be plotted
    visualizer = VisualizerClosedLoop(dataset, dynamics_model, plot_keys, args.verbosity, savedir=args.savedir)
    emulator = dynamics_model
    simulator = ClosedLoopSimulator(model=model, dataset=dataset, emulator=emulator)
    trainer = Trainer(model, dataset, optimizer, logger=logger, visualizer=visualizer,
                      simulator=simulator, epochs=args.epochs)
    best_model = trainer.train()
    trainer.evaluate(best_model)
    logger.clean_up()
