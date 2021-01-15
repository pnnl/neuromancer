"""
TODO: include readme on experiment setup

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
import glob
import random
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
from neuromancer.signals import NoiseGenerator, SignalGenerator, WhiteNoisePeriodicGenerator, PeriodicGenerator, WhiteNoiseGenerator, AddGenerator


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=int, default=None,
                        help="Gpu to use")
    ##################
    # OPTIMIZATION PARAMETERS
    opt_group = parser.add_argument_group('OPTIMIZATION PARAMETERS')
    opt_group.add_argument('-epochs', type=int, default=5)
    opt_group.add_argument('-lr', type=float, default=0.001,
                           choices=[3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01],
                           help='Step size for gradient descent.')
    opt_group.add_argument('-patience', type=int, default=100,
                           help='How many epochs to allow for no improvement in eval metric before early stopping.')
    opt_group.add_argument('-warmup', type=int, default=100,
                           help='Number of epochs to wait before enacting early stopping policy.')
    opt_group.add_argument('-skip_eval_sim', action='store_true',
                           help='Whether to run simulator during evaluation phase of training.')
    #################
    # DATA PARAMETERS
    data_group = parser.add_argument_group('DATA PARAMETERS')
    data_group.add_argument('-nsteps', type=int, default=32, choices=[4, 8, 16, 32, 64],
                            help='Number of steps for open loop during training.')
    data_group.add_argument('-system', type=str, default='flexy_air',
                            help='select particular dataset with keyword')
    data_group.add_argument('-nsim', type=int, default=100000,
                            help='Number of time steps for full dataset. (ntrain + ndev + ntest)'
                                 'train, dev, and test will be split evenly from contiguous, sequential, '
                                 'non-overlapping chunks of nsim datapoints, e.g. first nsim/3 art train,'
                                 'next nsim/3 are dev and next nsim/3 simulation steps are test points.'
                                 'None will use a default nsim from the selected dataset or emulator')
    data_group.add_argument('-norm', nargs='+', default=['U', 'D', 'Y'], choices=['U', 'D', 'Y', 'X'],
                            help='List of sequences to max-min normalize')
    mfiles = ['models/best_model_flexy1.pth',
              'models/best_model_flexy2.pth',
              'ape_models/best_model_blocknlin.pth']
    data_group.add_argument('-model_file', type=str, default=mfiles[0])
    # mfiles[2] - This model requires nsteps >=10
    ##################
    # POLICY PARAMETERS
    policy_group = parser.add_argument_group('POLICY PARAMETERS')
    policy_group.add_argument('-policy', type=str,
                              choices=['mlp', 'linear'], default='mlp')
    policy_group.add_argument('-n_hidden', type=int, default=20, choices=list(range(5, 50, 5)),
                              help='Number of hidden states')
    policy_group.add_argument('-n_layers', type=int, default=3, choices=list(range(1, 10)),
                             help='Number of hidden layers of single time-step state transition')
    policy_group.add_argument('-bias', action='store_true', help='Whether to use bias in the neural network models.')
    policy_group.add_argument('-policy_features', nargs='+', default=['Y_ctrl_p', 'Rf', 'Y_maxf', 'Y_minf'], help='Policy features')
    policy_group.add_argument('-activation', choices=['gelu', 'softexp'], default='gelu',
                              help='Activation function for neural networks')
    policy_group.add_argument('-perturbation', choices=['white_noise_sine_wave', 'white_noise'], default='white_noise')
    ##################
    # LINEAR PARAMETERS
    linear_group = parser.add_argument_group('LINEAR PARAMETERS')
    linear_group.add_argument('-linear_map', type=str,
                              choices=['linear', 'softSVD', 'pf'],
                              default='linear')
    linear_group.add_argument('-sigma_min', type=float, choices=[1e-5, 0.1, 0.2, 0.3, 0.4, 0.5], default=0.1)
    linear_group.add_argument('-sigma_max', type=float, choices=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
                              default=1.0)
    ##################
    # LAYERS
    layers_group = parser.add_argument_group('LAYERS PARAMETERS')
    layers_group.add_argument('-freeze', nargs='+', default=[''], help='sets requires grad to False')
    layers_group.add_argument('-unfreeze', default=['components.2'],
                              help='sets requires grad to True')
    ##################
    # WEIGHT PARAMETERS
    weight_group = parser.add_argument_group('WEIGHT PARAMETERS')
    weight_group.add_argument('-Q_con_x', type=float, default=1.0, choices=[0.1, 1.0, 10.0, 100.0],
                              help='Hidden state constraints penalty weight.')
    weight_group.add_argument('-Q_con_y', type=float, default=2.0, choices=[0.1, 1.0, 10.0, 100.0],
                              help='Observable constraints penalty weight.')
    weight_group.add_argument('-Q_dx', type=float, default=0.1, choices=[0.1, 1.0, 10.0, 100.0],
                              help='Penalty weight on hidden state difference in one time step.')
    weight_group.add_argument('-Q_sub', type=float, default=0.1, help='Linear maps regularization weight.',
                              choices=[0.1, 1.0, 10.0])
    weight_group.add_argument('-Q_y', type=float, default=1.0, choices=[0.1, 1.0, 10.0, 100.0],
                              help='Output tracking penalty weight')
    weight_group.add_argument('-Q_e', type=float, default=1.0, choices=[0.1, 1.0, 10.0, 100.0],
                              help='State estimator hidden prediction penalty weight')
    weight_group.add_argument('-Q_con_fdu', type=float, default=0.0, choices=[0.1, 1.0, 10.0, 100.0],
                              help='Penalty weight on control actions and disturbances.')
    weight_group.add_argument('-Q_con_u', type=float, default=10.0, choices=[0.1, 1.0, 10.0, 100.0],
                              help='Input constraints penalty weight.')
    weight_group.add_argument('-Q_r', type=float, default=1.0, choices=[0.1, 1.0, 10.0, 100.0],
                              help='Reference tracking penalty weight')
    weight_group.add_argument('-Q_du', type=float, default=0.1, choices=[0.1, 1.0, 10.0, 100.0],
                              help='control action difference penalty weight')
    # objective and constraints variations
    weight_group.add_argument('-con_tighten', choices=[0, 1], default=0)
    weight_group.add_argument('-tighten', type=float, default=0.05, choices=[0.1, 0.05, 0.01, 0.0],
                              help='control action difference penalty weight')
    weight_group.add_argument('-loss_clip', choices=[0, 1], default=0)
    weight_group.add_argument('-noise', choices=[0, 1], default=0)


    ####################
    # LOGGING PARAMETERS
    log_group = parser.add_argument_group('LOGGING PARAMETERS')
    log_group.add_argument('-savedir', type=str, default='test',
                           help="Where should your trained model and plots be saved (temp)")
    log_group.add_argument('-verbosity', type=int, default=1,
                           help="How many epochs in between status updates")
    log_group.add_argument('-exp', type=str, default='test',
                           help='Will group all run under this experiment name.')
    log_group.add_argument('-location', type=str, default='mlruns',
                           help='Where to write mlflow experiment tracking stuff')
    log_group.add_argument('-run', type=str, default='neuromancer',
                           help='Some name to tell what the experiment run was about.')
    log_group.add_argument('-logger', type=str, default='mlflow',
                           help='Logging setup to use')
    log_group.add_argument('-id', help='Unique run name')
    log_group.add_argument('-parent', help='ID of parent or none if from the Eve generation')
    log_group.add_argument('-train_visuals', action='store_true',
                           help='Whether to create visuals, e.g. animations during training loop')
    log_group.add_argument('-trace_movie', action='store_true',
                           help='Whether to plot an animation of the simulated and true dynamics')
    return parser


def logging(args):
    if args.logger == 'mlflow':
        Logger = loggers.MLFlowLogger(args=args, savedir=args.savedir, verbosity=args.verbosity,
                                      stdout=('nstep_dev_loss', 'loop_dev_loss', 'best_loop_dev_loss',
                                              'nstep_dev_ref_loss', 'loop_dev_ref_loss'), id=args.id)

    else:
        Logger = loggers.BasicLogger(args=args, savedir=args.savedir, verbosity=args.verbosity,
                                     stdout=('nstep_dev_loss', 'loop_dev_loss', 'best_loop_dev_loss',
                                             'nstep_dev_ref_loss', 'loop_dev_ref_loss'), id=args.id)
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
        ny = args.ny
        if not ny == dataset.data['Y'].shape[1]:
            new_sequences = {'Y': dataset.data['Y'][:, :1]}
            dataset.add_data(new_sequences, overwrite=True)
        dataset.min_max_norms['Ymin'] = dataset.min_max_norms['Ymin'][0]
        dataset.min_max_norms['Ymax'] = dataset.min_max_norms['Ymax'][0]

        nsim = dataset.data['Y'].shape[0]
        nu = dataset.data['U'].shape[1]
        new_sequences = {'Y_max': psl.Periodic(nx=1, nsim=nsim, numPeriods=30, xmax=0.9, xmin=0.6),
                         'Y_min': psl.Periodic(nx=1, nsim=nsim, numPeriods=25, xmax=0.4, xmin=0.1),
                         'U_max': np.ones([nsim, nu]), 'U_min': np.zeros([nsim, nu]),
                         'R': psl.Periodic(nx=1, nsim=nsim, numPeriods=20, xmax=0.8, xmin=0.2)
                         # 'Y_ctrl_': psl.RandomWalk(nx=ny, nsim=nsim, xmax=[1.0] * ny, xmin=[0.0] * ny, sigma=0.05)}
                         #'Y_ctrl_': psl.WhiteNoise(nx=ny, nsim=nsim, xmax=[1.0] * ny, xmin=[0.0] * ny)}
                         }
        dataset.add_data(new_sequences)
        # dataset.dims['Rf'] = (9000, 1)
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
    # device = 'cuda:0'

    ##########################################
    ########## PROBLEM COMPONENTS ############
    ##########################################
    # Learned dynamics system ID model setup
    load_model = torch.load(args.model_file, pickle_module=dill, map_location=torch.device(device))
    args.ny = load_model.components[1].fy.out_features
    dataset = dataset_load(args, device)
    print(dataset.dims)
    for k in range(len(load_model.components)):
        if load_model.components[k].name == 'dynamics':
            dynamics_model = load_model.components[k]
            dynamics_model.input_keys[2] = 'U_pred_policy'
            dynamics_model.fe = None
            dynamics_model.fyu = None
            dynamics_model.to(device)
        if load_model.components[k].name == 'estim':
            estimator = load_model.components[k]
            estimator.input_keys[0] = 'Y_ctrl_p'
            estimator.data_dims = dataset.dims
            estimator.data_dims['Y_ctrl_p'] = dataset.dims['Yp']
            estimator.nsteps = args.nsteps
            estimator.to(device)

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
                             linear_map=linmap,
                             nonlin=activation,
                             hsizes=[nh_policy] * args.n_layers,
                             input_keys=args.policy_features,
                             linargs={'sigma_min': args.sigma_min, 'sigma_max': args.sigma_max},
                             name='policy').to(device)

    signal_generator = WhiteNoisePeriodicGenerator(args.nsteps, args.ny, xmax=(0.8, 0.7), xmin=0.2,
                                                   min_period=1, max_period=20, name='Y_ctrl_').to(device)
    # reference_generator = PeriodicGenerator(args.nsteps, args.ny, xmax=0.7, xmin=0.3,
    #                                                min_period=1, max_period=20, name='R')
    # dynamics_generator = SignalGeneratorDynamics(dynamics_model, estimator, args.nsteps, xmax=1.0, xmin=0.0, name='Y_ctrl_')

    noise_generator = NoiseGenerator(ratio=0.05, keys=['Y_pred_dynamics'], name='_noise').to(device)

    components = [signal_generator, estimator, policy, dynamics_model, noise_generator]
    # components = [dynamics_generator, estimator, policy, dynamics_model]
    # components = [signal_generator, reference_generator, estimator, policy, dynamics_model]

    ##########################################
    ########## MULTI-OBJECTIVE LOSS ##########
    ##########################################
    # NOISE
    if args.noise:
        output_key = 'Y_pred_dynamics_noise'
    else:
        output_key = 'Y_pred_dynamics'

    reference_loss = Objective([output_key, 'Rf'], lambda pred, ref: F.mse_loss(pred[:, :, :1], ref),
                               weight=args.Q_r, name='ref_loss').to(device)
    regularization = Objective(['reg_error_policy'], lambda reg: reg,
                               weight=args.Q_sub).to(device)
    control_smoothing = Objective(['U_pred_policy'], lambda x: F.mse_loss(x[1:], x[:-1]),
                                  weight=args.Q_du, name='control_smoothing').to(device)
    observation_lower_bound_penalty = Objective([output_key, 'Y_minf'],
                                                lambda x, xmin: torch.mean(F.relu(-x[:, :, :1] + xmin)),
                                                weight=args.Q_con_y, name='observation_lower_bound').to(device)
    observation_upper_bound_penalty = Objective([output_key, 'Y_maxf'],
                                                lambda x, xmax: torch.mean(F.relu(x[:, :, :1] - xmax)),
                                                weight=args.Q_con_y, name='observation_upper_bound').to(device)
    inputs_lower_bound_penalty = Objective(['U_pred_policy', 'U_minf'], lambda x, xmin: torch.mean(F.relu(-x + xmin)),
                                           weight=args.Q_con_u, name='input_lower_bound').to(device)
    inputs_upper_bound_penalty = Objective(['U_pred_policy', 'U_maxf'], lambda x, xmax: torch.mean(F.relu(x - xmax)),
                                           weight=args.Q_con_u, name='input_upper_bound').to(device)

    # Constraints tightening
    if args.con_tighten:
        observation_lower_bound_penalty = Objective([output_key, 'Y_minf'],
                                                    lambda x, xmin: torch.mean(F.relu(-x[:, :, :1] + xmin+args.tighten)),
                                                    weight=args.Q_con_y, name='observation_lower_bound').to(device)
        observation_upper_bound_penalty = Objective([output_key, 'Y_maxf'],
                                                    lambda x, xmax: torch.mean(F.relu(x[:, :, :1] - xmax+args.tighten)),
                                                    weight=args.Q_con_y, name='observation_upper_bound').to(device)
        inputs_lower_bound_penalty = Objective(['U_pred_policy', 'U_minf'], lambda x, xmin: torch.mean(F.relu(-x + xmin+args.tighten)),
                                               weight=args.Q_con_u, name='input_lower_bound').to(device)
        inputs_upper_bound_penalty = Objective(['U_pred_policy', 'U_maxf'], lambda x, xmax: torch.mean(F.relu(x - xmax+args.tighten)),
                                               weight=args.Q_con_u, name='input_upper_bound').to(device)

    # LOSS clipping
    if args.loss_clip:
        reference_loss = Objective([output_key, 'Rf', 'Y_minf', 'Y_maxf'],
                                   lambda pred, ref, xmin, xmax: F.mse_loss(pred[:, :, :1]*torch.gt(ref, xmin).int()*torch.lt(ref, xmax).int(), ref*torch.gt(ref, xmin).int()*torch.lt(ref, xmax).int()),
                                   weight=args.Q_r, name='ref_loss').to(device)

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
    plot_keys = ['Y_pred', 'U_pred', 'x0_estim']  # variables to be plotted
    visualizer = VisualizerClosedLoop(dataset, policy, plot_keys, args.verbosity, savedir=args.savedir)
    emulator = dynamics_model
    # TODO: hacky solution for policy input keys compatibility with simulator
    policy.input_keys[0] = 'Yp'
    simulator = ClosedLoopSimulator(model=model, dataset=dataset, emulator=emulator, policy=policy)
    trainer = Trainer(model, dataset, optimizer, logger=logger, visualizer=visualizer,
                      simulator=simulator, epochs=args.epochs,
                      patience=args.patience, warmup=args.warmup)
    best_model = trainer.train()
    trainer.evaluate(best_model)
    logger.log_metrics({'alive': 0.0})
    # logger.clean_up()

    if False:
        model.load_state_dict(best_model)
        torch.save(model.components[2], './test/best_policy_flexy.pth', pickle_module=dill)
        torch.save(model.components[1], './test/best_estimator_flexy.pth', pickle_module=dill)
        torch.save(model.components[3], './test/best_dynamics_flexy.pth', pickle_module=dill)

# TODO: add noiser to control action
# TODO: UQ via ensemble methods
# TODO: adaptive constraints tightening based on UQ on the system dynamics model
# TODO: drop loss function value on reference if it is outside of the constraints
# TODO: add robust margins on constraints
# TODO: Constrained Neural Feedback Systems - paper title

