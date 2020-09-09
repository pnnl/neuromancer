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
from neuromancer.datasets import normalize
from neuromancer.signals import SignalGeneratorDynamics


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=int, default=None,
                        help="Gpu to use")
    # OPTIMIZATION PARAMETERS
    opt_group = parser.add_argument_group('OPTIMIZATION PARAMETERS')
    opt_group.add_argument('-epochs', type=int, default=5000)
    opt_group.add_argument('-lr', type=float, default=0.001,
                           choices=[3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01],
                           help='Step size for gradient descent.')
    opt_group.add_argument('-patience', type=int, default=50,
                           help='How many epochs to allow for no improvement in eval metric before early stopping.')
    opt_group.add_argument('-warmup', type=int, default=100,
                           help='Number of epochs to wait before enacting early stopping policy.')
    opt_group.add_argument('-skip_eval_sim', action='store_true',
                           help='Whether to run simulator during evaluation phase of training.')

    #################
    # DATA PARAMETERS
    data_group = parser.add_argument_group('DATA PARAMETERS')
    data_group.add_argument('-nsteps', type=int, default=4, choices=[4, 8, 16, 32, 64],
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
    data_group.add_argument('-model_file', type=str,  # choices=['/qfs/projects/deepmpc/best_flexy_models/best_blocknlin_nlinsearch/best_model.pth'],# choices=glob.glob('/qfs/projects/deepmpc/best_flexy_models/*/best_model.pth'),
                            default='../datasets/Flexy_air/best_model_flexy1.pth')
    mfiles = ['/qfs/projects/deepmpc/best_flexy_models/best_blocknlin_nlinsearch/best_model.pth',]
    ##################
    # POLICY PARAMETERS
    policy_group = parser.add_argument_group('POLICY PARAMETERS')
    policy_group.add_argument('-policy', type=str,
                              choices=['mlp', 'rnn', 'residual_mlp'], default='mlp')
    policy_group.add_argument('-n_hidden', type=int, default=20, choices=list(range(5, 50, 5)),
                              help='Number of hidden states')
    policy_group.add_argument('-n_layers', type=int, default=2, choices=list(range(1, 10)),
                             help='Number of hidden layers of single time-step state transition')
    policy_group.add_argument('-bias', action='store_true', help='Whether to use bias in the neural network models.')
    policy_group.add_argument('-policy_features', nargs='+', default=['Y_ctrl_p', 'Rf'], help='Policy features')
    policy_group.add_argument('-activation', choices=['gelu', 'softexp'], default='gelu',
                              help='Activation function for neural networks')
    policy_group.add_argument('-perturbation', choices=['white_noise_sine_wave'], default='white_noise')

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
    ##################
    # WEIGHT PARAMETERS
    weight_group = parser.add_argument_group('WEIGHT PARAMETERS')
    weight_group.add_argument('-Q_con_x', type=float, default=1.0, choices=[0.1, 1.0, 10.0],
                              help='Hidden state constraints penalty weight.')
    weight_group.add_argument('-Q_con_y', type=float, default=1.0, choices=[0.1, 1.0, 10.0],
                              help='Observable constraints penalty weight.')
    weight_group.add_argument('-Q_dx', type=float, default=0.1, choices=[0.1, 1.0, 10.0],
                              help='Penalty weight on hidden state difference in one time step.')
    weight_group.add_argument('-Q_sub', type=float, default=0.1, help='Linear maps regularization weight.',
                              choices=[0.1, 1.0, 10.0])
    weight_group.add_argument('-Q_y', type=float, default=1.0, choices=[0.1, 1.0, 10.0],
                              help='Output tracking penalty weight')
    weight_group.add_argument('-Q_e', type=float, default=1.0, choices=[0.1, 1.0, 10.0],
                              help='State estimator hidden prediction penalty weight')
    weight_group.add_argument('-Q_con_fdu', type=float, default=0.0, choices=[0.1, 1.0, 10.0],
                              help='Penalty weight on control actions and disturbances.')
    weight_group.add_argument('-Q_con_u', type=float, default=10.0, choices=[0.1, 1.0, 10.0],
                              help='Input constraints penalty weight.')
    weight_group.add_argument('-Q_r', type=float, default=1.0, choices=[0.1, 1.0, 10.0],
                              help='Reference tracking penalty weight')
    weight_group.add_argument('-Q_du', type=float, default=0.0, choices=[0.1, 1.0, 10.0],
                              help='control action difference penalty weight')

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
        new_sequences = {'Y_max': 0.8 * np.ones([nsim, 1]), 'Y_min': 0.2 * np.ones([nsim, 1]),
                         'U_max': np.ones([nsim, nu]), 'U_min': np.zeros([nsim, nu]),
                         # 'R': psl.Steps(nx=1, nsim=nsim, randsteps=60, xmax=0.7, xmin=0.3),
                         'R': psl.Periodic(nx=1, nsim=nsim, numPeriods=20, xmax=0.7, xmin=0.3)
                         # 'Y_ctrl_': psl.RandomWalk(nx=ny, nsim=nsim, xmax=[1.0] * ny, xmin=[0.0] * ny, sigma=0.05)}
                         #'Y_ctrl_': psl.WhiteNoise(nx=ny, nsim=nsim, xmax=[1.0] * ny, xmin=[0.0] * ny)}
                         }
        dataset.add_data(new_sequences)
    return dataset


class SignalGenerator(nn.Module):

    def __init__(self, nsteps, nx, xmax, xmin, name='signal', device='cpu'):
        super().__init__()
        self.nsteps, self.nx = nsteps, nx
        self.xmax, self.xmin, self.name = xmax, xmin, name
        self.iter = 0
        self.nsim = None
        self.device = device

    def get_xmax(self):
        return self.xmax

    def get_xmin(self):
        return self.xmin
# nstep X nsamples x nfeatures
#
#     end_step = data.shape[0] - nsteps
#     data = np.asarray([data[k:k+nsteps, :] for k in range(0, end_step)])  # nchunks X nsteps X nfeatures
#     return data.transpose(1, 0, 2)  # nsteps X nsamples X nfeatures

    def forward(self, data):
        key = list(data.keys())[0]
        self.nsim = data[key].shape[1] * data[key].shape[0] + self.nsteps
        nbatch = data[key].shape[1]

        xmax, xmin = self.get_xmax(), self.get_xmin()
        R = self.sequence_generator(self.nsim, xmax, xmin)
        R, _, _ = normalize(R)

        Rp, Rf = R[:-self.nsteps], R[self.nsteps:]

        if self.iter == 0:
            self.nstep_batch_size = nbatch
        if nbatch == self.nstep_batch_size:
            Rp = torch.tensor(Rp, dtype=torch.float32).view(nbatch, -1, self.nx).transpose(0, 1).to()
            Rf = torch.tensor(Rf, dtype=torch.float32).view(nbatch, -1, self.nx).transpose(0, 1)
        else:
            end_step = Rp.shape[0] - self.nsteps
            Rp = np.asarray([Rp[k:k+self.nsteps, :] for k in range(0, end_step)])  # nchunks X nsteps X nfeatures
            Rp = torch.tensor(Rp.transpose(1, 0, 2), dtype=torch.float32)  # nsteps X nsamples X nfeatures
            Rf = np.asarray([Rf[k:k + self.nsteps, :] for k in range(0, end_step)])  # nchunks X nsteps X nfeatures
            Rf = torch.tensor(Rf.transpose(1, 0, 2), dtype=torch.float32)  # nsteps X nsamples X nfeatures
        return {self.name + 'p': Rp.to(self.device), self.name + 'f': Rf.to(self.device)}


class WhiteNoisePeriodicGenerator(SignalGenerator):

    def __init__(self, nsteps, nx, xmax=(0.5, 0.1), xmin=0.0, min_period=5, max_period=30, name='period', device='cpu'):
        super().__init__(nsteps, nx, xmax, xmin, name=name, device=device)
        self.min_period = min_period
        self.max_period = max_period
        self.white_noise_generator = lambda nsim, xmin, xmax: psl.WhiteNoise(nx=self.nx, nsim=nsim,
                                                                             xmax=xmax, xmin=xmin)[:nsim]
        self.period_generator = lambda nsim, xmin, xmax: psl.Periodic(nx=self.nx, nsim=nsim,
                                                                      numPeriods=random.randint(self.min_period, self.max_period),
                                                                      xmax=xmax, xmin=xmin)[:nsim]
        self.sequence_generator = lambda nsim, xmin, xmax: self.period_generator(nsim, xmin, xmax) + self.white_noise_generator(nsim, xmin, 1.0 - xmax)

    def get_xmax(self):
        return random.uniform(*self.xmax)


    def forward(self, data):
        key = list(data.keys())[0]
        self.nsim = data[key].shape[1] * data[key].shape[0] + self.nsteps
        nbatch = data[key].shape[1]

        if self.max_period > self.nsim:
            self.max_period = self.nsim
        if self.min_period > self.nsim:
            self.min_period = self.nsim

        xmax, xmin = self.get_xmax(), self.get_xmin()
        R = self.sequence_generator(self.nsim, xmax, xmin)
        R, _, _ = normalize(R)

        Rp, Rf = R[:-self.nsteps], R[self.nsteps:]

        if self.iter == 0:
            self.nstep_batch_size = nbatch
        if nbatch == self.nstep_batch_size:
            Rp = torch.tensor(Rp, dtype=torch.float32).view(nbatch, -1, self.nx).transpose(0, 1).to()
            Rf = torch.tensor(Rf, dtype=torch.float32).view(nbatch, -1, self.nx).transpose(0, 1)
        else:
            end_step = Rp.shape[0] - self.nsteps
            Rp = np.asarray([Rp[k:k+self.nsteps, :] for k in range(0, end_step)])  # nchunks X nsteps X nfeatures
            Rp = torch.tensor(Rp.transpose(1, 0, 2), dtype=torch.float32)  # nsteps X nsamples X nfeatures
            Rf = np.asarray([Rf[k:k + self.nsteps, :] for k in range(0, end_step)])  # nchunks X nsteps X nfeatures
            Rf = torch.tensor(Rf.transpose(1, 0, 2), dtype=torch.float32)  # nsteps X nsamples X nfeatures
        return {self.name + 'p': Rp.to(self.device), self.name + 'f': Rf.to(self.device)}


class PeriodicGenerator(SignalGenerator):

    def __init__(self, nsteps, nx, xmax, xmin, min_period=5, max_period=30, name='period', device='cpu'):
        super().__init__(nsteps, nx, xmax, xmin, name=name, device=device)
        self.min_period = min_period
        self.max_period = max_period
        self.sequence_generator = lambda nsim: psl.Periodic(nx=self.nx, nsim=nsim, numPeriods=random.randint(self.min_period, self.max_period),
                                                            xmax=self.xmax, xmin=self.xmin)

class WhiteNoiseGenerator(SignalGenerator):
    def __init__(self, nsteps, nx, xmax, xmin, name='period', device='cpu'):
        super().__init__(nsteps, nx, xmax, xmin, name=name, device=device)
        self.sequence_generator = lambda nsim: psl.WhiteNoise(nx=self.nx, nsim=nsim,
                                                              xmax=self.xmax, xmin=self.xmin)

class AddGenerator(SignalGenerator):
    def __init__(self, SG1, SG2, nsteps, nx, xmax, xmin, name='period', device='cpu'):
        super().__init__(nsteps, nx, xmax, xmin, name=name, device=device)
        assert SG1.nsteps == SG2.nsteps, 'Nsteps must match to compose sequence generators'
        assert SG1.nx == SG2.nx, 'Nx must match to compose sequence generators'
        self.sequence_generator = lambda nsim: SG1.sequence_generator(nsim) + SG2.sequence_generator(nsim)


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
                             Linear=linmap,
                             nonlin=activation,
                             hsizes=[nh_policy] * args.n_layers,
                             input_keys=args.policy_features,
                             linargs=dict(),
                             name='policy').to(device)
    signal_generator = WhiteNoisePeriodicGenerator(args.nsteps, args.ny, xmax=(0.90, 0.8), xmin=0.0,
                                                   min_period=1, max_period=20, name='Y_ctrl_', device=device).to(device)
    # reference_generator = WhiteNoisePeriodicGenerator(args.nsteps, args.ny, xmax=(0.90, 0.8), xmin=0.0,
    #                                                min_period=1, max_period=20, name='R')
    # dynamics_generator = SignalGeneratorDynamics(dynamics_model, estimator, args.nsteps, xmax=1.0, xmin=0.0, name='Y_ctrl_')
    # components = [dynamics_generator, estimator, policy, dynamics_model]
    components = [signal_generator, estimator, policy, dynamics_model]

    ##########################################
    ########## MULTI-OBJECTIVE LOSS ##########
    ##########################################
    # TODO: reformulate Qdu constraint based on the feedback during real time control
    regularization = Objective(['reg_error_policy'], lambda reg: reg,
                               weight=args.Q_sub).to(device)
    reference_loss = Objective(['Y_pred_dynamics', 'Rf'], lambda pred, ref: F.mse_loss(pred[:, :, :1], ref),
                               weight=args.Q_r, name='ref_loss').to(device)
    # reference_loss = Objective(['Y_pred_dynamics', 'Rf'], lambda pred, ref: F.mse_loss(pred, ref),
    #                           weight=args.Q_r, name='ref_loss').to(device)
    control_smoothing = Objective(['U_pred_policy'], lambda x: F.mse_loss(x[1:], x[:-1]),
                                  weight=args.Q_du, name='control_smoothing').to(device)
    observation_lower_bound_penalty = Objective(['Y_pred_dynamics', 'Y_minf'],
                                                lambda x, xmin: torch.mean(F.relu(-x[:, :, :1] + xmin)),
                                                weight=args.Q_con_y, name='observation_lower_bound').to(device)
    observation_upper_bound_penalty = Objective(['Y_pred_dynamics', 'Y_maxf'],
                                                lambda x, xmax: torch.mean(F.relu(x[:, :, :1] - xmax)),
                                                weight=args.Q_con_y, name='observation_upper_bound').to(device)
    inputs_lower_bound_penalty = Objective(['U_pred_policy', 'U_minf'], lambda x, xmin: torch.mean(F.relu(-x + xmin)),
                                           weight=args.Q_con_u, name='input_lower_bound').to(device)
    inputs_upper_bound_penalty = Objective(['U_pred_policy', 'U_maxf'], lambda x, xmax: torch.mean(F.relu(x - xmax)),
                                           weight=args.Q_con_u, name='input_upper_bound').to(device)

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
    visualizer = VisualizerClosedLoop(dataset, dynamics_model, plot_keys, args.verbosity, savedir=args.savedir)
    emulator = dynamics_model
    simulator = ClosedLoopSimulator(model=model, dataset=dataset, emulator=emulator)
    trainer = Trainer(model, dataset, optimizer, logger=logger, visualizer=visualizer,
                      simulator=simulator, epochs=args.epochs,
                      patience=args.patience, warmup=args.warmup)
    best_model = trainer.train()
    trainer.evaluate(best_model)
    logger.log_metrics({'alive': 0.0})
    logger.clean_up()

    if False:
        model.load_state_dict(best_model)
        torch.save(model, '../datasets/Flexy_air/best_policy_flexy.pth', pickle_module=dill)
