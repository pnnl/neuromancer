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
from dataset import EmulatorDataset, FileDataset
import dynamics
import estimators
import emulators
import policies
import linear
import blocks
import logger
from visuals import Visualizer, VisualizerTrajectories
from trainer import Trainer
from problem import Problem, Objective
import torch.nn.functional as F
import plot
from dataset import unbatch_data
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default=None,
                        help="Gpu to use")
    # OPTIMIZATION PARAMETERS
    opt_group = parser.add_argument_group('OPTIMIZATION PARAMETERS')
    opt_group.add_argument('-epochs', type=int, default=1000)
    opt_group.add_argument('-lr', type=float, default=0.001,
                           help='Step size for gradient descent.')

    #################
    # DATA PARAMETERS
    data_group = parser.add_argument_group('DATA PARAMETERS')
    data_group.add_argument('-nsteps', type=int, default=32,
                            help='control policy prediction horizon.')
    data_group.add_argument('-system_data', type=str, choices=['emulator', 'datafile'],
                            default='emulator',
                            help='source type of the dataset')
    data_group.add_argument('-system', default='Reno_full',
                            help='select particular dataset with keyword')
    data_group.add_argument('-nsim', type=int, default=8640,
                            help='Number of time steps for full dataset. (ntrain + ndev + ntest)'
                                 'train, dev, and test will be split evenly from contiguous, sequential, '
                                 'non-overlapping chunks of nsim datapoints, e.g. first nsim/3 art train,'
                                 'next nsim/3 are dev and next nsim/3 simulation steps are test points.'
                                 'None will use a default nsim from the selected dataset or emulator')
    data_group.add_argument('-norm', choices=['UDY', 'U', 'Y', None], type=str, default='UDY')

    ##################
    # MODEL PARAMETERS
    model_group = parser.add_argument_group('MODEL PARAMETERS')
    model_group.add_argument('-n_hidden', type=int, default=10, help='Number of hidden states')
    model_group.add_argument('-n_layers', type=int, default=2,
                             help='Number of hidden layers of single time-step state transition')
    model_group.add_argument('-policy', type=str,
                             choices=['rnn', 'mlp', 'linear', 'residual_mlp'], default='mlp')
    model_group.add_argument('-linear_map', type=str, choices=list(linear.maps.keys()),
                             default='pf')
    model_group.add_argument('-bias', action='store_true', help='Whether to use bias in the neural network models.')
    model_group.add_argument('-system_id', required=True, help='path to pytorch pretrained dynamics and state estimator model from system ID')
    model_group.add_argument('-policy_features', nargs='+', default=['x0'], help='Policy features')

    ##################
    # Weight PARAMETERS
    weight_group = parser.add_argument_group('WEIGHT PARAMETERS')
    weight_group.add_argument('-Q_con_y', type=float, default=1.0, help='Output constraints penalty weight.')
    weight_group.add_argument('-Q_con_u', type=float, default=1.0, help='Input constraints penalty weight.')
    weight_group.add_argument('-Q_sub', type=float, default=0.2, help='Linear maps regularization weight.')
    weight_group.add_argument('-Q_r', type=float, default=1.0, help='Reference tracking penalty weight')

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


# TODO: add default Open and Closed loop visualizers to visuals.py
class VisualizerOpen(Visualizer):

    def __init__(self, dataset, model, verbosity):
        self.model = model
        self.dataset = dataset
        self.verbosity = verbosity
        self.anime = plot.Animator(dataset.dev_loop['Yp'].detach().cpu().numpy(), model)

    def train_plot(self, outputs, epoch):
        if epoch % self.verbosity == 0:
            self.anime(outputs['loop_dev_Y_pred'], outputs['loop_dev_Yf'])

    def train_output(self):
        self.anime.make_and_save(os.path.join(args.savedir, 'eigen_animation.mp4'))
        return dict()

    def eval(self, outputs):
        dsets = ['train', 'dev', 'test']
        Ypred = [unbatch_data(outputs[f'nstep_{dset}_Y_pred']).squeeze(1).detach().cpu().numpy() for dset in dsets]
        Ytrue = [unbatch_data(outputs[f'nstep_{dset}_Yf']).squeeze(1).detach().cpu().numpy() for dset in dsets]
        plot.pltOL(Y=np.concatenate(Ytrue), Ytrain=np.concatenate(Ypred),
                   figname=os.path.join(args.savedir, 'nstep_OL.png'))

        Ypred = [outputs[f'loop_{dset}_Y_pred'].squeeze(1).detach().cpu().numpy() for dset in dsets]
        Ytrue = [outputs[f'loop_{dset}_Yf'].squeeze(1).detach().cpu().numpy() for dset in dsets]
        plot.pltOL(Y=np.concatenate(Ytrue), Ytrain=np.concatenate(Ypred),
                   figname=os.path.join(args.savedir, 'open_OL.png'))

        plot.trajectory_movie(np.concatenate(Ytrue).transpose(1, 0),
                              np.concatenate(Ypred).transpose(1, 0),
                              figname=os.path.join(args.savedir, f'open_movie.mp4'),
                              freq=self.verbosity)
        return dict()


def logging(args):
    if args.logger == 'mlflow':
        Logger = logger.MLFlowLogger(args, args.savedir, args.verbosity)
    else:
        Logger = logger.BasicLogger(savedir=args.savedir, verbosity=args.verbosity)
    device = f'cuda:{args.gpu}' if (args.gpu is not None) else 'cpu'
    return Logger, device


def dataset_load(args, sequences):
    if args.system_data == 'emulator':
        dataset = EmulatorDataset(system=args.system, nsim=args.nsim, sequences=sequences,
                                  norm=args.norm, nsteps=args.nsteps, device=device, savedir=args.savedir)
    else:
        dataset = FileDataset(system=args.system, nsim=args.nsim, sequences=sequences,
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


    'Y_min'

    dataset = dataset_load(args, sequences)


    ##########################################
    ########## PROBLEM COMPONENTS ############
    ##########################################

    system_id_problem = torch.load(args.system_id)
    estimator = system_id_problem.components[0]
    dynamics_model = system_id_problem.components[1]

    nx = dynamics_model.nx
    nu = dynamics_model.nu
    nd = dynamics_model.nd
    ny = dynamics_model.ny
    nh_policy = args.n_hidden
    dataset_keys = set(dataset.dev_data.keys())
    linmap = linear.maps[args.linear_map]

    # state estimator setup
    policy = {'linear': policies.LinearPolicy,
                 'mlp': policies.MLPEstimator,
                 'rnn': policies.RNNEstimator,
                 'residual_mlp': policies.ResMLPEstimator}[args.policy]({**dataset.dims, 'X': nx},
                                                                                   nsteps=args.nsteps,
                                                                                   bias=args.bias,
                                                                                   Linear=linmap,
                                                                                   nonlin=F.gelu,
                                                                                   hsizes=[nh_policy] * args.n_layers,
                                                                                   input_keys=set(args.policy_features),
                                                                                   linargs=dict(),
                                                                                   name='policy')

    components = [estimator, policy, dynamics_model]

    # component variables
    input_keys = set.union(*[comp.input_keys for comp in components])
    output_keys = set.union(*[comp.output_keys for comp in components])
    plot_keys = {'Y_pred', 'X_pred', 'U_pred'}  # variables to be plotted

    ##########################################
    ########## MULTI-OBJECTIVE LOSS ##########
    ##########################################
    regularization = Objective(['policy_reg_error'], lambda reg: reg,
                               weight=args.Q_sub)
    reference_loss = Objective(['Y_pred', 'Rf'], F.mse_loss, weight=args.Q_r)
    #  TODO: expand control dataset with constraints and references
    observation_lower_bound_penalty = Objective(['Y_pred', 'Y_min'], lambda x, xmin: torch.mean(F.relu(-x + -xmin)),
                                                weight=args.Q_con_x)
    observation_upper_bound_penalty = Objective(['Y_pred', 'Y_max'], lambda x, xmax: torch.mean(F.relu(x - xmax)),
                                                weight=args.Q_con_y)
    inputs_lower_bound_penalty = Objective(['U_pred', 'U_min'], lambda x, xmin: torch.mean(F.relu(-x + -xmin)),
                                                weight=args.Q_con_x)
    inputs_upper_bound_penalty = Objective(['U_pred', 'U_max'], lambda x, xmax: torch.mean(F.relu(x - xmax)),
                                           weight=args.Q_con_y)

    objectives = [regularization, reference_loss]  # estimator_loss
    constraints = [observation_lower_bound_penalty, observation_upper_bound_penalty,
                   inputs_lower_bound_penalty, inputs_upper_bound_penalty]

    # TODO: CL training
    # 1, joint system ID and control - need system ID dataset + control dataset with some performance metric  to track
    #       need two instances of the dynamical model and state estimator
    # 2, fixed model and do only policy optization - need control dataset, and only control objectives
    #       requires_grad = False for the loaded model - add some option for loading selected component parameters
    # 3, online learning with subset of the parameter updates

    # TODO: CL evaluation
    # 1, using emulator
    # 2, using learned model

    ##########################################
    ########## OPTIMIZE SOLUTION ############
    ##########################################
    model = Problem(objectives, constraints, components).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    visualizer = VisualizerTrajectories(dataset, dynamics_model, plot_keys, args.verbosity)
    trainer = Trainer(model, dataset, optimizer, logger=logger, visualizer=visualizer, epochs=args.epochs)
    best_model = trainer.train()
    trainer.evaluate(best_model)
    # TODO: add simulator class for dynamical models? - evaluates open and closed loop
    # simulator = Simulator(best_model, dataset, emulator=emulators.systems[args.system], logger=logger, visualizer=visualizer)
    logger.clean_up()
