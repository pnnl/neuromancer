"""
Script for training block dynamics models for system identification.
Current block structure supported are black_box, hammerstein, and hammerstein weiner
Basic model options are:
    + prior on the linear maps of the neural network
    + state estimator
    + input non-linear map type for inputs and outputs
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
# python imports
import os
import argparse
from copy import deepcopy
import time
# plotting imports
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
# ml imports
import mlflow
import torch
import numpy as np
# local imports
import plot
import dataset
import dynamics
import estimators
import policies
import loops
import linear
import blocks
import emulators
from trainer import Problem
import logger
import visuals


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
    data_group.add_argument('-system', default='flexy_air',
                            help='select particular dataset with keyword')
    # data_group.add_argument('-system', choices=list(emulators.systems.keys()), default='flexy_air',
    #                         help='select particular dataset with keyword')
    data_group.add_argument('-nsim', type=int, default=6000,
                            help='Number of time steps for full dataset. (ntrain + ndev + ntest)'
                                 'train, dev, and test will be split evenly from contiguous, sequential, '
                                 'non-overlapping chunks of nsim datapoints, e.g. first nsim/3 art train,'
                                 'next nsim/3 are dev and next nsim/3 simulation steps are test points.'
                                 'None will use a default nsim from the selected dataset or emulator')
    data_group.add_argument('-norm', choices=['UDY', 'U', 'Y', None], type=str, default='UDY')
    data_group.add_argument('-loop', type=str, choices=['closed', 'open'], default='closed',
                            help='Defines open or closed loop for learning dynamics or control, respectively')
    data_group.add_argument('-adaptive', type=str, choices=[True, False], default=False,
                            help='extra option for closed loop'
                            'True = simultaneus policy optimization and system ID'
                            'False = policy optimization with given estimator and dynamics model parameters')

    ##################
    # MODEL PARAMETERS
    model_group = parser.add_argument_group('MODEL PARAMETERS')
    model_group.add_argument('-ssm_type', type=str, choices=['blackbox', 'hw', 'hammerstein', 'blocknlin'],
                             default='blocknlin')
    model_group.add_argument('-nx_hidden', type=int, default=5, help='Number of hidden states per output')
    model_group.add_argument('-n_layers', type=int, default=2, help='Number of hidden layers of single time-step state transition')
    model_group.add_argument('-state_estimator', type=str,
                             choices=['rnn', 'mlp', 'linear'], default='rnn')
    model_group.add_argument('-policy', type=str,
                             choices=['rnn', 'mlp', 'linear'], default='mlp')
    model_group.add_argument('-linear_map', type=str, choices=list(linear.maps.keys()),
                             default='linear')
    model_group.add_argument('-nonlinear_map', type=str, default='mlp',
                             choices=['mlp', 'rnn', 'linear', 'residual_mlp'])
    model_group.add_argument('-bias', action='store_true', help='Whether to use bias in the neural network models.')

    ##################
    # Weight PARAMETERS
    weight_group = parser.add_argument_group('WEIGHT PARAMETERS')
    weight_group.add_argument('-Q_con_u', type=float,  default=0.2, help='Hidden input constraints penalty weight.')
    weight_group.add_argument('-Q_con_x', type=float,  default=0.2, help='Hidden state constraints penalty weight.')
    weight_group.add_argument('-Q_dx_ud', type=float,  default=0.0,
                              help='Maximal influence of u and d on hidden state in one time step penalty weight.')
    weight_group.add_argument('-Q_dx', type=float,  default=0.0,
                              help='Penalty weight on hidden state difference in one time step.')
    weight_group.add_argument('-Q_sub', type=float,  default=0.2, help='Linear maps regularization weight.')
    weight_group.add_argument('-Q_y', type=float,  default=1.0, help='Output tracking penalty weight')
    weight_group.add_argument('-Q_e', type=float,  default=1.0, help='State estimator hidden prediction penalty weight')

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
    log_group.add_argument('-logger', choices=['mlflow', 'wandb', 'stdout'],
                           help='Logging setup to use')
    log_group.add_argument('-make_movie', action='store_true',
                           help='Make movie of best trace prediction at end of training')
    log_group.add_argument('-freq', type=int, default=10,
                           help='Frequency to create frames for reference tracking movie.')
    return parser.parse_args()


def arg_setup():
    args = parse_args()
    os.system(f'mkdir {args.savedir}')
    if args.logger == 'wandb':
        import wandb
    if args.logger in ['mlflow', 'wandb']:
        mlflow.set_tracking_uri(args.location)
        mlflow.set_experiment(args.exp)
        mlflow.start_run(run_name=args.run)
        params = {k: str(getattr(args, k)) for k in vars(args) if getattr(args, k)}
        mlflow.log_params(params)
    device = f'cuda:{args.gpu}' if (args.gpu is not None) else 'cpu'
    return args, device


def model_setup(args, device, nx, ny, nu, nd):
    linmap = linear.maps[args.linear_map]
    nonlinmap = {'linear': linmap,
                 'mlp': blocks.MLP,
                 'rnn': blocks.RNN,
                 'residual_mlp': blocks.ResMLP}[args.nonlinear_map]
    # state space model setup
    dynamics_model = {'blackbox': dynamics.blackbox,
                      'blocknlin': dynamics.blocknlin,
                      'hammerstein': dynamics.hammerstein,
                      'hw': dynamics.hw}[args.ssm_type](args, linmap, nonlinmap, nx, nu, nd, ny, args.n_layers)
    # state space model weights
    dynamics_model.Q_dx, dynamics_model.Q_dx_ud, dynamics_model.Q_con_x, dynamics_model.Q_con_u, dynamics_model.Q_sub = \
        args.Q_dx, args.Q_dx_ud, args.Q_con_x, args.Q_con_u, args.Q_sub
    # state estimator setup
    estimator = {'linear': estimators.LinearEstimator,
                 'mlp': estimators.MLPEstimator,
                 'rnn': estimators.RNNEstimator,
                 'kf': estimators.LinearKalmanFilter}[args.state_estimator](ny, nx,
                                                                            bias=args.bias,
                                                                            hsizes=[nx]*args.n_layers,
                                                                            num_layers=2, Linear=linmap,
                                                                            ss_model=dynamics_model)

    model = loops.OpenLoop(model=dynamics_model, estim=estimator, Q_e=args.Q_e).to(device)
    nweights = sum([i.numel() for i in list(model.parameters()) if i.requires_grad])
    if args.logger in ['mlflow', 'wandb']:
        mlflow.log_param('Parameters', nweights)
    return model


def evaluate(args, best_model, dataset, logger, visualizer):

    plt.style.use('classic')
    with torch.no_grad():
        ########################################
        ########## NSTEP TRAIN RESPONSE ########
        ########################################
        model.load_state_dict(best_model)
        all_output = dict()
        for dset, dname in zip([dataset.train_data, dataset.dev_data, dataset.test_data], ['train', 'dev', 'test']):
            output = model.n_step(dset)
            logger.log_metrics(output, args.epochs)
            all_output = {**all_output, **output}
        visualizer.plot(all_output, dataset, best_model)

        ########################################
        ########## OPEN LOOP RESPONSE ##########
        ########################################
        all_output = dict()
        for data, dname in zip([dataset.train_loop, dataset.dev_loop, dataset.test_loop], ['train', 'dev', 'test']):
            output = model.loop_step(data)
            logger.log_metrics(output, args.epochs)
            all_output = {**all_output, **output}
        visualizer.plot(all_output, dataset, best_model)
        plots = visualizer.output()
        logger.log_artifacts(plots)


if __name__ == '__main__':

    ########## DATA AND MODEL SETUP ############
    args, device = arg_setup()
    dataset, nx, ny, nu, nd, norms = dataset.data_setup(args=args, device='cpu')
    model = model_setup(args, device, nx, ny, nu, nd)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    logger = logger.BasicLogger(args.savedir, args.verbosity)
    visualizer = visuals.NoOpVisualizer()

    ########## PROBLEM SOLUTION ############
    prob = Problem(model, dataset, optimizer, logger=logger, visualizer=visualizer, epochs=args.epochs)
    best_model = prob.train()

    ########## MODEL PEFROMANCE ############
    evaluate(args, best_model, dataset, logger, visualizer)
