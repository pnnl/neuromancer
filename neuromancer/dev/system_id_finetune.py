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
import torch
import torch.nn.functional as F
import torch.nn as nn

# code ecosystem imports
from slim import maps

# local imports
from neuromancer.datasets import EmulatorDataset, FileDataset, systems
from neuromancer.dynamics import BlockSSM, BlackSSM
import neuromancer.loggers as loggers
from neuromancer.visuals import VisualizerOpen, VisualizerTrajectories
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem, Objective
from neuromancer.activations import activations
from neuromancer.blocks import blocks
from neuromancer.estimators import estimators
from neuromancer.simulators import OpenLoopSimulator
from neuromancer.operators import operators

""" python system_id.py -system flexy_air -epochs 10 -nx_hidden 2
0 -ssm_type blackbox -state_estimator mlp -nonlinear_map residual_mlp -n_layers 2 -nsim 10000 -nsteps 32 -lr 0.001
Namespace(Q_con_fdu=0.0, Q_con_x=1.0, Q_dx=0.2, Q_e=1.0, Q_sub=0.2, Q_y=1.0, activation='gelu', bias=False, epochs=10, exp='test', gpu=None, linear_map='linear', location='mlrun
s', logger='stdout', lr=0.001, n_layers=2, nonlinear_map='residual_mlp', norm=['U', 'D', 'Y'], nsim=10000, nsteps=32
"""


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=int, default=None,
                        help="Gpu to use")
    # OPTIMIZATION PARAMETERS
    opt_group = parser.add_argument_group('OPTIMIZATION PARAMETERS')
    opt_group.add_argument('-epochs', type=int, default=0)
    opt_group.add_argument('-lr', type=float, default=0.001,
                           help='Step size for gradient descent.')
    opt_group.add_argument('-eval_metric', type=str, default='loop_dev_loss',
                           help='Metric for model selection and early stopping.')
    opt_group.add_argument('-patience', type=int, default=5,
                           help='How many epochs to allow for no improvement in eval metric before early stopping.')
    opt_group.add_argument('-warmup', type=int, default=0,
                           help='Number of epochs to wait before enacting early stopping policy.')
    opt_group.add_argument('-skip_eval_sim', action='store_true',
                           help='Whether to run simulator during evaluation phase of training.')

    #################
    # DATA PARAMETERS
    data_group = parser.add_argument_group('DATA PARAMETERS')
    data_group.add_argument('-nsteps', type=int, default=32,
                            help='Number of steps for open loop during training.')
    data_group.add_argument('-system', type=str, default='flexy_air', choices=list(systems.keys()),
                            help='select particular dataset with keyword')
    data_group.add_argument('-nsim', type=int, default=10000,
                            help='Number of time steps for full dataset. (ntrain + ndev + ntest)'
                                 'train, dev, and test will be split evenly from contiguous, sequential, '
                                 'non-overlapping chunks of nsim datapoints, e.g. first nsim/3 art train,'
                                 'next nsim/3 are dev and next nsim/3 simulation steps are test points.'
                                 'None will use a default nsim from the selected dataset or emulator')
    data_group.add_argument('-norm', nargs='+', default=['U', 'D', 'Y'], choices=['U', 'D', 'Y'],
                            help='List of sequences to max-min normalize')
    data_group.add_argument('-batch_type', default='batch', choices=['mh', 'batch'], help='option for creating batches of time series data')
    
    ##################
    # MODEL PARAMETERS
    model_group = parser.add_argument_group('MODEL PARAMETERS')
    model_group.add_argument('-model_file', type=str, default='../datasets/Flexy_air/best_model_flexy1.pth')
    model_group.add_argument('-ssm_type', type=str, choices=['blackbox', 'block'],
                             default='blackbox')
    model_group.add_argument('-xmin', type=float, default=-0.2, help='Constraint on minimum state value')
    model_group.add_argument('-xmax', type=float, default=1.2, help='Constraint on maximum state value')
    model_group.add_argument('-dxudmin', type=float, default=-0.05,
                             help='Constraint on contribution of U and D to state')
    model_group.add_argument('-dxudmax', type=float, default=0.05,
                             help='Constraint on contribution of U and D to state')
    model_group.add_argument('-koopman', type=int, default=0,
                             help='Whether to enforce regularization so that fy is inverse of state estimator')

    ##################
    # Weight PARAMETERS
    weight_group = parser.add_argument_group('WEIGHT PARAMETERS')
    weight_group.add_argument('-Q_con_x', type=float,  default=1.0, help='Hidden state constraints penalty weight.')
    weight_group.add_argument('-Q_dx', type=float,  default=0.2,
                              help='Penalty weight on hidden state difference in one time step.')
    weight_group.add_argument('-Q_sub', type=float,  default=0.2, help='Linear maps regularization weight.')
    weight_group.add_argument('-Q_y', type=float,  default=1.0, help='Output tracking penalty weight')
    weight_group.add_argument('-Q_e', type=float,  default=1.0, help='State estimator hidden prediction penalty weight')
    weight_group.add_argument('-Q_con_fdu', type=float,  default=0.0, help='Penalty weight on control actions and disturbances.')

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
    log_group.add_argument('-logger', type=str, choices=['mlflow', 'stdout'], default='stdout',
                           help='Logging setup to use')
    log_group.add_argument('-train_visuals', action='store_true',
                           help='Whether to create visuals, e.g. animations during training loop')
    log_group.add_argument('-trace_movie', action='store_true',
                           help='Whether to plot an animation of the simulated and true dynamics')
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


# def dataset_load(args, device):
#     if systems[args.system] == 'emulator':
#         dataset = EmulatorDataset(system=args.system, nsim=args.nsim, batch_type=args.batch_type,
#                                   norm=args.norm, nsteps=args.nsteps, device=device, savedir=args.savedir)
#     else:
#         dataset = FileDataset(system=args.system, nsim=args.nsim, batch_type=args.batch_type,
#                               norm=args.norm, nsteps=args.nsteps, device=device, savedir=args.savedir)
#     return dataset

def dataset_load(args, device):
    if systems[args.system] == 'emulator':
        dataset = EmulatorDataset(system=args.system, nsim=args.nsim,
                                  norm=args.norm, nsteps=args.nsteps, device=device, savedir=args.savedir)
    else:
        dataset = FileDataset(system=args.system, nsim=args.nsim,
                              norm=args.norm, nsteps=args.nsteps, device=device, savedir=args.savedir)
        new_sequences = {'Y': dataset.data['Y'][:, :1]}
        dataset.min_max_norms['Ymin'] = dataset.min_max_norms['Ymin'][0]
        dataset.min_max_norms['Ymax'] = dataset.min_max_norms['Ymax'][0]
        dataset.add_data(new_sequences, overwrite=True)
    return dataset


def get_loss(args, dataset, components):
    estimator, dynamics = components
    estimator_loss = Objective([f'X_pred_{dynamics.name}', f'x0_{estimator.name}'],
                               lambda X_pred, x0: F.mse_loss(X_pred[-1, :-1, :], x0[1:]),
                               weight=args.Q_e, name='arrival_cost')
    regularization = Objective([f'reg_error_{estimator.name}', f'reg_error_{dynamics.name}'],
                               lambda reg1, reg2: reg1 + reg2, weight=args.Q_sub, name='reg_error')
    reference_loss = Objective([f'Y_pred_{dynamics.name}', 'Yf'], F.mse_loss, weight=args.Q_y,
                               name='ref_loss')
    state_smoothing = Objective([f'X_pred_{dynamics.name}'], lambda x: F.mse_loss(x[1:], x[:-1]), weight=args.Q_dx,
                                name='state_smoothing')
    observation_lower_bound_penalty = Objective([f'Y_pred_{dynamics.name}'],
                                                lambda x: torch.mean(F.relu(-x + args.xmin)), weight=args.Q_con_x,
                                                name='y_low_bound_error')
    observation_upper_bound_penalty = Objective([f'Y_pred_{dynamics.name}'],
                                                lambda x: torch.mean(F.relu(x - args.xmax)), weight=args.Q_con_x,
                                                name='y_up_bound_error')

    objectives = [regularization, reference_loss, estimator_loss]
    constraints = [state_smoothing, observation_lower_bound_penalty, observation_upper_bound_penalty]

    if args.ssm_type != 'blackbox':
        if 'U' in dataset.data:
            inputs_max_influence_lb = Objective([f'fU_{dynamics.name}'], lambda x: torch.mean(F.relu(-x + args.dxudmin)),
                                                weight=args.Q_con_fdu,
                                                name='input_influence_lb')
            inputs_max_influence_ub = Objective([f'fU_{dynamics.name}'], lambda x: torch.mean(F.relu(x - args.dxudmax)),
                                                weight=args.Q_con_fdu, name='input_influence_ub')
            constraints += [inputs_max_influence_lb, inputs_max_influence_ub]
        if 'D' in dataset.data:
            disturbances_max_influence_lb = Objective([f'fD_{dynamics.name}'], lambda x: torch.mean(F.relu(-x + args.dxudmin)),
                                                      weight=args.Q_con_fdu, name='dist_influence_lb')
            disturbances_max_influence_ub = Objective([f'fD_{dynamics.name}'], lambda x: torch.mean(F.relu(x - args.dxudmax)),
                                                      weight=args.Q_con_fdu, name='dist_influence_ub')
            constraints += [disturbances_max_influence_lb, disturbances_max_influence_ub]
    return objectives, constraints


class Decoder(nn.Module):
    """
    Implements the component interface
    """
    def __init__(self, fy):
        super().__init__()
        self.fy = fy

    def forward(self, data):
        return {'yhat': self.fy(data['x0_estim'])}


if __name__ == '__main__':
    ###############################
    ########## LOGGING ############
    ###############################
    args = parse().parse_args()
    print({k: str(getattr(args, k)) for k in vars(args) if getattr(args, k)})
    logger, device = logging(args)

    ###############################
    ########## DATA ###############
    ###############################
    dataset = dataset_load(args, device)

    ##########################################
    ########## PROBLEM COMPONENTS ############
    ##########################################
    problem = torch.load(args.model_file, pickle_module=dill)
    estimator, dynamics_model = problem.components
    components = [estimator, dynamics_model]

    ##########################################
    ########## MULTI-OBJECTIVE LOSS ##########
    ##########################################
    objectives, constraints = get_loss(args, dataset, components)
    if args.koopman:
        components.append(Decoder(dynamics_model.fy))
        autoencoder_loss = Objective(['Yp', 'yhat'], lambda Y, yhat: F.mse_loss(Y[-1], yhat), name='inverse')
        objectives.append(autoencoder_loss)
    ##########################################
    ########## OPTIMIZE SOLUTION ############
    ##########################################
    model = Problem(objectives, constraints, components).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    visualizer = VisualizerOpen(dataset, dynamics_model, args.verbosity, args.savedir,
                                training_visuals=args.train_visuals, trace_movie=args.trace_movie)
    simulator = OpenLoopSimulator(model=model, dataset=dataset, eval_sim=not args.skip_eval_sim)
    trainer = Trainer(model, dataset, optimizer, logger=logger, visualizer=visualizer,
                      simulator=simulator, epochs=args.epochs, eval_metric=args.eval_metric,
                      patience=args.patience, warmup=args.warmup)
    best_model = trainer.train()
    trainer.evaluate(best_model)
    logger.clean_up()
