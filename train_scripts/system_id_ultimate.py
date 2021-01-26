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
    opt_group.add_argument('-epochs', type=int, default=5000)
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
    data_group.add_argument('-system', type=str, default='Reno_ROM40', choices=list(systems.keys()),
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
    model_group.add_argument('-ssm_type', type=str, choices=['blackbox', 'block'],
                             default='blackbox')
    model_group.add_argument('-residual', type=int, choices=[0, 1], default=0,
                             help='Whether to model state space output as residual from previous state')
    model_group.add_argument('-xoe', type=str, choices=[k for k in operators], default='addmul',
                             help='Block aggregation operator for fX and fE')
    model_group.add_argument('-xou', type=str, choices=[k for k in operators], default='addmul',
                             help='Block aggregation operator for fX and fU')
    model_group.add_argument('-xod', type=str, choices=[k for k in operators], default='addmul',
                             help='Block aggregation operator for fX and fD')
    model_group.add_argument('-xmin', type=float, default=-0.2, help='Constraint on minimum state value')
    model_group.add_argument('-xmax', type=float, default=1.2, help='Constraint on maximum state value')
    model_group.add_argument('-dxudmin', type=float, default=-0.05,
                             help='Constraint on contribution of U and D to state')
    model_group.add_argument('-dxudmax', type=float, default=0.05,
                             help='Constraint on contribution of U and D to state')
    model_group.add_argument('-koopman', type=int, default=0,
                             help='Whether to enforce regularization so that fy is inverse of state estimator')

    ##################
    # fxud PARAMETERS
    fxud_group = parser.add_argument_group('fxud PARAMETERS')
    fxud_group.add_argument('-fxud', type=str, default='mlp', choices=[k for k in blocks],
                          help='Main transition dynamics block type.')
    fxud_group.add_argument('-fxud_hidden', type=int, default=20,
                          help='fxud hidden state dimension.')
    fxud_group.add_argument('-fxud_layers', type=int, default=2,
                          help='Number of hidden layers of single time-step state transition')
    fxud_group.add_argument('-fxud_map', type=str, choices=[k for k in maps], default='linear',
                          help='Linear map fxud uses as subcomponents')
    fxud_group.add_argument('-fxud_bias', action='store_true',
                          help='Whether to use bias in the fxud network.')
    fxud_group.add_argument('-fxud_act', choices=[k for k in activations], default='softexp',
                          help='Activation function for fxud network')
    fxud_group.add_argument('-fxud_sigma_min', type=float, default=0.1)
    fxud_group.add_argument('-fxud_sigma_max', type=float, default=1.0)

    ##################
    # FX PARAMETERS
    fx_group = parser.add_argument_group('FX PARAMETERS')
    fx_group.add_argument('-fx', type=str, default='mlp', choices=[k for k in blocks],
                          help='Main transition dynamics block type.')
    fx_group.add_argument('-fx_hidden', type=int, default=20, 
                          help='fx hidden state dimension.')
    fx_group.add_argument('-fx_layers', type=int, default=2,
                          help='Number of hidden layers of single time-step state transition')
    fx_group.add_argument('-fx_map', type=str, choices=[k for k in maps], default='linear',
                          help='Linear map fx uses as subcomponents')
    fx_group.add_argument('-fx_bias', action='store_true', 
                          help='Whether to use bias in the fx network.')
    fx_group.add_argument('-fx_act', choices=[k for k in activations], default='softexp',
                          help='Activation function for fx network')
    fx_group.add_argument('-fx_sigma_min', type=float, default=0.1)
    fx_group.add_argument('-fx_sigma_max', type=float, default=1.0)

    ##################
    # FU PARAMETERS
    fu_group = parser.add_argument_group('fu PARAMETERS')
    fu_group.add_argument('-fu', type=str, default='mlp', choices=[k for k in blocks],
                          help='Main transition dynamics block type.')
    fu_group.add_argument('-fu_hidden', type=int, default=20,
                          help='fu hidden state dimension.')
    fu_group.add_argument('-fu_layers', type=int, default=2,
                          help='Number of hidden layers of single time-step state transition')
    fu_group.add_argument('-fu_map', type=str, choices=[k for k in maps], default='linear',
                          help='Linear map fu uses as subcomponents')
    fu_group.add_argument('-fu_bias', action='store_true',
                          help='Whether to use bias in the fu network.')
    fu_group.add_argument('-fu_act', choices=[k for k in activations], default='softexp',
                          help='Activation function for fu network')
    fu_group.add_argument('-fu_sigma_min', type=float, default=0.1)
    fu_group.add_argument('-fu_sigma_max', type=float, default=1.0)

    ##################
    # fd PARAMETERS
    fd_group = parser.add_argument_group('fd PARAMETERS')
    fd_group.add_argument('-fd', type=str, default='mlp', choices=[k for k in blocks],
                          help='Main transition dynamics block type.')
    fd_group.add_argument('-fd_hidden', type=int, default=20,
                          help='fd hidden state dimension.')
    fd_group.add_argument('-fd_layers', type=int, default=2,
                          help='Number of hidden layers for fd')
    fd_group.add_argument('-fd_map', type=str, choices=[k for k in maps], default='linearor',
                          help='Linear map fd uses as subcomponents')
    fd_group.add_argument('-fd_bias', action='store_true',
                          help='Whether to use bias in the fd network.')
    fd_group.add_argument('-fd_act', choices=[k for k in activations], default='softexp',
                          help='Activation fdnction for fd network')
    fd_group.add_argument('-fd_sigma_min', type=float, default=0.1)
    fd_group.add_argument('-fd_sigma_max', type=float, default=1.0)

    ##################
    # fe PARAMETERS
    fe_group = parser.add_argument_group('fe PARAMETERS')
    fe_group.add_argument('-fe', type=str, default=None, choices=[k for k in blocks],
                          help='Main transition dynamics block type.')
    fe_group.add_argument('-fe_hidden', type=int, default=20,
                          help='fe hidden state dimension.')
    fe_group.add_argument('-fe_layers', type=int, default=2,
                          help='Number of hidden layers for fe')
    fe_group.add_argument('-fe_map', type=str, choices=[k for k in maps], default='linear',
                          help='Linear map fe uses as subcomponents')
    fe_group.add_argument('-fe_bias', action='store_true',
                          help='Whether to use bias in the fe network.')
    fe_group.add_argument('-fe_act', choices=[k for k in activations], default='softexp',
                          help='Activation function for fe network')
    fe_group.add_argument('-fe_sigma_min', type=float, default=0.1)
    fe_group.add_argument('-fe_sigma_max', type=float, default=1.0)

    ##################
    # fy PARAMETERS
    fy_group = parser.add_argument_group('fy PARAMETERS')
    fy_group.add_argument('-fy', type=str, default='mlp', choices=[k for k in blocks],
                          help='Main transition dynamics block type.')
    fy_group.add_argument('-fy_hidden', type=int, default=20,
                          help='fy hidden state dimension.')
    fy_group.add_argument('-fy_layers', type=int, default=2,
                          help='Number of hidden layers for fy')
    fy_group.add_argument('-fy_map', type=str, choices=[k for k in maps], default='linear',
                          help='Linear map fy uses as subcomponents')
    fy_group.add_argument('-fy_bias', action='store_true',
                          help='Whether to use bias in the fy network.')
    fy_group.add_argument('-fy_act', choices=[k for k in activations], default='softexp',
                          help='Activation function for fy network')
    fy_group.add_argument('-fy_sigma_min', type=float, default=0.1)
    fy_group.add_argument('-fy_sigma_max', type=float, default=1.0)

    ##################
    # STATE ESTIMATOR PARAMETERS
    est_group = parser.add_argument_group('STATE ESTIMATOR PARAMETERS')
    est_group.add_argument('-est', type=str,
                            choices=[k for k in estimators], default='mlp')
    est_group.add_argument('-est_keys', nargs='+', default=['Yp'],
                           help='Keys defining input to the state estimator.')
    est_group.add_argument('-est_input_window', type=int, default=1,
                            help="Number of previous time steps measurements to include in state estimator input")
    est_group.add_argument('-est_hidden', type=int, default=20,
                           help='estimator hidden state dimension.')
    est_group.add_argument('-est_layers', type=int, default=2,
                           help='Number of hidden layers for state estimator network')
    est_group.add_argument('-est_map', type=str, choices=[k for k in maps], default='linear',
                           help='Linear map state estimator uses as subcomponents')
    est_group.add_argument('-est_bias', action='store_true',
                           help='Whether to use bias in the state estimator network.')
    est_group.add_argument('-est_act', choices=[k for k in activations], default='softexp',
                           help='Activation function for state estimator network')
    est_group.add_argument('-est_sigma_min', type=float, default=0.1)
    est_group.add_argument('-est_sigma_max', type=float, default=1.0)

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


def dataset_load(args, device):
    if systems[args.system] == 'emulator':
        dataset = EmulatorDataset(system=args.system, nsim=args.nsim, batch_type=args.batch_type,
                                  norm=args.norm, nsteps=args.nsteps, device=device, savedir=args.savedir)
    else:
        dataset = FileDataset(system=args.system, nsim=args.nsim, batch_type=args.batch_type,
                              norm=args.norm, nsteps=args.nsteps, device=device, savedir=args.savedir)
    return dataset


def get_components(args, dataset):
    nx = dataset.dims['Y'][-1] * args.fx_hidden
    ny, nu, nd = [dataset.dims[k][-1] if k in dataset.dims else 0 for k in ['Yf', 'Uf', 'Df']]
    fd, fu, fe = None, None, None
    estimator = estimators[args.est]({**dataset.dims, 'x0': (nx,)},
                                     nsteps=args.nsteps,
                                     window_size=args.est_input_window,
                                     bias=args.est_bias,
                                     Linear=maps[args.est_map],
                                     nonlin=activations[args.est_act],
                                     hsizes=[args.est_hidden] * args.est_layers,
                                     input_keys=args.est_keys,
                                     linargs={'sigma_min': args.est_sigma_min, 'sigma_max': args.est_sigma_max},
                                     name='estim')
    fy = blocks[args.fy](nx, ny,
                         hsizes=[args.fy_hidden] * args.fy_layers,
                         bias=args.fy_bias,
                         Linear=maps[args.fy_map],
                         nonlin=activations[args.fy_act],
                         linargs={'sigma_min': args.fy_sigma_min, 'sigma_max': args.fy_sigma_max})
    if args.fe is not None:
        fe = blocks[args.fe](nx, nx,
                             hsizes=[args.fe_hidden] * args.fe_layers,
                             bias=args.fe_bias,
                             Linear=maps[args.fe_map],
                             nonlin=activations[args.fe_act],
                             linargs={'sigma_min': args.fe_sigma_min, 'sigma_max': args.fe_sigma_max})
    dynamics_key_updates = {'x0': f'x0_{estimator.name}'}

    if args.ssm_type == 'blackbox':
        fxud = blocks[args.fxud](nx + nu + nd, nx,
                                 hsizes=[args.fu_hidden] * args.fu_layers,
                                 bias=args.fu_bias,
                                 Linear=maps[args.fu_map],
                                 nonlin=activations[args.fu_act],
                                 linargs={'sigma_min': args.fu_sigma_min, 'sigma_max': args.fu_sigma_max})
        dynamics_model = BlackSSM(fxud, fy, fe=fe, xoe=operators[args.xoe],
                                  name='dynamics', residual=args.residual, input_keys=dynamics_key_updates)
    elif args.ssm_type == 'block':

        fx = blocks[args.fxud](nx, nx,
                               hsizes=[nx] * args.fx_layers,
                               bias=args.fx_bias,
                               Linear=maps[args.fx_map],
                               nonlin=activations[args.fx_act],
                               linargs={'sigma_min': args.fx_sigma_min, 'sigma_max': args.fx_sigma_max})
        if nd:
            fd = blocks[args.fd](nd, nx,
                                 hsizes=[args.fd_hidden] * args.fd_layers,
                                 bias=args.fd_bias,
                                 Linear=maps[args.fd_map],
                                 nonlin=activations[args.fd_act],
                                 linargs={'sigma_min': args.fd_sigma_min, 'sigma_max': args.fd_sigma_max})
        if nu:
            fu = blocks[args.fu](nu, nx,
                                 hsizes=[args.fu_hidden] * args.fu_layers,
                                 bias=args.fu_bias,
                                 Linear=maps[args.fu_map],
                                 nonlin=activations[args.fu_act],
                                 linargs={'sigma_min': args.fu_sigma_min, 'sigma_max': args.fu_sigma_max})
        dynamics_model = BlockSSM(fx, fy, fu=fu, fd=fd, fe=fe,
                                  xou=operators[args.xou],
                                  xod=operators[args.xod],
                                  xoe=operators[args.xoe],
                                  residual=args.residual, name='dynamics',
                                  input_keys=dynamics_key_updates)
    return estimator, dynamics_model


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
    print(dataset.dims)
    estimator, dynamics_model = get_components(args, dataset)
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
