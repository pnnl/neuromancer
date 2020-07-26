"""
Script for solving multiparametric programming problem
"""
import argparse
import torch
from datasets import DatasetMPP
import linear
import logger
import policies
from visuals import VisualizerMPP
from trainer import TrainerMPP
from problem import Problem, Objective
import torch.nn.functional as F
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default=None,
                        help="Gpu to use")
    # OPTIMIZATION PARAMETERS
    opt_group = parser.add_argument_group('OPTIMIZATION PARAMETERS')
    opt_group.add_argument('-epochs', type=int, default=1000)
    opt_group.add_argument('-lr', type=float, default=0.001,
                           help='Step size for gradient descent.')
    ##################
    # MODEL PARAMETERS
    model_group = parser.add_argument_group('MODEL PARAMETERS')
    model_group.add_argument('-nx_hidden', type=int, default=20, help='Number of hidden states per output')
    model_group.add_argument('-n_layers', type=int, default=2, help='Number of hidden layers of single time-step state transition')
    model_group.add_argument('-linear_map', type=str, choices=list(linear.maps.keys()),
                             default='softSVD')
    model_group.add_argument('-bias', action='store_true', help='Whether to use bias in the neural network models.')
    ##################
    # Weight PARAMETERS
    weight_group = parser.add_argument_group('WEIGHT PARAMETERS')
    weight_group.add_argument('-Q_con_z', type=float,  default=1.0, help='ineqiality penalty weight.')
    weight_group.add_argument('-Q_sub', type=float,  default=0.2, help='Linear maps regularization weight.')
    weight_group.add_argument('-Q_z', type=float,  default=1.0, help='Quadratic loss weight')
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

if __name__ == '__main__':
    ###############################
    ########## LOGGING ############
    ###############################
    args = parse_args()
    logger, device = logging(args)

    ###############################
    ########## DATA ###############
    ###############################
    # TODO: implement these examples
    # https: // yalmip.github.io / tutorial / multiparametricprogramming /
    # https: // www.mpt3.org / ParOpt / ParOpt

    nsim = 5000
    nz = 3   # number of optimized variables
    nx = 2   # number of parameters
    A = torch.randn(15, nz)
    b = torch.randn(15, 1)
    E = torch.randn(15, nx)
    data_sequences = {'X': np.random.uniform(-1, 1, [nsim, nx]), 'Z_ref': np.ones([nsim, nz])}
    # TODO: adjust dataset to allow creating empty dataset for arbitrary problem
    #  for static problem we don't need future and past
    dataset = DatasetMPP(sequences=data_sequences, device=device)
    dataset.add_variable({'Z': nz})

    ##########################################
    ########## PROBLEM COMPONENTS ############
    ##########################################
    linmap = linear.maps[args.linear_map]
    # TODO: extend solution map to take nonlinmap as an argument
    solution_map = policies.SolutionMap(dataset.dims,
                             bias=args.bias,
                             Linear=linmap,
                             nonlin=F.gelu,
                             hsizes=[args.nx_hidden] * args.n_layers,
                             input_keys=['X'],
                             output_keys=['Z'],
                             linargs=dict(),
                             name='sol_map')
    components = [solution_map]
    # component variables
    input_keys = list(set.union(*[set(comp.input_keys) for comp in components]))
    output_keys = list(set.union(*[set(comp.output_keys) for comp in components]))
    dataset_keys = list(set(dataset.train_data.keys()))
    plot_keys = ['X', 'Z']   # variables to be plotted

    ##########################################
    ########## MULTI-OBJECTIVE LOSS ##########
    ##########################################
    # min_W ||z-z_ref||^2
    # s.t:  A * z - b + E * x <= 0
    #       z = f_W(x)
    regularization = Objective(['sol_map_reg_error'], lambda reg1: reg1, weight=args.Q_sub)
    quadratic_loss = Objective(['Z', 'Z_ref'], F.mse_loss, weight=args.Q_z)
    ineq_constraint = Objective(['Z', 'X'], lambda z, x: torch.mean(F.relu(torch.mm(A, z.T)-b + torch.mm(E, x.T) - 0)),
                                weight=args.Q_con_z)

    objectives = [regularization, quadratic_loss]
    constraints = [ineq_constraint]

    ##########################################
    ########## OPTIMIZE SOLUTION ############
    ##########################################
    model = Problem(objectives, constraints, components).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    visualizer = VisualizerMPP(dataset, model, args.verbosity, args.savedir)
    trainer = TrainerMPP(model, dataset, optimizer, logger=logger, visualizer=visualizer, epochs=args.epochs)
    best_model, best_model_full = trainer.train()
    trainer.evaluate(best_model)
    logger.clean_up()

    # test model solution Z for particular values of parameters X
    dataset.test_data['X'] = torch.tensor([[0.1, 0.2]])
    dataset.test_data['Z_ref'] = torch.ones([1, nz])
    output = best_model_full(dataset.test_data)
    print(output['test_Z'])


# TODO: verify concrete problem against MPT3 solution
# https://www.mpt3.org/ParOpt/ParOpt
# https://yalmip.github.io/tutorial/multiparametricprogramming/