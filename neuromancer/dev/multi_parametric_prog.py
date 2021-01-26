"""
Script for solving multiparametric programming problem

    # TODO: implement these examples
    # https: // yalmip.github.io / tutorial / multiparametricprogramming /
    # https: // www.mpt3.org / ParOpt / ParOpt

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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


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
    model_group.add_argument('-nx_hidden', type=int, default=10, help='Number of hidden states per output')
    model_group.add_argument('-n_layers', type=int, default=2, help='Number of hidden layers of single time-step state transition')
    model_group.add_argument('-linear_map', type=str, choices=list(linear.maps.keys()),
                             default='softSVD')
    model_group.add_argument('-bias', action='store_true', help='Whether to use bias in the neural network models.')
    ##################
    # Weight PARAMETERS
    weight_group = parser.add_argument_group('WEIGHT PARAMETERS')
    weight_group.add_argument('-Q_con_z', type=float,  default=20.0, help='ineqiality penalty weight.')
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
    # problem for verification
    A = torch.tensor([[1.2744, 1.1040, -0.4325],
                  [-2.1320, 0.1314, 0.7086],
                  [-1.5300, -0.0948, 0.5043],
                  [-0.7049, 0.7533, 1.0351],
                  [1.4853, -1.7566, -1.4814],
                  [-0.0624, -0.1863, 0.2591],
                  [-0.9235, -2.4554, 0.3520],
                  [0.8160, -0.2640, -0.4121],
                  [0.4489, 0.9223, 0.3343],
                  [-0.4656, 0.0921, -2.3878],
                  [0.3906, -1.7249, -0.7722],
                  [-0.3648, 0.0905, -0.6322],
                  [0.1259, 0.7538, 1.2620],
                  [-0.1345, -0.0701, 1.1517],
                  [0.1549, 0.7001, 0.2606]])

    b = torch.tensor([[ 0.1682],
        [0.6524],
        [0.7404],
        [0.3220],
        [ 0.4458],
        [0.7967],
        [ 0.2437],
        [0.2755],
        [ 0.7987],
        [0.4895],
        [ 0.3452],
        [ 0.0668],
        [ 0.2678],
        [0.1654],
        [0.4187]])

    E = torch.tensor([[ 0.3778,  0.1215],
        [ 0.8601,  2.0487],
        [ 0.4992, -1.6091],
        [-0.1726,  0.3682],
        [-0.2376,  0.0034],
        [-2.5233, -1.1872],
        [-0.2398, -1.6971],
        [-0.0191, -1.2109],
        [-0.6369,  0.9359],
        [-0.8605,  0.1141],
        [ 0.0110,  1.7910],
        [-0.4991, -0.5109],
        [-0.9192,  1.0788],
        [ 1.3376,  1.3809],
        [ 0.5816,  2.0210]])
    nsim = 100000
    nz = 3   # number of optimized variables
    nx = 2   # number of parameters
    # A = torch.randn(15, nz)
    # b = torch.rand(15, 1)
    # E = torch.randn(15, nx)
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
                             nonlin=F.relu,
                             hsizes=[args.nx_hidden*nx] * args.n_layers,
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


    # plot solution surface
    dataset.test_data['Z_ref'] = torch.ones([1, nz])
    neval = 100
    X = np.linspace(-1, 1, neval)
    Y = np.linspace(-1, 1, neval)
    Xplt, Xplt = np.meshgrid(X, Y)
    Zplt = np.zeros([neval,neval])
    for i in range(neval):
        for j in range(neval):
            dataset.test_data['X'] = torch.tensor([[X[i], Y[j]]])
            output = best_model_full(dataset.test_data)
            Zplt[i,j] = output['test_Z'][:,0].detach().numpy()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(Xplt, Xplt, Zplt, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='z', labelsize=16)
    ax.set_title('solution map', fontsize=22)
    ax.set_xlabel('x_1', fontsize=22)
    ax.set_ylabel('x_2', fontsize=22)
    ax.set_zlabel('z_1', fontsize=22)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()


    # test model solution Z for particular values of parameters X
    dataset.test_data['X'] = torch.tensor([[0.1, 0.2]])
    dataset.test_data['Z_ref'] = torch.ones([1, nz])
    output = best_model_full(dataset.test_data)
    print(output['test_Z'])
    # constraints violations eval
    constr_residual = torch.mean(F.relu(torch.mm(A, output['test_Z'].T) - b + torch.mm(E, dataset.test_data['X'].T) - 0))
    print(constr_residual)

# TODO: verify concrete problem against MPT3 solution
# https://www.mpt3.org/ParOpt/ParOpt
# https://yalmip.github.io/tutorial/multiparametricprogramming/