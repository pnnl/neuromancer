"""
Tutorial script for formulating and solving multiparametric linear programs
using Neuromancer's data-driven differentiable parametric optimization (D3PO) approach

mpLP problem formulation:
    J(\Theta) = min_x c*x
                s.t. Ax \le b + E*\Theta
                      \Theta \in F

    x = decision variable
    A, b, E = fixed problem parameters
    F = set of admissible parameters \Theta

The objective is to find an explicit optimizer (solution map) x = h(\Theta) that optimizes the value function J(\Theta)

We consider a tutorial mpLP problem with benchmark solution using Yalmip and MPT3 toolbox in Matlab
    https://yalmip.github.io/tutorial/multiparametricprogramming/

Further reading:
    Parametric programming
        https://en.wikipedia.org/wiki/Parametric_programming
        https://www.frontiersin.org/articles/10.3389/fceng.2020.620168/full
        http://www.optimization-online.org/DB_FILE/2018/04/6587.pdf
    Parametric linear programming
        https://scholar.rose-hulman.edu/cgi/viewcontent.cgi?article=1022&context=math_mstr
        https://deepblue.lib.umich.edu/bitstream/handle/2027.42/47907/10107_2005_Article_BF01581642.pdf?sequence=1
    Multiparametric (MPT3) toolbox - solving parametric programming problems in Matlab
        https://ieeexplore.ieee.org/abstract/document/6669862


    TODO: create method for dataset name = 'optim' for static optimization problems
    # TODO: custom class for static solution maps: constructed via component class wrapper on blocks
    # solution_map = sol_map(linmap,
    #         nonlinmap,
    #         bias=args.bias,
    #         n_layers=args.n_layers,
    #         activation=activation,
    #         input_keys={'theta': 'thetap'},
    #         name='sol_map')

"""

import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import cm

import torch
import slim
from neuromancer import blocks
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem, Objective
import neuromancer.arg as arg
from neuromancer.datasets import Dataset
from neuromancer.constraint import Variable
from neuromancer.activations import activations
from neuromancer import policies
from neuromancer.loggers import BasicLogger


def arg_mpLP_problem(prefix=''):
    """
    Command line parser for mpLP problem definition arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers
                         are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = arg.ArgParser(prefix=prefix, add_help=False)
    gp = parser.group("mpLP")
    gp.add("-Q", type=float, default=1.0,
           help="loss function weight.")  # tuned value: 1.0
    gp.add("-Q_sub", type=float, default=0.0,
           help="regularization weight.")
    gp.add("-Q_con", type=float, default=50.0,
           help="constraints penalty weight.")  # tuned value: 50.0
    gp.add("-nx_hidden", type=int, default=20,
           help="Number of hidden states of the solution map")
    gp.add("-n_layers", type=int, default=2,
           help="Number of hidden layers of the solution map")
    gp.add("-bias", action="store_true",
           help="Whether to use bias in the neural network block component models.")
    gp.add("-data_seed", type=int, default=1337,
           help="Random seed used for simulated data")
    gp.add("-epochs", type=int, default=200,
           help='Number of training epochs')
    gp.add("-lr", type=float, default=0.01,
           help="Step size for gradient descent.")
    gp.add("-patience", type=int, default=100,
           help="How many epochs to allow for no improvement in eval metric before early stopping.")
    gp.add("-warmup", type=int, default=100,
           help="Number of epochs to wait before enacting early stopping policy.")
    return parser


def plot_loss(model, dataset, xmin=-2, xmax=2, save_path=None):
    """
    plots loss function for problem with 2 parameters
    :param model:
    :param dataset:
    :param xmin:
    :param xmax:
    :param save_path:
    :return:
    """
    x = torch.arange(xmin, xmax, 0.1)
    y = torch.arange(xmin, xmax, 0.1)
    xx, yy = torch.meshgrid(x, y)
    dataset_plt = copy.deepcopy(dataset)
    dataset_plt.dims['nsim'] = 1
    Loss = np.ones([x.shape[0], y.shape[0]]) * np.nan

    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            # check loss
            X = torch.stack([x[[i]], y[[j]]]).reshape(1, 1, -1)
            if dataset.nsteps == 1:
                dataset_plt.train_data['thetap'] = X
                step = model(dataset_plt.train_data)
                Loss[i, j] = step['nstep_train_loss'].detach().numpy()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx.detach().numpy(), yy.detach().numpy(), Loss,
                           cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    ax.set(ylabel='$x_1$')
    ax.set(xlabel='$x_2$')
    ax.set(zlabel='$L$')
    ax.set(title='Loss landscape')
    # plt.colorbar(surf)
    if save_path is not None:
        plt.savefig(save_path + '/loss.pdf')


def plot_solution(model, xmin=-2, xmax=2, save_path=None):
    """
    plots solution landscape for problem with 2 parameters and 1 decision variable
    :param net:
    :param xmin:
    :param xmax:
    :param save_path:
    :return:
    """
    x = torch.arange(xmin, xmax, 0.1)
    y = torch.arange(xmin, xmax, 0.1)
    xx, yy = torch.meshgrid(x, y)
    features = torch.stack([xx, yy]).transpose(0, 2)
    uu = model.net(features)
    plot_u = uu.detach().numpy()[:, :, 0]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx.detach().numpy(), yy.detach().numpy(), plot_u,
                           cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    ax.set(ylabel='$x_1$')
    ax.set(xlabel='$x_2$')
    ax.set(zlabel='$u$')
    ax.set(title='Solution landscape')
    if save_path is not None:
        plt.savefig(save_path + '/solution.pdf')


if __name__ == "__main__":
    """
    # # #  optimization problem hyperparameters
    """
    parser = arg.ArgParser(parents=[arg.log(),
                                    arg_mpLP_problem()])
    args, grps = parser.parse_arg_groups()
    args.bias = True
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"

    np.random.seed(args.data_seed)

    """
    # # #  LP problem matrices
    """
    n_x = 1         # number of decision variables
    n_theta = 2     # number of parameters
    n_con = 5       # number of constraints

    # fixed problem parameters
    A = torch.randn(n_con, n_x)
    b = torch.randn(n_con)
    E = torch.randn(n_con, n_theta)
    c = torch.randn(1, n_x)

    """
    # # #  Dataset 
    """
    #  randomly sampled parameters theta generating superset of:
    #  theta_samples.min() <= theta <= theta_samples.max()
    nsim = 50000  # number of datapoints: increase sample density for more robust results
    sequences = {"theta": 0.5 * np.random.randn(nsim, n_theta)}
    dataset = Dataset(nsim=nsim, device='cpu', sequences=sequences, name='openloop')

    # # # TODO manual fix for static datasets
    # TODO: do we need to have datasets specific to static optimization?
    #  to support batches, we don't need to have loop datasets
    dataset.train_data['thetap'] = dataset.train_data['thetap'][0, :, :]
    dataset.dev_data['thetap'] = dataset.dev_data['thetap'][0, :, :]
    dataset.test_data['thetap'] = dataset.test_data['thetap'][0, :, :]
    dataset.train_loop['thetap'] = dataset.train_loop['thetap'][:, 0, :]
    dataset.dev_loop['thetap'] = dataset.dev_loop['thetap'][:, 0, :]
    dataset.test_loop['thetap'] = dataset.test_loop['thetap'][:, 0, :]

    """
    # # #  mpLP problem formulation in Neuromancer
    """
    # define solution map as MLP policy
    dataset.dims['U'] = (nsim, n_x)  # defining expected dimensions of the solution variable: internal policy key 'U'
    activation = activations['relu']
    linmap = slim.maps['linear']
    sol_map = policies.MLPPolicy(
        {'thetap': (n_theta,), 'nu': (n_x,), **dataset.dims},
        bias=args.bias,
        linear_map=linmap,
        nonlin=activation,
        hsizes=[args.nx_hidden] * args.n_layers,
        input_keys=["thetap"],
        name='sol_map',
    )

    # variables
    x = Variable(f"U_pred_{sol_map.name}")  # decision variable as output from the solution map
    theta = Variable('thetap')  # sampled parametric variable
    # objective function:  J = c*x
    loss = args.Q * (x@c.t() == 0)  # weighted loss to be penalized
    loss.name = 'loss'
    # constraints: C = [ A*x <= b + E*theta, x >= 0]
    con_1 = args.Q_con * (x@A.t() <= b + theta@E.t())
    # con_2 = args.Q_con * (x >= 0)

    objectives = [loss]
    constraints = [con_1]
    components = [sol_map]

    # constrained optimization problem construction
    model = Problem(objectives, constraints, components)
    model = model.to(device)

    """
    # # # Metrics and Logger
    """
    args.savedir = 'test_mpLP'
    args.verbosity = 1
    metrics = ["nstep_dev_loss",
               "nstep_dev_ref_loss"]
    logger = BasicLogger(args=args, savedir=args.savedir, verbosity=args.verbosity, stdout=metrics)
    logger.args.system = 'mpLP'

    """
    # # #  mpLP problem solution in Neuromancer
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    trainer = Trainer(
        model,
        dataset,
        optimizer,
        epochs=args.epochs,
        patience=args.patience,
        warmup=args.warmup,
        eval_metric="nstep_dev_loss",
        logger=logger,
    )

    # Train mpLP solution map
    best_model = trainer.train()
    best_outputs = trainer.test(best_model)

    # plots
    plot_loss(model, dataset, xmin=-2, xmax=2, save_path=None)
    plot_solution(sol_map, xmin=-2, xmax=2, save_path=None)