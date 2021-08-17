"""
Tutorial script for formulating and solving multiparametric quadratic programs (mpQP)
using Neuromancer's data-driven differentiable parametric optimization (D3PO) approach

mpLP problem formulation:
    J(\Theta) = min_x ||x||^2
                s.t. Ax \le b + E*\Theta
                      \Theta \in F

    x = decision variable
    A, b, E = fixed problem parameters
    F = set of admissible parameters \Theta

The objective is to find an explicit optimizer (solution map) x = h(\Theta)
optimizing the value function J(\Theta)

We consider a mpLP problem with benchmark solution using Yalmip and MPT3 toolbox in Matlab
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
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import slim

from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
import neuromancer.arg as arg
from neuromancer.constraint import Variable
from neuromancer.activations import activations
from neuromancer import policies
from neuromancer.loggers import BasicLogger
from neuromancer.dataset import normalize_data, split_static_data, StaticDataset
from neuromancer.plot import plot_loss_mpp, plot_solution_mpp


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
    gp.add("-data_seed", type=int, default=408,
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


def get_dataloaders(data, norm_type=None, split_ratio=None, num_workers=0):
    """This will generate dataloaders for a given dictionary of data.
    Dataloaders are hard-coded for full-batch training to match NeuroMANCER's training setup.

    :param data: (dict str: np.array or list[dict str: np.array]) data dictionary or list of data
        dictionaries; if latter is provided, multi-sequence datasets are created and splits are
        computed over the number of sequences rather than their lengths.
    :param norm_type: (str) type of normalization; see function `normalize_data` for more info.
    :param split_ratio: (list float) percentage of data in train and development splits; see
        function `split_sequence_data` for more info.
    """

    if norm_type is not None:
        data, _ = normalize_data(data, norm_type)
    train_data, dev_data, test_data = split_static_data(data, split_ratio)

    train_data = StaticDataset(
        train_data,
        name="train",
    )
    dev_data = StaticDataset(
        dev_data,
        name="dev",
    )
    test_data = StaticDataset(
        test_data,
        name="test",
    )

    train_data = DataLoader(
        train_data,
        batch_size=len(train_data),
        shuffle=False,
        collate_fn=train_data.collate_fn,
        num_workers=num_workers,
    )
    dev_data = DataLoader(
        dev_data,
        batch_size=len(dev_data),
        shuffle=False,
        collate_fn=dev_data.collate_fn,
        num_workers=num_workers,
    )
    test_data = DataLoader(
        test_data,
        batch_size=len(test_data),
        shuffle=False,
        collate_fn=test_data.collate_fn,
        num_workers=num_workers,
    )

    return (train_data, dev_data, test_data), train_data.dataset.dims


if __name__ == "__main__":
    """
    # # #  optimization problem hyperparameters
    """
    parser = arg.ArgParser(parents=[arg.log(),
                                    arg_mpLP_problem()])
    args, grps = parser.parse_arg_groups()
    args.bias = True
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"

    """
    # # #  LP problem matrices
    """
    n_x = 1         # number of decision variables
    n_theta = 2     # number of parameters
    n_con = 5       # number of constraints

    # fixed problem parameters
    A = torch.randn(n_x, n_con).t()
    b = torch.randn(n_con)
    E = torch.randn(n_theta, n_con).t()

    # A = torch.randn(n_con, n_x)
    # b = torch.randn(n_con)
    # E = torch.randn(n_con, n_theta)


    """
    # # #  Dataset 
    """
    #  randomly sampled parameters theta generating superset of:
    #  theta_samples.min() <= theta <= theta_samples.max()
    np.random.seed(args.data_seed)
    nsim = 50000  # number of datapoints: increase sample density for more robust results
    sequences = {"theta": 0.5 * np.random.randn(nsim, n_theta)}

    nstep_data, dims = get_dataloaders(sequences)
    train_data, dev_data, test_data = nstep_data

    """
    # # #  mpLP problem formulation in Neuromancer
    """
    # define solution map as MLP policy
    dims['U'] = (nsim, n_x)  # defining expected dimensions of the solution variable: internal policy key 'U'
    activation = activations['relu']
    linmap = slim.maps['linear']
    sol_map = policies.MLPPolicy(
        {**dims},
        bias=args.bias,
        linear_map=linmap,
        nonlin=activation,
        hsizes=[args.nx_hidden] * args.n_layers,
        input_keys=["theta"],
        name='sol_map',
    )

    # variables
    x = Variable(f"U_pred_{sol_map.name}")  # decision variable as output from the solution map
    theta = Variable('theta')  # sampled parametric variable
    # objective function
    loss = args.Q * (x == 0)^2  # weighted loss to be penalized
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
    args.savedir = 'test_mpQP'
    args.verbosity = 1
    metrics = ["dev_loss"]
    logger = BasicLogger(args=args, savedir=args.savedir, verbosity=args.verbosity, stdout=metrics)
    logger.args.system = 'mpQP'

    """
    # # #  mpQP problem solution in Neuromancer
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # define trainer
    trainer = Trainer(
        model,
        train_data,
        dev_data,
        test_data,
        optimizer,
        logger=logger,
        epochs=args.epochs,
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="test_loss",
        eval_metric="dev_loss",
        patience=args.patience,
        warmup=args.warmup,
        device=device,
    )

    # Train mpLP solution map
    best_model = trainer.train()
    best_outputs = trainer.test(best_model)

    # plots
    plot_loss_mpp(model, train_data, xmin=-2, xmax=2, save_path=None)
    plot_solution_mpp(sol_map, xmin=-2, xmax=2, save_path=None)