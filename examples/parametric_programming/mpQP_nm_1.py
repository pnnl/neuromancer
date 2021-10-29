"""
Solve Quadratic Programming (QP) problem using Neuromancer toolbox:
minimize     x^2+y^2
subject to   x+y-p >= 0

problem parameters:            p
problem decition variables:    x, y
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import slim
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects

from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
import neuromancer.arg as arg
from neuromancer.constraint import Variable, Objective, Loss
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
    gp.add("-Q_con", type=float, default=20.0,
           help="constraints penalty weight.")  # tuned value: 20.0
    gp.add("-nx_hidden", type=int, default=40,
           help="Number of hidden states of the solution map")
    gp.add("-n_layers", type=int, default=2,
           help="Number of hidden layers of the solution map")
    gp.add("-bias", action="store_true",
           help="Whether to use bias in the neural network block component models.")
    gp.add("-data_seed", type=int, default=408,
           help="Random seed used for simulated data")
    gp.add("-epochs", type=int, default=800,
           help='Number of training epochs')
    gp.add("-lr", type=float, default=0.001,
           help="Step size for gradient descent.")
    gp.add("-patience", type=int, default=200,
           help="How many epochs to allow for no improvement in eval metric before early stopping.")
    gp.add("-warmup", type=int, default=200,
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
    # # #  Dataset 
    """
    #  randomly sampled parameters theta generating superset of:
    #  theta_samples.min() <= theta <= theta_samples.max()
    np.random.seed(args.data_seed)
    nsim = 10000  # number of datapoints: increase sample density for more robust results
    samples = {"p": np.random.uniform(low=5.0, high=15.0, size=(nsim, 1))}
    nstep_data, dims = get_dataloaders(samples)
    train_data, dev_data, test_data = nstep_data

    """
    # # #  mpLP problem formulation in Neuromancer
    """
    n_var = 2           # number of decision variables
    # define solution map as MLP policy
    dims['U'] = (nsim, n_var)  # defining expected dimensions of the solution variable: internal policy key 'U'
    activation = activations['relu']
    linmap = slim.maps['linear']
    sol_map = policies.MLPPolicy(
        {**dims},
        bias=args.bias,
        linear_map=linmap,
        nonlin=activation,
        hsizes=[args.nx_hidden] * args.n_layers,
        input_keys=["p"],
        name='sol_map',
    )

    # variables
    x = Variable(f"U_pred_{sol_map.name}", name='x')[:, :, [0]]
    y = Variable(f"U_pred_{sol_map.name}", name='y')[:, :, [1]]
    # sampled parameters
    p = Variable('p')

    # objective function
    loss = Objective(x**2 + y**2, weight=args.Q, name='loss')
    # constraints
    con_1 = args.Q_con * (x + y - p >= 0)

    # constrained optimization problem construction
    objectives = [loss]
    constraints = [con_1]
    components = [sol_map]
    model = Problem(objectives, constraints, components)
    # model = model.to(device)

    """
    # # # Metrics and Logger
    """
    args.savedir = 'test_mpQP_1'
    args.verbosity = 1
    metrics = ["dev_loss"]
    logger = BasicLogger(args=args, savedir=args.savedir, verbosity=args.verbosity, stdout=metrics)
    logger.args.system = 'mpQP_1'

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

    # # # plots
    # plot_loss_mpp(model, train_data, xmin=-2, xmax=2, save_path=None)
    # plot_solution_mpp(sol_map, xmin=-2, xmax=2, save_path=None)

    params = [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    p = 10.0
    x1 = np.arange(-1.0, 10.0, 0.05)
    y1 = np.arange(-1.0, 10.0, 0.05)
    xx, yy = np.meshgrid(x1, y1)
    fig, ax = plt.subplots(3,3)
    row_id = 0
    column_id = 0
    for i, p in enumerate(params):
        if i % 3 == 0 and i != 0:
            row_id += 1
            column_id = 0
        print(column_id)
        print(row_id)
        # eval objective and constraints
        J = xx ** 2 + yy ** 2
        c1 = xx + yy - p
        # Plot
        cp = ax[row_id,column_id].contourf(xx, yy, J,
                         alpha=0.6)
        ax[row_id,column_id].set_title(f'QP p={p}')
        cg1 = ax[row_id,column_id].contour(xx, yy, c1, [0], colors='mediumblue', alpha=0.7)
        plt.setp(cg1.collections,
                 path_effects=[patheffects.withTickedStroke()], alpha=0.7)
        fig.colorbar(cp, ax=ax[row_id,column_id])
        params = torch.tensor([p])
        xy_optim = model.components[0].net(params).detach().numpy()
        print(xy_optim[0])
        print(xy_optim[1])
        ax[row_id,column_id].plot(xy_optim[0], xy_optim[1], 'r*', markersize=10)
        column_id +=1
    plt.show()

    p = 10.0
    x1 = np.arange(-1.0, 10.0, 0.05)
    y1 = np.arange(-1.0, 10.0, 0.05)
    xx, yy = np.meshgrid(x1, y1)
    # eval objective and constraints
    J = xx ** 2 + yy ** 2
    c1 = xx + yy - p
    # Plot
    fig, ax = plt.subplots(1,1)
    cp = ax.contourf(xx, yy, J,
                     alpha=0.6)
    fig.colorbar(cp)
    ax.set_title('Quadratic problem')
    cg1 = ax.contour(xx, yy, c1, [0], colors='mediumblue', alpha=0.7)
    plt.setp(cg1.collections,
             path_effects=[patheffects.withTickedStroke()], alpha=0.7)

    params = torch.tensor([p])
    xy_optim = model.components[0].net(params).detach().numpy()
    print(xy_optim[0])
    print(xy_optim[1])
    ax.plot(xy_optim[0], xy_optim[1], 'r*', markersize=10)