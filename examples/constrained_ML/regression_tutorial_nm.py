"""
Tutorial script for regression in neuromancer

"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import slim
import psl

from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
import neuromancer.arg as arg
from neuromancer.constraint import Variable, Objective, Loss
from neuromancer.activations import activations
from neuromancer import policies
from neuromancer.loggers import BasicLogger
from neuromancer.dataset import read_file, normalize_data, split_static_data, StaticDataset
from neuromancer.component import Function
import neuromancer.blocks as blocks


def arg_reg_problem(prefix=''):
    """
    Command line parser for regression problem definition arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers
                         are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = arg.ArgParser(prefix=prefix, add_help=False)
    gp = parser.group("mpLP")
    gp.add("-Q", type=float, default=1.0,
           help="loss function weight.")
    gp.add("-Q_sub", type=float, default=0.0,
           help="regularization weight.")
    gp.add("-Q_con", type=float, default=1.0,
           help="constraints penalty weight.")
    gp.add("-nx_hidden", type=int, default=20,
           help="Number of hidden states of the solution map")
    gp.add("-n_layers", type=int, default=4,
           help="Number of hidden layers of the solution map")
    gp.add("-data_seed", type=int, default=408,
           help="Random seed used for simulated data")
    gp.add("-epochs", type=int, default=2000,
           help='Number of training epochs')
    gp.add("-lr", type=float, default=0.01,
           help="Step size for gradient descent.")
    gp.add("-patience", type=int, default=200,
           help="How many epochs to allow for no improvement in eval metric before early stopping.")
    gp.add("-warmup", type=int, default=200,
           help="Number of epochs to wait before enacting early stopping policy.")
    gp.add("-norm_type", type=str, default="zero-one", choices=
           ["zero-one", "one-one", "zscore", None],
           help="Normalization of the dataset.")
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
    # # #  Optimization problem hyperparameters
    """
    parser = arg.ArgParser(parents=[arg.log(),
                                    arg_reg_problem()])
    args, grps = parser.parse_arg_groups()
    args.bias = True
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"

    """
    # # #  Dataset 
    """
    #  randomly sampled data
    np.random.seed(args.data_seed)
    nsim = 1000   # number of datapoints: increase sample density for more robust results
    x_data = np.linspace((0,), (10,), nsim)
    # # # noisy sine wave with trend
    # y_data = 0.5*np.linspace((0,), (10,), nsim) + 0.5*np.random.rand(nsim, 1) + \
    #          psl.Periodic(nx=1, nsim=nsim, numPeriods=15, xmax=1.5, xmin=0.1)[:nsim, :]
    # noisy sine wave
    y_data = 0.5*np.random.rand(nsim, 1) + \
             psl.Periodic(nx=1, nsim=nsim, numPeriods=15, xmax=1.5, xmin=0.1)[:nsim, :]
    samples = {"x": x_data,
               "y": y_data}
    nstep_data, dims = get_dataloaders(samples, norm_type=args.norm_type)
    train_data, dev_data, test_data = nstep_data

    """
    # # #  Constrained regression problem formulation in Neuromancer
    """
    n_var = 1
    # define solution map as MLP
    dims['U'] = (nsim, n_var)  # defining expected dimensions of the solution variable: internal policy key 'U'
    activation = activations['gelu']
    linmap = slim.maps['linear']
    sol_map = policies.MLPPolicy(
        {**dims},
        bias=args.bias,
        linear_map=linmap,
        nonlin=activation,
        hsizes=[args.nx_hidden] * args.n_layers,
        input_keys=["x"],
        name='regressor',
    )

    # dataset variables
    x = Variable('x')
    y = Variable('y')
    y_hat = Variable(f"U_pred_{sol_map.name}")

    # objective function
    loss_1 = args.Q*(y_hat - y == 0)^2
    loss_1.name = 'loss'

    # constraints
    con_1 = args.Q_con*(y_hat <= 1.5)
    con_2 = args.Q_con*(-0.5 <= y_hat)

    # constrained optimization problem construction
    objectives = [loss_1]
    components = [sol_map]
    constraints = []
    # constraints = [con_1, con_2]
    model = Problem(objectives, constraints, components)
    model = model.to(device)

    """
    # # # Metrics and Logger
    """
    args.savedir = 'test_regression'
    args.verbosity = 1
    metrics = ["train_loss", "dev_loss"]
    logger = BasicLogger(args=args, savedir=args.savedir, verbosity=args.verbosity, stdout=metrics)
    logger.args.system = 'test_regression'

    """
    # # #  Constrained least squares problem solution in Neuromancer
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
        # eval_metric="dev_loss",
        eval_metric="train_loss",
        patience=args.patience,
        warmup=args.warmup,
        device=device,
    )

    # learn solution
    best_model = trainer.train()
    best_outputs = trainer.test(best_model)


    """
    # # #  Plots
    """
    # get the dict of the whole dataset
    if args.norm_type is not None:
        samples, _ = normalize_data(samples, norm_type=args.norm_type)
    all_data = StaticDataset(samples, name="dataset")
    all_data_dict = all_data.get_full_batch()
    # normalized targets
    x_d = all_data_dict['x']
    y_d = all_data_dict['y']
    net_out = sol_map(all_data_dict)
    y_pred = net_out[f"U_pred_{sol_map.name}"][0, :, :].detach().numpy()
    # plot regressors
    fig, ax = plt.subplots(1, 1)
    ax.plot(y_d, 'o')
    ax.plot(y_pred)
    ax.set(ylabel='$y$')
    ax.set(xlabel='$x$')

