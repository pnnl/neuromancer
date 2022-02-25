"""
Multi-parametric Quadratic Programming (mpQP) problem using Neuromancer:
minimize     x^2 + y^2
subject to
           -x - y + p1 <= 0,
           x + y - p1 - 5 <= 0,
           x - y + p2 - 5 <= 0,
           -x + y - p2 <= 0

problem parameters:            p1, p2
problem decition variables:    x, y
"""


import numpy as np
import torch
from torch.utils.data import DataLoader
import slim
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import cvxpy as cp
import numpy as np

from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
import neuromancer.arg as arg
from neuromancer.constraint import Variable
from neuromancer.activations import activations
from neuromancer.loggers import BasicLogger, MLFlowLogger
from neuromancer.dataset import normalize_data, split_static_data, StaticDataset
from neuromancer.loss import PenaltyLoss, BarrierLoss, AugmentedLagrangeLoss
from neuromancer.solvers import GradientProjection
from neuromancer.maps import ManyToMany
from neuromancer import blocks


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
    gp.add("-Q_con", type=float, default=100.0,
           help="constraints penalty weight.")  # tuned value: 100.0
    gp.add("-nx_hidden", type=int, default=80,
           help="Number of hidden states of the solution map")
    gp.add("-n_layers", type=int, default=4,
           help="Number of hidden layers of the solution map")
    gp.add("-bias", action="store_true",
           help="Whether to use bias in the neural network block component models.")
    gp.add("-data_seed", type=int, default=408,
           help="Random seed used for simulated data")
    gp.add("-epochs", type=int, default=1000,
           help='Number of training epochs')
    gp.add("-lr", type=float, default=0.001,
           help="Step size for gradient descent.")
    gp.add("-patience", type=int, default=100,
           help="How many epochs to allow for no improvement in eval metric before early stopping.")
    gp.add("-warmup", type=int, default=100,
           help="Number of epochs to wait before enacting early stopping policy.")
    gp.add("-loss", type=str, default='penalty',
           choices=['penalty', 'augmented_lagrange', 'barrier'],
           help="type of the loss function.")
    gp.add("-barrier_type", type=str, default='log10',
           choices=['log', 'log10', 'inverse'],
           help="type of the barrier function in the barrier loss.")
    gp.add("-eta", type=float, default=0.99,
           help="eta in augmented lagrangian.")
    gp.add("-sigma", type=float, default=2.0,
           help="sigma in augmented lagrangian.")
    gp.add("-mu_init", type=float, default=1.,
           help="mu_init in augmented lagrangian.")
    gp.add("-mu_max", type=float, default=1.,
           help="mu_max in augmented lagrangian.")
    gp.add("-inner_loop", type=int, default=1,
           help="inner loop in augmented lagrangian")
    gp.add("-proj_grad", default=True, choices=[True, False],
           help="Whether to use projected gradient update or not.")
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


def get_loss(objectives, constraints, train_data, args):
    if args.loss == 'penalty':
        loss = PenaltyLoss(objectives, constraints)
    elif args.loss == 'barrier':
        loss = BarrierLoss(objectives, constraints, barrier=args.barrier_type)
    elif args.loss == 'augmented_lagrange':
        optimizer_args = {'inner_loop': args.inner_loop, "eta": args.eta, 'sigma': args.sigma,
                          'mu_init': args.mu_init, "mu_max": args.mu_max}
        loss = AugmentedLagrangeLoss(objectives, constraints, train_data, **optimizer_args)
    return loss


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
    nsim = 9000  # number of datapoints: increase sample density for more robust results
    samples = {"p1": np.random.uniform(low=1.0, high=11.0, size=(nsim, 1)),
               "p2": np.random.uniform(low=1.0, high=11.0, size=(nsim, 1))}
    data, dims = get_dataloaders(samples)
    train_data, dev_data, test_data = data

    """
    # # #  mpQP primal solution map architecture
    """
    f1 = blocks.MLP(insize=2, outsize=1,
                bias=True,
                linear_map=slim.maps['linear'],
                nonlin=activations['relu'],
                hsizes=[args.nx_hidden] * args.n_layers)
    f2 = blocks.MLP(insize=2, outsize=1,
                bias=True,
                linear_map=slim.maps['linear'],
                nonlin=activations['relu'],
                hsizes=[args.nx_hidden] * args.n_layers)
    sol_map = ManyToMany([f1, f2],
            input_keys=["p1", "p2"],
            output_keys=["x", "y"],
            name='primal_map')

    """
    # # #  mpQP objective and constraints formulation in Neuromancer
    """
    # variables
    x = Variable("x")
    y = Variable("y")
    # sampled parameters
    p1 = Variable('p1')
    p2 = Variable('p2')

    # objective function
    f = x ** 2 + y ** 2
    obj = f.minimize(weight=args.Q, name='obj')
    # constraints
    g1 = -x - y + p1
    con_1 = (g1 <= 0)
    con_1.name = 'c1'
    g2 = x + y - p1 - 5
    con_2 = (g2 <= 0)
    con_2.name = 'c2'
    g3 = x - y + p2 - 5
    con_3 = (g3 <= 0)
    con_3.name = 'c3'
    g4 = -x + y - p2
    con_4 = (g4 <= 0)
    con_4.name = 'c4'

    """
    # # #  mpQP problem formulation in Neuromancer
    """
    # list of objectives, constraints, and components (solution maps)
    objectives = [obj]
    constraints = [args.Q_con*con_1, args.Q_con*con_2,
                   args.Q_con*con_3, args.Q_con*con_4]
    components = [sol_map]

    if args.proj_grad:  # use projected gradient update
        project_keys = ["x", "y"]
        projection = GradientProjection(constraints, input_keys=project_keys,
                                        num_steps=5, name='proj')
        components.append(projection)

    # create constrained optimization loss
    loss = get_loss(objectives, constraints, train_data, args)
    # construct constrained optimization problem
    problem = Problem(components, loss, grad_inference=args.proj_grad)
    # plot computational graph
    problem.plot_graph()

    """
    # # # Metrics and Logger
    """
    args.savedir = 'test_mpQP_2'
    args.verbosity = 1
    metrics = ["train_loss", "train_obj", "train_mu_scaled_penalty_loss", "train_con_lagrangian",
               "train_mu", "train_c1", "train_c2", "train_c3", "train_c4"]
    if args.logger == 'stdout':
        Logger = BasicLogger
    elif args.logger == 'mlflow':
        Logger = MLFlowLogger
    logger = Logger(args=args, savedir=args.savedir, verbosity=args.verbosity, stdout=metrics)
    logger.args.system = 'mpQP_2'

    """
    # # #  mpQP problem solution in Neuromancer
    """
    optimizer = torch.optim.AdamW(problem.parameters(), lr=args.lr)

    # define trainer
    trainer = Trainer(
        problem,
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


    """
    CVXPY benchmark
    """
    # Define and solve the CVXPY problem.
    x = cp.Variable(1)
    y = cp.Variable(1)
    p1 = 10.0  # problem parameter
    p2 = 10.0  # problem parameter

    def QP_param(p1, p2):
        prob = cp.Problem(cp.Minimize(x ** 2 + y ** 2),
                          [-x - y + p1 <= 0,
                           x + y - p1 - 5 <= 0,
                           x - y + p2 - 5 <= 0,
                           -x + y - p2 <= 0])
        return prob

    """
    Plots
    """
    # test problem parameters
    params = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
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
        # eval objective and constraints
        J = xx ** 2 + yy ** 2
        c1 = xx + yy - p
        c2 = -xx - yy + p + 5
        c3 = -xx + yy - p + 5
        c4 = xx - yy + p

        # Plot constraints and loss contours
        cp_plot = ax[row_id, column_id].contourf(xx, yy, J, 50, alpha=0.4)
        ax[row_id, column_id].set_title(f'QP p={p}')
        cg1 = ax[row_id, column_id].contour(xx, yy, c1, [0], colors='mediumblue', alpha=0.7)
        cg2 = ax[row_id, column_id].contour(xx, yy, c2, [0], colors='mediumblue', alpha=0.7)
        cg3 = ax[row_id, column_id].contour(xx, yy, c3, [0], colors='mediumblue', alpha=0.7)
        cg4 = ax[row_id, column_id].contour(xx, yy, c4, [0], colors='mediumblue', alpha=0.7)
        plt.setp(cg1.collections,
                 path_effects=[patheffects.withTickedStroke()], alpha=0.7)
        plt.setp(cg2.collections,
                 path_effects=[patheffects.withTickedStroke()], alpha=0.7)
        plt.setp(cg3.collections,
                 path_effects=[patheffects.withTickedStroke()], alpha=0.7)
        plt.setp(cg4.collections,
                 path_effects=[patheffects.withTickedStroke()], alpha=0.7)
        fig.colorbar(cp_plot, ax=ax[row_id, column_id])

        # Solve QP
        prob = QP_param(p, p)
        prob.solve()

        # Solve via neuromancer
        datapoint = {}
        datapoint['p1'] = torch.tensor([[p]])
        datapoint['p2'] = torch.tensor([[p]])
        datapoint['name'] = "test"
        model_out = problem(datapoint)
        x_nm = model_out['test_' + "x"][0, :].detach().numpy()
        y_nm = model_out['test_' + "y"][0, :].detach().numpy()

        print(f'primal solution QP x={x.value}, y={y.value}')
        print(f'parameter p={p, p}')
        print(f'primal solution DPP x1={x_nm}, x2={y_nm}')
        print(f' f: {model_out["test_" + f.key]}')
        print(f' g1: {model_out["test_" + g1.key]}')
        print(f' g2: {model_out["test_" + g2.key]}')
        print(f' g3: {model_out["test_" + g3.key]}')
        print(f' g4: {model_out["test_" + g4.key]}')

        # Plot optimal solutions
        ax[row_id, column_id].plot(x.value, y.value, 'g*', markersize=10)
        ax[row_id, column_id].plot(x_nm, y_nm, 'r*', markersize=10)
        column_id += 1
    plt.show()

