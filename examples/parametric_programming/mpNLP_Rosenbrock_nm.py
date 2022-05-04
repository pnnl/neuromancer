"""
Solve the Rosenbrock problem, formulated as the NLP using Neuromancer toolbox:
minimize     (1-x)^2 + a*(y-x^2)^2
subject to   (p/2)^2 <= x^2 + y^2 <= p^2
             x>=y

problem parameters:             a, p
problem decition variables:     x, y

https://en.wikipedia.org/wiki/Rosenbrock_function
"""


import numpy as np
import torch
from torch.utils.data import DataLoader
import slim
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import cvxpy as cp
import numpy as np
from casadi import *

from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
import neuromancer.arg as arg
from neuromancer.constraint import Variable
from neuromancer.activations import activations
from neuromancer.loggers import BasicLogger, MLFlowLogger
from neuromancer.dataset import get_static_dataloaders
from neuromancer.loss import get_loss
from neuromancer.solvers import GradientProjection
from neuromancer.maps import Map
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
           help="constraints penalty weight.")  # tuned value: 1.0
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
    gp.add("-mu_max", type=float, default=1000.,
           help="mu_max in augmented lagrangian.")
    gp.add("-inner_loop", type=int, default=1,
           help="inner loop in augmented lagrangian")
    gp.add("-proj_grad", default=False, choices=[True, False],
           help="Whether to use projected gradient update or not.")
    return parser


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
    nsim = 20000  # number of datapoints: increase sample density for more robust results
    samples = {"a": np.random.uniform(low=0.2, high=1.5, size=(nsim, 1)),
               "p": np.random.uniform(low=0.5, high=2.0, size=(nsim, 1))}
    data, dims = get_static_dataloaders(samples)
    train_data, dev_data, test_data = data

    """
    # # #  mpNLP primal solution map architecture
    """
    func = blocks.MLP(insize=2, outsize=2,
                    bias=True,
                    linear_map=slim.maps['linear'],
                    nonlin=activations['relu'],
                    hsizes=[args.nx_hidden] * args.n_layers)
    sol_map = Map(func,
            input_keys=["a", "p"],
            output_keys=["x"],
            name='primal_map')
    """
    # # #  mpNLP objective and constraints formulation in Neuromancer
    """
    # variables
    x = Variable("x")[:, [0]]
    y = Variable("x")[:, [1]]
    # sampled parameters
    p = Variable('p')
    a = Variable('a')

    # objective function
    f = (1-x)**2 + a*(y-x**2)**2
    obj = f.minimize(weight=args.Q, name='obj')

    # constraints
    con_1 = (x >= y)
    con_2 = ((p/2)**2 <= x**2+y**2)
    con_3 = (x**2+y**2 <= p**2)
    # g1 = y - x
    # con_1 = (g1 <= 0)
    # g2 = -x**2-y**2 - (p/2)**2
    # con_2 = (g2 <= 0)
    # g3 = x**2+y**2 - p**2
    # con_3 = (g3 <= 0)
    con_1.name = 'c1'
    con_2.name = 'c2'
    con_3.name = 'c3'

    # constrained optimization problem construction
    objectives = [obj]
    constraints = [args.Q_con*con_1, args.Q_con*con_2, args.Q_con*con_3]
    components = [sol_map]

    if args.proj_grad:  # use projected gradient update
        project_keys = ["x"]
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
    args.savedir = 'test_mpNLP_Rosebnrock'
    args.verbosity = 1
    metrics = ["train_loss", "train_obj", "train_mu_scaled_penalty_loss", "train_con_lagrangian",
               "train_mu", "train_c1", "train_c2", "train_c3"]
    if args.logger == 'stdout':
        Logger = BasicLogger
    elif args.logger == 'mlflow':
        Logger = MLFlowLogger
    logger = Logger(args=args, savedir=args.savedir, verbosity=args.verbosity, stdout=metrics)
    logger.args.system = 'mpmpNLP_Rosebnrock'

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
    # load best model dict
    problem.load_state_dict(best_model)

    """
    CasADi benchmark
    """
    # selected parameters
    p = 1.0
    a = 1.0

    # instantiate casadi optimizaiton problem class
    opti = casadi.Opti()
    # define variables
    x = opti.variable()
    y = opti.variable()
    # define objective and constraints
    opti.minimize((1 - x) ** 2 + a * (y - x ** 2) ** 2)
    opti.subject_to(x >= y)
    opti.subject_to((p / 2) ** 2 <= x ** 2 + y ** 2)
    opti.subject_to(x ** 2 + y ** 2 <= p ** 2)
    # select IPOPT solver and solve the NLP
    opti.solver('ipopt')
    sol = opti.solve()
    print(sol.value(x))
    print(sol.value(y))

    """
    Plots
    """
    a = 1.0
    p = 1.0
    x1 = np.arange(-0.5, 1.5, 0.02)
    y1 = np.arange(-0.5, 1.5, 0.02)
    xx, yy = np.meshgrid(x1, y1)

    # eval objective and constraints
    J = (1 - xx) ** 2 + a * (yy - xx ** 2) ** 2
    c1 = xx - yy
    c2 = xx ** 2 + yy ** 2 - (p / 2) ** 2
    c3 = -(xx ** 2 + yy ** 2) + p ** 2

    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(xx, yy, J,
                     levels=[0, 0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0],
                     alpha=0.6)
    fig.colorbar(cp)
    ax.set_title('Rosenbrock problem')
    cg1 = ax.contour(xx, yy, c1, [0], colors='mediumblue', alpha=0.7)
    plt.setp(cg1.collections,
             path_effects=[patheffects.withTickedStroke()], alpha=0.7)
    cg2 = ax.contour(xx, yy, c2, [0], colors='mediumblue', alpha=0.7)
    plt.setp(cg2.collections,
             path_effects=[patheffects.withTickedStroke()], alpha=0.7)
    cg3 = ax.contour(xx, yy, c3, [0], colors='mediumblue', alpha=0.7)
    plt.setp(cg3.collections,
             path_effects=[patheffects.withTickedStroke()], alpha=0.7)

    # Solution to mpNLP via Neuromancer
    datapoint = {}
    datapoint['a'] = torch.tensor([[a]])
    datapoint['p'] = torch.tensor([[p]])
    datapoint['name'] = "test"
    model_out = problem(datapoint)
    x_nm = model_out['test_' + "x"][0, 0].detach().numpy()
    y_nm = model_out['test_' + "x"][0, 1].detach().numpy()
    print(x_nm)
    print(y_nm)

    # optimal points CasADi vs Neuromancer
    ax.plot(sol.value(x), sol.value(y), 'g*', markersize=10)
    ax.plot(x_nm, y_nm, 'r*', markersize=10)
