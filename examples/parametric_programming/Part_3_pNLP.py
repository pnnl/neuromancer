"""
Learning to optimize a set of 2D parametric nonlinear programming problems (pNLP) using Neuromancer.

Problem formulation:
    minimize     f
    subject to   (p/2)^2 <= x^2 + y^2 <= p^2
                 x>=y

problem parameters:             p
problem decition variables:     x, y
Search domain:                  -5.0 <= x, y <= 5.0

Set of objective functions f:
    Rosenbrock:        f = (1 - x)**2 + (y - x**2)**2
    GomezLevy:         f = 4*x**2 - 2.1*x**4 + 1/3*x**6 + x*y - 4*y**2 + 4*y**4
    Himelblau:         f = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    Styblinski-Tang:   f = x**4 -15*x**2 + 5*x + y**4 -15*y**2 + 5*y
    Simionescu:        f = 0.1*x*y
    McCormick:         f = sin(x + y) + (x - y)**2 - 1.5*x + 2.5*y +1
    Three-hump-camel:  f = 2*x**2 - 1.05*x**4 + (x**6)/6 + x*y + y**2
    Beale:             f = (1.5 - x + x*y)**2 + (2.25 -x + x*y**2)**2 + (2.625 -x + x*y**3)**2

See the description of the test problems here:
https://en.wikipedia.org/wiki/Test_functions_for_optimization
https://en.wikipedia.org/wiki/Rosenbrock_function
https://en.wikipedia.org/wiki/Himmelblau%27s_function
"""

import torch
import torch.nn as nn
import neuromancer.slim as slim
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from casadi import *
import casadi

from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.constraint import variable
from neuromancer.dataset import DictDataset
from neuromancer.loss import PenaltyLoss, BarrierLoss, AugmentedLagrangeLoss
from neuromancer.modules import blocks
from neuromancer.system import Node
import neuromancer.arg as arg


def arg_pNLP_problem(prefix=''):
    """
    Command line parser for pNLP problem definition arguments

    :param prefix: (str) Optional prefix for command line arguments to resolve naming conflicts when multiple parsers
                         are bundled as parents.
    :return: (arg.ArgParse) A command line parser
    """
    parser = arg.ArgParser(prefix=prefix, add_help=False)
    gp = parser.group("pNLP")
    gp.add("-obj_fun", type=str, default='Himelblau',
           choices=['Rosenbrock', 'GomezLevy', 'Himelblau', 'Styblinski-Tang',
                    'Simionescu', 'McCormick', 'Three-hump-camel', 'Beale'],
           help="select objective function to be optimized.")
    gp.add("-Q", type=float, default=1.0,
           help="loss function weight.")
    gp.add("-Q_con", type=float, default=100.0,
           help="constraints penalty weight.")
    gp.add("-n_hidden", type=int, default=80,
           help="Number of hidden states of the solution map")
    gp.add("-n_layers", type=int, default=4,
           help="Number of hidden layers of the solution map")
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
    return parser


if __name__ == "__main__":

    """
    # # #  optimization problem hyperparameters
    """
    parser = arg.ArgParser(parents=[arg.log(),
                                    arg_pNLP_problem()])
    args, grps = parser.parse_arg_groups()
    device = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"

    """
    # # #  Dataset
    """
    np.random.seed(args.data_seed)
    nsim = 5000  # number of datapoints: increase sample density for more robust results
    # create dictionaries with sampled datapoints with uniform distribution
    p_low, p_high = 0.5, 6.0
    samples_train = {"p": torch.FloatTensor(nsim, 1).uniform_(p_low, p_high)}
    samples_dev = {"p": torch.FloatTensor(nsim, 1).uniform_(p_low, p_high)}
    samples_test = {"p": torch.FloatTensor(nsim, 1).uniform_(p_low, p_high)}
    # create named dictionary datasets
    train_data = DictDataset(samples_train, name='train')
    dev_data = DictDataset(samples_dev, name='dev')
    test_data = DictDataset(samples_test, name='test')
    # create torch dataloaders for the Trainer
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, num_workers=0,
                                               collate_fn=train_data.collate_fn, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=32, num_workers=0,
                                             collate_fn=dev_data.collate_fn, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, num_workers=0,
                                             collate_fn=test_data.collate_fn, shuffle=True)
    # note: training quality will depend on the DataLoader parameters such as batch size and shuffle

    """
    # # #  pNLP primal solution map architecture
    """
    # define neural architecture for the solution map
    func = blocks.MLP(insize=1, outsize=2,
                    linear_map=slim.maps['linear'],
                    nonlin=nn.ReLU,
                    hsizes=[args.n_hidden] * args.n_layers )
    # define symbolic solution map with concatenated features (problem parameters)
    sol_map = Node(func, ['p'], ['x'], name='map')

    """
    # # #  pNLP objective and constraints formulation in Neuromancer
    """
    # variables
    x = variable("x")[:, [0]]
    y = variable("x")[:, [1]]
    # sampled parameters
    p = variable('p')

    # list of nonlinear objective functions defined using Neuromancer variable
    obj_opt = {'Rosenbrock': (1 - x)**2 + (y - x**2)**2,
               'GomezLevy': 4 * x ** 2 - 2.1 * x ** 4 + 1 / 3 * x ** 6 + x * y - 4 * y ** 2 + 4 * y ** 4,
               'Himelblau': (x**2 + y - 11)**2 + (x + y**2 - 7)**2,
               'Styblinski-Tang': x**4 -15*x**2 + 5*x + y**4 -15*y**2 + 5*y,
               'Simionescu': 0.1*x*y,
               'McCormick': torch.sin(x + y) + (x - y)**2 - 1.5*x + 2.5*y +1,
               'Three-hump-camel': 2*x**2 - 1.05*x**4 + (x**6)/6 + x*y + y**2,
               'Beale': (1.5 - x + x*y)**2 + (2.25 -x + x*y**2)**2 + (2.625 -x + x*y**3)**2 }
    # select objective function
    f = obj_opt[args.obj_fun]
    obj = f.minimize(weight=args.Q, name='obj')

    # define constraints
    con_1 = args.Q_con*(x >= y)
    con_2 = args.Q_con*((p/2)**2 <= x**2+y**2)
    con_3 = args.Q_con*(x**2+y**2 <= p**2)
    con_1.name = 'c1'
    con_2.name = 'c2'
    con_3.name = 'c3'

    # lists of objective terms, constraints, and trainable components
    objectives = [obj]
    constraints = [con_1, con_2, con_3]
    components = [sol_map]

    # choose loss function type for optimization
    if args.loss == 'penalty':
        loss = PenaltyLoss(objectives, constraints)
    elif args.loss == 'barrier':
        loss = BarrierLoss(objectives, constraints, barrier=args.barrier_type)
    elif args.loss == 'augmented_lagrange':
        optimizer_args = {'inner_loop': args.inner_loop, "eta": args.eta, 'sigma': args.sigma,
                          'mu_init': args.mu_init, "mu_max": args.mu_max}
        loss = AugmentedLagrangeLoss(objectives, constraints, train_data, **optimizer_args)

    # construct constrained optimization problem
    # construct constrained optimization problem
    problem = Problem(components, loss)

    """
    # # #  pNLP problem solution in Neuromancer
    """
    optimizer = torch.optim.AdamW(problem.parameters(), lr=args.lr)
    # define trainer
    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        test_loader,
        optimizer,
        epochs=args.epochs,
        patience=args.patience,
        warmup=args.warmup,
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="test_loss",
        eval_metric="dev_loss",
    )

    # Train mpLP solution map
    best_model = trainer.train()
    best_outputs = trainer.test(best_model)
    # load best model dict
    problem.load_state_dict(best_model)

    """
    CasADi benchmark
    """
    # instantiate casadi optimizaiton problem class
    opti = casadi.Opti()
    # define variables
    x = opti.variable()
    y = opti.variable()
    p_opti = opti.parameter()
    # CasADi formulation of objectives
    obj_opt_cas = {'Rosenbrock': (1 - x) ** 2 + (y - x ** 2) ** 2,
                   'GomezLevy': 4 * x ** 2 - 2.1 * x ** 4 + 1 / 3 * x ** 6 + x * y - 4 * y ** 2 + 4 * y ** 4,
                   'Himelblau': (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2,
                   'Styblinski-Tang': x ** 4 - 15 * x ** 2 + 5 * x + y ** 4 - 15 * y ** 2 + 5 * y,
                   'Simionescu': 0.1 * x * y,
                   'McCormick': np.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1,
                   'Three-hump-camel': 2 * x ** 2 - 1.05 * x ** 4 + (x ** 6) / 6 + x * y + y ** 2,
                   'Beale': (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2}
    f = obj_opt_cas[args.obj_fun]
    opti.minimize(f)
    # define objective and constraints
    opti.subject_to(x >= y)
    opti.subject_to((p_opti / 2) ** 2 <= x ** 2 + y ** 2)
    opti.subject_to(x ** 2 + y ** 2 <= p_opti ** 2)
    # select IPOPT solver and solve the NLP
    opti.solver('ipopt')

    # set parametric value and solve a single instance NLP problem via CasADi
    p = 3.
    opti.set_value(p_opti, p)
    opti.solve()
    # solve NLP instance via CasADi
    sol = opti.solve()
    print('CasADi solution:')
    print(sol.value(x))
    print(sol.value(y))

    # solve NLP instance via Neuromancer
    datapoint = {'p': torch.tensor([[p]]), 'name': 'test'}
    model_out = problem(datapoint)
    x_nm = model_out['test_' + "x"][0, 0].detach().numpy()
    y_nm = model_out['test_' + "x"][0, 1].detach().numpy()
    print('Neuromancer solution:')
    print(x_nm)
    print(y_nm)

    """
    Plots
    """
    x1 = np.arange(-5., 5., 0.02)
    y1 = np.arange(-5., 5., 0.02)
    xx, yy = np.meshgrid(x1, y1)
    # eval and plot objective
    obj_opt_plot = {'Rosenbrock': (1 - xx) ** 2 + (yy - xx ** 2) ** 2,
                   'GomezLevy': 4 * xx ** 2 - 2.1 * xx ** 4 + 1 / 3 * xx ** 6 + xx * yy - 4 * yy ** 2 + 4 * yy ** 4,
                   'Himelblau': (xx ** 2 + yy - 11) ** 2 + (xx + yy ** 2 - 7) ** 2,
                   'Styblinski-Tang': xx ** 4 - 15 * xx ** 2 + 5 * xx + yy ** 4 - 15 * yy ** 2 + 5 * yy,
                   'Simionescu': 0.1 * xx * yy,
                   'McCormick': np.sin(xx + yy) + (xx - yy) ** 2 - 1.5 * xx + 2.5 * yy + 1,
                   'Three-hump-camel': 2 * xx ** 2 - 1.05 * xx ** 4 + (xx ** 6) / 6 + xx * yy + yy ** 2,
                   'Beale': (1.5 - xx + xx * yy) ** 2 + (2.25 - xx + xx * yy ** 2) ** 2 + (2.625 - xx + xx * yy ** 3) ** 2}
    J = obj_opt_plot[args.obj_fun]
    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(xx, yy, J, levels=1000, alpha=0.6)
    fig.colorbar(cp)
    ax.set_title(args.obj_fun+' pNLP')
    # eval  and plot  constraints
    c1 = xx - yy
    c2 = xx ** 2 + yy ** 2 - (p / 2) ** 2
    c3 = -(xx ** 2 + yy ** 2) + p ** 2
    cg1 = ax.contour(xx, yy, c1, [0], colors='mediumblue', alpha=0.7)
    plt.setp(cg1.collections,
             path_effects=[patheffects.withTickedStroke()], alpha=0.7)
    cg2 = ax.contour(xx, yy, c2, [0], colors='mediumblue', alpha=0.7)
    plt.setp(cg2.collections,
             path_effects=[patheffects.withTickedStroke()], alpha=0.7)
    cg3 = ax.contour(xx, yy, c3, [0], colors='mediumblue', alpha=0.7)
    plt.setp(cg3.collections,
             path_effects=[patheffects.withTickedStroke()], alpha=0.7)

    # plot optimal solutions CasADi vs Neuromancer
    ax.plot(sol.value(x), sol.value(y), 'g*', markersize=10, label='CasADi')
    ax.plot(x_nm, y_nm, 'r*', fillstyle='none', markersize=10, label='NeuroMANCER')
    plt.legend(bbox_to_anchor=(1.0, 0.15))
    plt.show(block=True)
