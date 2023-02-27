"""
Solve the Himmelblau problem, formulated as the NLP using Neuromancer toolbox:
minimize     (x**2 + y - 11)**2 + (x + y**2 - 7)**2
subject to   (p/2)^2 <= x^2 + y^2 <= p^2
             x>=y

problem parameters:             a, p
problem decision variables:     x, y

https://en.wikipedia.org/wiki/Himmelblau%27s_function
"""


import torch
import neuromancer.slim as slim
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np
from casadi import *
import time

from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
import neuromancer.arg as arg
from neuromancer.constraint import variable
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
    gp.add("-epochs", type=int, default=300,
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
    samples = {"p": np.random.uniform(low=2.0, high=6.0, size=(nsim, 1))}
    data, dims = get_static_dataloaders(samples)
    train_data, dev_data, test_data = data

    """
    # # #  mpNLP primal solution map architecture
    """
    func = blocks.MLP(insize=1, outsize=2,
                    bias=True,
                    linear_map=slim.maps['linear'],
                    nonlin=activations['relu'],
                    hsizes=[args.nx_hidden] * args.n_layers)
    sol_map = Map(func,
            input_keys=["p"],
            output_keys=["x"],
            name='primal_map')
    """
    # # #  mpNLP objective and constraints formulation in Neuromancer
    """
    # variables
    x = variable("x")[:, [0]]
    y = variable("x")[:, [1]]
    # sampled parameters
    p = variable('p')

    # objective function
    f = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    obj = f.minimize(weight=args.Q, name='obj')

    # constraints
    con_1 = (x >= y)
    con_2 = ((p/2)**2 <= x**2+y**2)
    con_3 = (x**2+y**2 <= p**2)
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
    args.savedir = 'test_mpNLP_Himmelblau'
    args.verbosity = 1
    metrics = ["train_loss", "train_obj", "train_mu_scaled_penalty_loss", "train_con_lagrangian",
               "train_mu", "train_c1", "train_c2", "train_c3"]
    if args.logger == 'stdout':
        Logger = BasicLogger
    elif args.logger == 'mlflow':
        Logger = MLFlowLogger
    logger = Logger(args=args, savedir=args.savedir, verbosity=args.verbosity, stdout=metrics)
    logger.args.system = 'mpmpNLP_Himmelblau'

    """
    # # #  mpNLP problem solution in Neuromancer
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
    # instantiate casadi optimizaiton problem class
    def NLP_param(p):
        opti = casadi.Opti()
        # define variables
        x = opti.variable()
        y = opti.variable()
        p_opti = opti.parameter()
        # define objective and constraints
        opti.minimize((x**2 + y - 11)**2 + (x + y**2 - 7)**2)
        opti.subject_to(x >= y)
        opti.subject_to((p_opti / 2) ** 2 <= x ** 2 + y ** 2)
        opti.subject_to(x ** 2 + y ** 2 <= p_opti ** 2)
        # select IPOPT solver and solve the NLP
        opti.solver('ipopt')
        # set parametric values
        opti.set_value(p_opti, p)
        return opti, x, y

    # selected parameters
    p = 4.0
    # construct casadi problem
    opti, x, y = NLP_param(p)
    # solve NLP via casadi
    sol = opti.solve()
    print(sol.value(x))
    print(sol.value(y))

    """
    Plots
    """
    x1 = np.arange(-5.0, 5.0, 0.02)
    y1 = np.arange(-5.0, 5.0, 0.02)
    xx, yy = np.meshgrid(x1, y1)

    # eval objective and constraints
    J = (xx**2 + yy - 11)**2 + (xx + yy**2 - 7)**2
    c1 = xx - yy
    c2 = xx ** 2 + yy ** 2 - (p / 2) ** 2
    c3 = -(xx ** 2 + yy ** 2) + p ** 2

    levels = [0, 1.0, 10.0, 20., 50., 100., 200., 400., 1000.]
    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(xx, yy, J, levels=levels, alpha=0.6)

    fig.colorbar(cp)
    ax.set_title('Himmelblau problem')
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
    # plt.savefig('figs/mpNLP_Himmelblau_nm.png')
    plt.show(block=True)
    plt.interactive(False)

    """
    Benchmark Solution
    """
    def eval_constraints(x, y, p):
        """
        evaluate mean constraints violations
        """
        con_1_viol = np.maximum(0, y - x)
        con_2_viol = np.maximum(0, (p/2)**2 - (x**2+y**2))
        con_3_viol = np.maximum(0, x**2+y**2 - p**2)
        con_viol = con_1_viol + con_2_viol + con_3_viol
        con_viol_mean = np.mean(con_viol)
        return con_viol_mean

    def eval_objective(x, y):
        obj_value_mean = np.mean((x**2 + y - 11)**2 + (x + y**2 - 7)**2)
        return obj_value_mean

    # fix random seeds
    torch.manual_seed(args.data_seed)
    np.random.seed(args.data_seed)

    # select n number of random samples to evaluate
    n_samples = 1000
    idx = np.random.randint(0, nsim, n_samples)
    p = samples['p'][idx]

    # create named dictionary for neuromancer
    datapoint = {}
    datapoint['p'] = torch.tensor(p).float()
    datapoint['name'] = "test"

    # Solve via neuromancer
    t = time.time()
    model_out = problem(datapoint)
    nm_time = time.time() - t
    x_nm = model_out['test_' + "x"][:, [0]].detach().numpy()
    y_nm = model_out['test_' + "x"][:, [1]].detach().numpy()

    # Solve via solver
    t = time.time()
    x_solver, y_solver = [], []
    for i in range(0, n_samples):
        prob, x, y = NLP_param(p[i])
        sol = prob.solve()
        x_solver.append(sol.value(x))
        y_solver.append(sol.value(y))
    solver_time = time.time() - t
    x_solver = np.asarray(x_solver)
    y_solver = np.asarray(y_solver)

    # Evaluate neuromancer solution
    print(f'Solution for {n_samples} problems via Neuromancer obtained in {nm_time:.4f} seconds')
    nm_con_viol_mean = eval_constraints(x_nm, y_nm, p)
    print(f'Neuromancer mean constraints violation {nm_con_viol_mean:.4f}')
    nm_obj_mean = eval_objective(x_nm, y_nm)
    print(f'Neuromancer mean objective value {nm_obj_mean:.4f}')

    # Evaluate solver solution
    print(f'Solution for {n_samples} problems via solver obtained in {solver_time:.4f} seconds')
    solver_con_viol_mean = eval_constraints(x_solver, y_solver, p)
    print(f'Solver mean constraints violation {solver_con_viol_mean:.4f}')
    solver_obj_mean = eval_objective(x_solver, y_solver)
    print(f'Solver mean objective value {solver_obj_mean:.4f}')

    # neuromancer solver comparison
    speedup_factor = solver_time/nm_time
    print(f'Solution speedup factor {speedup_factor:.4f}')

    # Difference in primal optimizers
    dx = (x_solver - x_nm)[:,0]
    dy = (y_solver - y_nm)[:,0]
    err_x = np.mean(dx**2)
    err_y = np.mean(dy**2)
    err_primal = err_x + err_y
    print('MSE primal optimizers:', err_primal)

    # Difference in objective
    err_obj = np.abs(solver_obj_mean - nm_obj_mean) / solver_obj_mean * 100
    print(f'mean objective value discrepancy: {err_obj:.2f} %')

    # stats to log
    stats = {"n_samples": n_samples,
             "nm_time": nm_time,
             "nm_con_viol_mean": nm_con_viol_mean,
             "nm_obj_mean": nm_obj_mean,
             "solver_time": solver_time,
             "solver_con_viol_mean": solver_con_viol_mean,
             "solver_obj_mean": solver_obj_mean,
             "speedup_factor": speedup_factor,
             "err_primal": err_primal,
             "err_obj": err_obj}