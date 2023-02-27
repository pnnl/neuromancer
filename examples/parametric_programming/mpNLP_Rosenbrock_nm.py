"""
Solve the parametric Rosenbrock problem, formulated as the NLP using Neuromancer toolbox:
minimize     (1-x)^2 + a*(y-x^2)^2
subject to   (p/2)^2 <= x^2 + y^2 <= p^2
             x>=y

problem parameters:             a, p
problem decition variables:     x, y

https://en.wikipedia.org/wiki/Rosenbrock_function
"""


import torch
import neuromancer.slim as slim
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib import cm
import numpy as np
from casadi import *
import time
import casadi
import os
import imageio.v2 as imageio

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
    gp.add("-epochs", type=int, default=400,
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
    nsim = 10000  # number of datapoints: increase sample density for more robust results
    a_low, a_high, p_low, p_high = 0.2, 1.2, 0.5, 2.0
    samples = {"a": np.random.uniform(low=a_low, high=a_high, size=(nsim, 1)),
               "p": np.random.uniform(low=p_low, high=p_high, size=(nsim, 1))}
    data, dims = get_static_dataloaders(samples)
    train_data, dev_data, test_data = data

    # visualize taining and test samples for 2D parametric space
    a_train = train_data.dataset.get_full_batch()['a'].numpy()
    p_train = train_data.dataset.get_full_batch()['p'].numpy()
    a_test = test_data.dataset.get_full_batch()['a'].numpy()
    p_test = test_data.dataset.get_full_batch()['p'].numpy()
    plt.figure()
    plt.scatter(a_train, p_train, s=1., c='blue', marker='o')
    plt.scatter(a_test, p_test, s=1., c='red', marker='o')
    plt.title('Sampled parametric space for training')
    plt.xlim(a_low, a_high)
    plt.ylim(p_low, p_high)
    plt.grid(True)
    plt.xlabel('a')
    plt.ylabel('p')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show(block=True)
    plt.interactive(False)

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
    x = variable("x")[:, [0]]
    y = variable("x")[:, [1]]
    # sampled parameters
    p = variable('p')
    a = variable('a')

    # objective function
    f = (1-x)**2 + a*(y-x**2)**2
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
    def NLP_param(a, p):
        opti = casadi.Opti()
        # define variables
        x = opti.variable()
        y = opti.variable()
        p_opti = opti.parameter()
        a_opti = opti.parameter()
        # define objective and constraints
        opti.minimize((1 - x) ** 2 + a_opti * (y - x ** 2) ** 2)
        opti.subject_to(x >= y)
        opti.subject_to((p_opti / 2) ** 2 <= x ** 2 + y ** 2)
        opti.subject_to(x ** 2 + y ** 2 <= p_opti ** 2)
        # select IPOPT solver and solve the NLP
        opti.solver('ipopt')
        # set parametric values
        opti.set_value(p_opti, p)
        opti.set_value(a_opti, a)
        return opti, x, y

    # selected parameters for a single instance problem
    p = 1.0
    a = 1.0
    # construct casadi problem
    opti, x, y = NLP_param(a, p)
    # solve NLP via casadi
    sol = opti.solve()
    print(sol.value(x))
    print(sol.value(y))

    """
    Plots
    """
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
    # plt.savefig('figs/mpNLP_Rosenbrock_nm.png')
    plt.show(block=True)
    plt.interactive(False)

    """
    Sensitivity analysis visualisation
    """
    a_range = np.arange(a_low, a_high, 0.1)
    p_range = np.arange(p_low, p_high, 0.1)
    a_mesh, p_mesh = np.meshgrid(a_range, p_range)
    J_mesh_casadi = np.nan*np.ones([p_range.shape[0], a_range.shape[0]])
    J_mesh_nm = np.nan*np.ones([p_range.shape[0], a_range.shape[0]])
    x_opt_casadi = np.nan*np.ones([p_range.shape[0], a_range.shape[0]])
    y_opt_casadi = np.nan*np.ones([p_range.shape[0], a_range.shape[0]])
    x_opt_nm = np.nan*np.ones([p_range.shape[0], a_range.shape[0]])
    y_opt_nm = np.nan*np.ones([p_range.shape[0], a_range.shape[0]])
    filenames = []
    savedir = './Rosebnrock_plots/'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    for i, p in enumerate(p_range):
        for j, a in enumerate(a_range):
            a = numpy.around(a, 1).tolist()
            p = numpy.around(p, 1).tolist()
            print(f'solving the case for p={p}, a={a}')
            # construct casadi problem
            opti, x, y = NLP_param(a, p)
            # solve NLP via casadi
            sol = opti.solve()
            # eval objective and constraints over the whole x, y domain for plotting
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
            # optimal points CasADi vs Neuromancer
            xy_cas = ax.plot(sol.value(x), sol.value(y), 'g*', markersize=10)
            nx_nm = ax.plot(x_nm, y_nm, 'r*', markersize=10)
            # save single instance fig of the solution
            figure_path = os.path.join(savedir, f'mpNLP_Rosenbrock_p={p}_a={a}.png')
            filenames.append(figure_path)
            if not os.path.exists(figure_path):
                plt.savefig(figure_path)
            # objective function values as function of varying params
            J_mesh_casadi[i,j] = (1 - sol.value(x)) ** 2 + a * (sol.value(y) - sol.value(x) ** 2) ** 2
            J_mesh_nm[i,j] = (1 - x_nm) ** 2 + a * (y_nm - x_nm ** 2) ** 2
            # primal solutions as function of varying paramrs
            x_opt_casadi[i,j] = sol.value(x)
            y_opt_casadi[i,j] = sol.value(y)
            x_opt_nm[i,j] = x_nm
            y_opt_nm[i,j] = y_nm

    # generate gif for parametric sensitivities of the solution
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    gif_path = os.path.join(savedir, 'Rosenbrock_sensitivity.gif')
    imageio.mimsave(gif_path, images)

    # plot objective values as function of varying paramters
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(p_mesh, a_mesh, J_mesh_casadi,
                           cmap=cm.viridis, linewidth=0, antialiased=False)
    ax.set(xlabel='$p$')
    ax.set(ylabel='$a$')
    ax.set(zlabel='$objective value$')
    ax.set(title='Casadi')
    figure_path = os.path.join(savedir, f'objective_Casadi.png')
    if not os.path.exists(figure_path):
        plt.savefig(figure_path)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(p_mesh, a_mesh, J_mesh_nm,
                           cmap=cm.viridis, linewidth=0, antialiased=False)
    ax.set(xlabel='$p$')
    ax.set(ylabel='$a$')
    ax.set(zlabel='$objective value$')
    ax.set(title='Neuromancer')
    figure_path = os.path.join(savedir, f'objective_neuromancer.png')
    if not os.path.exists(figure_path):
        plt.savefig(figure_path)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(p_mesh, a_mesh, J_mesh_casadi-J_mesh_nm,
                           cmap=cm.viridis, linewidth=0, antialiased=False)
    ax.set(xlabel='$p$')
    ax.set(ylabel='$a$')
    ax.set(zlabel='$objective value$')
    ax.set(title='Neuromancer error')
    figure_path = os.path.join(savedir, f'objective_error.png')
    if not os.path.exists(figure_path):
        plt.savefig(figure_path)
    plt.figure()
    plt.imshow(J_mesh_casadi, aspect='equal')
    plt.xlabel('a')
    plt.ylabel('p')
    plt.title('Casadi')
    plt.colorbar()
    figure_path = os.path.join(savedir, f'objective_casadi_planar.png')
    if not os.path.exists(figure_path):
        plt.savefig(figure_path)
    plt.figure()
    plt.imshow(J_mesh_nm, aspect='equal')
    plt.xlabel('a')
    plt.ylabel('p')
    plt.title('Neuromancer')
    plt.colorbar()
    figure_path = os.path.join(savedir, f'objective_nm_planar.png')
    if not os.path.exists(figure_path):
        plt.savefig(figure_path)
    plt.figure()
    plt.imshow(J_mesh_casadi-J_mesh_nm, aspect='equal')
    plt.xlabel('a')
    plt.ylabel('p')
    plt.title('Neuromancer error')
    plt.colorbar()
    figure_path = os.path.join(savedir, f'objective_error_planar.png')
    if not os.path.exists(figure_path):
        plt.savefig(figure_path)

    # plot primal solution values as function of varying paramters
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(p_mesh, a_mesh, x_opt_casadi,
                           cmap=cm.viridis, linewidth=0, antialiased=False)
    ax.set(xlabel='$p$')
    ax.set(ylabel='$a$')
    ax.set(zlabel='$x$')
    ax.set(title='Casadi')
    figure_path = os.path.join(savedir, f'x_param_Casadi.png')
    if not os.path.exists(figure_path):
        plt.savefig(figure_path)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(p_mesh, a_mesh, x_opt_nm,
                           cmap=cm.viridis, linewidth=0, antialiased=False)
    ax.set(xlabel='$p$')
    ax.set(ylabel='$a$')
    ax.set(zlabel='$x$')
    ax.set(title='Neuromancer')
    figure_path = os.path.join(savedir, f'x_param_neuromancer.png')
    if not os.path.exists(figure_path):
        plt.savefig(figure_path)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(p_mesh, a_mesh, y_opt_casadi,
                           cmap=cm.viridis, linewidth=0, antialiased=False)
    ax.set(xlabel='$p$')
    ax.set(ylabel='$a$')
    ax.set(zlabel='$y$')
    ax.set(title='Casadi')
    figure_path = os.path.join(savedir, f'y_param_Casadi.png')
    if not os.path.exists(figure_path):
        plt.savefig(figure_path)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(p_mesh, a_mesh, y_opt_nm,
                           cmap=cm.viridis, linewidth=0, antialiased=False)
    ax.set(xlabel='$p$')
    ax.set(ylabel='$a$')
    ax.set(zlabel='$y$')
    ax.set(title='Neuromancer')
    figure_path = os.path.join(savedir, f'y_param_neuromancer.png')
    if not os.path.exists(figure_path):
        plt.savefig(figure_path)


    """
    Benchmark solution performance
    """
    def eval_constraints(x, y, p):
        """
        evaluate mean constraints violations
        """
        con_1_viol = np.maximum(0, y - x)
        con_2_viol = np.maximum(0, (p / 2) ** 2 - (x ** 2 + y ** 2))
        con_3_viol = np.maximum(0, x ** 2 + y ** 2 - p ** 2)
        con_viol = con_1_viol + con_2_viol + con_3_viol
        con_viol_mean = np.mean(con_viol)
        return con_viol_mean

    def eval_objective(x, y, a):
        obj_value_mean = np.mean((1 - x) ** 2 + a * (y - x ** 2) ** 2)
        return obj_value_mean

    # fix random seeds
    torch.manual_seed(args.data_seed)
    np.random.seed(args.data_seed)

    # select n number of random samples to evaluate
    n_samples = 1000
    idx = np.random.randint(0, nsim, n_samples)
    a = samples['a'][idx]
    p = samples['p'][idx]

    # create named dictionary for neuromancer
    datapoint = {}
    datapoint['a'] = torch.tensor(a).float()
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
        prob, x, y = NLP_param(a[i], p[i])
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
    nm_obj_mean = eval_objective(x_nm, y_nm, a)
    print(f'Neuromancer mean objective value {nm_obj_mean:.4f}')

    # Evaluate solver solution
    print(f'Solution for {n_samples} problems via solver obtained in {solver_time:.4f} seconds')
    solver_con_viol_mean = eval_constraints(x_solver, y_solver, p)
    print(f'Solver mean constraints violation {solver_con_viol_mean:.4f}')
    solver_obj_mean = eval_objective(x_solver, y_solver, a)
    print(f'Solver mean objective value {solver_obj_mean:.4f}')

    # neuromancer solver comparison
    speedup_factor = solver_time / nm_time
    print(f'Solution speedup factor {speedup_factor:.4f}')

    # Difference in primal optimizers
    dx = (x_solver - x_nm)[:, 0]
    dy = (y_solver - y_nm)[:, 0]
    err_x = np.mean(dx ** 2)
    err_y = np.mean(dy ** 2)
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
