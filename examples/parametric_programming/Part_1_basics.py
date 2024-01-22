"""
Learning to optimize parametric nonlinear programming problem (pNLP) using Neuromancer.

Formulation of the parametric Rosenbrock problem:

    minimize     (1-x)^2 + a*(y-x^2)^2
    subject to   (p/2)^2 <= x^2 + y^2 <= p^2
                 x>=y

    problem parameters:             a, p
    problem decition variables:     x, y

https://en.wikipedia.org/wiki/Rosenbrock_function
"""

import torch
import torch.nn as nn
import numpy as np
import neuromancer.slim as slim
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from casadi import *
import casadi
import time

from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.constraint import variable
from neuromancer.dataset import DictDataset
from neuromancer.loss import PenaltyLoss
from neuromancer.modules import blocks
from neuromancer.system import Node


if __name__ == "__main__":

    """
    # # #  Dataset
    """
    # We synthetically sample a distribution of the optimization problem parameters
    # Each element in the set of sampled parameters represents one instance of the problem
    data_seed = 408
    np.random.seed(data_seed)
    nsim = 5000  # number of datapoints: increase sample density for more robust results
    # create dictionaries with sampled datapoints with uniform distribution
    a_low, a_high, p_low, p_high = 0.2, 1.2, 0.5, 2.0
    samples_train = {"a": torch.FloatTensor(nsim, 1).uniform_(a_low, a_high),
                     "p": torch.FloatTensor(nsim, 1).uniform_(p_low, p_high)}
    samples_dev = {"a": torch.FloatTensor(nsim, 1).uniform_(a_low, a_high),
                   "p": torch.FloatTensor(nsim, 1).uniform_(p_low, p_high)}
    samples_test = {"a": torch.FloatTensor(nsim, 1).uniform_(a_low, a_high),
                   "p": torch.FloatTensor(nsim, 1).uniform_(p_low, p_high)}
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

    # visualize taining and test samples for 2D parametric space
    a_train = samples_train['a'].numpy()
    p_train = samples_train['p'].numpy()
    a_dev = samples_dev['a'].numpy()
    p_dev = samples_dev['p'].numpy()
    plt.figure()
    plt.scatter(a_train, p_train, s=2., c='blue', marker='o')
    plt.scatter(a_dev, p_dev, s=2., c='red', marker='o')
    plt.title('Sampled parametric space for training')
    plt.xlim(a_low, a_high)
    plt.ylim(p_low, p_high)
    plt.grid(True)
    plt.xlabel('a')
    plt.ylabel('p')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    plt.show(block=True)

    """
    # # #  pNLP primal solution map architecture
    """
    # define neural architecture for the trainable solution map
    func = blocks.MLP(insize=2, outsize=2,
                    bias=True,
                    linear_map=slim.maps['linear'],
                    nonlin=nn.ReLU,
                    hsizes=[80] * 4)
    # wrap neural net into symbolic representation of the solution map via the Node class:
    # sol_map(a, p) -> x
    sol_map = Node(func, ['a', 'p'], ['x'], name='map')

    """
    # # #  pNLP variables, objective, and constraints formulation in Neuromancer
    
    variable is a basic symbolic abstraction in Neuromancer
       x = variable("variable_name")                      (instantiates new variable)  
    variable construction supports:
       algebraic expressions:     x**2 + x**3 + 5     (instantiates new variable)  
       slicing:                   x[:, i]             (instantiates new variable)  
       pytorch callables:         torch.sin(x)        (instantiates new variable)  
       constraints definition:    x <= 1.0            (instantiates Constraint object) 
       objective definition:      x.minimize()        (instantiates Objective object) 
    to visualize computational graph of the variable use x.show() method          
    """

    # define decision variables
    x = variable("x")[:, [0]]
    y = variable("x")[:, [1]]
    # problem parameters sampled in the dataset
    p = variable('p')
    a = variable('a')

    # objective function
    f = (1-x)**2 + a*(y-x**2)**2
    obj = f.minimize(weight=1.0, name='obj')

    # constraints
    Q_con = 1.  # constraint penalty weights
    con_1 = Q_con*(x >= y)
    con_2 = Q_con*((p/2)**2 <= x**2+y**2)
    con_3 = Q_con*(x**2+y**2 <= p**2)
    con_1.name = 'c1'
    con_2.name = 'c2'
    con_3.name = 'c3'

    # constrained optimization problem construction
    objectives = [obj]
    constraints = [con_1, con_2, con_3]
    components = [sol_map]

    # create penalty method loss function
    loss = PenaltyLoss(objectives, constraints)
    # construct constrained optimization problem
    problem = Problem(components, loss)
    # plot computational graph
    problem.show()

    """
    # # #  pNLP problem solution in Neuromancer
    """
    optimizer = torch.optim.AdamW(problem.parameters(), lr=0.001)
    # define trainer
    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        test_loader,
        optimizer,
        epochs=200,
        patience=100,
        warmup=100,
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="test_loss",
        eval_metric="dev_loss",
    )

    # Train problem solution map
    best_model = trainer.train()
    best_outputs = trainer.test(best_model)
    # load best model dict
    problem.load_state_dict(best_model)

    """
    CasADi benchmark
    """
    # instantiate casadi optimizaiton problem class
    def NLP_param(a, p, opti_silent=False):
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
        if opti_silent:
            opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        else:
            opts = {}
        opti.solver('ipopt', opts)
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

    # Solution to pNLP via Neuromancer
    datapoint = {'a': torch.tensor([[a]]), 'p': torch.tensor([[p]]),
                 'name': 'test'}
    model_out = problem(datapoint)
    x_nm = model_out['test_' + "x"][0, 0].detach().numpy()
    y_nm = model_out['test_' + "x"][0, 1].detach().numpy()
    print(x_nm)
    print(y_nm)

    # plot optimal solutions CasADi vs Neuromancer
    ax.plot(sol.value(x), sol.value(y), 'g*', markersize=10, label='CasADi')
    ax.plot(x_nm, y_nm, 'r*', fillstyle='none', markersize=10, label='NeuroMANCER')
    plt.legend(bbox_to_anchor=(1.0, 0.15))
    plt.show(block=True)

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
        obj_value_mean = np.mean((1 - x) ** 2 + a * (y - x ** 2) ** 2)
        return obj_value_mean

    # select n number of random samples to evaluate
    n_samples = 1000
    idx = np.random.randint(0, nsim, n_samples)
    p_samples = samples_test['p'][idx]
    a_samples = samples_test['a'][idx]

    # create named dictionary for neuromancer
    datapoint = {'a': a_samples, 'p': p_samples, 'name': 'test'}

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
        prob, x, y = NLP_param(p_samples[i].numpy(), a_samples[i].numpy(), opti_silent=True)
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
    print(f'Neuromancer mean objective value {nm_obj_mean:.4f}\n')

    # Evaluate solver solution
    print(f'Solution for {n_samples} problems via solver obtained in {solver_time:.4f} seconds')
    solver_con_viol_mean = eval_constraints(x_solver, y_solver, p)
    print(f'Solver mean constraints violation {solver_con_viol_mean:.4f}')
    solver_obj_mean = eval_objective(x_solver, y_solver)
    print(f'Solver mean objective value {solver_obj_mean:.4f}\n')

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
