"""
Learning to optimize parametric Quadratic Programming (pQP) and
parametric Quadratically Constrained Quadratic Programming (pQCQP)
problems using Neuromancer.

Problem formulation pQP:
    minimize     x^2 + y^2
    subject to
               -x - y + p1 <= 0,
               x + y - p1 - 5 <= 0,
               x - y + p2 - 5 <= 0,
               -x + y - p2 <= 0

Problem formulation pQCQP:
    minimize     x^2 + y^2
    subject to
           -x - y + p1 <= 0,
           x^2 + y^2 <= p2^2

    problem parameters:            p1, p2
    problem decition variables:    x, y
"""


import cvxpy as cp
import numpy as np
import time
import torch
import torch.nn as nn
import neuromancer.slim as slim
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects

from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.constraint import variable
from neuromancer.dataset import DictDataset
from neuromancer.loss import PenaltyLoss
from neuromancer.modules import blocks
from neuromancer.system import Node


if __name__ == "__main__":

    problem_type = 'pQP'   # select from 'pQP' or 'pQCQP'

    """
    # # #  Dataset
    """
    data_seed = 408
    np.random.seed(data_seed)
    nsim = 3000  # number of datapoints: increase sample density for more robust results
    # create dictionaries with sampled datapoints with uniform distribution
    p_low, p_high = 1.0, 11.0
    samples_train = {"p1": torch.FloatTensor(nsim, 1).uniform_(p_low, p_high),
                     "p2": torch.FloatTensor(nsim, 1).uniform_(p_low, p_high)}
    samples_dev = {"p1": torch.FloatTensor(nsim, 1).uniform_(p_low, p_high),
                   "p2": torch.FloatTensor(nsim, 1).uniform_(p_low, p_high)}
    samples_test = {"p1": torch.FloatTensor(nsim, 1).uniform_(p_low, p_high),
                   "p2": torch.FloatTensor(nsim, 1).uniform_(p_low, p_high)}
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
    a_train = samples_train['p1'].numpy()
    p_train = samples_train['p2'].numpy()
    a_dev = samples_dev['p1'].numpy()
    p_dev = samples_dev['p2'].numpy()
    plt.figure()
    plt.scatter(a_train, p_train, s=2., c='blue', marker='o')
    plt.scatter(a_dev, p_dev, s=2., c='red', marker='o')
    plt.title('Sampled parametric space for training')
    plt.xlim(p_low, p_high)
    plt.ylim(p_low, p_high)
    plt.grid(True)
    plt.xlabel('p1')
    plt.ylabel('p2')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    plt.show(block=True)


    """
    # # #  pQP primal solution map architecture
    """
    # define neural architecture for the solution map
    func = blocks.MLP(insize=2, outsize=2,
                    bias=True,
                    linear_map=slim.maps['linear'],
                    nonlin=nn.ReLU,
                    hsizes=[80] * 4)
    # define symbolic solution map
    sol_map = Node(func, ['p1', 'p2'], ['x'], name='map')
    # trainable components of the problem solution
    components = [sol_map]

    """
    # # #  mpQP objective and constraints formulation in Neuromancer
    """
    # variables
    x = variable("x")[:, [0]]
    y = variable("x")[:, [1]]
    # sampled parameters
    p1 = variable('p1')
    p2 = variable('p2')

    # objective function
    f = x ** 2 + y ** 2
    obj = f.minimize(weight=1.0, name='obj')
    objectives = [obj]

    # constraints
    Q_con = 100.
    g1 = -x - y + p1
    con_1 = Q_con * (g1 <= 0)
    con_1.name = 'c1'
    if problem_type == 'pQP':  # constraints for QP
        g2 = x + y - p1 - 5
        con_2 = Q_con*(g2 <= 0)
        con_2.name = 'c2'
        g3 = x - y + p2 - 5
        con_3 = Q_con*(g3 <= 0)
        con_3.name = 'c3'
        g4 = -x + y - p2
        con_4 = Q_con*(g4 <= 0)
        con_4.name = 'c4'
        constraints = [con_1, con_2, con_3, con_4]
    elif problem_type == 'pQCQP':  # constraints for QCQP
        g2 = x**2+y**2 - p2**2
        con_2 = Q_con*(g2 <= 0)
        con_2.name = 'c2'
        constraints = [con_1, con_2]

    """
    # # #  pQP problem formulation in Neuromancer
    """
    # create penalty method loss function
    loss = PenaltyLoss(objectives, constraints)
    # construct constrained optimization problem
    problem = Problem(components, loss)

    """
    # # #  pQP problem solution in Neuromancer
    """
    optimizer = torch.optim.AdamW(problem.parameters(), lr=0.001)
    # define trainer
    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        test_loader,
        optimizer,
        epochs=400,
        patience=100,
        warmup=100,
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="test_loss",
        eval_metric="dev_loss",
    )

    # Train solution map
    best_model = trainer.train()
    best_outputs = trainer.test(best_model)
    # load best model dict
    problem.load_state_dict(best_model)

    """
    CVXPY benchmarks
    """
    # Define the CVXPY problems.

    def QP_param(p1, p2):
        x = cp.Variable(1)
        y = cp.Variable(1)
        prob = cp.Problem(cp.Minimize(x ** 2 + y ** 2),
                          [-x - y + p1 <= 0,
                           x + y - p1 - 5 <= 0,
                           x - y + p2 - 5 <= 0,
                           -x + y - p2 <= 0])
        return prob, x, y

    def QCQP_param(p1, p2):
        x = cp.Variable(1)
        y = cp.Variable(1)
        prob = cp.Problem(cp.Minimize(x ** 2 + y ** 2),
                      [-x - y + p1 <= 0,
                       x ** 2 + y ** 2 - p2 ** 2 <= 0])
        return prob, x, y

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

        # eval and plot objective and constraints
        J = xx ** 2 + yy ** 2
        cp_plot = ax[row_id, column_id].contourf(xx, yy, J, 50, alpha=0.4)
        ax[row_id, column_id].set_title(f'QP p={p}')
        if problem_type == 'pQP':  # constraints for QP
            c1 = xx + yy - p
            c2 = -xx - yy + p + 5
            c3 = -xx + yy - p + 5
            c4 = xx - yy + p
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
        if problem_type == 'pQCQP':  # constraints for QCQP
            c1 = xx + yy - p
            c2 = - xx**2 - yy**2 + p**2
            cg1 = ax[row_id, column_id].contour(xx, yy, c1, [0], colors='mediumblue', alpha=0.7)
            cg2 = ax[row_id, column_id].contour(xx, yy, c2, [0], colors='mediumblue', alpha=0.7)
            plt.setp(cg1.collections,
                     path_effects=[patheffects.withTickedStroke()], alpha=0.7)
            plt.setp(cg2.collections,
                     path_effects=[patheffects.withTickedStroke()], alpha=0.7)
        fig.colorbar(cp_plot, ax=ax[row_id,column_id])

        # Solve CVXPY problem
        if problem_type == 'pQP':
            prob, x, y = QP_param(p, p)
        elif problem_type == 'pQCQP':
            prob, x, y = QCQP_param(p, p)
        prob.solve()

        # Solve via neuromancer
        datapoint = {'p1': torch.tensor([[p]]), 'p2': torch.tensor([[p]]),
                     'name': 'test'}
        model_out = problem(datapoint)
        x_nm = model_out['test_' + "x"][0, 0].detach().numpy()
        y_nm = model_out['test_' + "x"][0, 1].detach().numpy()

        print(f'primal solution {problem_type} x={x.value}, y={y.value}')
        print(f'parameter p={p, p}')
        print(f'primal solution Neuromancer x1={x_nm}, x2={y_nm}')
        print(f' f: {model_out["test_" + f.key]}')
        print(f' g1: {model_out["test_" + g1.key]}')
        print(f' g2: {model_out["test_" + g2.key]}')
        if problem_type == 'pQP':
            print(f' g3: {model_out["test_" + g3.key]}')
            print(f' g4: {model_out["test_" + g4.key]}')

        # Plot optimal solutions
        ax[row_id, column_id].plot(x.value, y.value, 'g*', markersize=10)
        ax[row_id, column_id].plot(x_nm, y_nm, 'r*', markersize=10)
        column_id += 1
    plt.show()
    plt.show(block=True)
    plt.interactive(False)

    """
    Benchmark Solution
    """

    def eval_constraints(x, y, p1, p2):
        """
        evaluate mean constraints violations
        """
        con_1_viol = np.maximum(0, -x - y + p1)
        con_2_viol = np.maximum(0, x + y - p1 - 5)
        con_3_viol = np.maximum(0, x - y + p2 - 5)
        con_4_viol = np.maximum(0, -x + y - p2)
        con_viol = con_1_viol + con_2_viol + con_3_viol + con_4_viol
        con_viol_mean = np.mean(con_viol)
        return con_viol_mean

    def eval_objective(x, y, a1=1, a2=1):
        obj_value_mean = np.mean(a1 * x**2 + a2 * y**2)
        return obj_value_mean

    # Solve via neuromancer
    t = time.time()
    samples_test['name'] = 'test'
    model_out = problem(samples_test)
    nm_time = time.time() - t
    x_nm = model_out['test_' + "x"][:, [0]].detach().numpy()
    y_nm = model_out['test_' + "x"][:, [1]].detach().numpy()

    # Solve via solver
    t = time.time()
    x_solver, y_solver = [], []
    for i in range(0, nsim):
        p1 = samples_test['p1'][i].detach().numpy()
        p2 = samples_test['p2'][i].detach().numpy()
        prob, x, y = QP_param(p1, p2)
        prob.solve(solver='ECOS_BB', verbose=False)
        prob.solve()
        x_solver.append(x.value)
        y_solver.append(y.value)
    solver_time = time.time() - t
    x_solver = np.asarray(x_solver)
    y_solver = np.asarray(y_solver)

    # Evaluate neuromancer solution
    print(f'Solution for {nsim} problems via Neuromancer obtained in {nm_time:.4f} seconds')
    nm_con_viol_mean = eval_constraints(x_nm, y_nm, p1, p2)
    print(f'Neuromancer mean constraints violation {nm_con_viol_mean:.4f}')
    nm_obj_mean = eval_objective(x_nm, y_nm)
    print(f'Neuromancer mean objective value {nm_obj_mean:.4f}')

    # Evaluate solver solution
    print(f'Solution for {nsim} problems via solver obtained in {solver_time:.4f} seconds')
    solver_con_viol_mean = eval_constraints(x_solver, y_solver, p1, p2)
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
    stats = {"nsim": nsim,
             "nm_time": nm_time,
             "nm_con_viol_mean": nm_con_viol_mean,
             "nm_obj_mean": nm_obj_mean,
             "solver_time": solver_time,
             "solver_con_viol_mean": solver_con_viol_mean,
             "solver_obj_mean": solver_obj_mean,
             "speedup_factor": speedup_factor,
             "err_primal": err_primal,
             "err_obj": err_obj}