"""
Learning to optimize parametric Quadratic Programming (pQP)
Problem formulation pQP:
    minimize     x^2 + y^2
    subject to
               -x - y + p1 <= 0,
               x + y - p1 - 1 <= 0,
               x - y + p2 - 1 <= 0,
               -x + y - p2 <= 0

An initial prediction of the solution is made by a neural network,
then a final layer is added to apply a trainable optimization routine for improved solutions.

This work utilizes the results from: https://arxiv.org/abs/2404.00882

problem parameters:            p1, p2
problem decision variables:    x, y
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

from neuromancer.modules import lopo




if __name__ == "__main__":

    """
    # # #  Dataset
    """
    data_seed = 408
    np.random.seed(data_seed)
    nsim = 2000 # number of datapoints: increase sample density for more robust results
    # create dictionaries with sampled datapoints with uniform distribution
    p_low, p_high = -2.0, 2.0
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
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, num_workers=0,
                                                collate_fn=train_data.collate_fn, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=100, num_workers=0,
                                                collate_fn=dev_data.collate_fn, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, num_workers=0,
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
    # define symbolic solution map with concatenated features (problem parameters)
    xi = lambda p1, p2: torch.cat([p1, p2], dim=-1)
    features = Node(xi, ['p1', 'p2'], ['xi'], name='features')
    sol_map = Node(func, ['xi'], ['x'], name='map')
    # trainable components of the problem solution
    components = [features, sol_map]
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
    g2 = x + y - p1 - 1
    con_2 = Q_con*(g2 <= 0)
    con_2.name = 'c2'
    g3 = x - y + p2 - 1
    con_3 = Q_con*(g3 <= 0)
    con_3.name = 'c3'
    g4 = -x + y - p2
    con_4 = Q_con*(g4 <= 0)
    con_4.name = 'c4'
    constraints = [con_1, con_2, con_3, con_4]
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
    optimizer = torch.optim.AdamW(problem.parameters(), lr=1e-3)
    # define trainer
    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        test_loader,
        optimizer,
        epochs=200,
        patience=200,
        warmup=100,
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="test_loss",
        eval_metric="dev_loss",
    )
    # Train solution map
    best_model = trainer.train()

    '''
    Add on a correction layer
    '''
    '''
    DEFINE THE OBJECTIVE
    '''
    def f_obj(x,parms):
        return torch.pow( x[0] ,2) + torch.pow( x[1], 2)
    '''
    DEFINE THE CONSTRAINTS
    '''
    def F_ineq(x,parms):
        c_1 = -x[0] - x[1] + parms[0]
        c_2 = x[0] + x[1] - parms[0] - 1 
        c_3 = x[0] - x[1] + parms[1] - 1 
        c_4 = -x[0] + x[1] - parms[1]
        return torch.stack((c_1,c_2,c_3,c_4))
    ''' 
    SET PROBLEM PARAMETERS
    '''
    num_steps = 15 # number of DR iterations to take
    x_dim = 2 # dimension of primal variable
    n_ineq = 4 #number of inequality constraints
    parm_dim = 2 #dimension of parameters for problem
    ''' 
    INITIALIZE THE SOLVER
    Can choose either DR or ADMM here
    '''
    solver = 'DR'
    #solver = 'ADMM'

    #Define a metric that will be trained
    lb_P = 1.0/5.0
    ub_P = 5.0
    scl_lb_P = 0.005
    scl_ub_P = 1.0
    n_dim = x_dim + n_ineq
    metric = lopo.ParaMetricDiagonal(n_dim,parm_dim,ub_P,lb_P,scl_upper_bound=scl_ub_P,scl_lower_bound=scl_lb_P)

    if solver == 'DR':
        solver = lopo.DRSolver(
            f_obj = f_obj, 
            F_ineq = F_ineq,
            x_dim = x_dim, 
            n_ineq = n_ineq, 
            JF_fixed=True,
            Hf_fixed = True,
            num_steps = num_steps,
            metric = metric
            )
    if solver == 'ADMM':
        solver = lopo.ADMMSolver(
            f_obj = f_obj, 
            F_ineq = F_ineq,
            x_dim = x_dim, 
            n_ineq = n_ineq, 
            JF_fixed=True,
            Hf_fixed = True,
            num_steps = num_steps,
            metric = metric
            )
    '''
    REMAP THROUGH SOLVER CORRECTION
    '''
    sol_map = Node(func, ['xi'], ['x_predicted'], name='map')
    correction = Node(solver,['x_predicted','xi'],['x','cnv_gap'])
    components = [features, sol_map, correction]
    ### ADD A CONVERGENCE PENALTY TO TRAIN SOLVER
    cnv_gap = variable("cnv_gap")
    f_cnv = cnv_gap
    cnv_obj = f_cnv.minimize(weight=1e8, name='cnv_obj')
    objectives = [cnv_obj]
    constraints = []
    # create loss function
    loss = PenaltyLoss(objectives, constraints)
    # construct constrained optimization problem
    problem = Problem(components, loss)
    '''  
    TRAIN THE METRIC
    '''
    optimizer = torch.optim.AdamW(solver.parameters(), lr=1e-2)
    # define trainer
    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        test_loader,
        optimizer,
        epochs=20,
        patience=200,
        warmup=100,
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="test_loss",
        eval_metric="dev_loss",
    )
    # Train solution map
    best_model = trainer.train()


    '''
    Compare to CVXPY Solver
    '''
    """
    CVXPY benchmarks
    """
    # Define the CVXPY problems.
    def QP_param(p1, p2):
        x = cp.Variable(1)
        y = cp.Variable(1)
        prob = cp.Problem(cp.Minimize(x ** 2 + y ** 2),
                            [-x - y + p1 <= 0,
                            x + y - p1 - 1 <= 0,
                            x - y + p2 - 1 <= 0,
                            -x + y - p2 <= 0])
        return prob, x, y
    """
    Plots
    """
    import matplotlib as mpl
    plt.rcParams.update({'font.size': 14})
    # test problem parameters
    params = [-1.25, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.25]
    x1 = np.arange(-1.5, 1.5, 0.01)
    y1 = np.arange(-1.5, 1.5, 0.01)
    xx, yy = np.meshgrid(x1, y1)
    fig, ax = plt.subplots(3,3,sharex = True,sharey=True)
    row_id = 0
    column_id = 0
    for i, p in enumerate(params):
        if i % 3 == 0 and i != 0:
            row_id += 1
            column_id = 0
        # eval and plot objective and constraints
        J = xx ** 2 + yy ** 2
        cp_plot = ax[row_id, column_id].contourf(xx, yy, J, 50, alpha=0.4,cmap = mpl.colormaps['bone'],linewidth = 10)
        ax[row_id, column_id].set_title(f'p={p}')
        c1 = xx + yy - p
        c2 = -xx - yy + p + 1
        c3 = -xx + yy - p + 1
        c4 = xx - yy + p
        cg1 = ax[row_id, column_id].contour(xx, yy, c1, [0], colors='k', alpha=0.5)
        cg2 = ax[row_id, column_id].contour(xx, yy, c2, [0], colors='k', alpha=0.5)
        cg3 = ax[row_id, column_id].contour(xx, yy, c3, [0], colors='k', alpha=0.5)
        cg4 = ax[row_id, column_id].contour(xx, yy, c4, [0], colors='k', alpha=0.5)
        plt.setp(cg1.collections,
                    path_effects=[patheffects.withTickedStroke()], alpha=0.5)
        plt.setp(cg2.collections,
                    path_effects=[patheffects.withTickedStroke()], alpha=0.5)
        plt.setp(cg3.collections,
                    path_effects=[patheffects.withTickedStroke()], alpha=0.5)
        plt.setp(cg4.collections,
                    path_effects=[patheffects.withTickedStroke()], alpha=0.5)
        # Solve CVXPY problem
        prob, x, y = QP_param(p, p)
        prob.solve()
        # Solve via neuromancer
        datapoint = {'p1': torch.tensor([[p]]), 'p2': torch.tensor([[p]]),
                        'name': 'test'}
        model_out = problem(datapoint)
        x_nm = model_out['test_' + "x"][0, 0].detach().numpy()
        y_nm = model_out['test_' + "x"][0, 1].detach().numpy()
        print(f'primal solution x={x.value}, y={y.value}')
        print(f'parameter p={p, p}')
        print(f'primal solution Neuromancer x1={x_nm}, x2={y_nm}')
        # Plot optimal solutions
        ax[row_id, column_id].plot(x.value, y.value, 'g*', markersize=10)
        ax[row_id, column_id].plot(x_nm, y_nm, 'r*', markersize=10)
        column_id += 1
    fig.tight_layout()
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
        con_2_viol = np.maximum(0, x + y - p1 - 1)
        con_3_viol = np.maximum(0, x - y + p2 - 1)
        con_4_viol = np.maximum(0, -x + y - p2)
        con_viol = con_1_viol + con_2_viol + con_3_viol + con_4_viol
        con_viol_mean = np.mean(con_viol)
        return con_viol_mean
    def eval_objective(x, y, a1=1, a2=1):
        obj_value_mean = np.mean(a1 * x**2 + a2 * y**2)
        return obj_value_mean
    # Solve via neuromancer
    with torch.no_grad():
        t = time.time()
        samples_test['name'] = 'test'
        model_out = problem(samples_test)
        nm_time = time.time() - t
    x_nm = model_out['test_' + "x"][:, [0]].detach().numpy()
    y_nm = model_out['test_' + "x"][:, [1]].detach().numpy()
    x_nm_noDR = model_out['test_x_predicted'][:, [0]].detach().numpy()
    y_nm_noDR = model_out['test_x_predicted'][:, [1]].detach().numpy()
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
    p1_vec = samples_test['p1'].detach().numpy()
    p2_vec = samples_test['p2'].detach().numpy()

    # Evaluate solver solution
    print(f'Solution for {nsim} problems via CVXPY obtained in {solver_time:.4f} seconds')
    solver_con_viol_mean = eval_constraints(x_solver, y_solver, p1_vec, p2_vec)
    print(f'CVXPY mean constraints violation {solver_con_viol_mean:.4f}')
    solver_obj_mean = eval_objective(x_solver, y_solver)
    print(f'CVXPY mean objective value {solver_obj_mean:.4f}\n')

    # Evaluate neuromancer solution no DR
    nm_con_viol_mean = eval_constraints(x_nm_noDR, y_nm_noDR, p1_vec, p2_vec)
    print(f'Neuromancer mean constraints violation no correction {nm_con_viol_mean:.4f}')
    nm_obj_mean = eval_objective(x_nm_noDR, y_nm_noDR)
    print(f'Neuromancer mean objective value no correction {nm_obj_mean:.4f}')

    # Difference in primal optimizers
    dx = (x_solver - x_nm_noDR)[:,0]
    dy = (y_solver - y_nm_noDR)[:,0]
    err_x = np.mean(dx**2)
    err_y = np.mean(dy**2)
    err_primal = err_x + err_y
    print('MSE primal optimizers no correction:', err_primal)

    # Difference in objective
    err_obj = np.abs(solver_obj_mean - nm_obj_mean) / solver_obj_mean * 100
    print(f'mean objective value discrepancy np correction: {err_obj:.2f} % \n')



    # Evaluate neuromancer solution
    print(f'Solution for {nsim} problems with correction obtained in {nm_time:.4f} seconds')
    nm_con_viol_mean = eval_constraints(x_nm, y_nm, p1_vec, p2_vec)
    print(f'mean constraints violation with correction {nm_con_viol_mean:.4f}')
    nm_obj_mean = eval_objective(x_nm, y_nm)
    print(f'mean objective value with correction {nm_obj_mean:.4f}')


    # neuromancer solver comparison
    speedup_factor = solver_time/nm_time
    print(f'Solution speedup factor {speedup_factor:.4f}')

    # Difference in primal optimizers0
    dx = (x_solver - x_nm)[:,0]
    dy = (y_solver - y_nm)[:,0]
    err_x = np.mean(dx**2)
    err_y = np.mean(dy**2)
    err_primal = err_x + err_y
    err_primal = np.mean((dx**2 + dy**2))
    rel_err_primal = np.mean((dx**2 + dy**2))/np.mean((x_solver**2 + y_solver**2 ))
    print('MSE primal optimizers:', rel_err_primal)

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

