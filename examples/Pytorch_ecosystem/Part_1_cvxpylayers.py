"""
This example demonstrates integration of Cvxpylayers into Neuromancer

We will demonstrate this capability on learning to optimize
for parametric nonlinear programming problem (pNLP).

Formulation of the parametric Rosenbrock problem:

    minimize     (1-x)^2 + p*(y-x^2)^2
    subject to   Ax <= b

    varying problem parameters:     p, b
    fixed problem parameters:       A
    problem decition variables:     x, y

https://en.wikipedia.org/wiki/Rosenbrock_function


We will use neuromancer to minimize the objective function,
and cvxpy layers to project the solution onto feasible resion.

"""

import torch
import torch.nn as nn
import numpy as np
import cvxpy
from cvxpylayers.torch import CvxpyLayer

# plotting
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import linprog

# benchmarking
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
    # # #  Problem dimensions
    """
    nx = 2              #  number of decision variables
    n_con = 4           #  number of constraints
    n_p = 1             #  number of objective parameters

    # generate fixed parameters of the inequality constraints: Ax <= b
    torch.manual_seed(7)
    A = torch.FloatTensor(n_con, nx).uniform_(-4, 4)
    x0 = torch.full([nx], 0.5)       # controls center of the polytope
    s0 = torch.full([n_con], 0.2)    # controls offset from the center of the polytope
    b0 = A.mv(x0) + s0

    """
    # # #  Dataset
    """
    # We synthetically sample a distribution of the optimization problem parameters
    # Each element in the set of sampled parameters represents one instance of the problem
    data_seed = 408
    np.random.seed(data_seed)
    nsim = 1000  # number of datapoints: increase sample density for more robust results
    # create dictionaries with sampled datapoints with uniform distribution
    p_low, p_high = 0.2, 5.0,
    b_low, b_high = 0.0, 1.0,
    # we sample objective and constraints parameters
    samples_train = {"p": torch.FloatTensor(nsim, n_p).uniform_(p_low, p_high),
                     "b_param": torch.FloatTensor(nsim, n_con).uniform_(b_low, b_high)}
    samples_dev = {"p": torch.FloatTensor(nsim, n_p).uniform_(p_low, p_high),
                   "b_param": torch.FloatTensor(nsim, n_con).uniform_(b_low, b_high)}
    samples_test = {"p": torch.FloatTensor(nsim, n_p).uniform_(p_low, p_high),
                    "b_param": torch.FloatTensor(nsim, n_con).uniform_(b_low, b_high)}
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
    # define neural architecture for the trainable solution map
    # mapping problem parameters to decitionv ariables
    func = blocks.MLP(insize=n_con+n_p, outsize=nx,
                    bias=True,
                    linear_map=torch.nn.Linear,
                    nonlin=nn.ReLU,
                    hsizes=[80] * 4)
    # wrap neural net into symbolic representation of the solution map via the Node class:
    # sol_map(p, bparam) -> xy
    sol_map = Node(func, ['p', 'b_param'], ['xy'], name='map')

    """
    # # #  Neuromancer variables and objective function 
    """
    # define decision variables
    xy = variable("xy")
    x = variable("xy")[:, [0]]
    y = variable("xy")[:, [1]]
    # problem parameters sampled in the dataset
    p = variable('p')
    b_param = variable('b_param')

    # objective function
    f = (1-x)**2 + p*(y-x**2)**2
    nm_obj = f.minimize(weight=10.0, name='obj')

    """
    # # #  cvxpy layer
    """
    # cvxpy projection problem
    A_cvxpy = A.detach().numpy()
    b0_cvxpy = b0.detach().numpy()
    b_cvxpy = cvxpy.Parameter(n_con)
    xy_net = cvxpy.Parameter(nx)      # primal decision from neural net
    xy_cvxpy = cvxpy.Variable(nx)     # cvxpy decision variable
    cvxpy_obj = cvxpy.Minimize(1.0 * cvxpy.sum_squares(xy_net - xy_cvxpy))
    cvxpy_cons = [xy_cvxpy@A_cvxpy.T <= b0_cvxpy + b_cvxpy]
    cvxpy_prob = cvxpy.Problem(cvxpy_obj, cvxpy_cons)

    # cvxpy layer
    cvxpy_layer = CvxpyLayer(cvxpy_prob,
                       parameters=[b_cvxpy, xy_net],
                       variables=[xy_cvxpy])
    # symbolic wrapper: sol_map(bparam, xy) -> xy
    project = Node(cvxpy_layer, ['b_param', 'xy'], ['xy_cvx'], name='proj')

    # corrected variable by the cvxpy layer
    xy_cvx = variable("xy_cvx")
    # cvxpy-supervised loss for the neural net
    residual = torch.abs(xy - xy_cvx)
    cvxp_loss = 1.*(residual == 0)

    """
    # # #  Differentiable Parametric optimization problem
    """
    # constrained optimization problem construction
    objectives = [nm_obj, cvxp_loss]
    constraints = []
    components = [sol_map, project]

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
        epochs=10,
        patience=10,
        warmup=10,
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
    Plots
    """
    p = torch.FloatTensor(n_p).uniform_(p_low, p_high)
    b_param = torch.rand(n_con)
    b = b0 + b_param

    x1 = np.arange(-0.5, 1.5, 0.02)
    y1 = np.arange(-0.5, 1.5, 0.02)
    xx, yy = np.meshgrid(x1, y1)

    x_flat = xx.flatten()
    y_flat = yy.flatten()
    xy_samples = np.stack([x_flat, y_flat])
    # sampled constraints Ax - b <= 0
    A_np = A.detach().numpy()
    b_np = b.detach().numpy()
    C_samples = np.subtract(np.matmul(A_np, xy_samples).T, b_np).T
    C_samples = C_samples.reshape(n_con, x1.shape[0], y1.shape[0])

    feasible = (C_samples <= 0.0)
    feasible_region = feasible.sum(0) == n_con

    # eval objective and constraints
    p_np = p.detach().numpy()
    J = (1 - xx) ** 2 + p_np * (yy - xx ** 2) ** 2

    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(xx, yy, J,
                     levels=[0, 0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 16],
                     alpha=0.6)
    fig.colorbar(cp)
    ax.set_title('Rosenbrock problem')
    for k in range(n_con):
        cg1 = ax.contour(xx, yy, -C_samples[k], [0], colors='mediumblue', alpha=0.7)
        plt.setp(cg1.collections,
                 path_effects=[patheffects.withTickedStroke()], alpha=0.7)

    # Solution to pNLP via Neuromancer
    datapoint = {'p': p, 'b_param': b_param,
                 'name': 'test'}
    # evaluate neuromancer model
    model_out = problem.step(datapoint)

    # neural net solution
    x_nm_net = model_out["xy"][0].detach().numpy()
    y_nm_net = model_out["xy"][1].detach().numpy()
    print(x_nm_net)
    print(y_nm_net)

    # cvxpy projected solution
    x_nm_cvx = model_out["xy_cvx"][0].detach().numpy()
    y_nm_cvx = model_out["xy_cvx"][1].detach().numpy()
    print(x_nm_cvx)
    print(y_nm_cvx)

    # plot optimal solutions CasADi vs Neuromancer
    ax.plot(x_nm_net, y_nm_net, 'r*', fillstyle='none', markersize=15, label='net')
    ax.plot(x_nm_cvx, y_nm_cvx, 'g*', fillstyle='none', markersize=10, label='cvxpy')
    plt.legend(bbox_to_anchor=(1.0, 0.15))
    plt.show(block=True)

