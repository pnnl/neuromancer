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
import numpy as np
from casadi import *
import casadi

from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.constraint import variable
from neuromancer.activations import activations
from neuromancer.dataset import DictDataset
from neuromancer.loss import PenaltyLoss
from neuromancer import blocks
from neuromancer.system import Node


if __name__ == "__main__":

    """
    # # #  Dataset
    """
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
    # # #  mpNLP primal solution map architecture
    """
    # define neural architecture for the solution map
    func = blocks.MLP(insize=2, outsize=2,
                    bias=True,
                    linear_map=slim.maps['linear'],
                    nonlin=activations['relu'],
                    hsizes=[80] * 4)
    # define symbolic solution map with concatenated features (problem parameters)
    xi = lambda a, p: torch.cat([a, p], dim=-1)
    features = Node(xi, ['a', 'p'], ['xi'], name='features')
    sol_map = Node(func, ['xi'], ['x'], name='map')

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
    obj = f.minimize(weight=1.0, name='obj')

    # constraints
    Q_con = 100.  # constraint penalty weights
    con_1 = Q_con*(x >= y)
    con_2 = Q_con*((p/2)**2 <= x**2+y**2)
    con_3 = Q_con*(x**2+y**2 <= p**2)
    con_1.name = 'c1'
    con_2.name = 'c2'
    con_3.name = 'c3'

    # constrained optimization problem construction
    objectives = [obj]
    constraints = [con_1, con_2, con_3]
    components = [features, sol_map]

    # create penalty method loss function
    loss = PenaltyLoss(objectives, constraints)
    # construct constrained optimization problem
    problem = Problem(components, loss)
    # plot computational graph
    problem.plot_graph()
    plt.show(block=True)

    """
    # # #  mpNLP problem solution in Neuromancer
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
    datapoint = {'a': torch.tensor([[a]]), 'p': torch.tensor([[p]]),
                 'name': 'test'}
    model_out = problem(datapoint)
    x_nm = model_out['test_' + "x"][0, 0].detach().numpy()
    y_nm = model_out['test_' + "x"][0, 1].detach().numpy()
    print(x_nm)
    print(y_nm)

    # plot optimal solutions CasADi vs Neuromancer
    ax.plot(sol.value(x), sol.value(y), 'g*', markersize=10)
    ax.plot(x_nm, y_nm, 'r*', markersize=10)
    plt.show(block=True)

