"""
Solve Quadratic Programming (QP) problem using CVXPY toolbox:
minimize     x^2+y^2
subject to   g <= 0

problem parameters:            p
problem decition variables:    x, y

Reference material:
    https://www.cvxpy.org/examples/basic/quadratic_program.html
"""

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects


# Define and solve the CVXPY problem.
x = cp.Variable(1)
y = cp.Variable(1)
p1 = 10.0    # problem parameter
p2 = 10.0    # problem parameter

def QP_param(p1, p2):
    prob = cp.Problem(cp.Minimize(x**2+y**2),
                     [-x - y + p1 <= 0,
                      x + y - p1 - 5 <= 0,
                      x - y + p2 - 5 <= 0,
                      -x + y - p2 <= 0])
    return prob

# test solution
prob = QP_param(p1, p2)
prob.solve()

# # # plots
# test problem parameters
params = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
p = 10.0
x1 = np.arange(-1.0, 10.0, 0.05)
y1 = np.arange(-1.0, 10.0, 0.05)
xx, yy = np.meshgrid(x1, y1)
fig, ax = plt.subplots(3, 3)
row_id = 0
column_id = 0
for i, p in enumerate(params):
    if i % 3 == 0 and i != 0:
        row_id += 1
        column_id = 0
    print(column_id)
    print(row_id)
    # eval objective and constraints
    J = xx ** 2 + yy ** 2
    c1 = xx + yy - p
    c2 = -xx - yy + p + 5
    c3 = -xx + yy - p + 5
    c4 = xx - yy + p
    # Plot
    cp_plot = ax[row_id, column_id].contourf(xx, yy, J, 50,
                                        alpha=0.6)
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
    print(f'primal solution x={x.value}, y={y.value}')
    # print(f'dual solution mu={mu_optim}')
    ax[row_id, column_id].plot(x.value, y.value, 'r*', markersize=10)
    column_id += 1
plt.show()

