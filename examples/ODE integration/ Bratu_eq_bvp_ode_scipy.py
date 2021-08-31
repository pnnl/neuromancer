"""

Bratu’s problem:
y'' + k * exp(y) = 0
y(0) = y(1) = 0

We rewrite the equation as a first-order system:
y1' = y2
y2' = -exp(y1)

"""


import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt


# define Bratu equation
def fun(x, y):
    return np.vstack((y[1], -np.exp(y[0])))

# define boundary condition residuals
def bc(ya, yb):
    return np.array([ya[0], yb[0]])

# Define the initial mesh with 5 nodes:
x = np.linspace(0, 1, 5)

# This problem is known to have two solutions.
# To obtain both of them, we use two different initial guesses for y.
# We denote them by subscripts a and b.
y_a = np.zeros((2, x.size))
y_b = np.zeros((2, x.size))
y_b[0] = 3

# solve boundary value problem from two initial conditions
res_a = solve_bvp(fun, bc, x, y_a)
res_b = solve_bvp(fun, bc, x, y_b)

# plot solution
x_plot = np.linspace(0, 1, 100)
y_plot_a = res_a.sol(x_plot)[0]
y_plot_b = res_b.sol(x_plot)[0]
plt.plot(x_plot, y_plot_a, label='y_a')
plt.plot(x_plot, y_plot_b, label='y_b')
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()