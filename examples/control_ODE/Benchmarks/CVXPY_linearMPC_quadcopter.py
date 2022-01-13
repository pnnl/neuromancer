"""
linear-quadratic MPC
example from: https://osqp.org/docs/examples/mpc.html
"""

from cvxpy import *
import numpy as np
import scipy as sp
from scipy import sparse
from pylab import *
import time

# Discrete time model of a quadcopter
Ad = sparse.csc_matrix([
    [1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0., 0., 0., 0.1, 0., 0., 0.],
    [0.0488, 0., 0., 1., 0., 0., 0.0016, 0., 0., 0.0992, 0., 0.],
    [0., -0.0488, 0., 0., 1., 0., 0., -0.0016, 0., 0., 0.0992, 0.],
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.0992],
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
    [0.9734, 0., 0., 0., 0., 0., 0.0488, 0., 0., 0.9846, 0., 0.],
    [0., -0.9734, 0., 0., 0., 0., 0., -0.0488, 0., 0., 0.9846, 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.9846]
])
Bd = sparse.csc_matrix([
    [0., -0.0726, 0., 0.0726],
    [-0.0726, 0., 0.0726, 0.],
    [-0.0152, 0.0152, -0.0152, 0.0152],
    [-0., -0.0006, -0., 0.0006],
    [0.0006, 0., -0.0006, 0.0000],
    [0.0106, 0.0106, 0.0106, 0.0106],
    [0, -1.4512, 0., 1.4512],
    [-1.4512, 0., 1.4512, 0.],
    [-0.3049, 0.3049, -0.3049, 0.3049],
    [-0., -0.0236, 0., 0.0236],
    [0.0236, 0., -0.0236, 0.],
    [0.2107, 0.2107, 0.2107, 0.2107]])
[nx, nu] = Bd.shape

# Constraints
u0 = 10.5916
umin = np.array([9.6, 9.6, 9.6, 9.6]) - u0
umax = np.array([13., 13., 13., 13.]) - u0
xmin = np.array([-np.pi / 6, -np.pi / 6, -np.inf, -np.inf, -np.inf, -1.,
                 -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
xmax = np.array([np.pi / 6, np.pi / 6, np.inf, np.inf, np.inf, np.inf,
                 np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

# Objective function
Q = sparse.diags([0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.])
QN = Q
R = 0.1 * sparse.eye(4)

# Initial and reference states
x0 = np.zeros(12)
x0 = 0.2*np.ones(12)
xr = np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

# Prediction horizon
N = 10

# Define problem
u = Variable((nu, N))
x = Variable((nx, N + 1))
x_init = Parameter(nx)
objective = 0
constraints = [x[:, 0] == x_init]
for k in range(N):
    objective += quad_form(x[:, k] - xr, Q) + quad_form(u[:, k], R)
    constraints += [x[:, k + 1] == Ad @ x[:, k] + Bd @ u[:, k]]
    constraints += [xmin <= x[:, k], x[:, k] <= xmax]
    constraints += [umin <= u[:, k], u[:, k] <= umax]
objective += quad_form(x[:, N] - xr, QN)
prob = Problem(Minimize(objective), constraints)

# Simulate in closed loop
nsim = 50
X = [x0]
U = []
times = []
for i in range(nsim):
    x_init.value = x0
    start_time = time.time()
    prob.solve(solver=OSQP, warm_start=True)
    sol_time = time.time() - start_time
    times.append(sol_time)
    x0 = Ad.dot(x0) + Bd.dot(u[:,0].value)
    X.append(x0)
    U.append(u[:,0].value)
Xnp = np.asarray(X)
Unp = np.asarray(U)

mean_sol_time = np.mean(times)
max_sol_time = np.max(times)
print(f'mean sol time {mean_sol_time}')
print(f'max sol time {max_sol_time}')

ref = np.ones([nsim+1, 1])
u_min = umin*np.ones([nsim+1, umin.shape[0]])
u_max = umax*np.ones([nsim+1, umax.shape[0]])

fig, ax = plt.subplots(2, 1)
ax[0].plot(Xnp, label='x', linewidth=2)
ax[0].plot(ref, 'k--', label='r', linewidth=2)
ax[0].set(ylabel='$x$')
ax[0].set(xlabel='time')
ax[0].grid()
ax[0].set_xlim(0, nsim)
ax[1].plot(Unp, label='u', drawstyle='steps', linewidth=2)
ax[1].plot(u_min, 'k--', label='r', linewidth=2)
ax[1].plot(u_max, 'k--', label='r', linewidth=2)
ax[1].set(ylabel='$u$')
ax[1].set(xlabel='time')
ax[1].grid()
ax[1].set_xlim(0, nsim)
plt.tight_layout()