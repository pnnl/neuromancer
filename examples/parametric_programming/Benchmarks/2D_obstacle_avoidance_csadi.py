"""
Solve the Rosenbrock problem, formulated as the NLP using CasADi toolbox:
minimize     (1-x)^2 + a*(y-x^2)^2
subject to   (p/2)^2 <= x^2 + y^2 <= p^2
             x>=y

problem parameters:             a, p
problem decition variables:     x, y

Reference material:
    https://en.wikipedia.org/wiki/Rosenbrock_function
    https://web.casadi.org/blog/opti/
    https://web.casadi.org/blog/nlp_sens/
    https://github.com/casadi/casadi/blob/master/docs/examples/python/rosenbrock.py

"""

from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.patheffects as patheffects
import matplotlib.patches as mpatches
import matplotlib.path as mpath


# instantiate casadi optimizaiton problem class
opti = casadi.Opti()

N = 20

X = opti.variable(2, N+1)  # state trajectory
x1 = X[0,:]
x2 = X[1,:]
U = opti.variable(2, N)    # control trajectory

# p = opti.parameter()
p = 0.5
b = 3.0
c = 0.3
d = 0.2
# 0.3 <= c < 0.7
# 0.0 <= d <= 0.4
# 1.0 <= b <= 3.0

# system dynamics
A = MX(np.array([[1.0, 0.1],
            [0.0, 1.0]]))
B = MX(np.array([[1.0, 0.0],
            [0.0, 1.0]]))

x_init = [0.0, 0.0]
x_final = [1.0, 0.2]
# initial conditions
opti.subject_to(x1[:,0] == x_init[0])
opti.subject_to(x2[:,0] == x_init[1])
# terminal condition
opti.subject_to(x1[:,N] == x_final[0])
opti.subject_to(x2[:,N] == x_final[1])

for k in range(N):
    opti.subject_to((p/2)**2 <= b*(x1[:,k]-c)**2+(x2[:,k]-d)**2)
    opti.subject_to(X[:,k+1] == A@X[:,k] + B@U[:,k])

opti.subject_to(opti.bounded(-1.0, U, 1.0))

# define objective
opti.minimize(sumsqr(U))

# select IPOPT solver and solve the NLP
opti.solver('ipopt')
sol = opti.solve()

print(sol.value(x1))
print(sol.value(x2))

x1 = np.arange(-0.5, 1.5, 0.02)
x2 = np.arange(-0.5, 1.5, 0.02)
xx, yy = np.meshgrid(x1, x2)

# eval objective and constraints
c2 = b*(xx -c)**2+(yy-d)**2 - (p/2)**2

# Plot
fig, ax = plt.subplots(1,1)
cg2 = ax.contour(xx, yy, c2, [0], colors='mediumblue', alpha=0.5)
plt.setp(cg2.collections, facecolor='mediumblue')
ax.plot(x_final[0], x_final[1], 'r*', markersize=10)
ax.plot(x_init[0], x_init[1], 'g*', markersize=10)
# plot trajectory
ax.plot(sol.value(X[0,:]), sol.value(X[1,:]), '*--')

fig, ax = plt.subplots(2, 1)
ax[0].plot(sol.value(X).transpose())
ax[1].plot(sol.value(U).transpose())