"""
Solve Quadratic Programming (QP) problem using CasADi toolbox:
minimize     x^2+y^2
subject to   x+y-p >= 0

problem parameters:            p
problem decition variables:    x, y

Reference material:
    https://web.casadi.org/docs/#quadratic-programming
"""

from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.patheffects as patheffects

p = 10.0    # problem parameter

# # # casadi opti stack problem definition # # #
opti = casadi.Opti()
# define variables
x = opti.variable()
y = opti.variable()
# define objective and constraints
opti.minimize(x**2+y**2)
opti.subject_to(x+y-p>=0)
# solve the problem
opti.solver('ipopt')
sol = opti.solve()
print(sol.value(x))
print(sol.value(y))

# # # Alternative casadi problem definition with QP solver # # #
x = SX.sym('x')
y = SX.sym('y')
qp = {'x': vertcat(x,y), 'f': x**2+y**2, 'g': x+y-p}
# solver object
S = qpsol('S', 'qpoases', qp)
print(S)
# solve the problem
r = S(lbg=0)
x_opt = r['x']
print('x_opt: ', x_opt)

# # # Plot the solution # # #

x1 = np.arange(-1.0, 10.0, 0.05)
y1 = np.arange(-1.0, 10.0, 0.05)
xx, yy = np.meshgrid(x1, y1)

# eval objective and constraints
J = xx**2+yy**2
c1 = xx+yy-p

# Plot
fig, ax = plt.subplots(1,1)
cp = ax.contourf(xx, yy, J,
                 alpha=0.6)
fig.colorbar(cp)
ax.set_title('Quadratic problem')
cg1 = ax.contour(xx, yy, c1, [0], colors='mediumblue', alpha=0.7)
plt.setp(cg1.collections,
         path_effects=[patheffects.withTickedStroke()], alpha=0.7)
# ax.plot(sol.value(x), sol.value(y), 'r*', markersize=10)
ax.plot(x_opt[0], x_opt[1], 'r*', markersize=10)
