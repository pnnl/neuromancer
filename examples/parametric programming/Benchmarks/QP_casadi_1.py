"""
Solve the Rosenbrock problem, formulated as the NLP using CasADi toolbox:
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


# instantiate casadi optimizaiton problem class
opti = casadi.Opti()

# define variables
x = opti.variable()
y = opti.variable()
p = 10.0

# define objective and constraints
opti.minimize(x**2+y**2)
opti.subject_to(x+y-p>=0)


# S = casadi.qpsol('S', 'qpoases', qp)
# select qpoases solver and solve the QLP
opti.solver('ipopt')
sol = opti.solve()

print(sol.value(x))
print(sol.value(y))

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
ax.plot(sol.value(x), sol.value(y), 'r*', markersize=10)
