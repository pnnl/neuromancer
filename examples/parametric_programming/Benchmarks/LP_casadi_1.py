"""
Solve Linear Programming (LP) problem using CasADi toolbox:
minimize     a1*x-a2*y
subject to   x+y-p1 >= 0
             -2*x+y+p2 >= 0
             x-2*y+p3 >= 0

problem parameters:            a1, a2, p1, p2, p3
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

# problem parameters
a1 = 0.5
a2 = 1.0
p1 = 6.0
p2 = 8.0
p3 = 9.0

# # # casadi problem definition with QP solver # # #
x = SX.sym('x')
y = SX.sym('y')
qp = {'x': vertcat(x,y), 'f': a1*x+a2*y, 'g': vertcat(x+y-p1, -2*x+y+p2, x-2*y+p3)}
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
J = a1*xx+a2*yy
c1 = xx+yy-p1
c2 = -2*xx+yy+p2
c3 = xx-2*yy+p3

# Plot
fig, ax = plt.subplots(1,1)
cp = ax.contourf(xx, yy, J,
                 alpha=0.6)
fig.colorbar(cp)
ax.set_title('Linear problem')
cg1 = ax.contour(xx, yy, c1, [0], colors='mediumblue', alpha=0.7)
plt.setp(cg1.collections,
         path_effects=[patheffects.withTickedStroke()], alpha=0.7)
cg2 = ax.contour(xx, yy, c2, [0], colors='mediumblue', alpha=0.7)
plt.setp(cg2.collections,
         path_effects=[patheffects.withTickedStroke()], alpha=0.7)
cg3 = ax.contour(xx, yy, c3, [0], colors='mediumblue', alpha=0.7)
plt.setp(cg3.collections,
         path_effects=[patheffects.withTickedStroke()], alpha=0.7)
ax.plot(x_opt[0], x_opt[1], 'r*', markersize=10)
