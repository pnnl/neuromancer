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


# instantiate casadi optimizaiton problem class
opti = casadi.Opti()

# define variables
x = opti.variable()
y = opti.variable()
# p = opti.parameter()
p = 1.0
a = 1.0

# define objective and constraints
opti.minimize((1-x)**2+a*(y-x**2)**2)
opti.subject_to(x>=y)
# opti.subject_to(x>=0)
opti.subject_to((p/2)**2 <= x**2+y**2)
opti.subject_to(x**2+y**2 <= p**2)
# opti.subject_to(x**2+y**2==1)

# select IPOPT solver and solve the NLP
opti.solver('ipopt')
sol = opti.solve()

print(sol.value(x))
print(sol.value(y))

x1 = np.arange(-0.5, 1.5, 0.02)
y1 = np.arange(-0.5, 1.5, 0.02)
xx, yy = np.meshgrid(x1, y1)

# eval objective and constraints
J = (1-xx)**2+a*(yy-xx**2)**2
c1 = xx-yy
c2 = xx**2+yy**2 - (p/2)**2
c3 = -(xx**2+yy**2) + p**2
# c4 = xx**2+yy**2-1


# Plot
fig, ax = plt.subplots(1,1)
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
# cg4 = ax.contour(xx, yy, c4, [0], colors='orangered', alpha=0.7)
ax.plot(sol.value(x), sol.value(y), 'r*', markersize=10)
