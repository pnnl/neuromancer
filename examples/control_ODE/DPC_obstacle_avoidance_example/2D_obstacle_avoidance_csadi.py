"""
Solve obstacle avoidance control problem in Casadi
minimize     u_k^2
subject to   (p/2)^2 <= b(x[0]-c)^2 + (x[1]-d)^2
             x_k+1 = Ax_k + Bu_k

problem parameters:             p, b, c, d
problem decition variables:     x, u

"""

from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import time


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

times = []
for k in range(50):
    start_time = time.time()
    opti.solver('ipopt')
    sol = opti.solve()
    sol_time = time.time() - start_time
    times.append(sol_time)
print(f'mean solution time: {np.mean(times)}')
print(f'max solution time: {np.max(times)}')



# print(sol.value(x1))
# print(sol.value(x2))

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
print(f'energy use: {np.mean(sol.value(U)**2)}')