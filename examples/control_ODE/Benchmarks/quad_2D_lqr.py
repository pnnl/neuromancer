"""
benchmark 2D quadcopter LQR example from:
https://github.com/charlestytler/QuadcopterSim/blob/master/quad2D_lqr.py
"""

from pylab import *
import numpy as np
import scipy.linalg as splin

# Constants
m = 2
I = 1
d = 0.2
g = 9.8  # m/s/s
DTR = 1 / 57.3
RTD = 57.3


# Nonlinear Dynamics Equations of Motion
def f(x, u):
    # idx  0,1,2,3,4,5
    # x = [u,v,q,x,y,theta]
    xnew = zeros(6)
    xnew[0] = -1 * x[2] * x[1] + 1 / m * (u[0] + u[1]) * math.sin(x[5])
    xnew[1] = x[2] * x[0] + 1 / m * (u[0] + u[1]) * math.cos(x[5]) - g
    xnew[2] = 1 / I * (u[0] - u[1]) * d
    xnew[3] = x[0]
    xnew[4] = x[1]
    xnew[5] = x[2]
    return xnew


# 4th Order Runge Kutta Calculation
def RK4(x, u, dt):
    K1 = f(x, u)
    K2 = f(x + K1 * dt / 2, u)
    K3 = f(x + K2 * dt / 2, u)
    K4 = f(x + K3 * dt, u)
    xest = x + 1 / 6 * (K1 + 2 * K2 + 2 * K3 + K4) * dt
    return xest


tstep = .01  # sec

t = arange(0, 80, tstep)
x = zeros((6, size(t)))
xc = zeros((6, size(t)))
u = zeros((2, size(t)))

# Initial Conditions
x[4, 0] = 20  # y 20m
x[5, 0] = 20 * DTR  # theta 20deg

# Calculate equilibrium values
ue = 0
ve = 0
qe = 0
theta_e = 0
T1e = g * m / 2 / math.cos(theta_e)
T2e = g * m / 2 / math.cos(theta_e)

# Initial inputs
u[0, 0] = T1e
u[1, 0] = T2e

# Create Jacobian matrix
A = np.array([[0, -qe, -ve, 0, 0, g],
              [qe, 0, ue, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0]])

# Create linear Input matrix
B = np.array([[0, 0],
              [1 / m, 1 / m],
              [d / I, -d / I],
              [0, 0],
              [0, 0],
              [0, 0]])

Q = np.diag([1, 1, 10, 1, 2, 100])
R = np.diag([1, 1])

S = np.matrix(splin.solve_continuous_are(A, B, Q, R))
K = np.matrix(splin.inv(R) * (B.T * S))

for k in range(1, size(t)):  # run for 60 sec

    # State truth
    x[:, k] = RK4(x[:, k - 1], u[:, k - 1], tstep)

    # Command vector
    latc = 0
    vertc = 20
    if (t[k] > 40):
        latc = 20
        vertc = 10
    if (t[k] > 60):
        latc = -5
        vertc = 25

    xc[:, k] = [0, 0, 0, latc, vertc, 0]

    # State error
    e = x[:, k] - xc[:, k]

    feedback = np.matmul(K,e)
    u[:, k] = np.array([T1e, T2e]).T - feedback.A1

    # Limits
    u[0, k] = max(0, min(30, u[0, k]))
    u[1, k] = max(0, min(30, u[1, k]))

figure(1)
subplot(311)
plot(t, x[4, :], 'b', label='x')
ylabel('y [m]')
legend(loc='best')
subplot(312)
plot(t, x[3, :], 'b', label='x')
ylabel('x [m]')
# plot(t,x[2,:]*57.3,'b',label='x')
# ylabel('pitch rate [deg/s]')
subplot(313)
plot(t, x[5, :] * 57.3, 'b', label='x')
ylabel('theta [deg]')
figure(2)
plot(x[3, :], x[4, :], 'b', label='x')
ylabel('y [m]');
xlabel('x [m]')
legend(loc='best')
figure(3)
plot(t, u[0, :], 'b', label='T1')
plot(t, u[1, :], 'g', label='T2')
xlabel('Time (sec)')
legend(loc='best')
show()