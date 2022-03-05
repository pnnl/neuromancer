
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.io import loadmat


"""
Ground truth state space model (SSM)
"""

# discretized SSM matrices with 5min sampling 
#A = np.matrix([[0.9950,0.0017,0.0000,0.0031],[0.0007,0.9957,0.0003,0.0031],
#              [0.0000,0.0003,0.9834,0.0000],[0.2015,0.4877,0.0001,0.2571]])
A = np.matrix([[0.9950,0.0017,0.0000,0.0031],[0.0007,0.9957,0.0003,0.0031],
              [0.0000,0.0003,0.9834,0.0000],[0.2015,0.4877,0.0100,0.2571]])
B = np.matrix([[1.7586e-06],[1.7584e-06],
              [1.8390e-10],[5.0563e-04]])
E = np.matrix([[0.0002,0.0000,0.0000],[0.0002,0.0000,0.0000],
              [0.0163,0.0000,0.0000],[0.0536,0.0005,0.0001]])
C = np.matrix([[0.0,0.0,0.0,1.0]])

Model_matrices = [A,B,E,C]
Model_matrices_labels = ['A','B','E','C']

# plot original matrices
fig=plt.figure()
fig.set_size_inches(8, 2)
columns = 4
rows = 1
for i in range(1, columns*rows +1):
    ax = fig.add_subplot(rows, columns, i)
    plt.imshow(Model_matrices[i-1])
    plt.colorbar()
    plt.tight_layout() 
    ax.title.set_text(Model_matrices_labels[i-1])

plt.show()

# dimensions, initial conditions, disturbance and input signals
nx = 4
ny = 1
nu = 1
nd = 3

# TODO: start simdays one week later

# Train set 
samples_day = 288 # 288 samples per day with 5 min sampling
umax = 4e3
start_day = 7
start_sim = samples_day*start_day

Sim_days = 7 # number of simulated days
U_day = umax*np.sin(np.arange(0, 2*np.pi,2*np.pi/samples_day)) #  daily control profile
U = np.matlib.repmat(U_day, 1, Sim_days) #  Sim_days control profile

N_sim =  U.shape[1] # number of simulation steps train phase

file = loadmat('../TimeSeries/disturb.mat') # load disturbance file
Disturb = file['D']
D = Disturb[:,start_sim:start_sim+N_sim]

Y = np.zeros((ny,N_sim)) # output trajectory placeholders
X = np.zeros((nx,N_sim+1)) # state trajectory placeholders
X[:,0] =  20*np.ones(nx) # initial conditions

# open loop response LTI SSM
for k in range(0,N_sim):
    x0 = np.asmatrix(X[:,k]).T
    d = np.asmatrix(D[:,k]).T
    u = np.asmatrix(U[:,k]).T 
    # LTI SSM
    xn = A*x0 + B*u + E*d
    y = C*x0
    
    X[:,k+1] = xn.ravel()
    Y[:,k] = y.ravel()
    
    # plot inputs u and disturbances d
plt.figure()
plt.subplot(411)
plt.plot(U.T)
plt.ylabel('U')
plt.xlabel('time steps')
plt.title('Train set trajectories')
plt.show()
plt.subplot(412)
plt.plot(D.T)
plt.ylabel('D')
plt.xlabel('time steps')
plt.show()
# plot states and outputs
plt.subplot(413)
plt.plot(X.T)
plt.ylabel('X')
plt.xlabel('time steps')
plt.show()
plt.subplot(414)
plt.plot(Y.T)
plt.ylabel('Y')
plt.xlabel('time steps')
plt.show()

# standard score normalization
U_std =  np.std(U, axis=1)
U_mean =  np.mean(U, axis=1)
D_std =  np.std(D, axis=1)
D_mean =  np.mean(D, axis=1)

# Min-Max Feature scaling
U_max=  np.amax(U, axis=1)
U_min=  np.amin(U, axis=1)
D_max=  np.amax(D, axis=1)
D_min=  np.amin(D, axis=1)

U_scale = np.subtract(U.T,U_min)/(U_max.ravel()-U_min.ravel())
D_scale = np.subtract(D.T,D_min)/(D_max.ravel()-D_min.ravel())

# Test set     
#Sim_days_test = 7 # number of simulated days

# three week scenario
N_sim_start = start_sim
#N_sim_start = N_sim
Sim_days_test = 21
U_t = np.matlib.repmat(U_day, 1, Sim_days_test) #  Sim_days control profile

N_sim_t = U_t.shape[1] # number of simulation steps test phase

D_t = Disturb[:,N_sim_start:N_sim_start+N_sim_t]

Y_t = np.zeros((ny,N_sim_t)) # output trajectory placeholders
X_t = np.zeros((nx,N_sim_t+1)) # state trajectory placeholders
#X_t[:,0] =  20*np.ones(nx) # initial conditions
X_t[:,0] =  X[:,-1]


# Min-Max Feature scaling
U_t_max=  np.amax(U_t, axis=1)
U_t_min=  np.amin(U_t, axis=1)
D_t_max=  np.amax(D_t, axis=1)
D_t_min=  np.amin(D_t, axis=1)

U_t_scale = np.subtract(U_t.T,U_min)/(U_t_max.ravel()-U_t_min.ravel())
D_t_scale = np.subtract(D_t.T,D_min)/(D_t_max.ravel()-D_t_min.ravel())



# open loop response LTI SSM
for k in range(0,N_sim_t):
    x0 = np.asmatrix(X_t[:,k]).T
    d = np.asmatrix(D_t[:,k]).T
    u = np.asmatrix(U_t[:,k]).T 
    # LTI SSM
    xn = A*x0 + B*u + E*d
    y = C*x0
    
    X_t[:,k+1] = xn.ravel()
    Y_t[:,k] = y.ravel()
    
    # plot inputs u and disturbances d
plt.figure()
plt.subplot(411)
plt.plot(U_t.T)
plt.ylabel('U')
plt.xlabel('time steps')
plt.title('Test set trajectories')
plt.show()
plt.subplot(412)
plt.plot(D_t.T)
plt.ylabel('D')
plt.xlabel('time steps')
plt.show()
# plot states and outputs
plt.subplot(413)
plt.plot(X_t.T)
plt.ylabel('X')
plt.xlabel('time steps')
plt.show()
plt.subplot(414)
plt.plot(Y_t.T)
plt.ylabel('Y')
plt.xlabel('time steps')
plt.show()



