"""
LTI SSM Policy training N-step
physics informed policy gradient
"""

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.model_selection import train_test_split
import time

"""
Ground truth state space model (SSM)
"""
A = np.matrix([[0.9950,0.0017,0.0000,0.0031],[0.0007,0.9957,0.0003,0.0031],
              [0.0000,0.0003,0.9834,0.0000],[0.2015,0.4877,0.0100,0.2571]])
B = np.matrix([[1.7586e-06],[1.7584e-06],
              [1.8390e-10],[5.0563e-04]])
E = np.matrix([[0.0002,0.0000,0.0000],[0.0002,0.0000,0.0000],
              [0.0163,0.0000,0.0000],[0.0536,0.0005,0.0001]])
C = np.matrix([[0.0,0.0,0.0,1.0]])
# model dimensions
nx = 4
ny = 1
nu = 1
nd = 3


"""
Generate data for training, validation and testing
"""
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
# Training set open loop response LTI SSM
for k in range(0,N_sim):
    x0 = np.asmatrix(X[:,k]).T
    d = np.asmatrix(D[:,k]).T
    u = np.asmatrix(U[:,k]).T 
    # LTI SSM
    xn = A*x0 + B*u + E*d
    y = C*x0
    
    X[:,k+1] = xn.ravel()
    Y[:,k] = y.ravel()

# Validation + Test set     
N_sim_start = start_sim
Sim_days_test = 21
U_t = np.matlib.repmat(U_day, 1, Sim_days_test) #  Sim_days control profile
N_sim_t = U_t.shape[1] # number of simulation steps test phase
D_t = Disturb[:,N_sim_start:N_sim_start+N_sim_t]
Y_t = np.zeros((ny,N_sim_t)) # output trajectory placeholders
X_t = np.zeros((nx,N_sim_t+1)) # state trajectory placeholders
X_t[:,0] =  X[:,-1]
#  Validation + Test set  open loop response LTI SSM
for k in range(0,N_sim_t):
    x0 = np.asmatrix(X_t[:,k]).T
    d = np.asmatrix(D_t[:,k]).T
    u = np.asmatrix(U_t[:,k]).T 
    # LTI SSM
    xn = A*x0 + B*u + E*d
    y = C*x0
    
    X_t[:,k+1] = xn.ravel()
    Y_t[:,k] = y.ravel()


"""
Algorithm 1 -Constrained model predictive control policy with N>1
"""  
# Model definition
class NeuroSSM_policy_N_step(nn.Module):
    def __init__(self, nx, nu, ny, nd, nr, N, n_hidden):
        super().__init__()
        # model layers weights
        self.A = nn.Linear(nx,nx, bias=False)
        self.B = nn.Linear(nu,nx, bias=False)
        self.E = nn.Linear(nd,nx, bias=False)
        self.C = nn.Linear(nx,ny, bias=False)
        # positive policy layers weights
        self.policy1 = nn.Linear(nx+nd*N+nr*N, n_hidden, bias=False)
        self.policy2 = nn.Linear(n_hidden, nu*N, bias=False)
        
        self.N = N
        self.nr = nr
        self.nd = nd
        
        # initialize model weights
        with torch.no_grad():
            self.A.weight.copy_(torch.from_numpy(A))
            self.B.weight.copy_(torch.from_numpy(B))
            self.E.weight.copy_(torch.from_numpy(E))
            self.C.weight.copy_(torch.from_numpy(C))
  
        # fix first 4 layers of model weights
        child_counter = 0
        for child in self.children():
            child_counter += 1
            if child_counter <= 4:
                for param in child.parameters():
                    param.requires_grad = False         
            
    def forward(self,x,D,R,YMIN,YMAX,UMIN,UMAX):
        x = x.view(x.size(0), -1) # initial state comming from KF or direct measurements    
        
        X = torch.tensor([])
        Y = torch.tensor([]) 
        U = torch.tensor([])        
        Sy_min = torch.tensor([]) 
        Sy_max = torch.tensor([]) 
        Su_min = torch.tensor([]) 
        Su_max = torch.tensor([]) 
          
        D_xi = D.reshape(-1,self.N*self.nd)     
        R_xi = R.reshape(-1,self.N*self.nr)        
                    
        xi = torch.cat((x,D_xi,R_xi),1)
        u_hidden = F.relu(self.policy1(xi))
        U =  self.policy2(u_hidden)  
        U = U.unsqueeze(2)
        
        for k in range(0,N):
            ymin = YMIN[:,k,:]  
            ymax = YMAX[:,k,:]     
            umin = UMIN[:,k,:]    
            umax = UMAX[:,k,:]    
            
            d = D[:,k,:]  
            u = U[:,k,:] 
             
            # predict step
            x = self.A(x) + self.B(u) + self.E(d)   
            y = self.C(x)                        
            # control input constraints Algo 1 lines 4-6  
            s_umin = F.relu(-u+umin)  #      control lower bound u>= umin 
            s_umax = F.relu(u-umax) #      control upper bound u <= umax          
            # controlled state constraints Algo 1 lines 8-10 
            s_ymin = F.relu(-y+ymin) #       state lower bound  u>= umin
            s_ymax = F.relu(y-ymax) #      state upper bound u <= umax            
            
            Sy_min = torch.cat((Sy_min, s_ymin), 0)
            Sy_max = torch.cat((Sy_max, s_ymax), 0)
            Su_min = torch.cat((Su_min, s_umin), 0)
            Su_max = torch.cat((Su_max, s_umax), 0)
            
            X = torch.cat((X, x), 0)
            Y = torch.cat((Y, y), 0)   
        
        return X, Y, U, Sy_min, Sy_max, Su_min, Su_max
    
"""
# Model instantiation
"""          
N = 8
n_hidden = 10*N # number of hidden neurons
nr = 1      # number of outputs/references
ny = 1
model = NeuroSSM_policy_N_step(nx,nu,ny,nd,nr, N, n_hidden)
model = model.float()
model.parameters

for param in model.named_parameters():
    print(param)

"""
# Dataset creation
"""  
#R = np.ones(Y.shape) +20 # reference signal
R_day = 20+2*np.sin(np.arange(0, 2*np.pi,2*np.pi/samples_day)) #  daily control profile
R = np.matlib.repmat(R_day, 1, Sim_days) #  Sim_days control profile

ymin_val = 19
ymax_val = 25
ymin = ymin_val*torch.ones(samples_day*Sim_days,ny)
ymax = ymax_val*torch.ones(samples_day*Sim_days,ny)
umin_val = 0
umax_val = 5000
umin = umin_val*torch.ones(samples_day*Sim_days,nu)
umax = umax_val*torch.ones(samples_day*Sim_days,nu)

# slack variables targets
sx_targets = torch.zeros(samples_day*Sim_days,nx)
sy_targets = torch.zeros(samples_day*Sim_days,ny)
su_targets = torch.zeros(samples_day*Sim_days,nu)
u_targets = torch.zeros(samples_day*Sim_days,nu)

# dataset
R_train = R.T.reshape((-1,N,nu))
D_train = D.T.reshape((-1,N,nd))
YMIN = ymin.T.reshape((-1,N,ny))
YMAX = ymax.T.reshape((-1,N,ny))
UMIN = umin.T.reshape((-1,N,nu))
UMAX = umax.T.reshape((-1,N,nu))
X_train = X[:,range(0,N_sim,N)].T

a = 10
b = 10   
X0_train = a +b*np.random.rand(X_train.shape[0],X_train.shape[1])   

X0_in = torch.from_numpy(X0_train).float() 
D_in = torch.from_numpy(D_train).float()
R_in = torch.from_numpy(R_train).float()

R_target_use = torch.tensor([])
for k in range(R_in.shape[1]):
    R_target_use = torch.cat((R_target_use, R_in[:,k]), 0) 

"""
# MSE citerion and optimizer
"""  
# objective and optimizer
criterion = nn.MSELoss()  # we'll convert this to RMSE later
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

"""
# Weights of the multi-objective loss - equation (16)
""" 
# constraints weight
Q_con_u = 5e-7
Q_con_x = 50
Q_con_y = 50
Q_u = 1e-7
Q_u = 1e-6
Q_ref = 20

""" 
Policy Optimization Training Loop using Algorithm 1
""" 
epochs = 30000
losses = []

for i in range(epochs):
    i+=1  
   
    X_pred, Y_pred, U_pred, Sy_min, Sy_max, Su_min, Su_max = model.forward(X0_in,D_in,R_in,YMIN,YMAX,UMIN,UMAX)
    loss = Q_ref*criterion(Y_pred, R_target_use)  + Q_u*criterion(U_pred.flatten(), u_targets) \
    + Q_con_y*criterion(Sy_min, sy_targets)   + Q_con_y*criterion(Sy_max, sy_targets)  \
    + Q_con_u*criterion(Su_min, su_targets)   + Q_con_u*criterion(Su_max, su_targets)
    losses.append(loss)
    
    if i%1000 == 1:
        print(f'epoch: {i:2}  loss: {loss.item():10.8f}')
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch');



replicate_paper = False

"""
load model parameters of trained policy from Section 3.7.
""" 
if replicate_paper:
    model = torch.load('Trained_DeepMPCPolicy_N_'+str(N)+'_sec_3_7.pt')
# else:
#     torch.save(model, 'Trained_DeepMPCPolicy_N_'+str(N)+'_sec_3_7.pt')


"""
Running closed-loop control simulations from Section 3.7.
""" 
Eval_runs  = 20 
CPU_mean_time = np.zeros(Eval_runs)
CPU_max_time = np.zeros(Eval_runs)


ymin_val = 19
ymax_val = 25
ymin = ymin_val*torch.ones(samples_day*Sim_days_test,ny)
ymax = ymax_val*torch.ones(samples_day*Sim_days_test,ny)
umin_val = 0
umax_val = 5000
umin = umin_val*torch.ones(samples_day*Sim_days_test,nu)
umax = umax_val*torch.ones(samples_day*Sim_days_test,nu)
R_t = np.matlib.repmat(R_day, 1, Sim_days_test) #  Sim_days control profile


    # Control Validation closed loop
for run in range(0,Eval_runs):
    StepTime = np.zeros(N_sim_t)
    
    U_torch = np.zeros((nu,N_sim_t-N)) # input trajectory placeholders
    X_cl = np.zeros((nx,N_sim_t+1-N)) # state trajectory placeholders
    X_cl[:,0] =  20*np.ones(nx) # initial conditions
    Y_cl = np.zeros((ny,N_sim_t-N)) # output trajectory placeholders
    
    
    for k in range(0,N_sim_t-N):
        x0 = np.asmatrix(X_cl[:,k])
        d = np.asmatrix(D_t[:,k:k+N]).T
        r = np.asmatrix(R_t[:,k:k+N]).T
        
        umin_k = np.asmatrix(umin[k:k+N,:])
        umax_k = np.asmatrix(umax[k:k+N,:])
        ymin_k = np.asmatrix(ymin[k:k+N,:])
        ymax_k = np.asmatrix(ymax[k:k+N,:])
        
        x0_in = torch.from_numpy(x0).float()
        d_in = torch.from_numpy(d).float().unsqueeze(0)
        r_in = torch.from_numpy(r).float().unsqueeze(0)
        
        umin_in = torch.from_numpy(umin_k).float().unsqueeze(0)
        umax_in = torch.from_numpy(umax_k).float().unsqueeze(0)
        ymin_in = torch.from_numpy(ymin_k).float().unsqueeze(0)
        ymax_in = torch.from_numpy(ymax_k).float().unsqueeze(0)
        
        start_step_time = time.time()
        X_pred, Y_pred, U_pred, Sy_min, Sy_max, Su_min, Su_max  = model.forward(x0_in,d_in,r_in,ymin_in,ymax_in,umin_in,umax_in)
        StepTime[k] = time.time() - start_step_time

        if N>1:
            U_torch[:,k] = U_pred[:,1].T.detach().numpy().ravel()
            d0 = d[1,:].T
        else:
            U_torch[:,k] = U_pred.T.detach().numpy().ravel()
            d0 = d.T
        
        x0 = x0.T        
    
        # LTI SSM
        xn = A*x0 + B*U_torch[:,k] + E*d0
        y = C*x0
        
        X_cl[:,k+1] = xn.ravel()
        Y_cl[:,k] = y.ravel()
        
    CPU_mean_time[run] = np.mean(StepTime)
    CPU_max_time[run] = np.amax(StepTime)  
 
CPU_mean_time_paper = np.mean(CPU_mean_time)*1e3 
CPU_max_time_paper = np.amax(CPU_max_time)*1e3      

plt.figure()
plt.subplot(411)
plt.plot(X_cl.T, '--', label='X True')
plt.title('Closed-loop control validation: train set')
plt.ylabel('X')
plt.xlabel('time steps')
plt.show()
plt.subplot(412)
plt.plot(R_t.T, '--', label='R')
plt.plot(Y_cl.T, label='Y')
plt.ylabel('Y')
plt.xlabel('time steps')
plt.legend()
plt.show()
plt.subplot(413)
plt.plot(U_torch.T, label='U Trained')
# plt.plot(U.T, '--', label='U True')
plt.ylabel('U')
plt.xlabel('time steps')
plt.legend()
plt.show()
plt.subplot(414)
plt.plot(D_t.T)
# plt.plot(U.T, '--', label='U True')
plt.ylabel('D')
plt.xlabel('time steps')
plt.legend()
plt.show()



# show policy weights
fig, (plt1, plt2, plt3) = plt.subplots(1, 3)
fig.suptitle('Pytorch linear SSM weights')
img1 = plt1.imshow(model.policy1.weight.data)
fig.colorbar(img1, ax=plt1)
plt1.title.set_text('policy1')
img2 = plt2.imshow(model.policy2.weight.data)
fig.colorbar(img2, ax=plt2)
plt2.title.set_text('policy2')
img3 = plt3.imshow(torch.mm(model.policy2.weight.data,F.relu(model.policy1.weight.data)))
fig.colorbar(img3, ax=plt3)
img3 = plt3.title.set_text('Effective Policy')
fig.tight_layout() 



