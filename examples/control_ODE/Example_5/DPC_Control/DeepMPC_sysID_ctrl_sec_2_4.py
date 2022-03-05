"""
Prototype Code for Deep MPC policy from  Section 2.4. Constrained Control Policy Learning
Results from 3.4. Constrained Continuous Control
"""


"""
# Dependencies
"""   
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.model_selection import train_test_split
import math
from numpy import linalg as LA
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
Algorithm 1 -Constrained model predictive control policy
"""  
class NeuroSSM_policy_one_step_con_DR(nn.Module):
    def __init__(self, nx, nu, nd, nr, n_hidden):
        super().__init__()
        # model layers weights
        self.A = nn.Linear(nx,nx, bias=False)
        self.B = nn.Linear(nu,nx, bias=False)
        self.E = nn.Linear(nd,nx, bias=False)
        self.C = nn.Linear(nx,ny, bias=False)
        #  policy layers weights
        self.policy1 = nn.Linear(nx+nd+nr+2*ny+2*nu, n_hidden, bias=False)
        self.policy2 = nn.Linear(n_hidden, nu, bias=False)

        # initialize model weights from the ground truth model
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
            
    def forward(self,x,d,r,ymin,ymax,umin,umax,w_mean,w_var):
        x = x.view(x.size(0), -1)
        d = d.view(d.size(0), -1)
        r = r.view(r.size(0), -1)       
        ymin = ymin.view(ymin.size(0), -1)    
        ymax = ymax.view(ymax.size(0), -1)    
        umin = umin.view(umin.size(0), -1)    
        umax = umax.view(umax.size(0), -1)  
               
        # Domain Randomization
        w = (w_mean-w_var)+(2*w_var)*torch.rand(nx) # additive state uncertainty       
        
        # Output constraints eval for policy input
        y = self.C(x)
        symin = F.relu(-y+ymin)  #       state lower bound  u>= umin
        symax = F.relu(y-ymax) #      state upper bound u <= umax
               
        # policy inputs: Algo 1 - line 2
        xi = torch.cat((x,d,r,symin,symax,umin,umax),1)
       # control policy evaluation: Algo 1 - line 3
        u_hidden = F.relu(self.policy1(xi))
        u =  self.policy2(u_hidden)         

        #   state update Algo 1 line 7
        x = self.A(x) + self.B(u) + self.E(d) + w
        y = self.C(x)
             
        # control input constraints Algo 1 lines 4-6  
        s_umin = F.relu(-u+umin)  #      control lower bound u>= umin 
        s_umax = F.relu(u-umax) #      control upper bound u <= umax
        
        # controlled state constraints Algo 1 lines 8-10 
        s_ymin = F.relu(-y+ymin) #       state lower bound  u>= umin
        s_ymax = F.relu(y-ymax) #      state upper bound u <= umax
               
        #         Algo 1 line 11
        return x, y, u, s_ymin, s_ymax, s_umin, s_umax
    
    
"""
# Model instantiation
"""      
# define additive uncertainty
w_mean_train = 0 # disturbance mean
w_var_train = 0.0 # disturbance variance
n_hidden = 10 # number of hidden neurons
nr = 1      # number of outputs/references
model = NeuroSSM_policy_one_step_con_DR(nx,nu,nd,nr,n_hidden)
model = model.float()
model.parameters
for param in model.named_parameters():
    print(param)


"""
# Dataset creation
"""  
a = 0
b = 25   
X0 = a +b*np.random.rand(X.shape[0],X.shape[1])   
a = 15
b = 25  
R_day = 20+2*np.sin(np.arange(0, 2*np.pi,2*np.pi/samples_day)) #  daily control profile
R_day_train = 15+10*np.sin(np.arange(0, 2*np.pi,2*np.pi/samples_day)) #  daily control profile
R = np.matlib.repmat(R_day_train, 1, Sim_days) #  Sim_days control profile
R  = a +b*np.random.rand(R.shape[0],R.shape[1])   

Features = np.concatenate((X0[:,:-2].T, D[:,:-1].T, R[:,:-1].T), axis=1)       # tranining on randomized trajectory
Targets = R[:,:-1].T 

# divide dataset to train and test set
Features_train, Features_test, Targets_train, Targets_test = train_test_split(Features,Targets,test_size=0.2,shuffle=False)
Features_train = torch.FloatTensor(Features_train)
Features_test = torch.FloatTensor(Features_test)
Targets_train = torch.FloatTensor(Targets_train)
Targets_test = torch.FloatTensor(Targets_test)
# split features - train and test sets
X_train = Features_train[:,0:nx]
D_train = Features_train[:,nx:-1]
R_train = Features_train[:,-1:]
X_test= Features_test[:,0:nx]
D_test = Features_test[:,nx:-1]
R_test = Features_test[:,-1:]
# slack variables
sx_targets = torch.zeros(Targets_train.shape[0],nx)
sy_targets = torch.zeros(Targets_train.shape[0],ny)
su_targets = torch.zeros(Targets_train.shape[0],nu)
u_targets = torch.zeros(Targets_train.shape[0],nu)
# costraints bounds
xmin_val = 19
xmax_val = 25
xmin = xmin_val*torch.ones(Targets_train.shape[0],nx)
xmax = xmax_val*torch.ones(Targets_train.shape[0],nx)
ymin_val = 19
ymax_val = 25
ymin = ymin_val*torch.ones(Targets_train.shape[0],ny)
ymax = ymax_val*torch.ones(Targets_train.shape[0],ny)
umin_val = 0
umax_val = 5000
umin = umin_val*torch.ones(Targets_train.shape[0],nu)
umax = umax_val*torch.ones(Targets_train.shape[0],nu)
        
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
#  multi-objective loss  equation (16)
    x_pred, y_pred, u_pred, s_ymin, s_ymax, s_umin, s_umax = model.forward(X_train, D_train, R_train, ymin, ymax, umin, umax,w_mean_train,w_var_train)
    loss = Q_ref*criterion(y_pred, Targets_train)  + Q_u*criterion(u_pred, u_targets) \
    + Q_con_y*criterion(s_ymin, sy_targets)   + Q_con_y*criterion(s_ymax, sy_targets)  \
    + Q_con_u*criterion(s_umin, su_targets)   + Q_con_u*criterion(s_umax, su_targets)  
    losses.append(loss)
    
    if i%1000 == 1:
        print(f'epoch: {i:2}  loss: {loss.item():10.8f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch');

## save entire model
#torch.save(model, 'Trained_DeepMPCPolicy_sec_3_4.pt')


"""
# Parameters of the simulation from Section 3.4.
""" 

run_sim = True
replicate_paper = False # load the policy used in the paper
Eval_runs  = 20  # number of randomized closed-loop simulations, Paper value: 20
# plotting flags
save_fig = True
paper_plot = True

# uncertainty  ad constraints flags in simulation
u_clipping = True # hard control bounds
param_uncertainty =  False
add_uncertainty = True
robust_relaxation = False
if add_uncertainty:
    w_mean = 0
    w_var = 0.1
else:
    w_mean = 0
    w_var = 0.0
if param_uncertainty:
    theta_mean = 0
    theta_var = 0.01
else:
    theta_mean = 0
    theta_var = 0.00    


"""
load model parameters of trained policy from Section 3.5.
""" 
if replicate_paper:
    model = torch.load('Trained_DeepMPCPolicy_sec_3_4.pt')

     
"""
Running closed-loop control simulations from Section 3.4.
""" 
if run_sim:
    xmin_ = xmin_val*torch.ones(1,nx)
    xmax_ = xmax_val*torch.ones(1,nx)
    ymin_ = ymin_val*torch.ones(1,ny)
    ymax_ = ymax_val*torch.ones(1,ny)
    umin_ = umin_val*torch.ones(1,nu)
    umax_ = umax_val*torch.ones(1,nu)
    ## Control Validation closed loop: test set
    R_t = np.matlib.repmat(R_day, 1, Sim_days_test) #  Sim_days control profile
    X_torch = np.zeros((nx,N_sim_t+1)) # state trajectory placeholders
    X_torch[:,0] =  20*np.ones(nx) # initial conditions
    Y_torch = np.zeros((ny,N_sim_t)) # output trajectory placeholders
    U_torch = np.zeros((nu,N_sim_t)) # input trajectory placeholders
    X_cl = np.zeros((nx,N_sim_t+1)) # state trajectory placeholders
    X_cl[:,0] =  20*np.ones(nx) # initial conditions
    Y_cl = np.zeros((ny,N_sim_t)) # output trajectory placeholders    
    
    DATA_X_cl = np.ndarray(shape=(X_cl.shape[0],X_cl.shape[1],Eval_runs))
    DATA_Y_cl = np.ndarray(shape=(Y_cl.shape[0],Y_cl.shape[1],Eval_runs))
    DATA_U_torch = np.ndarray(shape=(U_torch.shape[0],U_torch.shape[1],Eval_runs))
    DATA_X_torch = np.ndarray(shape=(X_torch.shape[0],X_torch.shape[1],Eval_runs))
    DATA_Y_torch = np.ndarray(shape=(Y_torch.shape[0],Y_torch.shape[1],Eval_runs))
    DATA_R_t = np.ndarray(shape=(R_t.shape[0],R_t.shape[1],Eval_runs)) 
    
    CPU_mean_time = np.zeros(Eval_runs)
    CPU_max_time = np.zeros(Eval_runs)

    for run in range(0,Eval_runs):
        StepTime = np.zeros(N_sim_t)
        for k in range(0,N_sim_t):
            x0 = np.asmatrix(X_cl[:,k])
            d = np.asmatrix(D_t[:,k])
            r = np.asmatrix(R_t[:,k])
            w = (w_mean-w_var)+(2*w_var)*np.asmatrix(np.random.rand(nx,1)) # state noise signal
            theta = (1+theta_mean-theta_var)+(2*theta_var)*np.asmatrix(np.random.rand(nx,nx))            
            x0_torch = torch.from_numpy(x0).float()
            d_torch = torch.from_numpy(d).float()
            r_torch = torch.from_numpy(r).float()
        
            start_step_time = time.time()  
            xn_pt, y_pt, u, s_ymin, s_ymax, s_umin, s_umax = model(x0_torch,d_torch,r_torch,ymin_,ymax_,umin_,umax_,w_mean_train,w_var_train)              
            StepTime[k] = time.time() - start_step_time
            
            if u_clipping:
                if u < umin_:
                    u = umin_                
                elif u > umax_:
                    u = umax_
        
            X_torch[:,k+1] = xn_pt.detach().numpy().ravel()
            Y_torch[:,k] = y_pt.detach().numpy().ravel()
            U_torch[:,k] = u.detach().numpy().ravel()
            Y_torch[:,k] = y_pt.detach().numpy().ravel()
        
            # ground truth LTI SSM
            xn = np.multiply(theta,A)*x0.T + B*U_torch[:,k] + E*d.T + w
            y = C*x0.T
            
            X_cl[:,k+1] = xn.ravel()
            Y_cl[:,k] = y.ravel()
                  
        CPU_mean_time[run] = np.mean(StepTime)
        CPU_max_time[run] = np.amax(StepTime)
            
        # various evaluations with uncertainty    
        DATA_X_cl[:,:,run] = X_cl
        DATA_Y_cl[:,:,run] = Y_cl
        DATA_U_torch[:,:,run] = U_torch
        DATA_X_torch[:,:,run] = X_torch   
        DATA_Y_torch[:,:,run] = Y_torch
        DATA_R_t[:,:,run] = R_t
        
    CPU_mean_time_paper = np.mean(CPU_mean_time)*1e3 
    CPU_max_time_paper = np.amax(CPU_max_time)*1e3    
    
    DATA_X_cl_max = np.amax(DATA_X_cl, axis = 2)
    DATA_X_cl_min = np.amin(DATA_X_cl, axis = 2)
    DATA_X_cl_mmean = np.mean(DATA_X_cl, axis = 2)
    
    DATA_Y_cl_max = np.amax(DATA_Y_cl, axis = 2)
    DATA_Y_cl_min = np.amin(DATA_Y_cl, axis = 2)
    DATA_Y_cl_mmean = np.mean(DATA_Y_cl, axis = 2)
    
    DATA_U_torch_max = np.amax(DATA_U_torch, axis = 2)
    DATA_U_torch_min = np.amin(DATA_U_torch, axis = 2)
    DATA_U_torch_mmean = np.mean(DATA_U_torch, axis = 2)
    
    time_x = np.linspace(0, Sim_days_test, DATA_X_cl_mmean.shape[1], endpoint=False)
    time_y = np.linspace(0, Sim_days_test, DATA_Y_cl_mmean.shape[1], endpoint=False)
    time_u = np.linspace(0, Sim_days_test, DATA_U_torch_mmean.shape[1], endpoint=False)
    
    xmin = xmin_val*np.ones((time_x.shape[0],nx))
    xmax = xmax_val*np.ones((time_x.shape[0],nx))
    ymin = ymin_val*np.ones((time_y.shape[0],ny))
    ymax = ymax_val*np.ones((time_y.shape[0],ny))
    umin = umin_val*np.ones((time_u.shape[0],ny))
    umax = umax_val*np.ones((time_u.shape[0],ny))
       
    #    eval control statistics
    test_start = 14*samples_day
    test_end = DATA_Y_cl_mmean.shape[1]
    DATA_Y_cl_torch = torch.from_numpy(DATA_Y_cl[:,test_start:test_end]).float()  
    DATA_s_ymin_real = F.relu(-DATA_Y_cl_torch+ymin_).detach().numpy()
    DATA_s_ymax_real = F.relu(DATA_Y_cl_torch-ymax_).detach().numpy()
    MAE_constr = np.mean(DATA_s_ymin_real) + np.mean(DATA_s_ymax_real)
    MSE_model = np.mean(np.square(DATA_Y_cl[:,test_start:test_end]-DATA_Y_torch[:,test_start:test_end]))
    MSE_ref = np.mean(np.square(DATA_Y_cl[:,test_start:test_end]-DATA_R_t[:,test_start:test_end]))
    MA_energy =  np.mean(np.absolute(DATA_U_torch_mmean[:,test_start:test_end]))
    
    
"""
Plot - not included in paper
"""     
if paper_plot:
    fig, ax = plt.subplots(4,1,figsize=(8, 8))
    for i in range(0,nx):
        ax[0].plot(time_x,DATA_X_cl_mmean[i,:].T)
        ax[0].fill_between(time_x, np.squeeze(DATA_X_cl_min[i,:].T), np.squeeze(DATA_X_cl_max[i,:].T), alpha=0.2)
    ax[0].set(ylabel='X')
    ax[0].tick_params(labelbottom=False)
    ax[0].axvspan(7, 14, facecolor='grey', alpha=0.25, zorder=-100)
    ax[0].axvspan(14, 21, facecolor='grey', alpha=0.5, zorder=-100)
    ax[0].margins(0,0.1)
            
    ax[1].plot(time_y,R_t.T, '--', label='R')
    ax[1].plot(time_y,DATA_Y_cl_mmean.T, label='Y')
    ax[1].fill_between(time_y, np.squeeze(DATA_Y_cl_min.T), np.squeeze(DATA_Y_cl_max.T), color='orange', alpha=0.2)
    ax[1].plot(time_y,ymin, 'k--')
    ax[1].plot(time_y,ymax, 'k--')
    ax[1].set(ylabel='Y')
    ax[1].tick_params(labelbottom=False)
    ax[1].axvspan(7, 14, facecolor='grey', alpha=0.2, zorder=-100)
    ax[1].axvspan(14, 21, facecolor='grey', alpha=0.4, zorder=-100)
    ax[1].margins(0,0.1)
    
    ax[2].plot(time_u,DATA_U_torch_mmean.T, label='U')
    ax[2].fill_between(time_u, np.squeeze(DATA_U_torch_min.T), np.squeeze(DATA_U_torch_max.T), color='blue', alpha=0.2)
    ax[2].plot(time_u,umin, 'k--')
    ax[2].plot(time_u,umax, 'k--')
    ax[2].set(ylabel='U')
    ax[2].tick_params(labelbottom=False)
    ax[2].axvspan(7, 14, facecolor='grey', alpha=0.2, zorder=-100)
    ax[2].axvspan(14, 21, facecolor='grey', alpha=0.4, zorder=-100)
    #plt.tight_layout()
    ax[2].margins(0,0.1)
    
    ax[3].plot(time_u,D_t[1:3,:].T)
    ax[3].tick_params(labelbottom=True)
    ax[3].set_xlabel('Day')
    ax[3].set(ylabel='D')
    ax[3].tick_params(labelbottom=True)
    ax[3].axvspan(7, 14, facecolor='grey', alpha=0.2, zorder=-100)
    ax[3].axvspan(14, 21, facecolor='grey', alpha=0.4, zorder=-100)
    ax[3].margins(0,0.1)
    ax[3].text(1, 1200, '             Train                ',
                bbox={'facecolor': 'white', 'alpha': 0.5})
    ax[3].text(8, 1200, '           Validation           ',
                bbox={'facecolor': 'grey', 'alpha': 0.25})
    ax[3].text(15, 1200, '              Test                ',
                   bbox={'facecolor': 'grey', 'alpha': 0.5})    
    ax4 = ax[3].twinx()  # instantiate a second axes that shares the same x-axis
    ax4.plot(time_u,D_t[0,:].T, color ='green')
    ax4.tick_params(axis='y', labelcolor='green')
    ax4.margins(0,0.1)   
    steps = range(0, Sim_days_test+1, 1)
    days = np.array(list(range(Sim_days_test+1)))
    ax[3].set(xticks=steps,xticklabels=days)   
    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    
    if save_fig:
        plt.savefig('DeepMPC_sysID_ctrl_sec_2_4.pdf')



