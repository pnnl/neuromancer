"""
Prototype Code for Deep MPC policy from  Section 2.5. Joint System Identification and Control Learning
Results from Section 3.5. Simultaneous System Identification and Control
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
# Consrained Layer Weights
"""   
# dominant eigenvalue constraints
class ConstrainedLinear9(nn.Module):
    def __init__(self, nx, nu):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(nx,nu))
        self.scalar = nn.Parameter(torch.rand(nx,nx))  # matrix scaling to allow for different row sums
        
    def effective_W(self):    
        s_clapmed = 1 - 0.1*F.sigmoid(self.scalar)     # constrain sum of rows to be in between 0.9 and 1
        w_sofmax = s_clapmed*F.softmax(self.weight, dim = 1)
        return w_sofmax.T

    def forward(self,x):  
        return torch.mm(x, self.effective_W())
    
#  nonnehative weights
class ConstrainedLinear_LB(nn.Module):
    def __init__(self, nx, nu):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(nx,nu))
        
    def effective_W(self):    
        w_LB = F.relu(self.weight)
        return w_LB

    def forward(self,x):      
        return torch.mm(x, self.effective_W())
    
    
class Compensator(nn.Module):
    def __init__(self, nx, nu):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(nx,nu))       

    def forward(self):      
        return self.weight                    
      
"""
Algorithm 2 - Joined model and control policy learning
"""    
class NeuroSSM_policy_one_step_con_DR(nn.Module):
    def __init__(self, nx, nu, nd, nr, n_hidden):
        super().__init__()
        # constrained model layers weights     
        self.A = ConstrainedLinear9(nx,nx)
        self.B = ConstrainedLinear_LB(nu,nx)
        self.E = ConstrainedLinear_LB(nd,nx) 
        self.C = ConstrainedLinear_LB(nx,ny)
        # policy layers weights
        self.policy1 = nn.Linear(nx+nd+nr+2*ny+2*nu, n_hidden, bias=False)
        self.policy2 = nn.Linear(n_hidden, nu, bias=False)
        
#        thresholds on maximal change of rate of state variables
        dxmax_val = 0.5
        dxmin_val = -0.5       
        self.dxmax = nn.Parameter(dxmax_val*torch.ones(1,nx), requires_grad=False)
        self.dxmin = nn.Parameter(dxmin_val*torch.ones(1,nx), requires_grad=False)  
        
#        online learning correction terms
        self.x0_correct = Compensator(1,nx)
        self.u_correct = Compensator(1,nu)         
        for param in self.x0_correct.parameters():
            param.requires_grad = False        
        for param in self.u_correct.parameters():
            param.requires_grad = False 
                      
    def forward(self,x,x0_sysID,u_sysID,d,r,ymin,ymax,umin,umax,w_mean,w_var):
        x = x.view(x.size(0), -1)
        x0_sysID = x0_sysID.view(x0_sysID.size(0), -1)
        u_sysID = u_sysID.view(u_sysID.size(0), -1)
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
        
        # policy inputs: Algo 2 - line 2
        xi = torch.cat((x,d,r,symin,symax,umin,umax),1)
        # control policy evaluation: Algo 2 - line 3
        u_hidden = F.relu(self.policy1(xi))
        u =  self.policy2(u_hidden)
      
#        correction terms - to be trained online
        u = u + self.u_correct() 
        x0_sysID = x0_sysID + self.x0_correct()
        x = x + self.x0_correct()      

#       system ID state update Algo 2 line 8
        x_sysID = self.A(x0_sysID) + self.B(u_sysID) + self.E(d) + w
        y_sysID = self.C(x_sysID)
        
        # system ID state residual Algo 2 line 9
        dx_sysID = x_sysID-x0_sysID  # one step state residual a.k.a smooth operator
        
        # penalties on max one-step infuence of controls and disturbances on states
        dx_u =  F.relu(-self.B(u_sysID)+self.dxmin) + F.relu(self.B(u_sysID)-self.dxmax) 
        dx_d =  F.relu(-self.E(d)+self.dxmin) + F.relu(self.E(d)-self.dxmax) 
        dx_ud = dx_u + dx_d
        
        #  control state update Algo 2 line 7
        x = self.A(x) + self.B(u) + self.E(d) + w
        y = self.C(x)
        
        # controlled state constraints Algo 2 lines 10-12  
        s_ymin = F.relu(-y+ymin) #       state lower bound  u>= umin
        s_ymax = F.relu(y-ymax) #      state upper bound u <= umax
        
        # control input constraints Algo 2 lines 4-6  
        s_umin = F.relu(-u+umin)  #      control lower bound u>= umin 
        s_umax = F.relu(u-umax) #      control upper bound u <= umax
        
#         Algo 2 line 13
        return x, y, u, s_ymin, s_ymax, s_umin, s_umax, y_sysID, dx_sysID, dx_ud


"""
# Model instantiation
"""   
# define additive uncertainty
w_mean_train = 0 # disturbance mean
w_var_train = 0.0 # disturbance variance

n_hidden = 10 # number of hidden neurons of the policy layers
nr = 1      # number of outputs/references
model = NeuroSSM_policy_one_step_con_DR(nx,nu,nd,nr,n_hidden)
model = model.float()
model.parameters
for param in model.named_parameters():
    print(param)

"""
# Dataset creation
"""   
# Randomized initial conditions and references
a = 0
b = 25   
X0 = a +b*np.random.rand(X.shape[0],X.shape[1])   
a = 15
b = 25  
R_day = 20+2*np.sin(np.arange(0, 2*np.pi,2*np.pi/samples_day)) #  daily control profile
R_day_train = 15+10*np.sin(np.arange(0, 2*np.pi,2*np.pi/samples_day)) #  daily control profile
R = np.matlib.repmat(R_day_train, 1, Sim_days) #  Sim_days control profile
R  = a +b*np.random.rand(R.shape[0],R.shape[1])   

# Features
Features = np.concatenate((X0[:,:-2].T, X[:,:-2].T, U[:,:-1].T, D[:,:-1].T, R[:,:-1].T), axis=1)       # tranining on randomized trajectory
# Targets
Targets =  np.concatenate((R[:,:-1].T, Y[:,1:].T), axis=1)   

# divide dataset to train and test set
Features_train, Features_test, Targets_train, Targets_test = train_test_split(Features,Targets,test_size=0.2,shuffle=False)
Features_train = torch.FloatTensor(Features_train)
Features_test = torch.FloatTensor(Features_test)
Targets_train = torch.FloatTensor(Targets_train)
Targets_test = torch.FloatTensor(Targets_test)
# train sets features
X_train = Features_train[:,0:nx]
X_sysID_train = Features_train[:,nx:2*nx]
U_sysID_train = Features_train[:,2*nx:2*nx+nu]
D_train = Features_train[:,2*nx+nu:-1]
R_train = Features_train[:,-1:]
# test sets features
X_test = Features_test[:,0:nx]
X_sysID_test = Features_test[:,nx:2*nx]
U_sysID_test = Features_test[:,2*nx:2*nx+nu]
D_test = Features_test[:,2*nx+nu:-1]
R_test = Features_test[:,-1:]
# targets
R_targets_train = Targets_train[:,0:ny]
Y_targets_train = Targets_train[:,ny:]
R_targets_test = Targets_test[:,0:ny]
Y_targets_test  = Targets_test[:,ny:]
# slack variables
sx_targets = torch.zeros(Targets_train.shape[0],nx)
sy_targets = torch.zeros(Targets_train.shape[0],ny)
su_targets = torch.zeros(Targets_train.shape[0],nu)
u_targets = torch.zeros(Targets_train.shape[0],nu)
dx_targets = torch.zeros(Targets_train.shape[0],nx)
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
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

"""
# Weights of the multi-objective loss - equation (17)
""" 
# constraints weight
Q_con_u = 5e-7
Q_con_x = 50
Q_con_y = 50
# control inputs weights
Q_u = 1e-6
Q_du = 1e-6
# reference tracking weights
Q_ref = 20
# system ID weights
Q_y_sysID = 100
Q_dx_sysID = 1
Q_dx_ud_sysID = 10


"""
Joint System ID and Policy Optimization Training Loop  using Algorithm 2
""" 
epochs = 30000
losses = []

for i in range(epochs):
    i+=1
#    Multi-objective loss function equation (17)
    x_pred, y_pred, u_pred, s_ymin, s_ymax, s_umin, s_umax, y_sysID, dx_sysID, dx_ud = model.forward(X_train, X_sysID_train, U_sysID_train, D_train, R_train, ymin, ymax, umin, umax,w_mean_train,w_var_train)
    loss = Q_ref*criterion(y_pred, R_targets_train)  + Q_u*criterion(u_pred, u_targets) \
    + Q_con_y*criterion(s_ymin, sy_targets)   + Q_con_y*criterion(s_ymax, sy_targets)  \
    + Q_con_u*criterion(s_umin, su_targets)   + Q_con_u*criterion(s_umax, su_targets)  \
    + Q_y_sysID*criterion(y_sysID, Y_targets_train) + Q_dx_sysID*criterion(dx_sysID, dx_targets) \
    + Q_dx_ud_sysID*criterion(dx_ud, sx_targets) 
    losses.append(loss)
    
    if i%1000 == 1:
        print(f'epoch: {i:2}  loss: {loss.item():10.8f}')
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch');


# save entire model
#torch.save(model, 'Trained_DeepMPCPolicy_sec_3_5.pt')


"""
# Parameters of the simulation from Section 3.5.
""" 
run_sim = True
online_learning = True
replicate_paper = True # load the policy used in the paper

Eval_runs  = 20  # number of randomized closed-loop simulations, Paper value: 20
train_epochs = 10 # number of iterations in one online batch, Paper value: 10

# increase penalties for constraints during online adaptation for increased safety
Q_con_u_on = 10*Q_con_u
Q_con_y_on = 10*Q_con_y

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
    model_on = torch.load('Trained_DeepMPCPolicy_sec_3_5.pt') 
    
    # fix all params
    for param in model_on.parameters():
        param.requires_grad = False
    # unfix compensators
    for param in model_on.policy2.parameters():   #  online policy correction param
        param.requires_grad = True 
    
    for param in model_on.x0_correct.parameters(): #  online model correction param
        param.requires_grad = True 
                
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_on.parameters()), lr=0.001)
     
"""
Running closed-loop control simulations from Section 3.5.
""" 
if run_sim:  
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
    DATA_Y_min = np.ndarray(shape=(ymin.T.shape[0],ymin.T.shape[1],Eval_runs)) 
    DATA_Y_max = np.ndarray(shape=(ymax.T.shape[0],ymax.T.shape[1],Eval_runs)) 
    
    u = torch.from_numpy(U_torch[:,0]).float() 
    
    xmin_ = xmin_val*torch.ones(1,nx)
    xmax_ = xmax_val*torch.ones(1,nx)
    ymin_ = ymin_val*torch.ones(1,ny)
    ymax_ = ymax_val*torch.ones(1,ny)
    umin_ = umin_val*torch.ones(1,nu)
    umax_ = umax_val*torch.ones(1,nu)
    
    CPU_mean_time = np.zeros(Eval_runs)
    CPU_max_time = np.zeros(Eval_runs)
    
    # initialize robust relaxations of constraints
    if robust_relaxation:
        ymax_viol = torch.zeros(1,ny)  
        ymin_viol = torch.zeros(1,ny)
    
    for run in range(0,Eval_runs):       
        #        reset model update at each eval
        model_on = torch.load('Trained_DeepMPCPolicy_sec_3_5.pt') 
        for param in model_on.parameters():
            param.requires_grad = False      
        for param in model_on.policy2.parameters():
            param.requires_grad = True      
        for param in model_on.x0_correct.parameters():
            param.requires_grad = True                    
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_on.parameters()), lr=0.001)

        StepTime = np.zeros(N_sim_t-14*samples_day)
        
        for k in range(0,N_sim_t):
            x0 = np.asmatrix(X_cl[:,k])
            d = np.asmatrix(D_t[:,k])
            r = np.asmatrix(R_t[:,k])
            w = (w_mean-w_var)+(2*w_var)*np.asmatrix(np.random.rand(nx,1)) # additive uncertainty
            theta = (1+theta_mean-theta_var)+(2*theta_var)*np.asmatrix(np.random.rand(nx,nx)) # parametric uncertainty

            x0_torch = torch.from_numpy(x0).float()
            d_torch = torch.from_numpy(d).float()
            r_torch = torch.from_numpy(r).float()
        
            start_step_time = time.time() 
            xn_pt, y_pt, u, s_ymin, s_ymax, s_umin, s_umax, y_sysID, dx_sysID, dx_ud = model_on(x0_torch,x0_torch,u,d_torch,r_torch,ymin_,ymax_,umin_,umax_,w_mean_train,w_var_train)   
            eval_time = time.time() - start_step_time
   
            if u_clipping:
                if u < umin_:
                    u = umin_                
                elif u > umax_:
                    u = umax_
        
            X_torch[:,k+1] = xn_pt.detach().numpy().ravel()
            Y_torch[:,k] = y_pt.detach().numpy().ravel()
            U_torch[:,k] = u.detach().numpy().ravel()
        
            # ground truth LTI SSM
            xn = np.multiply(theta,A)*x0.T + B*U_torch[:,k] + E*d.T + w
            y = C*x0.T
            
            X_cl[:,k+1] = xn.ravel()
            Y_cl[:,k] = y.ravel()
            
    #        evaluate robust relaxation of constraints
            if robust_relaxation:
                y_torch = torch.from_numpy(y).float()        
                s_ymin_real = F.relu(-y_torch+ymin_)
                s_ymax_real = F.relu(y_torch-ymax_)
                ymin_viol = torch.max(s_ymin_real.detach(), ymin_viol)
                ymax_viol = torch.max(s_ymax_real.detach(), ymax_viol)
                ymin_ = ymin_ + ymin_viol
                ymax_ = ymax_ - ymax_viol
              
    #        ONLINE training phase
            online_train_time = 0
            if online_learning and (k>14*samples_day):
                start_step_time_online = time.time() 
                y_target =  torch.from_numpy(np.asmatrix(Y_cl[:,k])).float()
                losses = []
                for i in range(train_epochs):
                    i+=1
                    x_pred, y_pred, u_pred, s_ymin, s_ymax, s_umin, s_umax, y_sysID, dx_sysID, dx_ud =  model_on(x0_torch,x0_torch,u.detach(),d_torch,r_torch,ymin_,ymax_,umin_,umax_,w_mean_train,w_var_train)    
                    loss = Q_ref*criterion(y_pred, r_torch)  + Q_u*criterion(u_pred, u_targets) \
                    + Q_con_y_on*criterion(s_ymin, sy_targets)   + Q_con_y_on*criterion(s_ymax, sy_targets)  \
                    + Q_con_u_on*criterion(s_umin, su_targets)   + Q_con_u_on*criterion(s_umax, su_targets)  \
                    + 1e-1*Q_y_sysID*criterion(y_sysID, y_target) + Q_dx_sysID*criterion(dx_sysID, dx_targets) \
                    + Q_dx_ud_sysID*criterion(dx_ud, sx_targets) 
                    losses.append(loss)                   
        
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                online_train_time = time.time() -start_step_time_online
            
#            evaluate only test set policy time            
            if (k>14*samples_day):
                StepTime[k-14*samples_day-1] =online_train_time + eval_time         
            
        CPU_mean_time[run] = np.mean(StepTime)
        CPU_max_time[run] = np.amax(StepTime)
        
        # various evaluations with uncertainty    
        DATA_X_cl[:,:,run] = X_cl
        DATA_Y_cl[:,:,run] = Y_cl
        DATA_U_torch[:,:,run] = U_torch
        DATA_X_torch[:,:,run] = X_torch
        DATA_Y_torch[:,:,run] = Y_torch
        DATA_R_t[:,:,run] = R_t
        DATA_Y_min[:,:,run] = ymin.T
        DATA_Y_max[:,:,run] = ymax.T
    
    CPU_mean_time_paper = np.mean(CPU_mean_time)*1e3 
    CPU_max_time_paper = np.amax(CPU_max_time)*1e3  
    
    DATA_X_cl_max = np.amax(DATA_X_cl, axis = 2)
    DATA_X_cl_min = np.amin(DATA_X_cl, axis = 2)
    DATA_X_cl_mmean = np.mean(DATA_X_cl, axis = 2)
    
    DATA_Y_cl_max = np.amax(DATA_Y_cl, axis = 2)
    DATA_Y_cl_min = np.amin(DATA_Y_cl, axis = 2)
    DATA_Y_cl_mmean = np.mean(DATA_Y_cl, axis = 2)
    
    DATA_X_torch_max = np.amax(DATA_X_torch, axis = 2)
    DATA_X_torch_min = np.amin(DATA_X_torch, axis = 2)
    DATA_X_torch_mmean = np.mean(DATA_X_torch, axis = 2)
    
    DATA_U_torch_max = np.amax(DATA_U_torch, axis = 2)
    DATA_U_torch_min = np.amin(DATA_U_torch, axis = 2)
    DATA_U_torch_mmean = np.mean(DATA_U_torch, axis = 2)
    
    DATA_Y_torch_max = np.amax(DATA_Y_torch, axis = 2)
    DATA_Y_torch_min = np.amin(DATA_Y_torch, axis = 2)
    DATA_Y_torch_mmean = np.mean(DATA_Y_torch, axis = 2)
    
    time_x = np.linspace(0, Sim_days_test, DATA_X_cl_mmean.shape[1], endpoint=False)
    time_y = np.linspace(0, Sim_days_test, DATA_Y_cl_mmean.shape[1], endpoint=False)
    time_u = np.linspace(0, Sim_days_test, DATA_U_torch_mmean.shape[1], endpoint=False)
    
    xmin = xmin_val*np.ones((time_x.shape[0],nx))
    xmax = xmax_val*np.ones((time_x.shape[0],nx))
    ymin = ymin_val*np.ones((time_y.shape[0],ny))
    ymax = ymax_val*np.ones((time_y.shape[0],ny))
    umin = umin_val*np.ones((time_u.shape[0],ny))
    umax = umax_val*np.ones((time_u.shape[0],ny))
     
#    eval control statistics for Table 5
    test_start = 14*samples_day
    test_end = DATA_Y_torch_mmean.shape[1]
    DATA_Y_cl_torch = torch.from_numpy(DATA_Y_cl[:,test_start:test_end]).float()  
    DATA_s_ymin_real = F.relu(-DATA_Y_cl_torch+ymin_).detach().numpy()
    DATA_s_ymax_real = F.relu(DATA_Y_cl_torch-ymax_).detach().numpy()
    MAE_constr = np.mean(DATA_s_ymin_real) + np.mean(DATA_s_ymax_real)
    MSE_model = np.mean(np.square(DATA_Y_cl[:,test_start:test_end]-DATA_Y_torch[:,test_start:test_end]))
    MSE_ref = np.mean(np.square(DATA_Y_cl[:,test_start:test_end]-DATA_R_t[:,test_start:test_end]))
    MA_energy =  np.mean(np.absolute(DATA_U_torch_mmean[:,test_start:test_end]))
    

"""
Plot for Figure 2
"""     
if paper_plot:   
    fig, ax = plt.subplots(3,1,figsize=(8, 5))         
    ax[0].plot(time_y,R_t.T, '--', label='R')
    ax[0].plot(time_y,DATA_Y_cl_mmean.T, label='Y')
    ax[0].fill_between(time_y, np.squeeze(DATA_Y_cl_min.T), np.squeeze(DATA_Y_cl_max.T), color='orange', alpha=0.2)
    ax[0].plot(time_y,DATA_Y_torch_mmean.T, label='Y')
    ax[0].fill_between(time_y, np.squeeze(DATA_Y_torch_min.T), np.squeeze(DATA_Y_torch_max.T), color='green', alpha=0.2)
    ax[0].plot(time_y,ymin, 'k--')
    ax[0].plot(time_y,ymax, 'k--')
    ax[0].set(ylabel='$x$')
    # ax[0].set(ylabel='$\mathcal{X}_4$ [$^\circ$C]')
    ax[0].tick_params(labelbottom=False)
    ax[0].axvspan(7, 14, facecolor='grey', alpha=0.2, zorder=-100)
    ax[0].axvspan(14, 21, facecolor='grey', alpha=0.4, zorder=-100)
    ax[0].margins(0,0.1)
    
    ax[1].plot(time_u,DATA_U_torch_mmean.T, label='U')
    ax[1].fill_between(time_u, np.squeeze(DATA_U_torch_min.T), np.squeeze(DATA_U_torch_max.T), color='blue', alpha=0.2)
    ax[1].plot(time_u,umin, 'k--')
    ax[1].plot(time_u,umax, 'k--')
    # ax[1].set(ylabel='$\mathcal{U}$ [kW]')
    ax[1].set(ylabel='$u$')
    ax[1].tick_params(labelbottom=False)
    ax[1].axvspan(7, 14, facecolor='grey', alpha=0.2, zorder=-100)
    ax[1].axvspan(14, 21, facecolor='grey', alpha=0.4, zorder=-100)
    ax[1].margins(0,0.1)
    ax[1].set(yticks=(0, 2000, 4000),yticklabels=('0', '2', '4'))
    
    # ax[2].plot(time_u,D_t[1:3,:].T)
    ax[2].plot(time_u,D_t.T)
    ax[2].tick_params(labelbottom=True)
    ax[2].set_xlabel('Day')
    ax[2].set(ylabel='$d$')
    # ax[2].set(ylabel='$\mathcal{D}_{(2,3)}$ [kW]')
    ax[2].tick_params(labelbottom=True)
    ax[2].axvspan(7, 14, facecolor='grey', alpha=0.2, zorder=-100)
    ax[2].axvspan(14, 21, facecolor='grey', alpha=0.4, zorder=-100)
    ax[2].margins(0,0.1)
    ax[2].text(1, 1200, '             Train                ',
                bbox={'edgecolor': 'none','facecolor': 'white', 'alpha': 0.0})
    ax[2].text(8, 1200, '           Validation           ',
                bbox={'edgecolor': 'none','facecolor': 'grey', 'alpha': 0.0})
    ax[2].text(15, 1200, '              Test                ',
                   bbox={'edgecolor': 'none','facecolor': 'grey', 'alpha': 0.0})
    # ax4 = ax[2].twinx()  # instantiate a second axes that shares the same x-axis
    # ax4.plot(time_u,D_t[0,:].T, color ='green')
    # ax4.tick_params(axis='y', labelcolor='green')
    # ax4.set_ylabel('$\mathcal{D}_1$ [$^\circ$C]', color='green')
    # ax4.margins(0,0.1)
    
    steps = range(0, Sim_days_test+1, 1)
    days = np.array(list(range(Sim_days_test+1)))+7
    ax[2].set(xticks=steps,xticklabels=days)
    ax[2].set(yticks=(0, 500, 1000),yticklabels=('0', '.5', '1'))
    
    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    
    if save_fig:
        plt.savefig('DeepMPC_sysID_ctrl_sec_2_5.pdf')
    

