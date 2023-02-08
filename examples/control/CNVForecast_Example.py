# Tutorial example for Forecast CNV Control Policy

import torch
import neuromancer as nm
import slim 
import numpy as np



from neuromancer import dataset
from neuromancer import estimators
from neuromancer import loss
from neuromancer import problem
from neuromancer import trainer
from neuromancer import constraint
from neuromancer import policies
from neuromancer import dynamics


from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


''' 
# # # Problem Definition

Linear time invariant dynamic system
with a delayed control action.

System state has two dimensions x[0] and x[1]

x[0]' = x[1]
x[1]' = u

goal is to control x[0] to follow a given trajectory using 
the control input u

'''


# LTI SSM
# x_k+1 = Ax_k + Bu_k
# y_k+1 = Cx_k+1
A = torch.tensor([[0.0, 1.0],
                  [0.0, 0.0]])
B = torch.tensor([[0.0],
                  [1.0]])
C = torch.tensor([[1.0, 0.0],
                  [0.0, 1.0]])





"""
# # #  Set Problem Parameters
"""



nsteps = 3 # number of forecast steps
kernel_size = 2 #Number of timesteps to include in convolutional layer
#Set kernel_size =1 to train a control without access to future trajectory information





# Bounds on control input
umin = -2.
umax = 2.


# Bounds on the state
xmin = -2.
xmax = 2.


nsim = 10000 # number of datapoints to sample




''' 
# # # Construct Trajectories to Track

trajectories are given by cosine waves
for each trial frequency and shift are varied.

traj =  scl_factor*cos(freq*(t - shift))

'''



freq_max = .1
freq_min = .01
shift_max = (2/freq_min)*np.pi
shift_min = 0
scl_factor = 1
Traj_seq_len = nsteps



Traj_parms = np.concatenate( (np.expand_dims(np.random.uniform(freq_min, freq_max,nsim),1),np.expand_dims(np.random.uniform(shift_min,shift_max,nsim),1) ),axis= 1 )
Traj_data_list = []
t_vals = np.arange(0,Traj_seq_len,1)
for i in range(nsim):
    Traj_data_list.append( scl_factor*np.cos(Traj_parms[i,0]*(t_vals - Traj_parms[i,1])) )
Traj_data = np.expand_dims(np.stack(Traj_data_list),2)
Traj_data = torch.tensor(Traj_data,dtype = torch.float32)






''' 
# # # Construct Random initial conditions

Initial conditions are taken as a random normal pertubation
from the trajectory to be tracked


'''

# problem dimensions
nx = 2  #dimension of the state
nu = 1  #dimension of the control


# Randomly Sample Initial Conditions
# (batch, y )
x0_data = 1*np.expand_dims(np.random.randn(nsim, nx),1)
x0_data = torch.tensor(x0_data,dtype = torch.float32) + torch.concat( (Traj_data[:,0,None,:],torch.zeros((nsim,1,1)) ),axis = 2)







''' 
# # # Construct Neuromancer Datasets for Training
'''

train_proportion = .8 #amount of data to use for training
batch_size = 5000 #batch size for training





train_split = np.floor(train_proportion*nsim).astype(int)
test_split = train_split + np.floor((1 - train_proportion)*nsim/2).astype(int)

train_datadict = {'Yp':x0_data[0:train_split], 'Df':Traj_data[0:train_split] }
test_datadict = {'Yp':x0_data[train_split:test_split], 'Df':Traj_data[train_split:test_split] }
dev_datadict = {'Yp':x0_data[test_split:], 'Df':Traj_data[test_split:] }

train_dataset = nm.dataset.DictDataset(train_datadict,name = 'train')
train_dataloader = DataLoader(train_dataset,batch_size = batch_size,collate_fn = train_dataset.collate_fn)
test_dataset = nm.dataset.DictDataset(test_datadict,name = 'test')
test_dataloader = DataLoader(test_dataset,batch_size = batch_size,collate_fn = test_dataset.collate_fn)
dev_dataset = nm.dataset.DictDataset(dev_datadict,name = 'dev')
dev_dataloader = DataLoader(dev_dataset,batch_size = batch_size,collate_fn = dev_dataset.collate_fn)
data_dims = {'Yp':x0_data.shape[1:],'x0':x0_data.shape[1:],'Df':Traj_data.shape[1:] ,'U':torch.tensor([Traj_seq_len,1])}







"""
# # #  System model and Control policy

A convolutional forecast policy is used to learn a control for the system.
The given trajectory to track is taken as input and 

"""



# Fully observable estimator as identity map: x0 = Yp[-1]
# x_0 = e(Yp)
# Yp = [y_-N, ..., y_0]
estimator = nm.estimators.FullyObservable(
               data_dims,
               input_keys=["Yp"], name='est')



policy = nm.policies.ConvolutionalForecastPolicy(
    {**data_dims,**{estimator.output_keys[0]:(nx,)} },
    input_keys = [estimator.output_keys[0],'Df'],
    nsteps = nsteps, 
    kernel_size = kernel_size,
    hsizes = [10])



dynamics_model = nm.dynamics.LinearSSM(A, B, C, name='mod',
                          input_key_map={'x0': estimator.output_keys[0],
                                         'Uf': policy.output_keys[0],'Df':'Df','Yf':policy.output_keys[0]} )
dynamics_model.requires_grad_(False)  # fix model parameters







"""
# # #  DPC objectives and constraints
"""

u = nm.constraint.variable(policy.output_keys[0])
y = nm.constraint.variable(dynamics_model.output_keys[2])
target = nm.constraint.variable('Df')
# objective weights
Qu = 0.0001
Qx = 10.
Q_con_x = 10.
Q_con_u = 100.
Qn = 1.
# objectives
action_loss = Qu * ((u == 0.) ^ 2)  # control penalty
regulation_loss = Qx * ((y[:,1:,0] == target[:,1:,0] )^ 2)  # target posistion

# constraints
state_lower_bound_penalty = Q_con_x * (y[:,:,0] > xmin)
state_upper_bound_penalty = Q_con_x * (y[:,:,0] < xmax)
inputs_lower_bound_penalty = Q_con_u * (u > umin)
inputs_upper_bound_penalty = Q_con_u * (u < umax)


# list of objectives and constraints
objectives = [regulation_loss, action_loss]
constraints = [
    state_lower_bound_penalty,
    state_upper_bound_penalty,
    inputs_lower_bound_penalty,
    inputs_upper_bound_penalty
]









"""
# # #  DPC problem = objectives + constraints + trainable components 
"""
# data (y_k) -> estimator (x_k) -> policy (u_k) -> dynamics (x_k+1, y_k+1)
components = [estimator, policy, dynamics_model]
# create constrained optimization loss
loss = nm.loss.PenaltyLoss(objectives, constraints)
# construct constrained optimization problem
problem = nm.problem.Problem(components, loss)
# plot computational graph
problem.plot_graph()








"""
# # #  DPC trainer 
"""
optimizer = torch.optim.AdamW(problem.parameters(), lr=1e-2)
trainer = nm.trainer.Trainer(
    problem,
    train_dataloader,
    dev_dataloader,
    test_dataloader,
    optimizer,
    epochs=500,
    train_metric="train_loss",
    dev_metric="dev_loss",
    test_metric="test_loss",
    eval_metric='dev_loss',
    patience = 10
)

# Train control policy
best_model = trainer.train()
best_outputs = trainer.test(best_model)









'''  
# # # TEST CLOSE LOOP RHC CONTROL


Test a receding horizon implementation, 
at each time step the open loop control is computed,
the dynamics are progressed one time step using the control policy, 
then this procedure is repeated.

'''


n_rollout_steps = 100




#generate a target trajectory
D_parms = np.concatenate( (np.expand_dims(np.random.uniform(freq_min, freq_max,1),1),np.expand_dims(np.random.uniform(shift_min,shift_max,1),1) ),axis= 1 )
t_vals = np.arange(0,n_rollout_steps,1)
Tj_data = scl_factor*np.cos(D_parms[0,0]*(t_vals - D_parms[0,1])) 
Tj_data = np.reshape(Tj_data,(len(Tj_data),1))
Tj_data = torch.tensor(Tj_data,dtype = torch.float32)


#generate random initial condition
x0_data = .2*torch.tensor(1*np.expand_dims(np.random.randn(1, nx),1),dtype = torch.float32)


D_rollout_dict = {'D':Tj_data}
sim_steps = 2

sim_dataset = nm.dataset.SequenceDataset(D_rollout_dict,nsteps = nsteps,name = 'sim',moving_horizon = True)
sim_dataloader = DataLoader(sim_dataset)




state_dict = {'Yp':x0_data}
y_traj = []
control_traj = []
target_traj = []

D_dict_iter = iter(sim_dataloader)

for i in range(len(sim_dataloader)):
    D_dict = next(D_dict_iter)
    sim_dict = {**state_dict,**D_dict}

    target_traj.append(D_dict['Df'][0,0,:].detach().numpy())

    e_dict = estimator(sim_dict)
    sim_dict = {**sim_dict, **e_dict}

    p_dict = policy(sim_dict)
    sim_dict = {**sim_dict,**p_dict}


    dyn_dict = dynamics_model(sim_dict)
    sim_dict = {**sim_dict,**dyn_dict}

    state_dict = {'Yp':sim_dict['Y_pred_mod'][:,0,None,:]}

    y_traj.append(sim_dict['Y_pred_mod'][0,0,:].detach().numpy())
    control_traj.append(sim_dict['U_pred_CNV_policy'][0,0,:].detach().numpy())


control_traj = np.stack(control_traj)
y_traj = np.stack(y_traj)
target_traj = np.stack(target_traj)




plt.figure()
plt.plot(y_traj[:,0])
plt.plot(target_traj)
plt.title('x[0] Solution')
plt.legend(['x[0]','target'])


plt.figure()
plt.plot(y_traj[:,0] - target_traj[:,0])
plt.title('Residual')







