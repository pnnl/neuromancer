


'''
This script trains a DED-DPC model

'''


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



import pandas as pd
import numpy as np

import dill

import neuromancer as nm
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem, Objective
from neuromancer.loggers import BasicLogger
from neuromancer.dataset import read_file, normalize_data, split_sequence_data, SequenceDataset



from neuromancer.DED import fpSplitSequenceDataset




'''
#####################
Set Training Parameters
'''


n_epochs = 10 #maximum number of training epochs

lrn_rate = 1e-3

device = 'cpu'




#Problem parameters


'''
Generator input bounds
'''

u_min = [.815,.425]
u_min_weight = 1.0 #how constraint penalty is weighted in the training objective

u_max = [2.445,1.275]
u_max_weight = 1.0


'''
Generator frequency bounds
'''
#omega_max = [.01,.01]
omega_max = [.314,.314]
omega_max_weight = 1e3


#omega_min = [-.01,-.01]
omega_min = [-.314,-.314]
omega_min_weight = 1e3


'''
Generator ramping constraint
'''

rmp_tol = .0005
rmp_cns_weight = 1e4


init_cntrl_weight = 1e4






'''
#####################
Specify Where to save trained model
'''

save_model_folder = 'saved_models/example_model'




'''
#####################
Specify KO model to load
'''


KO_save_model_folder = '../Koopman_system_ID/saved_models/KO_example_model'





'''
#####################
Specify training data location
'''

data_root = '../Data/Example_Data/'
meta_data_file = 'opt_metadata.csv'



nsteps = 5999 #length of the trajectories in the data


met_data = pd.read_csv(data_root+ meta_data_file)

n_files = met_data.shape[0]
file_idx = np.arange(1,n_files+1)

Dat_dict = []
for i in file_idx:
    path = data_root + f'trial_data/trial_{i}.csv'
    dat = read_file(path)
    Dat_dict.append(dat)



##Preprocess the state data for inputs to the state to observable map.
for j in range(len(Dat_dict)):
    Y_phi = np.zeros((6001,40))
    for i in range(6001):
        X0 = Dat_dict[j]['Y'][i,:]
        Ps0 = X0[12]
        z = X0[3:12] 
        zdff = np.reshape(z,[9,1]) - np.reshape(z,[1,9])
        X0 =np.concatenate( (np.reshape(X0[0:3],[3,]) , zdff[np.triu_indices(zdff.shape[0],k = 1)] ) )
        X0 = np.concatenate((np.reshape(X0,[39,1]),np.reshape([Ps0],(1,1)) ))  
        Y_phi[i,:] = np.ravel(X0)
    Dat_dict[j]['Y_phi'] = Y_phi
    
    jj = file_idx[j] - 1
    c = met_data[['cs','c1','c2']].loc[jj].values
    c = np.reshape(c,(1,3))
    cvec = np.tile(c,(6001,1))
    Dat_dict[j]['coefs'] = cvec



#Split data for training
train_data = Dat_dict[0:-1]
dev_data = Dat_dict[-1]
test_data = Dat_dict[-1]


train_data = fpSplitSequenceDataset(train_data,nsteps = 5999, psteps = 1, name = 'train')
dev_data = fpSplitSequenceDataset(dev_data,nsteps = 5999, psteps = 1, name = 'dev')
test_data = fpSplitSequenceDataset(test_data,nsteps = 5999, psteps = 1, name = 'test')

#### Construct torch dataloader objects which handle batching and pass data to the trainer.

train_data = DataLoader(
    train_data,
    batch_size=len(train_data),
    shuffle=False,
    collate_fn=train_data.collate_fn
)

dev_data = DataLoader(
    dev_data,
    batch_size=len(dev_data),
    shuffle=False,
    collate_fn=dev_data.collate_fn
)

test_data = DataLoader(
    test_data,
    batch_size=len(test_data),
    shuffle=False,
    collate_fn=test_data.collate_fn
)




metrics = ["nstep_train_ref_loss"]
logger = BasicLogger(stdout=metrics,savedir='saved_models')

#get dictionary of the variable keys and associated dimension
dims = train_data.dataset.dims
nx = dims["Y"][-1] # set the dimension of the true system state, here assuming estimator is fully observable
dims['x0'] =  (nx,) # add state dimension to the dictionary






'''
##############################
### Load the Koopman Dynamics model
###############################
'''



model_path = KO_save_model_folder + '/best_model.pth'
sv_model = torch.load(model_path, pickle_module=dill,
                             map_location=torch.device('cpu'))


estimator = sv_model.components[0]
StO_map = sv_model.components[1]
dynamics_model = sv_model.components[2]



'''
##############################
### Construct the Control Policy
###############################
'''
from neuromancer.DED import Direct_Conv_policy_fn
from neuromancer.DED import Forecast_Policy




#Construct the control policy
ndsteps  = 50   #down sampled time dimension

cnv_fn = Direct_Conv_policy_fn(6,2,12,16,nsteps,ndsteps) # control policy function
cnv_fn.to(device)


policy = Forecast_Policy(cnv_fn,input_keys={f"psi_0_{StO_map.name}": "x0"} ) #control policy component
policy.to(device)





'''
##############################
### Construct the Objective and Constraints
###############################
'''



'''
Objective Definition
'''

def Objective_fn(x,u,c):
    obj =  torch.linalg.norm(c[:,:,0]*x[:,:,2],ord=1,dim=0) + torch.linalg.norm(c[:,:,[1,2]]*u,ord=1,dim=0).sum(1)
    return torch.mean(obj)
     

reference_loss = Objective(
    [f"X_pred_{dynamics_model.name}", f"U_pred_{policy.name}","coefsf"], Objective_fn, weight=1.0, name="ref_loss"
)






'''
Generator Ramping Constraints
'''

c_rmp_tol = torch.tensor([rmp_tol],requires_grad=False,device=device)
def cntrl_rmp_fn(u,up):
    spp = list(u.shape)
    spp[0] = 1
    up = torch.reshape(up[0,:,:],spp)
    u = torch.cat((u,up),dim=0)
    output = torch.linalg.norm(F.relu(u[1:,:,:] - u[:-1,:,:] - c_rmp_tol ),ord=1,dim=0 ).sum(1) +  torch.linalg.norm(F.relu( - u[1:,:,:] + u[:-1,:,:] - rmp_tol),ord=1,dim=0 ).sum(1)      
    return torch.mean(output)

Control_ramping_cnstrnt = Objective(
        [f"U_pred_{policy.name}",'Up'],
        cntrl_rmp_fn,
        weight=rmp_cns_weight,
        name="ramping_cnstrnt",
    )







'''
Consistency of the initial control
'''

def cntrl_initial_cnd_fn(u,up):
    up_i = up[0,:,:]
    u_i = u[0,:,:]
    output = torch.linalg.norm(up_i - u_i,ord=2,dim=1)
    return torch.mean(output)

Control_initial_cnstrnt = Objective(
        [f"U_pred_{policy.name}",'Up'],
        cntrl_initial_cnd_fn,
        weight=init_cntrl_weight,
        name="cntrl_init_cnstrnt",
    )






'''
Bounds on Generator Inputs
'''

u_min = torch.tensor(u_min,requires_grad = False,device=device)
def cntrl_lb_fn(u):
    output =  torch.linalg.norm(F.relu( - u + u_min),ord=1,dim=0 ).sum(1)
    return torch.mean(output)

Control_lower_bound_penalty = Objective(
     [f"U_pred_{policy.name}"],
     cntrl_lb_fn,
     weight=u_min_weight,
     name="control_lower_bound",
 )


u_max = torch.tensor(u_max,requires_grad = False,device=device)
def cntrl_ub_fn(u):
    output =  torch.linalg.norm(F.relu( u - u_max),ord=1, dim=0 )
    return torch.mean(output)

Control_upper_bound_penalty = Objective(
     [f"U_pred_{policy.name}"],
     cntrl_ub_fn,
     weight=u_max_weight,
     name="control_upper_bound",
 )




'''
Bounds on Generator Frequencies
'''

omega_max = torch.tensor(omega_max,requires_grad=False,device=device)


def omega_ub_fn(x):
    output = torch.linalg.norm(F.relu( x[:,:,[0,1]] - omega_max),ord=1,dim=0 ).sum(1)
    return torch.mean(output)
    
omega_upper_bound_penalty = Objective(
     [f"X_pred_{dynamics_model.name}"],
     omega_ub_fn,
     weight=omega_max_weight,
     name="omega_upper_bound",
 )





omega_min = torch.tensor(omega_min,requires_grad=False,device = device)



def omega_lb_fn(x):
    output =  torch.linalg.norm(F.relu( -x[:,:,[0,1]] + omega_min),ord=1,dim=0 ).sum(1)
    return torch.mean(output)
    
omega_lower_bound_penalty = Objective(
     [f"X_pred_{dynamics_model.name}"],
     omega_lb_fn,
     weight=omega_min_weight,
     name="omega_lower_bound",
 )





'''
##############################
### Build The Neuromancer Problem
###############################
'''




objectives=[reference_loss]
constraints=[Control_ramping_cnstrnt,
             Control_lower_bound_penalty,
             Control_upper_bound_penalty,
             omega_upper_bound_penalty,
             omega_lower_bound_penalty,
             Control_initial_cnstrnt
             ]

components=[estimator,
            StO_map,
            policy,
            dynamics_model]


model = Problem(objectives, constraints, components)
model = model.to(device)





'''
##############################
### Train The Model
###############################
'''


optimizer = torch.optim.AdamW(model.parameters(), lr=lrn_rate)


logger = BasicLogger(stdout=metrics,savedir=save_model_folder)


trainer = Trainer(
    model,
    train_data,
    dev_data,
    test_data,
    optimizer,
    eval_metric="nstep_train_ref_loss",
    epochs=n_epochs,
    logger = logger,
    patience=50,
    device = device
)

best_model = trainer.train()




