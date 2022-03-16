


'''
This script constructs a Koopman Model of the swind dynamics in neuromancer
using saved weights and trains the model to fit swing dynamics trajectories.


'''


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



import pandas as pd
import numpy as np


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


n_epochs = 1 #maximum number of training epochs

lrn_rate = 1e-7

device = 'cpu'






'''
#####################
Specify Where to save trained model
'''

save_model_folder = 'saved_models/KO_example_model'







'''
#####################
Specify file with initial KO model parameters
'''
KO_model_file = '../Data/KO_model_parameters/KO_model_parms.npz'





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
train_data = Dat_dict[0:2]
dev_data = Dat_dict[1]
test_data = Dat_dict[1]


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
### build the Koopman Dynamics model
###############################
'''
from neuromancer.DED import State_to_Obs_map
from neuromancer.DED import Domain_aware_KO_state_transition
from neuromancer.DED import StatetoObservableMap
from neuromancer.DED import KoopmanUpdateModel




#Internal functions
fpsi = State_to_Obs_map(40,12,8,np.array([1,2,39]))
fpsi.to(device)


fxud = Domain_aware_KO_state_transition(11,8,device)
fxud.to(device)


##Set initial values for the KO Model

npzfile = np.load(KO_model_file,allow_pickle=True)

## get the observable and Koopman parameters
kdy=npzfile['arr_0']
K = kdy[0]
B = kdy[1]
phi_parms=kdy[2:8]

#Btop = np.array(np.concatenate( (np.zeros((2,8)),np.reshape([-1,-1,0,-1,0,-1,0,-1],(1,8)) ) ),dtype='float32')
#B = np.concatenate([Btop,B],0)

##Set initial values for fpsi
#L1
with torch.no_grad():fpsi.L1.weight.copy_(torch.tensor(phi_parms[0]))
with torch.no_grad():fpsi.L1.bias.copy_(torch.tensor(np.ravel(phi_parms[1])))
fpsi.L1.weight.requires_grad=True
fpsi.L1.bias.requires_grad = True
fpsi.L1.to(device)
#L2
with torch.no_grad():fpsi.L2.weight.copy_(torch.tensor(phi_parms[2][0,:,:]))
with torch.no_grad():fpsi.L2.bias.copy_(torch.tensor(np.ravel(phi_parms[3][0,:,:])))
fpsi.L2.weight.requires_grad=True
fpsi.L2.bias.requires_grad = True
fpsi.L2.to(device)
#L3
with torch.no_grad():fpsi.L3.weight.copy_(torch.tensor(phi_parms[2][1,:,:]))
with torch.no_grad():fpsi.L3.bias.copy_(torch.tensor(np.ravel(phi_parms[3][1,:,:])))
fpsi.L3.weight.requires_grad=True
fpsi.L3.bias.requires_grad = True
fpsi.L3.to(device)
#L4
with torch.no_grad():fpsi.L4.weight.copy_(torch.tensor(phi_parms[2][2,:,:]))
with torch.no_grad():fpsi.L4.bias.copy_(torch.tensor(np.ravel(phi_parms[3][2,:,:])))
fpsi.L4.weight.requires_grad=True
fpsi.L4.bias.requires_grad = True
fpsi.L4.to(device)
#L5
with torch.no_grad():fpsi.L5.weight.copy_(torch.tensor(phi_parms[4]))
with torch.no_grad():fpsi.L5.bias.copy_(torch.tensor(np.ravel(phi_parms[5])))
fpsi.L5.weight.requires_grad= True
fpsi.L5.bias.requires_grad = True
fpsi.L5.to(device)



##Set initial values for the KO Model


#K
with torch.no_grad():fxud.KO.weight.copy_(torch.tensor(K.T) )
with torch.no_grad():fxud.KO.W.copy_(torch.tensor(K.T) )
fxud.KO.weight.requires_grad = True
fxud.KO.W.requires_grad = True
fxud.KO.to(device)


#B
with torch.no_grad():fxud.B_bot.copy_(torch.tensor(B) )
fxud.B_bot.requires_grad = True
fxud.B_bot.to(device)

n_O= 11

fpsi.to(device)
fxud.to(device)




#Component Construction


estimator = nm.estimators.FullyObservable_MH(dims,input_keys={"Y_phif": "Yp"},nsteps=nsteps)
estimator.to(device)

StO_map = StatetoObservableMap(fpsi,input_keys={f"x0_{estimator.name}": "x0"})
StO_map.to(device)
StO_map.requires_grad = False


dynamics_model = KoopmanUpdateModel(fxud,fpsi,input_keys={f"psi_0_{StO_map.name}": "x0"})
dynamics_model.to(device)
dynamics_model.requires_grad = False






'''
##############################
### Construct the Objective 
###############################
'''

reg_weight = torch.tensor(1e-3)
reg_weight.to(device)

def ref_loss_fn(x_pred,x_true):
    loss =  torch.sum(  torch.sum( torch.pow(x_true - x_pred,2),dim=2 ),dim = 0  ) + reg_weight*dynamics_model.fxud.KO.reg_error()
    return torch.mean(loss) 
     


reference_loss = Objective(
    [f"X_pred_{dynamics_model.name}", f"X_true_{dynamics_model.name}"], 
    ref_loss_fn, 
    weight=1.0, 
    name="ref_loss"
)



'''
##############################
### Build The Neuromancer Problem
###############################
'''




objectives=[reference_loss]
constraints=[]

components=[estimator,
            StO_map,
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




