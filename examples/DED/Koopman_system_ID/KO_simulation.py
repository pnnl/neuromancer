


'''
This script simulates the Koopman model and compares it to a 'ground truth'
swing dynamics trajectory.


'''



import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


import matplotlib.pyplot as plt


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
Specify model to load
'''

save_model_folder = 'saved_models/KO_example_model'


decive = 'cpu'


'''
#####################
Specify data to load
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





#Specify data for simulation


sim_data = Dat_dict[0]


sim_data = fpSplitSequenceDataset(sim_data,nsteps = 5999, psteps = 1, name = 'sim')

sim_data = DataLoader(
    sim_data,
    batch_size=len(sim_data),
    shuffle=False,
    collate_fn=sim_data.collate_fn
)






'''
##############################
### Load the Koopman Dynamics model
###############################
'''



model_path = save_model_folder + '/best_model.pth'
sv_model = torch.load(model_path, pickle_module=dill,
                             map_location=torch.device('cpu'))


estimator = sv_model.components[0]
StO_map = sv_model.components[1]
dynamics_model = sv_model.components[2]




objectives=[]
constraints=[]
components=[estimator,
            StO_map,
            dynamics_model]

sim_model = Problem(objectives, constraints, components)



'''
##############################
### Simulate The Model
###############################
'''



sd = next(iter(sim_data))
output = sim_model(sd)



'''
Plot the Generator frequencies
'''
plt.figure(1)
plt.plot(output['nstep_sim_X_pred_KO_UM'][:,0,0:2].detach().numpy(),'--')
plt.plot(output['nstep_sim_X_true_KO_UM'][:,0,0:2].detach().numpy())
plt.xlabel('time step')
plt.ylabel('frequency (rad/s)')
plt.legend(['KO_w_1','KO_w_2','w_1','w_2'])



'''
Plot the Slack Bus
'''
plt.figure(2)
plt.plot(output['nstep_sim_X_pred_KO_UM'][:,0,3].detach().numpy())
plt.plot(output['nstep_sim_X_true_KO_UM'][:,0,3].detach().numpy())
plt.xlabel('time step')
plt.ylabel('Slack bus (p.u.)')
plt.legend(['KO_s','s'])






