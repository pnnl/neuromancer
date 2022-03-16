

'''
This script computes a DED-DPC control policy and compares it to the optimal
solution.

'''



# pytorch imports
import torch
import torch.nn as nn

# ecosystem imports
import slim

# local imports
import neuromancer.blocks as blocks
from neuromancer.dynamics import BlockSSM


import pandas as pd

import matplotlib.pyplot as plt

import torch.nn.functional as F

import psl

import neuromancer as nm

from neuromancer import blocks, estimators, dynamics, arg
from neuromancer.activations import activations

from neuromancer.trainer import Trainer
from neuromancer.problem import Problem, Objective

from neuromancer.loggers import BasicLogger, MLFlowLogger


import numpy as np


import neuromancer.blocks as blocks
from neuromancer.component import Component

import dill




from torch.utils.data import Dataset, DataLoader



from neuromancer.dataset import read_file, normalize_data, split_sequence_data, SequenceDataset

from neuromancer.DED import fpSplitSequenceDataset




'''
#####################
Load in a saved model
'''

save_root = '../DED_DPC/saved_models/'
save_folder = 'example_model'



'''
#####################
Load in a dataset
'''

data_root = '../Data/Example_Data/'
meta_data_file = 'opt_metadata.csv'



met_data = pd.read_csv(data_root+ meta_data_file)

n_files = met_data.shape[0]
file_idx = np.arange(1,n_files+1)



Dat_dict = []
for i in file_idx:
    path = data_root + f'trial_data/trial_{i}.csv'
    dat = read_file(path)
    Dat_dict.append(dat)

met_data = pd.read_csv(data_root+ meta_data_file)

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
    
    
    jj=file_idx[j] - 1
    c = met_data[['cs','c1','c2']].loc[jj].values
    c = np.reshape(c,(1,3))
    cvec = np.tile(c,(6001,1))
    Dat_dict[j]['coefs'] = cvec




model_path = save_root + save_folder + '/best_model.pth'
sv_model = torch.load(model_path, pickle_module=dill,
                             map_location=torch.device('cpu'))


estimator = sv_model.components[0]
StO_map = sv_model.components[1]
policy = sv_model.components[2]
dynamics_model = sv_model.components[3]


device = 'cpu'

objectives=[]
constraints=[]
components=[estimator,
            StO_map,
            policy,
            dynamics_model]

sim_model = Problem(objectives, constraints, components)





## sim data
sim_data = Dat_dict[0]

nsteps = 5999
sim_data = fpSplitSequenceDataset(sim_data,nsteps = 5999, psteps = 1, name = 'sim')
sim_data = DataLoader(
    sim_data,
    batch_size=1,
    shuffle=False,
    collate_fn=sim_data.collate_fn
)






sd = next(iter(sim_data))
output = sim_model(sd)





'''
Plot the Forecasted Loads
'''
plt.figure(1)
plt.plot(-output['nstep_sim_Df'][:,0,:].detach().numpy())
plt.xlabel('time step')
plt.ylabel('|P_i| (p.u.)')



'''
Plot the Generator Inputs
'''
plt.figure(2)
plt.plot(output['nstep_sim_U_pred_policy'][:,0,:].detach().numpy())
plt.plot(output['nstep_sim_Uf'][:,0,:].detach().numpy())
plt.xlabel('time step')
plt.ylabel('Generator Input (p.u.)')
plt.legend(['DPC_P_1','DPC_P_2','P_1','P_2'])




'''
Plot the Generator frequencies
'''
plt.figure(3)
plt.plot(output['nstep_sim_X_pred_KO_UM'][:,0,0:2].detach().numpy())
plt.xlabel('time step')
plt.ylabel('frequency (rad/s)')




