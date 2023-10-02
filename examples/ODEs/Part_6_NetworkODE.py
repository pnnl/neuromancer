# %% Numpy + plotting utilities + ordered dicts
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


# Standard PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Neuromancer imports
from neuromancer.psl.coupled_systems import *
from neuromancer.dynamics import integrators, ode, physics
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.problem import Problem
from neuromancer.loss import PenaltyLoss
from neuromancer.system import Node, System
from neuromancer.loggers import BasicLogger
from neuromancer.trainer import Trainer

# Fix seeds for reproducibility
np.random.seed(200)
torch.manual_seed(0)

# Define Network and datasets
adj = np.array([[0,1],[0,2],[0,3],[1,0],[1,3],[1,4],[2,0],[2,3],[3,0],[3,1],[3,2],[3,4],[4,1],[4,3]]).T
s = RC_Network(nx=5, adj=adj)
nsim = 500
sim, s_dev, s_test = [s.simulate(nsim=nsim) for _ in range(3)]

plt.figure()
plt.plot(sim['Time'],sim['X'])
plt.plot(sim['Time'],sim['U'])

nstep = 10

train_data = {'Y': sim['Y'], 'X': sim['X'], 'U': sim['U']}
dev_data = {'Y': s_dev['Y'], 'X': s_dev['X'], 'U': s_dev['U']}
test_data = {'Y': s_test['Y'], 'X': s_test['X'], 'U': s_test['U']}
for d in [train_data, dev_data]:
    d['X'] = d['X'].reshape(nsim//nstep, nstep, 5)
    d['Y'] = d['Y'].reshape(nsim//nstep, nstep, 5)
    d['U'] = d['U'].reshape(nsim//nstep, nstep, 6)
    d['xn'] = d['X'][:, 0:1, :]

train_dataset, dev_dataset, = [DictDataset(d, name=n) for d, n in zip([train_data, dev_data], ['train', 'dev'])]
train_loader, dev_loader, test_loader = [DataLoader(d, batch_size=nsim//nstep, collate_fn=d.collate_fn, shuffle=True) for d in [train_dataset, dev_dataset, dev_dataset]]


# Define the states
states = {}
states['T_1'] = 0
states['T_2'] = 1
states['T_3'] = 2
states['T_4'] = 3
states['T_5'] = 4
states['T_6'] = 5
states['T_7'] = 6
states['T_8'] = 7
states['T_9'] = 8
states['T_10'] = 9
states['T_11'] = 10

# Model construction
keys = list(states.keys())
zones = [physics.RCNode(in_keys=[keys[i]], state_keys=[keys[i]], 
                        C=nn.Parameter(torch.tensor(5.0)),scaling=1.0e-5) for i in range(5)]

heaters = [physics.SourceSink(state_keys=[keys[i+len(zones)]], in_keys=[keys[i+len(zones)]]) for i in range(5)] # define heaters

outside = [physics.SourceSink(state_keys=[keys[-1]], in_keys=[keys[-1]])]

# join lists:
agents = zones + heaters + outside

# Helper function for constructing couplings based on desired edge physics and an edge list:
def generate_deltaTemp_edges(physics,edge_list,agents):
    """
    Quick helper function to construct edge physics/objects from adj. list:
    """
    couplings = []
    for edge in edge_list:
        agent = physics(in_keys=[*agents[edge[1]].in_keys,*agents[edge[0]].in_keys],R=nn.Parameter(torch.tensor(50.0)),pins=[edge])
        couplings.append(agent)

    return couplings

couplings = generate_deltaTemp_edges(physics.DeltaTemp,list(adj.T),agents)    # Heterogeneous edges of same physics

# What do we have so far?
print(len(couplings))
# Let's take a look at one:
print(couplings[0])
# What's it connecting?
print(couplings[0].pins)

# Couple w/ outside temp:
outside_list = [[0,5],[1,5],[2,5],[3,5],[4,5]]
out_couplings = generate_deltaTemp_edges(physics.DeltaTemp,outside_list,agents)

# Couple w/ individual sources:
source_list = [[0,6],[1,7],[2,8],[3,9],[4,10]]
source_couplings = generate_deltaTemp_edges(physics.DeltaTemp,source_list,agents)

couplings += out_couplings + source_couplings

# Model ODE RHS instantiation
model_ode = ode.GeneralNetworkedODE(
    states=states,
    agents = agents,
    couplings = couplings,
    insize = s.nx+s.nu,
    outsize = s.nx)

# Integrator instantiation
fx_int = integrators.RK4(model_ode, h=1.0)

dynamics_model = System([Node(fx_int,['xn','U'],['xn'])])

x = variable("X")
xhat = variable("xn")[:, :-1, :]

reference_loss = ((xhat == x)^2)
reference_loss.name = "ref_loss"

objectives = [reference_loss]
constraints = []

# create constrained optimization loss
loss = PenaltyLoss(objectives, constraints)

# construct constrained optimization problem
problem = Problem([dynamics_model], loss)

optimizer = torch.optim.Adam(problem.parameters(), lr=0.01)
logger = BasicLogger(args=None, savedir='test', verbosity=1,
                     stdout=["dev_loss","train_loss"])

trainer = Trainer(
    problem,
    train_loader,
    dev_loader,
    test_data,
    optimizer,
    epochs=1000,
    patience=20,
    warmup=5,
    eval_metric="dev_loss",
    train_metric="train_loss",
    dev_metric="dev_loss",
    test_metric="dev_loss",
    logger=logger,
)

best_model = trainer.train()

u = torch.from_numpy(s_test['U']).float()
sol = torch.zeros((500,5))
ic = torch.tensor(s_test['X'][0,:5])
for j in range(sol.shape[0]-1):
    if j==0:
        sol[[0],:] = ic.float()
        sol[[j+1],:] = fx_int(sol[[j],:],u[[j],:])
    else:
        sol[[j+1],:] = fx_int(sol[[j],:],u[[j],:])

plt.figure()
plt.plot(sol.detach().numpy(),label='model', color = 'black')
plt.plot(s_test['X'][:,:5],label = 'data', color = 'red')
# %%
