# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy import interpolate
from neuromancer.psl.coupled_systems import *
from neuromancer.dynamics import integrators, ode, physics, interpolation
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.problem import Problem
from neuromancer.loss import PenaltyLoss
from neuromancer.system import Node, System
from collections import OrderedDict
from neuromancer.loggers import BasicLogger
from neuromancer.trainer import Trainer
np.random.seed(200)
torch.manual_seed(0)
device = 'cpu'

plt.rcParams["font.family"] = "serif"
#plt.rcParams["font.serif"] = ["Times"]
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 10})


params = {'legend.fontsize': 10,
         'axes.labelsize': 10,
         'axes.titlesize': 10,
         'xtick.labelsize': 10,
         'ytick.labelsize': 10}
plt.rcParams.update(params)

# %%
adj = np.array([[0,1],[0,2],[0,3],[1,0],[1,3],[1,4],[2,0],[2,3],[3,0],[3,1],[3,2],[3,4],[4,1],[4,3]]).T
s = RC_Network(nx=5, adj=adj)
nsteps = 10
nsim = 500
nx = 5
nu = 6

sim = s.simulate(nsim=nsim)

# %%
# Prepare the generated data from the previous cell for Training...
nstep = 10 #int(interp_sol.shape[0]*0.01) 

train_data = {'Y': sim['Y'], 'X': sim['X'], 'U': sim['U']}
dev_data = train_data
test_data = train_data
for d in [train_data, dev_data]:
    d['X'] = d['X'].reshape(nsim//nstep, nstep, 5)
    d['Y'] = d['Y'].reshape(nsim//nstep, nstep, 5)
    d['U'] = d['U'].reshape(nsim//nstep, nstep, 5 + 1)
    d['xn'] = d['X'][:, 0:1, :]

train_dataset, dev_dataset, = [DictDataset(d, name=n) for d, n in zip([train_data, dev_data], ['train', 'dev'])]
train_loader, dev_loader, test_loader = [DataLoader(d, batch_size=nsim//nstep, collate_fn=d.collate_fn, shuffle=True) for d in [train_dataset, dev_dataset, dev_dataset]]


# %% Helper function:
def generate_parameterized_edges(physics,edge_list):
    """
    Quick helper function to construct edge physics/objects from adj. list:
    """

    couplings = []
    if isinstance(physics,nn.Module): # is "physics" an instance or a class?
        # If we're in here, we expect one instance of "physics" for all edges in edge_list (homogeneous edges)
        physics.pins = edge_list
        couplings.append(physics)
        print(f'Broadcasting {physics} to all elements in edge list.')
    else: 
        # If we're in here, we expect different "physics" for each edge in edge_list (heterogeneous edges)
        for edge in edge_list:
            agent = physics(R=nn.Parameter(torch.tensor(50.0)),pins=[edge])
            couplings.append(agent)

        print(f'Assuming new {physics} for each element in edge list.')

    return couplings

# %%
adjacency = list(adj.T)                                                      # Interaction physics

# Define zones:
zones = [physics.RCNode(C=nn.Parameter(torch.tensor(5.0)),scaling=1.0e-5) for i in range(5)]  # heterogeneous population w/ identical physics

# Define heaters:
heaters = [physics.SourceSink() for i in range(5)]

# Define 'outside':
outside = [physics.SourceSink()]  

# Join lists:
agents = zones + heaters + outside

map = physics.map_from_agents(agents)                                           # Construct state mappings

# Define inter-node couplings:
couplings = generate_parameterized_edges(physics.DeltaTemp,adjacency)                   # Heterogeneous edges of same physics

# Couple w/ outside temp:
couplings.append(physics.DeltaTemp(R=nn.Parameter(torch.tensor(50.0)),pins=[[0,5]]))
couplings.append(physics.DeltaTemp(R=nn.Parameter(torch.tensor(50.0)),pins=[[1,5]]))
couplings.append(physics.DeltaTemp(R=nn.Parameter(torch.tensor(50.0)),pins=[[2,5]]))
couplings.append(physics.DeltaTemp(R=nn.Parameter(torch.tensor(50.0)),pins=[[3,5]]))
couplings.append(physics.DeltaTemp(R=nn.Parameter(torch.tensor(50.0)),pins=[[4,5]]))

# Couple w/ individual sources:
couplings.append(physics.DeltaTemp(R=nn.Parameter(torch.tensor(50.0)),pins=[[0,6]]))
couplings.append(physics.DeltaTemp(R=nn.Parameter(torch.tensor(50.0)),pins=[[1,7]]))
couplings.append(physics.DeltaTemp(R=nn.Parameter(torch.tensor(50.0)),pins=[[2,8]]))
couplings.append(physics.DeltaTemp(R=nn.Parameter(torch.tensor(50.0)),pins=[[3,9]]))
couplings.append(physics.DeltaTemp(R=nn.Parameter(torch.tensor(50.0)),pins=[[4,10]]))

# Define ODE system:
model_ode = ode.GeneralNetworkedODE(
    map = map,
    agents = agents,
    couplings = couplings,
    insize = nx+nu,
    outsize = nx,
    inductive_bias="compositional")

# %%
fx_int = integrators.Euler(model_ode, interp_u = lambda tq, t, u: u, h=1.0)

dynamics_model = System([Node(fx_int,['xn','U'],['xn'])])

# %%
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
    epochs=2000,
    patience=20,
    warmup=5,
    eval_metric="dev_loss",
    train_metric="train_loss",
    dev_metric="dev_loss",
    test_metric="dev_loss",
    logger=logger,
)

# %%
best_model = trainer.train()

# %%
u = torch.from_numpy(sim['U']).to(device).float()
sol = torch.zeros((500,5))
ic = torch.tensor(sim['X'][0,:5])
for j in range(sol.shape[0]-1):
    if j==0:
        sol[[0],:] = ic.float()
        sol[[j+1],:] = fx_int(sol[[j],:],u[[j],:])
    else:
        sol[[j+1],:] = fx_int(sol[[j],:],u[[j],:])

# %%
plt.plot(sol.detach().numpy(),label='model', color = 'black')
plt.plot(sim['X'][:,:5],label = 'data', color = 'red')
# %%
