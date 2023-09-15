# Numpy + plotting utilities + ordered dicts
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


# Standard PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Neuromancer imports
from neuromancer.psl.coupled_systems import *
from neuromancer.dynamics import integrators, ode, physics, interpolation
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

zones = [physics.RCNode(C=nn.Parameter(torch.tensor(5.0)),scaling=1.0e-5) for i in range(5)]  # heterogeneous population w/ identical physics

heaters = [physics.SourceSink() for i in range(5)] # define heaters

outside = [physics.SourceSink()]

# join lists:
agents = zones + heaters + outside

map = physics.map_from_agents(agents)
# Let's take a look at this 'map':
print(map)

# Helper function for constructing couplings based on desired edge physics and an edge list:
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

couplings = generate_parameterized_edges(physics.DeltaTemp,list(adj.T))    # Heterogeneous edges of same physics

# What do we have so far?
print(len(couplings))
# Let's take a look at one:
print(couplings[0])
# What's it connecting?
print(couplings[0].pins)

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

model_ode = ode.GeneralNetworkedODE(
    map = map,
    agents = agents,
    couplings = couplings,
    insize = s.nx+s.nu,
    outsize = s.nx,
    inductive_bias="compositional")

fx_int = integrators.RK2(model_ode, h=1.0)

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