# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

import neuromancer.slim as slim
from neuromancer.modules import blocks, activations
from neuromancer.dynamics import integrators, ode, physics
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.dataset import DictDataset
from neuromancer.loss import PenaltyLoss
from neuromancer.constraint import variable, Objective
from neuromancer.system import Node, System
from neuromancer.loggers import BasicLogger

from collections import OrderedDict
from abc import ABC, abstractmethod

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

data = np.float32(np.loadtxt('data/tanks.dat'))
data=data[1:497,]
area_data = np.loadtxt('data/area.dat')
time = np.float32(np.linspace(0.0,len(data[:,0])-1,len(data[:,0])).reshape(-1, 1))
U = time*0.0 + 0.5

train_data = {'Y': data[1:], 'X': data[1:], 'Time': time[1:], 'U': U[1:] }
dev_data = train_data
test_data = train_data

nsim = data.shape[0]
nx = data.shape[1]
nstep = 15

for d in [train_data, dev_data]:
    d['X'] = d['X'].reshape(nsim//nstep, nstep, nx)
    d['Y'] = d['Y'].reshape(nsim//nstep, nstep, nx)
    d['xn'] = d['X'][:, 0:1, :] # Add an initial condition to start the system loop
    d['Time'] = d['Time'].reshape(nsim//nstep, nstep, 1)
    d['U'] = d['U'].reshape(nsim//nstep, nstep, 1)

train_dataset, dev_dataset, = [DictDataset(d, name=n) for d, n in zip([train_data, dev_data], ['train', 'dev'])]
train_loader, dev_loader, test_loader = [DataLoader(d, batch_size=nsim//nstep, collate_fn=d.collate_fn, shuffle=True) for d in [train_dataset, dev_dataset, dev_dataset]]

nx = 4 # set the state dimension
nu = 1 # set the exogenous input dimension

# State names if we need them (we do)
states = {}
states['h_1'] = 0
states['h_2'] = 1
states['m_1'] = 2
states['m_2'] = 3
states['m'] = 4

plt.plot(time,data[:,0])
plt.xlim([0,500])
plt.ylim([0,40])
plt.xlabel("Time")
plt.ylabel("Height")
plt.show()

plt.plot(time,data[:,[2,3]])
plt.plot(time,data[:,2]+data[:,3])
plt.xlim([0,500])
plt.ylim([0,0.6])
plt.xlabel("Time")
plt.ylabel("Volumetric Flow")
plt.show()

############### Black-box Neural ODE Model ###############
#  define neural network of the NODE
fx = blocks.MLP(nx+nu, nx, bias=True,
                 linear_map=torch.nn.Linear,
                 nonlin=torch.nn.ReLU,
                 hsizes=[10, 10])

fxRK4 = integrators.RK4(fx, h=1.0)

dynamics_model = System([Node(fxRK4,['xn','U'],['xn'])])

x = variable("X")
xhat = variable("xn")[:, :-1, :]
reference_loss = ((xhat[:,:,[2,3]] == x[:,:,[2,3]])^2)
reference_loss.name = "ref_loss"

height_loss = (1.0e0*(xhat[:,:,0] == xhat[:,:,1])^2)
height_loss.name = "height_loss"

objectives = [reference_loss, height_loss]
constraints = []
# create constrained optimization loss
loss = PenaltyLoss(objectives, constraints)
# construct constrained optimization problem
problem = Problem([dynamics_model], loss)

optimizer = torch.optim.Adam(problem.parameters(), lr=0.01)

trainer = Trainer(
    problem,
    train_loader,
    dev_loader,
    test_loader,
    optimizer,
    epochs=10000,
    patience=20,
    warmup=50,
    eval_metric="dev_loss",
    train_metric="train_loss",
    dev_metric="dev_loss",
    test_metric="dev_loss",
    logger=None,
)

best_model = trainer.train()

def process_and_plot(integrator):

    # Roll out the model:
    end_step = len(data[:,0])
    sol = torch.zeros((end_step,5))
    sol[:,-1] = 0.5
    x0 = np.concatenate((data[0,:],U[0]))
    ic = torch.unsqueeze(torch.tensor(x0),0).float()
    t = 0
    for j in range(sol.shape[0]-1):
        if j==0:
            sol[[0],:] = ic
            sol[[j+1],:4] = integrator(sol[[0],:4],sol[[0],-1:])
        else:
            sol[[j+1],:4] = integrator(sol[[j],:4],sol[[j],-1:])
        t += time[1]-time[0]

    # plot the results
    plt.plot(time,sol.detach().numpy()[:,0],label="Tank #1")
    plt.plot(time,sol.detach().numpy()[:,1],label="Tank #2")
    plt.plot(time,data[:,0],label="Data",linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Height")
    plt.legend()
    plt.show()

    plt.plot(time,sol.detach().numpy()[:,2],label="Inflow #1")
    plt.plot(time,sol.detach().numpy()[:,3],label="Inflow #1")
    plt.plot(time,np.sum(sol.detach().numpy()[:,[2,3]],-1),label="In_1 + In_2")
    plt.plot(time,data[:,2],label="Data Inflow #1",linestyle="--")
    plt.plot(time,data[:,3],label="Data Inflow #2",linestyle="--")

    plt.xlim([0,500])
    plt.ylim([0,0.6])
    plt.xlabel("Time")
    plt.ylabel("Volumetric Flow")
    plt.legend()
    plt.show()

process_and_plot(fxRK4)

############### Black-box Neural DAE Model ############### 

# Class for 'black-box' differential state evolution
class BBNodeDiff(physics.Agent):
    def __init__(self, state_keys = None, in_keys = None, solver = None, profile = None):
        super().__init__(state_keys=state_keys)
        self.solver = solver
        self.in_keys = in_keys
        self.profile = profile

    def intrinsic(self, x, y):
        return self.profile(x)

    def algebra(self, x):
        return x[:,:len(self.state_keys)]

# Class for 'black-box' algebraic state evolution
class BBNodeAlgebra(physics.Agent):
    def __init__(self, state_keys = None, in_keys = None, solver = None, profile = None):
        super().__init__(state_keys=state_keys)
        self.solver = solver
        self.in_keys = in_keys
        self.profile = profile

    def intrinsic(self, x, y):
        return torch.zeros_like(x[:,:len(self.state_keys)])

    def algebra(self, x):
        # Learning the convex combination of stream outputs that equal the input
        param = torch.abs(self.solver(x[:,1:]))
        return torch.cat((x[:,[0]]*param,x[:,[0]]*(1.0 - param)),-1)

ode_rhs = blocks.MLP(insize=4, outsize=2, hsizes=[5],
                            linear_map=slim.maps['linear'],
                            nonlin=nn.LeakyReLU)

algebra_solver_bb = blocks.MLP(insize=4, outsize=1, hsizes=[5],
                            linear_map=slim.maps['linear'],
                            nonlin=nn.LeakyReLU)

# Define differential agent:
diff = BBNodeDiff(in_keys=["h_1","h_2","m_1","m_2"], state_keys=["h_1","h_2"], profile=ode_rhs)

# Define algebraic agent:
alg = BBNodeAlgebra(in_keys = ["m","h_1","h_2","m_1","m_2"], state_keys=["m_1","m_2"], solver=algebra_solver_bb)

agents = [diff, alg]

couplings = []

model_ode = ode.GeneralNetworkedODE(
    states=states,
    agents=agents,
    couplings=couplings,
    insize=nx+nu,
    outsize=nx,
)

model_algebra = ode.GeneralNetworkedAE(
    states=states,
    agents=agents,
    insize=nx+nu,
    outsize=nx ,
)

fx_int = integrators.EulerDAE(model_ode,algebra=model_algebra,h=1.0)
dynamics_model = System([Node(fx_int,['xn','U'],['xn'])])

# construct constrained optimization problem
problem = Problem([dynamics_model], loss)
optimizer = torch.optim.Adam(problem.parameters(), lr=0.005)

trainer = Trainer(
    problem,
    train_loader,
    dev_loader,
    test_loader,
    optimizer,
    epochs=10000,
    patience=50,
    warmup=50,
    eval_metric="dev_loss",
    train_metric="train_loss",
    dev_metric="dev_loss",
    test_metric="dev_loss",
    logger=None,
)

best_model = trainer.train()
process_and_plot(fx_int)

############### Gray-box DAE Model ###############

# Tank area - height profiles: These should map height to area. R^1 -> R^1.
tank_profile = blocks.MLP(insize=1, outsize=1, hsizes=[3],
                            linear_map=slim.maps['linear'],
                            nonlin=nn.Sigmoid)

# Surrogate for algebra solver: This should map 'algebraic state indices' to len(state names). 
algebra_solver = blocks.MLP(insize=4, outsize=1, hsizes=[3],
                            linear_map=slim.maps['linear'],
                            nonlin=nn.Sigmoid)
                            
# Individual components:
tank_1 = physics.MIMOTank(state_keys=["h_1"], in_keys=["h_1"], profile= lambda x: 3.0) # assume known area-height profile
tank_2 = physics.MIMOTank(state_keys=["h_2"], in_keys=["h_2"], profile=tank_profile)
pump = physics.SourceSink(state_keys=["m"], in_keys=["m"])

# Define algebraic agent:
manifold = physics.SIMOConservationNode(in_keys = ["m","h_1","h_2","m_1","m_2"], state_keys=["m_1","m_2"], solver=algebra_solver)

# Accumulate agents in list:
# index:   0       1        2       3 
agents = [pump, tank_1, tank_2, manifold]

couplings = []
# Couple w/ pipes:
couplings.append(physics.Pipe(in_keys = ["m"], pins = [[0,3]])) # Pump -> Manifold
couplings.append(physics.Pipe(in_keys = ["m_1"], pins = [[3,1]])) # Manifold -> tank_1
couplings.append(physics.Pipe(in_keys = ["m_2"], pins = [[3,2]])) # Manifold -> tank_2

model_ode = ode.GeneralNetworkedODE(
    states=states,
    agents=agents,
    couplings=couplings,
    insize=nx+nu,
    outsize=nx,
)

model_algebra = ode.GeneralNetworkedAE(
    states=states,
    agents=agents,
    insize=nx+nu,
    outsize=nx,
)

fx_dae = integrators.EulerDAE(model_ode,algebra=model_algebra,h=1.0)
dynamics_model = System([Node(fx_dae,['xn','U'],['xn'])])

# construct constrained optimization problem
problem = Problem([dynamics_model], loss)
optimizer = torch.optim.Adam(problem.parameters(), lr=0.005)

trainer = Trainer(
    problem,
    train_loader,
    dev_loader,
    test_loader,
    optimizer,
    epochs=10000,
    patience=50,
    warmup=50,
    eval_metric="dev_loss",
    train_metric="train_loss",
    dev_metric="dev_loss",
    test_metric="dev_loss",
    logger=None,
)

best_model = trainer.train()
process_and_plot(fx_dae)

# %%
