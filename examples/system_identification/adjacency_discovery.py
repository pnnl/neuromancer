# %%
"""
Reproduction of Chaos paper key result.
"""
import torch
import torch.nn as nn
from neuromancer.psl import plot
from neuromancer import psl
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader
import os

from neuromancer.system import Node, System
from neuromancer.dynamics import integrators, ode
from neuromancer.dynamics.ode import ode_param_systems_auto as systems
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.loggers import BasicLogger
from neuromancer.dataset import DictDataset
from neuromancer.constraint import Loss, variable
from neuromancer.loss import PenaltyLoss
from neuromancer.psl.base import ODE_Autonomous, cast_backend
import neuromancer.slim as slim
from neuromancer.modules import blocks, activations
import numpy as np

# %% Define our system for data generation:
class NetworkedOscillator(ODE_Autonomous):

    @property
    def params(self):
        self.N = 4
        self.nx = 2*self.N
        self.mu = 0.2
        self.x0 = np.random.rand(self.nx)
        self.A = np.round(np.random.rand(self.N,self.N))*(np.ones((self.N,self.N))-np.eye(self.N))
        variables = {'x0': self.x0}
        constants = {'ts': 0.1, 'N': self.N}
        parameters = {'A': self.A}
        meta = {}        

        return variables, constants, parameters, meta

    @cast_backend 
    def equations(self, t, x):
        # Derivatives
        dx = np.zeros(self.nx)
        for i in range(self.N):
            # odd numbers are velocity. even numbers are for position.
            dx[2*i] = x[2*i+1]          # d(position)/dt = velocity
            f = 11.0*x[2*i]**4 - 11.0*x[2*i] + 1 # really funky roots; i.e. multiple stable f.p. w/in parameter range of interest
            # d(velocity)/dt = (energy loss) + (energy gain) + (interactions)
            dx[2*i+1] += -x[2*i] - self.mu*(f*x[2*i+1])      # d(velocity)/dt = acceleration function
            for j in range(self.N):
                dx[2*i+1] += self.A[i,j]*(x[2*i+1] - x[2*j+1]) # self.A = self.A*self.mu has been implemented above
        return dx

def get_data(sys, nsim, nsteps, bs, ts):
    """
    :param nsteps: (int) Number of timesteps for each batch of training data
    :param sys: (psl.ODE_NonAutonomous)
    :param normalize: (bool) Whether to normalize the data

    """
    train_sim, dev_sim, test_sim = [sys.simulate(nsim=nsim, ts=ts) for i in range(3)]
    nx = sys.nx
    nbatch = nsim//nsteps
    length = (nsim//nsteps) * nsteps

    trainX = train_sim['X'][:length].reshape(nbatch, nsteps, nx)
    trainX = torch.tensor(trainX, dtype=torch.float32)
    train_data = DictDataset({'X': trainX, 'xn': trainX[:, 0:1, :]}, name='train')
    train_loader = DataLoader(train_data, batch_size=bs,
                              collate_fn=train_data.collate_fn, shuffle=True)

    devX = dev_sim['X'][:length].reshape(nbatch, nsteps, nx)
    devX = torch.tensor(devX, dtype=torch.float32)
    dev_data = DictDataset({'X': devX, 'xn': devX[:, 0:1, :]}, name='dev')
    dev_loader = DataLoader(dev_data, batch_size=bs,
                            collate_fn=dev_data.collate_fn, shuffle=True)

    testX = test_sim['X'][:length].reshape(1, nsim, nx)
    testX = torch.tensor(testX, dtype=torch.float32)
    test_data = {'X': testX, 'xn': testX[:, 0:1, :]}

    return train_loader, dev_loader, test_data

# %% Test and visualize:
nsim = 200
ts = 0.01
s = NetworkedOscillator()
train_loader, dev_loader, test_data = get_data(s, nsim = nsim, nsteps = 10, bs = 10, ts = ts)

# %% Custom hybrid ODE:
class NeuralNetworkedOscillators(ode.ODESystem):
    def __init__(self, node_block, coupling_block, insize, outsize):
        super().__init__(insize=insize, outsize=outsize)
        self.node_phys = node_block
        self.coupling_phys= coupling_block
        self.nx = insize
        self.N = self.nx//2
        self.sparse_map = slim.maps['linear'](self.N, self.N, bias=False)
        self.mu = torch.nn.Parameter(torch.tensor(0.2, requires_grad=True))
        torch.nn.init.zeros_(self.sparse_map.weight)
        self.A = torch.ones(self.N,self.N)

    def ode_equations(self, x):
        # Compute state derivatives
        self.A = self.mu*torch.sigmoid(self.sparse_map.effective_W() - torch.eye(self.N)*1e6)
        dx = torch.zeros_like(x)
        for i in range(self.N):
        # odd numbers are velocities. even numbers are positions.
            dx[:, 2*i] = x[:, 2*i+1] 
            dx[:, [2*i+1]] += self.node_phys(x[:,2*i:2*i+2])
            for j in range(self.N):
                dx[:, [2*i+1]] += self.A[i,j]*self.coupling_phys(torch.cat((x[:, 2*i:2*i+2], x[:, 2*j:2*j+2]),dim=1))
        return dx

    def reg_error(self):
        return torch.norm(self.A, 1)

# %%
node_physics = blocks.MLP(2, 1, linear_map=slim.maps['linear'],
                nonlin=nn.LeakyReLU,
                hsizes=[50,50])
coupling_physics = blocks.MLP(4, 1, linear_map=slim.maps['linear'],
                    nonlin=nn.LeakyReLU,
                    hsizes=[4]) 
ode_rhs = NeuralNetworkedOscillators(node_physics, coupling_physics, insize = s.nx, outsize = s.nx)
integrator = integrators.RK2(ode_rhs, h=ts)
dynamics_model = System([Node(integrator, ['xn'], ['xn'])], name = "dynamics")

# %% Constraints + losses:
x = variable("X")
xhat = variable('xn')[:, :-1, :]

reference_loss = ((xhat == x)^2)
reference_loss.name = "ref_loss"

xFD = (x[:, 1:, :] - x[:, :-1, :])
xhatFD = (xhat[:, 1:, :] - xhat[:, :-1, :])

fd_loss = ((xFD == xhatFD)^2)
fd_loss.name = 'FD_loss'

sparsity_loss = Loss(
        ["X"],
        lambda x: ode_rhs.reg_error(),
        weight=0.0,
        name="reg_error",
    )

# %%
objectives = [reference_loss,fd_loss,sparsity_loss]
constraints = []
# create constrained optimization loss
loss = PenaltyLoss(objectives, constraints)
# construct constrained optimization problem
problem = Problem([dynamics_model], loss)
# plot computational graph

# %%
optimizer = torch.optim.Adam(problem.parameters(), lr=0.005)
logger = BasicLogger(args=None, savedir='test', verbosity=1,
                        stdout=['dev_loss', 'train_loss'])

trainer = Trainer(
    problem,
    train_loader,
    dev_loader,
    test_data,
    optimizer,
    patience=50,
    warmup=10,
    epochs=3000,
    eval_metric="dev_loss",
    train_metric="train_loss",
    dev_metric="dev_loss",
    test_metric="dev_loss",
    logger=logger,
)

# %%
best_model = trainer.train()

# %% Adjacency post-training:
plt.pcolormesh(np.fliplr(ode_rhs.A.detach().numpy()),clim=(0,1))
plt.show()
plt.pcolormesh(np.round(np.fliplr(ode_rhs.A.detach().numpy())),clim=(0,1))
plt.show()
plt.pcolormesh(np.fliplr(s.A))                      
# %% Roll out from random IC for some time:
sol = torch.zeros((nsim*3,s.nx))
ic = torch.rand((1,s.nx))
for j in range(sol.shape[0]-1):
    if j==0:
        sol[[0],:] = ic.float()
        sol[[j+1],:] = integrator(sol[[j],:])
    else:
        sol[[j+1],:] = integrator(sol[[j],:])

plt.plot(sol[:,0].detach().numpy())
plt.plot(sol[:,1].detach().numpy())
# %%
