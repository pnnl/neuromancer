"""

For this example we demonstrate learning a model-based control policy for an unknown dynamical system.

**Offline off-policy control learning scenario.**

In a typical real world control setting, due to cost and operational concerns, there is not an opportunity to directly interact with the system to learn a controller. The presented scenario has three stages:

+ Stage 1: the system is perturbed for some amount of time to collect measurements representative of the system state space.
+ Stage 2: Learn a black-box neural ordinary differential equation NODE approximation of an unknown dynamical system given the time series data of system rollouts.
+ Stage 3: Learn neural control policy by differentiating closed-loop dynamical system (neural policy + NODE) using Differentiable predictive control (DPC) method.
In the following cells we walk through the three stage process of generating data, system identification, and control policy learning using neuromancer.


NODE paper: https://arxiv.org/abs/1806.07366
DPC paper: https://www.sciencedirect.com/science/article/pii/S0959152422000981

"""

"""
# # # # # # # # # # # # # # # # # # # # # 
#       Stage 1: data generation        #
# # # # # # # # # # # # # # # # # # # # # 
"""

"""Instantiate a system emulator from neuromancer.psl"""
from neuromancer.psl.nonautonomous import Actuator
from neuromancer.dataset import DictDataset
sys = Actuator()

"""Generate datasets representative of system behavior"""
# obtain time series of the system to be controlled
train_data, dev_data, test_data = [sys.simulate(nsim=1000) for i in range(3)]
# normalize the dataset
train_data, dev_data, test_data = [sys.normalize(d) for d in [train_data, dev_data, test_data]]
sys.show(train_data)

# Set up the data to be in samples of 10 contiguous time steps
# (100 samples with 10 time steps each last dim is dimension of the measured variable)
for d in [train_data, dev_data]:
    d['X'] = d['X'].reshape(100, 10, 3)
    d['U'] = d['U'].reshape(100, 10, 3)
    d['Y'] = d['Y'].reshape(100, 10, 3)
    d['xn'] = d['X'][:, 0:1, :]     # Add an initial condition to start the system loop
    d['Time'] = d['Time'].reshape(100, -1)

# create dataloaders
from torch.utils.data import DataLoader
train_dataset, dev_dataset, = [DictDataset(d, name=n) for d, n in zip([train_data, dev_data], ['train', 'dev'])]
train_loader, dev_loader = [DataLoader(d, batch_size=100, collate_fn=d.collate_fn, shuffle=True) for d in [train_dataset, dev_dataset]]

"""
# # # # # # # # # # # # # # # # # # # # # # # #
#       Stage 2: system identification        #
# # # # # # # # # # # # # # # # # # # # # # # #
"""

""" Define a black-box ODE model to identify the system from data"""
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.dynamics import integrators
import torch

# define neural ODE
dx = blocks.MLP(sys.nx + sys.nu, sys.nx, bias=True, linear_map=torch.nn.Linear, nonlin=torch.nn.ELU,
              hsizes=[20 for h in range(3)])
integrator = integrators.Euler(dx, h=torch.tensor(0.1))
system_nodel = Node(integrator, ['xn', 'U'], ['xn'], name='NODE')
model = System([system_nodel])
model.show()        # visualize computational graph of the NODE system ID model

"""Define the system identification optimization problem"""
from neuromancer.constraint import variable
from neuromancer.problem import Problem
from neuromancer.loss import PenaltyLoss

# Nstep rollout predictions from the model
xpred = variable('xn')[:, :-1, :]
# Ground truth data
xtrue = variable('X')
# define system identification loss function
loss = (xpred == xtrue) ^ 2
loss.update_name('system_id')
# construct differentiable optimization problem in Neuromancer
obj = PenaltyLoss([loss], [])
problem = Problem([model], obj)
problem.show()

"""Solve the system identification problem"""
from neuromancer.trainer import Trainer
import torch.optim as optim

opt = optim.Adam(model.parameters(), 0.001)
trainer = Trainer(problem, train_loader, dev_loader,
                  optimizer=opt,
                  epochs=1000,
                  patience=300,
                  train_metric='train_loss',
                  eval_metric='dev_loss')
best_model = trainer.train()

""" Evaluate the learned NODE system model on 1000 time step rollout"""
import torch
test_data = sys.normalize(sys.simulate(nsim=1000))
print({k: v.shape for k, v in test_data.items()})

test_data['X'] = test_data['X'].reshape(1, *test_data['X'].shape)
test_data['U'] = test_data['U'].reshape(1, *test_data['U'].shape)
test_data['xn'] = test_data['X'][:, 0:1, :]
test_data = {k: torch.tensor(v, dtype=torch.float32) for k, v in test_data.items()}
test_output = model(test_data)

import matplotlib.pyplot as plt
fix, ax = plt.subplots(nrows=3)
for v in range(3):
    ax[v].plot(test_output['xn'][0, :-1, v].detach().numpy(), label='pred')
    ax[v].plot(test_data['X'][0, :, v].detach().numpy(), label='true')
plt.legend()

"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       Stage 3: learning neural control policy         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
"""

""" Create a closed loop system using the system model and a parametrized control policy """
nx, nu = sys.nx, sys.nu
# define control policy
class Policy(torch.nn.Module):

    def __init__(self, insize, outsize):
        super().__init__()
        self.net = blocks.MLP(insize, outsize, bias=True, hsizes=[20, 20, 20])

    def forward(self, x, R):
        features = torch.cat([x, R], dim=-1)
        return self.net(features)

# fix model parameters
system_nodel.requires_grad_(False)

insize = 2*nx
policy = Policy(insize, nu)
policy_node = Node(policy, ['xn', 'R'], ['U'], name='policy')
cl_system = System([policy_node, system_nodel], name='cl_system')
cl_system.show()

""" Sample dataset of control parameters """
# For this simple Actuator system the same dataset can be used for learning a control policy as we used to learn the system model. Here we wish to optimize  controlling the system to some reference trajectory R.
train_dataset = DictDataset({'R': train_data['X'], 'X': train_data['X'], 'xn': train_data['xn']}, name='train')
dev_dataset = DictDataset({'R': dev_data['X'], 'X': train_data['X'], 'xn': dev_data['xn']}, name='dev')
train_loader, dev_loader = [DataLoader(d, batch_size=100, collate_fn=d.collate_fn, shuffle=True)
                            for d in [train_dataset, dev_dataset]]

""" Define objectives and of the optimal control problem """
tru = variable('xn')[:, 1:, :]  # system states
ref = variable('R')             # reference
u = variable('U')               # control action
# reference tracking objective
loss = (ref == tru) ^ 2
loss.update_name('tracking')

# differentiable optimal control problem
obj = PenaltyLoss([loss], [])
problem = Problem([cl_system], obj)
problem.show()

""" Optimize the control policy"""
opt = optim.Adam(policy.parameters(), 0.01)
logout = ['loss']
trainer = Trainer(problem, train_loader, dev_loader,
                  optimizer=opt,
                  epochs=1000,
                  patience=1000,
                  train_metric='train_loss',
                  eval_metric='dev_loss')

best_model = trainer.train()
trainer.model.load_state_dict(best_model)

""" Evaluate the learned control policy on the true system """
# With the optional pytorch backend for the original ODE system
# we can now swap out our learned model to evaluate the learned control policy
# on the original system.
sys.change_backend('torch')

# We will have to denormalize the policy actions according to the system stats
# Conversely we will have to normalize the system states according to the system stats
# to hand to the policy
def norm(x):
    return sys.normalize(x, key='X')

def denorm(u):
    return sys.denormalize(u, key='U')

normnode = Node(norm, ['xsys'], ['xn'], name='norm')
denormnode = Node(denorm, ['U'], ['u'], name='denorm')
sysnode = Node(sys, ['xsys', 'u'], ['xsys'], name='actuator')
test_system = System([normnode, policy_node, denormnode, sysnode])
test_system.show()

""" Evaluate on 1000 steps with a new reference trajectory distribution """
from neuromancer.psl.signals import sines, step, arma, spline
import numpy as np

# generate random sequence of step changes
references = step(nsim=1000, d=sys.nx, min=sys.stats['X']['min'], max=sys.stats['X']['max'])
test_data = {'R': torch.tensor(sys.normalize(references, key='X'), dtype=torch.float32).unsqueeze(0), 'xsys': sys.get_x0().reshape(1, 1, -1),
            'Time': (np.arange(1000)*sys.ts).reshape(1, 1000, 1)}
print({k: v.shape for k, v in test_data.items()})
test_system.nsteps=1000
with torch.no_grad():
    test_out = test_system(test_data)

print({k: v.shape for k, v in test_out.items()})
fix, ax = plt.subplots(nrows=3)
for v in range(3):
    ax[v].plot(test_data['R'][0, :, v].detach().numpy(), 'r--', label='reference')
    ax[v].plot(test_out['xn'][0, 1:, v].detach().numpy(), label='state')
plt.legend()
plt.savefig('control.png')





