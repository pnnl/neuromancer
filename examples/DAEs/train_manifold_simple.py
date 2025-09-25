"""
Implementation of the Tank-Manifold Property Inference Example from the paper:

Learning Neural Differential Algebraic Equations via Operator Splitting
James Koch, Madelyn Shapiro, Himanshu Sharma, Draguna Vrabie, and Jan Drgona
https://arxiv.org/abs/2403.12938

Neural Differential Algebraic Equations (Neural DAEs) are an extension of the canonical neural timestepper
for systems with algebraic constraints. Inspired by fractional-step methods,
this work leverages sequential sub-tasks to provide updates for algebraic states and differential states.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import requests

import neuromancer.slim as slim
from neuromancer.modules import blocks
from neuromancer.dynamics import integrators, ode
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.dataset import DictDataset
from neuromancer.loss import PenaltyLoss
from neuromancer.constraint import variable
from neuromancer.system import Node, System
from neuromancer.loggers import BasicLogger

# Set device and random seed
device = 'cpu'
seed = 136
np.random.seed(seed)
torch.manual_seed(seed)

"""
Load data files
"""

# Raw URL of area.dat
url = "https://raw.githubusercontent.com/pnnl/NeuralDAEs/master/training/area.dat"

# Download and save the file locally
response = requests.get(url)
with open("area.dat", "wb") as f:
    f.write(response.content)

# Raw URL of tanks_splits.dat
url = "https://raw.githubusercontent.com/pnnl/NeuralDAEs/master/training/tanks_splits.dat"

# Download and save the file locally
response = requests.get(url)
with open("tanks_splits.dat", "wb") as f:
    f.write(response.content)


"""
Dataset generation
"""

area_data = np.loadtxt('area.dat')

nx = 4
nu = 1
dt = 1.0

def add_snr(data, db):
    snr = 10 ** (db / 10)
    for i in range(data.shape[1]):
        signal_power = np.mean(data[:, i] ** 2)
        std_n = (signal_power / snr) ** 0.5
        if snr > 1e8:
            continue
        data[:, i] += np.random.normal(0, std_n, len(data[:, i]))
    return data

data = np.float32(np.loadtxt('tanks_splits.dat'))
data = data[:len(data) // 2, :]
data = data[1:, :]
db = 90
data = add_snr(data, db)
time = np.linspace(0.0, len(data)-1, len(data)).reshape(-1, 1).astype(np.float32) * dt
U = np.full_like(time, 0.5, dtype=np.float32)

train_data = {'X': torch.tensor(data),
              'U': torch.tensor(U),
              'Time': torch.tensor(time)}
dev_data = train_data.copy()
test_data = train_data.copy()

nsim = data.shape[0]
nstep = 5

for d in [train_data, dev_data]:
    d['X'] = d['X'].reshape(nsim // nstep, nstep, nx)
    d['U'] = d['U'].reshape(nsim // nstep, nstep, nu)
    d['xn'] = d['X'][:, 0:1, :]
    d['Time'] = d['Time'].reshape(nsim // nstep, -1)

test_data['X'] = test_data['X'].reshape(1, nsim, nx)
test_data['U'] = test_data['U'].reshape(1, nsim, 1)
test_data['xn'] = test_data['X'][:, 0:1, :]
test_data['Time'] = test_data['Time'].reshape(nsim, -1)

train_dataset, dev_dataset = [DictDataset(d, name=n) for d, n in zip([train_data, dev_data], ['train', 'dev'])]
train_loader, dev_loader, test_loader = [
    DataLoader(d, batch_size=nsim // nstep, collate_fn=d.collate_fn, shuffle=True)
    for d in [train_dataset, dev_dataset, dev_dataset]
]

"""
Define DAE system dynamics model

states - x:
h_1 - height of tank 1 - index 0
h_2 - height of tank 2 - index 1
m_1 - mass flow into tank 1 - index 2
m_2 - mass flow into tank 2 - index 3

inputs - u:
m   - mass flow into system
"""

# Define the algebra surrogate
class AlgebraSolver(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, u):
        if x.ndim == 1:
            x = x.unsqueeze(0)  # Ensure batch dimension

        # get differential states
        h1 = x[:, 0].unsqueeze(-1)
        h2 = x[:, 1].unsqueeze(-1)
        # get the mass flow input
        m = u[:, 0].unsqueeze(-1)

        # get the mass flow outputs through the algebra solver
        in_algebra = torch.cat([h1, h2], dim=-1)
        param = torch.abs(self.net(in_algebra))
        m1 = m * param
        m2 = m * (1.0 - param)

        # update algebra states
        x_new = x.clone()
        x_new[:, 2] = m1.squeeze(-1)
        x_new[:, 3] = m2.squeeze(-1)
        return x_new


# define the physics-structured ODE part of the system
class TankManifoldDynamics(ode.ODESystem):
    def __init__(self, net, insize=nx, outsize=nx):
        super().__init__(insize=insize, outsize=outsize)
        self.net = net

    def ode_equations(self, x):
        # get the differential and algebraic states
        h1 = x[:, 0].unsqueeze(-1)
        h2 = x[:, 1].unsqueeze(-1)
        m1 = x[:, 2].unsqueeze(-1)
        m2 = x[:, 3].unsqueeze(-1)
        # get the area
        A1 = 3.0 * torch.ones_like(h1)
        A2 = self.net(h2)
        # compute the differential states
        dh1 = m1 / A1
        dh2 = m2 / A2
        # update differential states
        dx = torch.zeros_like(x)
        dx[:, 0] = dh1.squeeze(-1)
        dx[:, 1] = dh2.squeeze(-1)
        return dx


"""
Instantiate the DAE model
"""
# define neural component of the algebra solver
algebra_net = blocks.MLP(insize=2, outsize=1, hsizes=[3],
                         linear_map=slim.maps['linear'], nonlin=nn.Sigmoid)
# instantiate the algebra solver and wrap it in a Node
algebra = Node(AlgebraSolver(algebra_net), ['xn', 'U'],
               ['x_alg'], name='algebra')

# define neural component of the ODE part
tank_profile = blocks.MLP(insize=1, outsize=1, hsizes=[5],
                          linear_map=slim.maps['linear'], nonlin=nn.Sigmoid)
# Instantiate ODE model
ode_dynamics = TankManifoldDynamics(tank_profile)
fx_int = integrators.RK4(ode_dynamics, h=dt)
ode_node = Node(fx_int, ['x_alg'],
                ['xn'], name='ode_node')

# Compose DAE system via operator splitting approach
dynamics_model = System([algebra, ode_node], nsteps=nstep)

"""
Define the optimization problem
"""
# track differential and algebraic states
x = variable("X")
xhat = variable("xn")[:, :-1, :]
reference_loss = ((xhat == x) ^ 2)
reference_loss.name = "ref_loss"

loss = PenaltyLoss([reference_loss], [])
problem = Problem([dynamics_model], loss)

"""
Define trainer and train the model
"""
optimizer = torch.optim.Adam(problem.parameters(), lr=0.001)
logger = BasicLogger(args=None, savedir='test', verbosity=10,
                     stdout=["dev_loss", "train_loss"])

trainer = Trainer(
    problem, train_loader, dev_loader, test_loader, optimizer,
    epochs=20000, patience=100, warmup=100,
    eval_metric="dev_loss", train_metric="train_loss",
    dev_metric="dev_loss", test_metric="dev_loss",
    logger=logger,
)
trained_model_dict = trainer.train()
torch.save(trained_model_dict, "manifold_" + str(db) + ".pth")

"""
Evaluate and plot the results
"""

# do the full simulation rollout
dynamics_model.nsteps = nsim
# Use model to predict trajectory on test data
test_outputs = dynamics_model(test_data)

# Retrieve predictions and ground truth
pred_traj = test_outputs['xn'][:, :-1, :]  # skip final since it's predicted
true_traj = test_data['X'][:, :-1, :]      # match prediction horizon

# Reshape to [nx, time]
pred_traj = pred_traj.detach().numpy().reshape(-1, nx).T
true_traj = true_traj.detach().numpy().reshape(-1, nx).T

# ---------------------------------------
# Plotting Style
# ---------------------------------------
plt.style.use('default')
plt.rcParams["font.family"] = "serif"
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({
    'font.size': 10,
    'legend.fontsize': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

# ---------------------------------------
# Tank Heights
# ---------------------------------------
plt.figure()
plt.plot(time, pred_traj[0], label="Tank #1")
plt.plot(time, pred_traj[1], label="Tank #2")
plt.plot(time[:-1], data[1:, 0], linestyle='--', label="Data Tank #1")
plt.xlim([0, 500])
plt.ylim([0, 40])
plt.xlabel("Time")
plt.ylabel("Height")
plt.legend()
plt.tight_layout()
plt.savefig(f'tanks_{db}.png')
plt.show()

# ---------------------------------------
# Mass Flows
# ---------------------------------------
plt.figure()
plt.plot(time, pred_traj[2], label="Inflow #1")
plt.plot(time, pred_traj[3], label="Inflow #2")
plt.plot(time, pred_traj[2] + pred_traj[3], label="In_1 + In_2")
plt.plot(time[:-1], data[1:, 2], linestyle='--', label="Data Inflow #1")
plt.plot(time[:-1], data[1:, 3], linestyle='--', label="Data Inflow #2")
plt.xlim([0, 500])
plt.ylim([0, 0.6])
plt.xlabel("Time")
plt.ylabel("Volumetric Flow")
plt.legend()
plt.tight_layout()
plt.savefig(f'flows_{db}.png')
plt.show()

# ---------------------------------------
# Area-Height Relationship
# ---------------------------------------
h = torch.linspace(0, 40, 401).unsqueeze(-1).float()
area = tank_profile(h).detach().numpy()

plt.figure()
plt.plot(h.numpy(), area, label="Model")
plt.plot(h.numpy(), area_data, label="Actual")
plt.ylim([0, 6])
plt.xlim([0, 35])
plt.xlabel("Height")
plt.ylabel("Area(h)")
plt.legend()
plt.tight_layout()
plt.savefig(f'areas_{db}.png')
plt.show()



###############
# Eval on new conditions with sinusoidal input
nsteps = 501
data = np.float32(np.loadtxt('tanks_splits.dat'))
data = data[nsteps:, ]
time = np.float32(np.linspace(0.0, len(data[:, 0]) - 1, len(data[:, 0])).reshape(-1, 1)) * dt
U = time * 0.0 + 0.5 + 0.25 * np.sin(time / 100.0)
U_torch = torch.tensor(U).reshape(1, -1, nu)
# update dictionary dataset
test_data['U'] = U_torch
test_data['X'] = torch.tensor(data).reshape(1, -1, nx)
test_data['xn'] = test_data['X'][:, 0:1, :]  # new initial state

# Use model to predict trajectory on test data
test_outputs = dynamics_model(test_data)

# Retrieve predictions and ground truth
pred_traj = test_outputs['xn'][:, :-1, :]  # skip final since it's predicted
true_traj = test_data['X'][:, :-1, :]      # match prediction horizon

# Reshape to [nx, time]
pred_traj = pred_traj.detach().numpy().reshape(-1, nx).T
true_traj = true_traj.detach().numpy().reshape(-1, nx).T

# ---------------------------------------
# Tank Heights
# ---------------------------------------
plt.figure()
plt.plot(time[:-1], pred_traj[0], label="Tank #1")
plt.plot(time[:-1], pred_traj[1], label="Tank #2")
plt.plot(time[:-1], data[1:, 0], linestyle='--', label="Data Tank #1")
plt.xlim([0, 500])
plt.ylim([0, 40])
plt.xlabel("Time")
plt.ylabel("Height")
plt.legend()
plt.tight_layout()
plt.savefig(f'tanks_{db}.png')
plt.show()

# ---------------------------------------
# Mass Flows
# ---------------------------------------
plt.figure()
plt.plot(time[:-1], pred_traj[2], label="Inflow #1")
plt.plot(time[:-1], pred_traj[3], label="Inflow #2")
plt.plot(time[:-1], pred_traj[2] + pred_traj[3], label="In_1 + In_2")
plt.plot(time[:-1], data[1:, 2], linestyle='--', label="Data Inflow #1")
plt.plot(time[:-1], data[1:, 3], linestyle='--', label="Data Inflow #2")
plt.xlim([0, 500])
plt.ylim([0, 0.8])
plt.xlabel("Time")
plt.ylabel("Volumetric Flow")
plt.legend()
plt.tight_layout()
plt.savefig(f'extrap_flows_{db}.png')
plt.show()

# ---------------------------------------
# Area-Height Relationship
# ---------------------------------------
h = torch.linspace(0, 40, 401).unsqueeze(-1).float()
area = tank_profile(h).detach().numpy()

plt.figure()
plt.plot(h.numpy(), area, label="Model")
plt.plot(h.numpy(), area_data, label="Actual")
plt.ylim([0, 6])
plt.xlim([0, 35])
plt.xlabel("Height")
plt.ylabel("Area(h)")
plt.legend()
plt.tight_layout()
plt.savefig(f'extrap_areas_{db}.png')
plt.show()