import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from neuromancer.dataset import DictDataset
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.psl.gym import BuildingEnv
from neuromancer.psl.signals import step as step_signal
from neuromancer.plot import pltCL
from ppo import Agent, Args, run as ppo_run
from nssm import NSSMTrainer

# Step 1: Define the physical system model using ODEs
env = BuildingEnv(simulator='SimpleSingleZone')
sys = env.model
nsim = 8000
sim = sys.simulate(nsim=nsim, x0=sys.get_x0(), U=sys.get_U(nsim + 1))

# Step 2: Collect real-world and simulated data
nsteps = 100
n_samples = 1000
x_min = 18.0
x_max = 22.0

list_xmin = [x_min + (x_max - x_min) * torch.rand(1, 1) * torch.ones(nsteps + 1, sys.ny) for _ in range(n_samples)]
xmin = torch.cat(list_xmin)
batched_xmin = xmin.reshape([n_samples, nsteps + 1, sys.ny])
batched_xmax = batched_xmin + 2.0

list_dist = [torch.tensor(sys.get_D(nsteps)) for _ in range(n_samples)]
batched_dist = torch.stack(list_dist, dim=0)

list_x0 = [torch.tensor(sys.get_x0().reshape(1, sys.nx)) for _ in range(n_samples)]
batched_x0 = torch.stack(list_x0, dim=0)

train_data = DictDataset({'x': batched_x0, 'y': batched_x0[:, :, [3]], 'ymin': batched_xmin, 'ymax': batched_xmax, 'd': batched_dist}, name='train')
dev_data = DictDataset({'x': batched_x0, 'y': batched_x0[:, :, [3]], 'ymin': batched_xmin, 'ymax': batched_xmax, 'd': batched_dist}, name='dev')

batch_size = 100
train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=train_data.collate_fn, shuffle=False)
dev_loader = DataLoader(dev_data, batch_size=batch_size, collate_fn=dev_data.collate_fn, shuffle=False)

# Step 3: Train the Neural State Space Model (NSSM)
nssm_trainer = NSSMTrainer(env, batch_size=100, epochs=1)
dynamics_model = nssm_trainer.train(nsim=2000, nsteps=2)

# Step 4: Pre-train the policy network using DPC
A = torch.tensor(sys.params[2]['A'])
B = torch.tensor(sys.params[2]['Beta'])
C = torch.tensor(sys.params[2]['C'])
E = torch.tensor(sys.params[2]['E'])
umin = torch.tensor(sys.umin)
umax = torch.tensor(sys.umax)

xnext = lambda x, u, d: x @ A.T + u @ B.T + d @ E.T
state_model = Node(xnext, ['x', 'u', 'd'], ['x'], name='SSM')
ynext = lambda x: x @ C.T
output_model = Node(ynext, ['x'], ['y'], name='y=Cx')

dist_model = lambda d: d[:, sys.d_idx]
dist_obsv = Node(dist_model, ['d'], ['d_obsv'], name='dist_obsv')

net = blocks.MLP_bounds(insize=sys.ny + 2 * sys.ny + sys.nd, outsize=sys.nu, hsizes=[32, 32], nonlin=nn.GELU, min=umin, max=umax)
policy = Node(net, ['y', 'ymin', 'ymax', 'd_obsv'], ['u'], name='policy')

cl_system = System([dist_obsv, policy, state_model, output_model], nsteps=nsteps, name='cl_system')

y = variable('y')
u = variable('u')
ymin = variable('ymin')
ymax = variable('ymax')

action_loss = 0.01 * (u == 0.0)
du_loss = 0.1 * (u[:, :-1, :] - u[:, 1:, :] == 0.0)
state_lower_bound_penalty = 50.0 * (y > ymin)
state_upper_bound_penalty = 50.0 * (y < ymax)

objectives = [action_loss, du_loss]
constraints = [state_lower_bound_penalty, state_upper_bound_penalty]

loss = PenaltyLoss(objectives, constraints)
problem = Problem([cl_system], loss)

epochs = 200
optimizer = torch.optim.AdamW(problem.parameters(), lr=0.001)
trainer = Trainer(problem, train_loader, dev_loader, optimizer=optimizer, epochs=epochs, train_metric='train_loss', eval_metric='dev_loss', warmup=epochs)
best_model = trainer.train()
trainer.model.load_state_dict(best_model)

# Step 5: Train the policy network using DRL
args = Args(
    env_id='SimpleSingleZone',
    total_timesteps=1000000,
    learning_rate=3e-4,
    num_envs=1,
    num_steps=2048,
    anneal_lr=True,
    gamma=0.99,
    gae_lambda=0.95,
    num_minibatches=32,
    update_epochs=10,
    norm_adv=True,
    clip_coef=0.2,
    clip_vloss=True,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=None
)

ppo_run()

# Step 6: Test the hybrid control system
nsteps_test = 2000
np_refs = step_signal(nsteps_test + 1, 1, min=x_min, max=x_max, randsteps=5)
ymin_val = torch.tensor(np_refs, dtype=torch.float32).reshape(1, nsteps_test + 1, 1)
ymax_val = ymin_val + 2.0
torch_dist = torch.tensor(sys.get_D(nsteps_test + 1)).unsqueeze(0)
x0 = torch.tensor(sys.get_x0()).reshape(1, 1, sys.nx)
data = {'x': x0, 'y': x0[:, :, [3]], 'ymin': ymin_val, 'ymax': ymax_val, 'd': torch_dist}
cl_system.nsteps = nsteps_test
trajectories = cl_system(data)

Umin = umin * np.ones([nsteps_test, sys.nu])
Umax = umax * np.ones([nsteps_test, sys.nu])
Ymin = trajectories['ymin'].detach().reshape(nsteps_test + 1, sys.ny)
Ymax = trajectories['ymax'].detach().reshape(nsteps_test + 1, sys.ny)

pltCL(Y=trajectories['y'].detach().reshape(nsteps_test + 1, sys.ny), R=Ymax, X=trajectories['x'].detach().reshape(nsteps_test + 1, sys.nx), D=trajectories['d'].detach().reshape(nsteps_test + 1, sys.nd), U=trajectories['u'].detach().reshape(nsteps_test, sys.nu), Umin=Umin, Umax=Umax, Ymin=Ymin, Ymax=Ymax)