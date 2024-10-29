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
from neuromancer.rl.gym_nssm import NSSMTrainer
from neuromancer.rl.gym_dpc import DPCTrainer  # Import the DPCTrainer class

# Step 1: Define the physical system model using ODEs
env = BuildingEnv(simulator='SimpleSingleZone')
sys = env.model

# Step 2: Collect data
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
dynamics_model = nssm_trainer.train(nsim=2000, nsteps=2, niters=1)

# Step 4: Pre-train the policy network using DPC
dpc_trainer = DPCTrainer(env, batch_size=100, epochs=200)
best_model = dpc_trainer.train(nsim=2000, nsteps=2, niters=5)

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
dpc_trainer.cl_system.nsteps = nsteps_test
trajectories = dpc_trainer.cl_system(data)

Umin = dpc_trainer.env.model.umin * np.ones([nsteps_test, sys.nu])
Umax = dpc_trainer.env.model.umax * np.ones([nsteps_test, sys.nu])
Ymin = trajectories['ymin'].detach().reshape(nsteps_test + 1, sys.ny)
Ymax = trajectories['ymax'].detach().reshape(nsteps_test + 1, sys.ny)

pltCL(Y=trajectories['y'].detach().reshape(nsteps_test + 1, sys.ny), R=Ymax, X=trajectories['x'].detach().reshape(nsteps_test + 1, sys.nx), D=trajectories['d'].detach().reshape(nsteps_test + 1, sys.nd), U=trajectories['u'].detach().reshape(nsteps_test, sys.nu), Umin=Umin, Umax=Umax, Ymin=Ymin, Ymax=Ymax)