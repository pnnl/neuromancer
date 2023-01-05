"""
Parameter estimation of Duffing ODE system using SGD
"""

import torch
import numpy as np
import slim
import psl
import matplotlib.pyplot as plt

from neuromancer import blocks, estimators, dynamics, integrators, ode
from neuromancer.interpolation import LinInterp_Offline
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.loggers import BasicLogger
from neuromancer.constraint import variable
from neuromancer.dataset import get_sequence_dataloaders
from neuromancer.loss import PenaltyLoss
import neuromancer.simulator as sim

# %%
torch.manual_seed(0)
device = "cpu"

# %% load the ground truth dataset by simulating ODE system
system = psl.systems['Duffing']
ts = 0.01
nsim = 1000
modelSystem = system(ts=ts, nsim=nsim)
raw = modelSystem.simulate()
psl.plot.pltOL(Y=raw['Y'])
psl.plot.pltPhase(X=raw['Y'])

# %% interpolate time for the ODE integrator of non-autonomous system
t = (np.arange(nsim+1)*ts).reshape(-1, 1)
raw['Time'] = t
t = torch.from_numpy(t)
interp_u = LinInterp_Offline(t, t)

# %% create sequence dataloader with given prediction horizon
#   check for dataset keys: train_data.dataset.get_full_batch().keys()
nsteps = 2
nstep_data, loop_data, dims = get_sequence_dataloaders(raw, nsteps,
                                moving_horizon=True)
train_data, dev_data, test_data = nstep_data    # (nstep, # batches, sys dim)
train_loop, dev_loop, test_loop = loop_data

# %% fully observable state estimator as identity mapping
nx = dims['X'][1]
estim = estimators.FullyObservable(
    {**train_data.dataset.dims, "x0": (nx,)},
    linear_map=slim.maps['identity'],
    input_keys=["Yp"])
# test forward pass of the estimator
estim(train_data.dataset.get_full_batch())

# %% Instantiate the dynamics model
duffing_sys = ode.DuffingParam()   # ODE RHS
# duffing_sys = blocks.MLP(insize=nx+1, outsize=nx, hsizes=[32, 32])  # MLP RHS
fx_int = integrators.RK4(duffing_sys, interp_u=interp_u, h=modelSystem.ts)  # ODE integrator
fy = slim.maps['identity'](nx, nx)  # output model
# component class mapping symbolic variables
dynamics_model = dynamics.ODENonAuto(fx_int, fy, 
input_key_map={"x0": estim.output_keys[0], 'Time': 'Timef'},
               extra_inputs=['Time'], name='dynamics', online_flag=False)

# %% symbolic NeuroMANCER variables
yhat = variable(dynamics_model.output_keys[2])
y = variable("Yf")
# composite variables
yFD = (y[:, 1:, :] - y[:, :-1, :])
yhatFD = (yhat[:, 1:, :] - yhat[:, :-1, :])

# %% NeuroMANCER constraints and objective terms
fd_loss = 2.0*((yFD == yhatFD)^2)
fd_loss.name = 'FD_loss'
reference_loss = ((yhat == y)^2)
reference_loss.name = "ref_loss"

# %% NeuroMANCER differentiable problem
objectives = [reference_loss, fd_loss]
constraints = []
components = [estim, dynamics_model]
# create constrained optimization loss
loss = PenaltyLoss(objectives, constraints)
# construct constrained parametric optimization problem
problem = Problem(components, loss)
# plot computational graph
problem.plot_graph()
problem = problem.to(device)

# %% trainer class
optimizer = torch.optim.Adam(problem.parameters(), lr=0.1)
logger = BasicLogger(args=None, savedir='test', verbosity=1,
                     stdout="nstep_dev_"+reference_loss.output_keys[0])
trainer = Trainer(
    problem,
    train_data,
    dev_data,
    test_data,
    optimizer,
    patience=20,
    warmup=100,
    epochs=200,
    eval_metric="nstep_dev_"+reference_loss.output_keys[0],
    train_metric="nstep_train_loss",
    dev_metric="nstep_dev_loss",
    test_metric="nstep_test_loss",
    logger=logger,
    device=device,
)
best_model = trainer.train()
best_outputs = trainer.test(best_model)

"""
Test open loop performance
"""
# NeuroMANCER sim.classes act as wrappers for arbitrary callables
#   to be jointly simulated via sim.SystemSimulator
#   this setup facilitates benchmarking and pipelines with heterogenous models
sim_steps = 400
nm_system = sim.DynamicsNeuromancer(dynamics_model,
                name='nm', input_key_map={'x': 'x_nm', 'Time': 'Time'})
psl_system = sim.DynamicsPSL(modelSystem,
                name='psl', input_key_map={'x': 'x_psl', 'Time': 'Time'})
psl_system.model.ts = ts
components = [nm_system, psl_system]
system = sim.SystemSimulator(components)
system.plot_graph()
x0 = np.asarray(modelSystem.x0)
data_init = {'x_psl': x0, 'x_nm': x0}
Time = raw['Time'][:sim_steps+1, :]
data_traj = {'Time': Time}
trajectories = system.simulate(nsim=sim_steps, data_init=data_init, data_traj=data_traj)
plt.close('all')
psl.plot.pltOL(Y=trajectories['y_psl'], Ytrain=trajectories['y_nm'])
plt.show(block=True)
plt.interactive(False)

