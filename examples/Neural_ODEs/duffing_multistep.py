# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import slim
import psl

from neuromancer import blocks, dynamics, arg, integrators, ode, estimators
from neuromancer.activations import activations
from neuromancer.visuals import VisualizerOpen
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.simulators import OpenLoopSimulator, MultiSequenceOpenLoopSimulator
from neuromancer.callbacks import SysIDCallback
from neuromancer.loggers import BasicLogger, MLFlowLogger
from neuromancer.dataset import get_sequence_dataloaders_multistep
from neuromancer.constraint import Variable
from neuromancer.loss import PenaltyLoss, BarrierLoss
from neuromancer.interpolation import LinInterp_Offline

import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)

# %%
device = "cpu"

# %%
def get_loss(objectives, constraints, type):
    if type == 'penalty':
        loss = PenaltyLoss(objectives, constraints)
    elif type == 'barrier':
        loss = BarrierLoss(objectives, constraints)
    return loss

# %%
"""

Get some data from the L-V system for prototyping.

"""
system = psl.systems['Duffing'] # non-autonomous system

ts = 0.01
nsim = 1000
modelSystem = system(ts=ts, nsim=nsim)
raw = modelSystem.simulate()
psl.plot.pltOL(Y=raw['Y'])
psl.plot.pltPhase(X=raw['Y'])

t = (np.arange(nsim)*ts).reshape(-1, 1)
raw['Time'] = t

t = torch.from_numpy(t)
interp_u = LinInterp_Offline(t, t)

#  Train, Development, Test sets - nstep and loop format
nsteps_rollouts = 1 # # of rollouts
input_window_size = 4
output_window_size = 1
nsteps_p = input_window_size + nsteps_rollouts - 1 # Xp and Xf have different nsteps
nsteps_f = output_window_size + nsteps_rollouts - 1
nstep_data, loop_data, dims = get_sequence_dataloaders_multistep(raw, nsteps_p, nsteps_f, moving_horizon=True)
train_data, dev_data, test_data = nstep_data
train_loop, dev_loop, test_loop = loop_data

# %% Identity mapping
nx = dims['X'][1]
estim = estimators.FullyObservable_MultiStep(
    {**train_data.dataset.dims, "x0": (nx,)},
    linear_map = slim.maps['identity'],
    input_keys = ["Yp"],
    nsteps = nsteps_p,
    window_size = input_window_size
)

estim(train_data.dataset.get_full_batch())

# %% Instantiate the blocks, dynamics model:
duffing_sys = ode.DuffingParam()
         
fx = integrators.MultiStep_PredictorCorrector(duffing_sys,h=ts, interp_u=interp_u)

fy = slim.maps['identity'](nx,nx)

dynamics_model = dynamics.ODENonAuto_MultiStep(fx, fy,
 input_key_map={"x0": f"x0_{estim.name}", "Time": "Timef", 'Yf': 'Yf'})

# %% Constraints + losses:
yhat = Variable(f"Y_pred_{dynamics_model.name}")
y = Variable("Yf")
x0 = Variable(f"x0_{estim.name}")
xhat = Variable(f"X_pred_{dynamics_model.name}")

yFD = (y[1:] - y[:-1])
yhatFD = (yhat[1:] - yhat[:-1])

fd_loss = 2.0*((yFD == yhatFD)^2)
fd_loss.name = 'FD_loss'

reference_loss = ((yhat == y)^2)
reference_loss.name = "ref_loss"

# %%
objectives = [reference_loss,fd_loss]
constraints = []
components = [estim, dynamics_model]
# create constrained optimization loss
loss = get_loss(objectives, constraints, 'penalty')
# construct constrained optimization problem
problem = Problem(components, loss)
# plot computational graph
#problem.plot_graph()
problem = problem.to(device)

# %%
optimizer = torch.optim.Adam(problem.parameters(), lr=0.1)
logger = BasicLogger(args=None, savedir= 'test', verbosity=1, stdout="nstep_dev_"+reference_loss.output_keys[0])

simulator = OpenLoopSimulator(
    problem, train_loop, dev_loop, test_loop, eval_sim=True, device=device,
) if isinstance(train_loop, dict) else MultiSequenceOpenLoopSimulator(
    problem, train_loop, dev_loop, test_loop, eval_sim=True, device=device,
)
visualizer = VisualizerOpen(
    dynamics_model,
    1,
    'test',
    training_visuals=False,
    trace_movie=False,
)
callback = SysIDCallback(simulator, visualizer)


trainer = Trainer(
    problem,
    train_data,
    dev_data,
    test_data,
    optimizer,
    callback=callback,
    patience=10,
    warmup=10,
    epochs=1,
    eval_metric="nstep_dev_"+reference_loss.output_keys[0],
    train_metric="nstep_train_loss",
    dev_metric="nstep_dev_loss",
    test_metric="nstep_test_loss",
    logger=logger,
    device=device,
)

# %%
best_model = trainer.train()
# %%
best_outputs = trainer.test(best_model)

# %%
