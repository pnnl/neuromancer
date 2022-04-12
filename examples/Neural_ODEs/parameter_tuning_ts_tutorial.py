# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import slim
import psl

from neuromancer import blocks, estimators, dynamics, arg, integrators, ode
from neuromancer.activations import activations
from neuromancer.visuals import VisualizerOpen
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.simulators import OpenLoopSimulator, MultiSequenceOpenLoopSimulator
from neuromancer.callbacks import SysIDCallback
from neuromancer.loggers import BasicLogger, MLFlowLogger
from neuromancer.dataset import get_sequence_dataloaders
from neuromancer.constraint import Variable
from neuromancer.loss import PenaltyLoss

torch.manual_seed(0)
# %%
device = "cpu"

# %%
"""

Get some data from the L-V system for prototyping.

"""
system = psl.systems['Brusselator1D']

modelSystem = system()
raw = modelSystem.simulate(ts=0.05)
psl.plot.pltOL(Y=raw['Y'])
psl.plot.pltPhase(X=raw['Y'])

#  Train, Development, Test sets - nstep and loop format
nsteps = 1
nstep_data, loop_data, dims = get_sequence_dataloaders(raw, nsteps, moving_horizon=False)
train_data, dev_data, test_data = nstep_data
train_loop, dev_loop, test_loop = loop_data

# %% Identity mapping
nx = 2
estim = estimators.FullyObservable(
    {**train_data.dataset.dims, "x0": (nx,)},
    linear_map=slim.maps['identity'],
    input_keys=["Yp"],
)

# %% Instantiate the blocks, dynamics model:

"""
The below definition can be found in ode.py. This file contains user-defined gray box
systems as a subclass of ODESystem(). 

class HybridBrusselatorMLP(ODESystem):
    def __init__(self, insize=2, outsize=2):
        super().__init__(insize=insize, outsize=outsize)
        self.alpha = nn.Parameter(torch.tensor([5.0]), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor([5.0]), requires_grad=True)
    
    def ode_equations(self, x): 

        x1 = x[:,[0]]
        x2 = x[:,[-1]]

        dx1 = self.alpha + x2*x1**2 - self.beta*x1 - x1
        dx2 = self.beta*x1 - x2*x1**2

        dx = torch.cat([dx1, dx2],dim=-1)

        return dx
"""

"""
class parameter_tune_sys(ode.ODESystem):
    def __init__(self, insize=2, outsize=2):
        super().__init__(insize=insize, outsize=outsize)
        self.alpha = torch.nn.Parameter(torch.tensor([5.0]), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.tensor([5.0]), requires_grad=True)
    
    def ode_equations(self, x): 

        x1 = x[:,[0]]
        x2 = x[:,[-1]]

        dx1 = self.alpha + x2*x1**2 - self.beta*x1 - x1
        dx2 = self.beta*x1 - x2*x1**2

        dx = torch.cat([dx1, dx2],dim=-1)

        return dx

brussels = parameter_tune_sys()
"""

brussels = ode.BrusselatorParam()
fxRK4 = integrators.RK4(brussels, h=0.05)
fy = slim.maps['identity'](nx, nx)
dynamics_model = dynamics.ODEAuto(fxRK4, fy, name='dynamics',
                                  input_key_map={"x0": f"x0_{estim.name}"})

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
objectives = [reference_loss, fd_loss]
constraints = []
components = [estim, dynamics_model]
# create constrained optimization loss
loss = PenaltyLoss(objectives, constraints, batch_second=True)
# construct constrained optimization problem
problem = Problem(components, loss)
# plot computational graph
#problem.plot_graph()
problem = problem.to(device)

# %%
optimizer = torch.optim.Adam(problem.parameters(), lr=0.1)
logger = BasicLogger(args=None, savedir='test', verbosity=1,
                     stdout="nstep_dev_"+reference_loss.output_keys[0])

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
    epochs=350,
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
print('alpha = '+str(brussels.alpha.item()))
print('beta = '+str(brussels.beta.item()))

# %%
