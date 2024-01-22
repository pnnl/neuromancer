"""
Neural Ordinary Differentiable predictive control (NO-DPC)

Stabilize underactuated nonlinear ODE system with explicit neural control policy via DPC algorithm

system: Van der Pol oscillator
see section V.A in for system equations: https://arxiv.org/abs/2203.14114
objective: stabilize towards origin

"""

import torch
import torch.nn as nn
import numpy as np

import neuromancer.psl as psl
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.modules.activations import activations
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.dynamics import ode, integrators
from neuromancer.plot import pltCL, pltPhase


if __name__ == "__main__":
    """
    # # #  Ground truth system model
    """
    gt_model = psl.nonautonomous.VanDerPolControl()
    # sampling rate
    ts = gt_model.params[1]['ts']
    # problem dimensions
    nx = gt_model.nx    # number of states
    nu = gt_model.nu    # number of control inputs
    nref = nx           # number of references
    # constraints bounds
    umin = -5.
    umax = 5.
    xmin = -4.
    xmax = 4.

    """
    # # #  Dataset 
    """
    nsteps = 50  # prediction horizon
    n_samples = 2000    # number of sampled scenarios
    # Training dataset generation
    train_data = DictDataset({'x': torch.randn(n_samples, 1, nx),
                              'r': torch.zeros(n_samples, nsteps+1, nx)}, name='train')
    # Development dataset generation
    dev_data = DictDataset({'x': torch.randn(n_samples, 1, nx),
                            'r': torch.zeros(n_samples, nsteps+1, nx)}, name='dev')
    # torch dataloaders
    batch_size = 200
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               collate_fn=train_data.collate_fn, shuffle=False)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size,
                                             collate_fn=dev_data.collate_fn, shuffle=False)

    """
    # # #  System model and Control policy in Neuromancer
    """
    # white-box ODE model with no-plant model mismatch
    vdp_ode = ode.VanDerPolControl()
    vdp_ode.mu = nn.Parameter(torch.tensor(gt_model.mu), requires_grad=False)

    # integrate continuous time ODE
    integrator = integrators.RK4(vdp_ode, h=torch.tensor(ts))
    # symbolic system model
    model = Node(integrator, ['x', 'u'], ['x'], name='model')

    # neural net control policy
    net = blocks.MLP_bounds(insize=nx + nref, outsize=nu, hsizes=[32, 32],
                        nonlin=activations['gelu'], min=umin, max=umax)
    policy = Node(net, ['x', 'r'], ['u'], name='policy')

    # closed-loop system model
    cl_system = System([policy, model], nsteps=nsteps,
                       name='cl_system')
    cl_system.show()

    """
    # # #  Differentiable Predictive Control objectives and constraints
    """
    # state and reference variables
    x = variable('x')
    ref = variable("r")
    # objectives
    regulation_loss = 100. * ((x == ref) ^ 2)  # target posistion
    # state bound constraints
    state_lower_bound_penalty = 10.0*(x > xmin)
    state_upper_bound_penalty = 10.0*(x < xmax)
    # state terminal penalties
    terminal_lower_bound_penalty = 20.0*(x[:, [-1], :] > ref-0.01)
    terminal_upper_bound_penalty = 20.0*(x[:, [-1], :] < ref+0.01)
    # objectives and constraints names for nicer plot
    regulation_loss.name = 'state_loss'
    state_lower_bound_penalty.name = 'x_min'
    state_upper_bound_penalty.name = 'x_max'
    terminal_lower_bound_penalty.name = 'x_N_min'
    terminal_upper_bound_penalty.name = 'x_N_max'
    # list of constraints and objectives
    objectives = [regulation_loss]
    constraints = [
        state_lower_bound_penalty,
        state_upper_bound_penalty,
        terminal_lower_bound_penalty,
        terminal_upper_bound_penalty,
    ]

    """
    # # #  Differentiable optimal control problem 
    """
    # data (x_k, r_k) -> parameters (xi_k) -> policy (u_k) -> dynamics (x_k+1)
    components = [cl_system]
    # create constrained optimization loss
    loss = PenaltyLoss(objectives, constraints)
    # construct constrained optimization problem
    problem = Problem(components, loss)
    # plot computational graph
    problem.show()

    """
    # # #  Solving the problem 
    """
    optimizer = torch.optim.AdamW(problem.parameters(), lr=0.002)
    #  Neuromancer trainer
    trainer = Trainer(
        problem,
        train_loader, dev_loader,
        optimizer,
        epochs=50,
        train_metric='train_loss',
        eval_metric='dev_loss',
        warmup=50,
    )
    # Train control policy
    best_model = trainer.train()
    # load best trained model
    trainer.model.load_state_dict(best_model)

    """
    Test Closed Loop System
    """
    print('\nTest Closed Loop System \n')
    nsteps = 100
    # generate initial data for closed loop simulation
    data = {'x': torch.randn(1, 1, nx, dtype=torch.float32),
            'r': torch.zeros(1, nsteps+1, nx, dtype=torch.float32)}
    cl_system.nsteps = nsteps
    # perform closed-loop simulation
    trajectories = cl_system(data)

    # constraints bounds
    Umin = umin * np.ones([nsteps, 1])
    Umax = umax * np.ones([nsteps, 1])
    Xmin = xmin * np.ones([nsteps+1, 1])
    Xmax = xmax * np.ones([nsteps+1, 1])
    # plot closed loop trajectories
    pltCL(Y=trajectories['x'].detach().reshape(nsteps + 1, 2),
          R=trajectories['r'].detach().reshape(nsteps + 1, 2),
          U=trajectories['u'].detach().reshape(nsteps, 1),
          Umin=Umin, Umax=Umax)
    # plot phase portrait
    pltPhase(X=trajectories['x'].detach().reshape(nsteps + 1, 2))
