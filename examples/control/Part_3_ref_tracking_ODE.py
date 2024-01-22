"""
Neural Ordinary Differentiable predictive control (NO-DPC)

Reference tracking of nonlinear ODE system with explicit neural control policy via DPC algorithm

system: Two Tank model
example inspired by: https://apmonitor.com/do/index.php/Main/LevelControl
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
    gt_model = psl.nonautonomous.TwoTank()
    # sampling rate
    ts = gt_model.params[1]['ts']
    # problem dimensions
    nx = gt_model.nx    # number of states
    nu = gt_model.nu    # number of control inputs
    nref = nx           # number of references
    # constraints bounds
    umin = 0
    umax = 1.
    xmin = 0
    xmax = 1.

    """
    # # #  Dataset 
    """
    nsteps = 50  # prediction horizon
    n_samples = 2000    # number of sampled scenarios

    #  sampled references for training the policy
    list_refs = [torch.rand(1, 1)*torch.ones(nsteps+1, nref) for k in range(n_samples)]
    ref = torch.cat(list_refs)
    batched_ref = ref.reshape([n_samples, nsteps+1, nref])
    # Training dataset
    train_data = DictDataset({'x': torch.rand(n_samples, 1, nx),
                              'r': batched_ref}, name='train')

    # references for dev set
    list_refs = [torch.rand(1, 1)*torch.ones(nsteps+1, nref) for k in range(n_samples)]
    ref = torch.cat(list_refs)
    batched_ref = ref.reshape([n_samples, nsteps+1, nref])
    # Development dataset
    dev_data = DictDataset({'x': torch.rand(n_samples, 1, nx),
                            'r': batched_ref}, name='dev')

    # torch dataloaders
    batch_size = 200
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               collate_fn=train_data.collate_fn,
                                               shuffle=False)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=batch_size,
                                             collate_fn=dev_data.collate_fn,
                                             shuffle=False)

    """
    # # #  System model and Control policy in Neuromancer
    """
    # white-box ODE model with no-plant model mismatch
    two_tank_ode = ode.TwoTankParam()
    two_tank_ode.c1 = nn.Parameter(torch.tensor(gt_model.c1), requires_grad=False)
    two_tank_ode.c2 = nn.Parameter(torch.tensor(gt_model.c2), requires_grad=False)

    # integrate continuous time ODE
    integrator = integrators.RK4(two_tank_ode, h=torch.tensor(ts))  # using 4th order runge kutta integrator
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
    # variables
    x = variable('x')
    ref = variable("r")
    # objectives
    regulation_loss = 5. * ((x == ref) ^ 2)  # target posistion
    # constraints
    state_lower_bound_penalty = 10.*(x > xmin)
    state_upper_bound_penalty = 10.*(x < xmax)
    terminal_lower_bound_penalty = 10.*(x[:, [-1], :] > ref-0.01)
    terminal_upper_bound_penalty = 10.*(x[:, [-1], :] < ref+0.01)
    # objectives and constraints names for nicer plot
    regulation_loss.name = 'ref_tracking'
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
    nodes = [cl_system]
    # create constrained optimization loss
    loss = PenaltyLoss(objectives, constraints)
    # construct constrained optimization problem
    problem = Problem(nodes, loss)
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
        optimizer=optimizer,
        epochs=100,
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
    nsteps = 750
    step_length = 150
    # generate reference
    np_refs = psl.signals.step(nsteps+1, 1, min=xmin, max=xmax, randsteps=5)
    R = torch.tensor(np_refs, dtype=torch.float32).reshape(1, nsteps+1, 1)
    torch_ref = torch.cat([R, R], dim=-1)
    # generate initial data for closed loop simulation
    data = {'x': torch.rand(1, 1, nx, dtype=torch.float32),
            'r': torch_ref}
    cl_system.nsteps = nsteps
    # perform closed-loop simulation
    trajectories = cl_system(data)

    # constraints bounds
    Umin = umin * np.ones([nsteps, nu])
    Umax = umax * np.ones([nsteps, nu])
    Xmin = xmin * np.ones([nsteps+1, nx])
    Xmax = xmax * np.ones([nsteps+1, nx])
    # plot closed loop trajectories
    pltCL(Y=trajectories['x'].detach().reshape(nsteps + 1, nx),
          R=trajectories['r'].detach().reshape(nsteps + 1, nref),
          U=trajectories['u'].detach().reshape(nsteps, nu),
          Umin=Umin, Umax=Umax, Ymin=Xmin, Ymax=Xmax)
    # plot phase portrait
    pltPhase(X=trajectories['x'].detach().reshape(nsteps + 1, nx))
