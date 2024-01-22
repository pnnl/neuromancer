"""
Modeling and control of a Two Tank system with neural ODEs
and differentiable predictive control (DPC)
"""

import torch
import numpy as np
import neuromancer.slim as slim
import neuromancer.psl as psl
import matplotlib.pyplot as plt

from neuromancer import blocks, estimators, dynamics, integrators, ode
from neuromancer.activations import activations
from neuromancer.trainer import Trainer
from neuromancer.problem import Problem
from neuromancer.dataset import get_sequence_dataloaders
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
import neuromancer.simulator as sim

import argparse
"""
# # #  System Identification via NODE
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cpu"
    system = psl.systems['TwoTank']
    ts = 4.0
    nsim = 10000
    modelSystem = system(ts=ts, nsim=nsim)
    modelSystem.ts = ts
    raw = modelSystem.simulate(ts=ts)
    raw['Y'] = raw['Y'][:-1, :]
    raw['X'] = raw['X'][:-1, :]
    # psl.plot.pltOL(Y=raw['Y'], U=raw['U'])
    # psl.plot.pltPhase(X=raw['Y'])

    # #%% Interpolant for offline mode
    interp_u = lambda tq, t, u: u

    #%%
    #  Train, Development, Test sets - nstep and loop format
    nsteps = 60  # nsteps rollouts in training
    nstep_data, loop_data, dims = get_sequence_dataloaders(raw, nsteps,
                                moving_horizon=True)
    train_data, dev_data, test_data = nstep_data
    train_loop, dev_loop, test_loop = loop_data

    # %% Identity mapping
    nx = dims['X'][1]
    estim = estimators.FullyObservable(
        {**train_data.dataset.dims, "x0": (nx,)},
        linear_map=slim.maps['identity'],
        input_keys=["Yp"],
    )
    estim(train_data.dataset.get_full_batch())

    # %% Instantiate the blocks, dynamics model:
    nu = dims['U'][1]
    black_box_ode = blocks.MLP(insize=nx+nu, outsize=nx, hsizes=[32, 32],
                               linear_map=slim.maps['linear'],
                               nonlin=activations['relu'])
    fx_int = integrators.RK4(black_box_ode, interp_u=interp_u, h=modelSystem.ts)
    fy = slim.maps['identity'](nx, nx)

    dynamics_model = dynamics.ODENonAuto(fx_int, fy, extra_inputs=['Uf'],
                    input_key_map={"x0": estim.output_keys[0],
                                   'Yf': 'Yf', 'Uf': 'Uf'},
                    name='dynamics', online_flag=False)

    # %% Constraints + losses:
    yhat = variable(dynamics_model.output_keys[2])
    y = variable("Yf")
    yFD = (y[:, 1:, :] - y[:, :-1, :])
    yhatFD = (yhat[:, 1:, :] - yhat[:, :-1, :])
    fd_loss = (yFD == yhatFD)^2
    fd_loss.name = 'FD_loss'
    reference_loss = 10*((yhat == y)^2)
    reference_loss.name = "ref_loss"
    Q_con = 10.
    state_lower_bound_penalty = Q_con * (yhat > -0.05)
    state_upper_bound_penalty = Q_con * (yhat < 1.2)
    state_lower_bound_penalty.name = 'y_min'
    state_upper_bound_penalty.name = 'y_max'
    # %%
    objectives = [reference_loss, fd_loss]
    constraints = [state_lower_bound_penalty,
                   state_upper_bound_penalty]
    components = [estim, dynamics_model]
    # create constrained optimization loss
    loss = PenaltyLoss(objectives, constraints)
    # construct constrained optimization problem
    problem = Problem(components, loss)
    # plot computational graph
    plt.figure()
    problem.plot_graph()
    problem = problem.to(device)

    # %%
    optimizer = torch.optim.Adam(problem.parameters(), lr=0.001)
    trainer_sysID = Trainer(
        problem,
        train_data,
        dev_data,
        test_data,
        optimizer,
        patience=100,
        warmup=10,
        epochs=args.epochs,
        eval_metric="nstep_dev_"+reference_loss.output_keys[0],
        train_metric="nstep_train_loss",
        dev_metric="nstep_dev_loss",
        test_metric="nstep_test_loss",
        device=device)
    # %%
    best_model_sysID = trainer_sysID.train()
    # %%
    best_outputs = trainer_sysID.test(best_model_sysID)
    # %%

    """
    Test open loop performance
    """
    nm_system = sim.DynamicsNeuromancer(dynamics_model,
                    name='nm', input_key_map={'x': 'x_nm', 'u': 'U'})
    psl_system = sim.DynamicsPSL(modelSystem,
                    name='psl', input_key_map={'x': 'x_psl', 'u': 'U'})
    psl_system.model.ts = ts
    components = [nm_system, psl_system]
    system_sim = sim.SystemSimulator(components)
    sim_steps = 2000
    modelSystem = system(ts=ts, nsim=sim_steps+1)
    raw = modelSystem.simulate()
    x0 = np.asarray(modelSystem.x0)
    data_init = {'x_psl': x0, 'x_nm': x0}
    U = raw['U'][:sim_steps+1, :]
    data_traj = {'U': U}
    trajectories = system_sim.simulate(nsim=sim_steps, data_init=data_init,
                                       data_traj=data_traj)
    psl.plot.pltOL(Y=trajectories['y_psl'],
                   Ytrain=trajectories['y_nm'], U=trajectories['U'])
    plt.show(block=False)
    plt.interactive(False)

    """
    # # #  Differentiable Predictive Control (DPC)
    """
    # problem dimensions
    nx = modelSystem.nx
    ny = modelSystem.nx
    nu = modelSystem.nu
    nref = nx  # number of references
    # constraints bounds
    umin = 0
    umax = 1.
    xmin = 0
    xmax = 1.

    # prediction horizon
    nsteps = 50
    # number of datapoints
    nsim = 10000
    repeats = int(np.ceil(nsim / nsteps))
    list_refs = [np.random.rand(1, nref) * np.ones([nsteps, 1]) for k in range(repeats)]
    Ref = np.concatenate(list_refs)
    sequences = {
        "R": Ref[:nsim, :],
        "Y": np.random.rand(nsim, nx),
    }
    nstep_data, loop_data, dims = get_sequence_dataloaders(sequences, nsteps, batch_size=32)
    train_data, dev_data, test_data = nstep_data
    train_loop, dev_loop, test_loop = loop_data

    # Fully observable estimator as identity map: x0 = Yp[-1]
    estimator = estimators.FullyObservable(
        {**dims, "x0": (nx,)},
        input_keys=["Yp"], name='est')
    # control policy
    policy = blocks.MLP_bounds(insize=nx + nref, outsize=nu, hsizes=[32, 32],
                               nonlin=activations['gelu'], min=umin, max=umax)
    # trained ODE system
    black_box_ode.requires_grad_(False)  # fix ODE model parameters
    # closed loop control ODE
    control_ode = ode.ControlODE(policy=policy, ode=black_box_ode,
                                 nx=nx, nu=nu, np=nref, u_con=[])

    # control_ode: ControlODE
    # fx_int: integrators.RK4 (Integrator)
    # black_box_ode: blocks.MLP

    # closed loop simulated via ODE integration of ControlODE class
    interp_u = lambda tq, t, u: u  # Interpolant for offline mode
    fx_int = integrators.RK4(control_ode, interp_u=interp_u, h=modelSystem.ts)
    fy = slim.maps['identity'](nx, nx)
    cl_model = dynamics.ODENonAuto(fx_int, fy, extra_inputs=['Rf'],
                                   input_key_map={"x0": estimator.output_keys[0]},
                                   name='closed_loop', online_flag=False)
    # weigt factors
    Qx = 5.
    Q_con_x = 10.
    Qn = 10.
    # variables
    y = variable(cl_model.output_keys[2])
    ref = variable("Rf")
    # objectives
    regulation_loss = Qx * ((y == ref) ^ 2)  # target posistion
    # constraints
    state_lower_bound_penalty = Q_con_x * (y > xmin)
    state_upper_bound_penalty = Q_con_x * (y < xmax)
    terminal_lower_bound_penalty = Qn * (y[:, [-1], :] > ref - 0.01)
    terminal_upper_bound_penalty = Qn * (y[:, [-1], :] < ref + 0.01)
    # objectives and constraints names for nicer plot
    regulation_loss.name = 'state_loss'
    state_lower_bound_penalty.name = 'x_min'
    state_upper_bound_penalty.name = 'x_max'
    terminal_lower_bound_penalty.name = 'y_N_min'
    terminal_upper_bound_penalty.name = 'y_N_max'
    # list of constraints and objectives
    objectives = [regulation_loss]
    constraints = [
        state_lower_bound_penalty,
        state_upper_bound_penalty,
        terminal_lower_bound_penalty,
        terminal_upper_bound_penalty,
    ]
    # data (y_k) -> estimator (x_k) -> policy (u_k) -> dynamics (x_k+1, y_k+1)
    components = [estimator, cl_model]
    # create constrained optimization loss
    loss = PenaltyLoss(objectives, constraints)
    # construct constrained optimization problem
    problem = Problem(components, loss)
    # plot computational graph
    plt.figure()
    problem.plot_graph()

    # device and optimizer
    device = "cpu"
    problem = problem.to(device)
    optimizer = torch.optim.AdamW(problem.parameters(), lr=0.001)
    # trainer
    cl_trainer = Trainer(
        problem,
        train_data,
        dev_data,
        test_data,
        optimizer,
        epochs=args.epochs,
        patience=100,
        train_metric="nstep_train_loss",
        dev_metric="nstep_dev_loss",
        test_metric="nstep_test_loss",
        eval_metric='nstep_dev_loss',
        warmup=10,
    )
    # Train control policy
    best_model_cl = cl_trainer.train()
    best_outputs = cl_trainer.test(best_model_cl)

    """
    Test Closed Loop System
    """
    print('\nTest Closed Loop System with PSL model \n')
    np.random.seed(20)
    sim_steps = 750
    # construct system simulator
    policy_cl = sim.ControllerPytorch(policy=cl_model.fx.block.policy,
                                      input_keys=['x', 'r'])
    system_psl = sim.DynamicsPSL(modelSystem, input_key_map={'x': 'x', 'u': 'u'})
    policy_cl.name = 'policy'
    system_psl.name = 'dynamics'
    components = [policy_cl, system_psl]
    cl_sim = sim.SystemSimulator(components)
    # plot system simulator computational graph
    plt.figure()
    cl_sim.plot_graph()
    # generate initial data for closed loop simulation
    data_init = {'x': np.random.rand(nx)}
    Ref_list = []
    step_length = 150
    for k in range(int(np.ceil(sim_steps / step_length))):
        Ref_list.append(np.random.rand() * np.ones([step_length + 1, nref]))
    R = np.concatenate(Ref_list)[:sim_steps + 1, :]
    data_traj = {'r': R}
    # simulate closed loop
    trajectories = cl_sim.simulate(nsim=sim_steps, data_init=data_init,
                                   data_traj=data_traj)
    # plot closed loop trajectories
    Umin = umin * np.ones([trajectories['y'].shape[0], 1])
    Umax = umax * np.ones([trajectories['y'].shape[0], 1])
    Ymin = xmin * np.ones([trajectories['y'].shape[0], 1])
    Ymax = xmax * np.ones([trajectories['y'].shape[0], 1])
    psl.plot.pltCL(Y=trajectories['y'], U=trajectories['u'], R=R,
                   Umin=Umin, Umax=Umax, Ymin=Ymin, Ymax=Ymax)
